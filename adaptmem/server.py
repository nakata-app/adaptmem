"""HTTP daemon — long-lived adaptmem process for cross-language consumers.

Run:
    pip install "adaptmem[server]"
    adaptmem serve --port 7800 --base-model all-MiniLM-L6-v2

Endpoint contract is documented in `docs/metis_integration.md`.

Design choices:
- A single corpus index lives in memory per `corpus_id`. Re-indexing is
  cheap (a `/reindex` call replaces the matrix).
- No persistence by default. Caller (metis, claimcheck, ...) owns the
  source markdown / database; daemon is a query cache.
- One global encoder shared across corpora — base_model swap requires
  daemon restart.
"""
from __future__ import annotations

import importlib.util
import time
from typing import Any

from adaptmem.core import AdaptMem


_STARTED_AT = time.time()


# ---- Pydantic schemas (top-level so FastAPI introspection works) ----------

if importlib.util.find_spec("pydantic") is not None:
    from pydantic import BaseModel

    class EmbedRequest(BaseModel):
        texts: list[str]

    class EmbedResponse(BaseModel):
        embeddings: list[list[float]]
        model: str
        dim: int

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 5
        corpus_id: str = "default"

    class SearchHit(BaseModel):
        id: str
        text: str
        score: float

    class SearchResponse(BaseModel):
        hits: list[SearchHit]
        model: str
        elapsed_ms: float

    class ReindexDoc(BaseModel):
        id: str
        text: str

    class ReindexRequest(BaseModel):
        corpus_id: str = "default"
        documents: list[ReindexDoc]

    class ReindexResponse(BaseModel):
        corpus_id: str
        n_docs: int
        elapsed_ms: float

    class HealthzResponse(BaseModel):
        ok: bool
        uptime_s: float

    class VersionResponse(BaseModel):
        adaptmem: str
        encoder: str
        corpora: list[str]


def _build_app() -> Any:
    """Build the FastAPI app + return (app, state). Imports gated to keep `[server]` optional."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as e:  # pragma: no cover — exercised only without `[server]`
        raise SystemExit(
            "adaptmem.server requires the [server] extras. Run\n"
            "    pip install \"adaptmem[server]\""
        ) from e

    app = FastAPI(title="adaptmem", description="Domain-tuned retrieval daemon")

    state: dict[str, Any] = {
        "encoder_name": None,
        "engine": None,        # AdaptMem instance (shared encoder)
        "corpora": {},         # corpus_id -> AdaptMem
        "device": None,
        # Prometheus-style counters: endpoint -> {"count": int, "duration_s_sum": float}
        "metrics": {},
    }

    def _record_metric(endpoint: str, duration_s: float) -> None:
        m = state["metrics"].setdefault(endpoint, {"count": 0, "duration_s_sum": 0.0})
        m["count"] += 1
        m["duration_s_sum"] += duration_s

    def _ensure_encoder() -> AdaptMem:
        """Return a (lazy) AdaptMem with a loaded base encoder."""
        if state["engine"] is None:
            from sentence_transformers import SentenceTransformer

            new_engine = AdaptMem(base_model=state["encoder_name"], device=state.get("device"))
            new_engine._model = SentenceTransformer(state["encoder_name"], device=state.get("device") or None)
            state["engine"] = new_engine
        result: AdaptMem = state["engine"]
        return result

    @app.middleware("http")
    async def _track_request(request: Any, call_next: Any) -> Any:
        # Skip the metrics endpoint itself so scrapes don't inflate counters.
        if request.url.path == "/metrics":
            return await call_next(request)
        t0 = time.perf_counter()
        response = await call_next(request)
        _record_metric(request.url.path, time.perf_counter() - t0)
        return response

    @app.get("/metrics")
    def metrics() -> Any:
        """Prometheus text format. No external dep — hand-rolled.

        Lines look like:
            adaptmem_request_total{endpoint="/embed"} 17
            adaptmem_request_duration_seconds_sum{endpoint="/embed"} 0.412
            adaptmem_uptime_seconds 123.4
        """
        from fastapi.responses import PlainTextResponse

        lines = [
            "# HELP adaptmem_uptime_seconds Daemon process uptime in seconds.",
            "# TYPE adaptmem_uptime_seconds gauge",
            f"adaptmem_uptime_seconds {time.time() - _STARTED_AT:.3f}",
            "# HELP adaptmem_corpora_total Number of indexed corpora.",
            "# TYPE adaptmem_corpora_total gauge",
            f"adaptmem_corpora_total {len(state['corpora'])}",
            "# HELP adaptmem_request_total Total requests by endpoint.",
            "# TYPE adaptmem_request_total counter",
            "# HELP adaptmem_request_duration_seconds_sum Cumulative request duration by endpoint.",
            "# TYPE adaptmem_request_duration_seconds_sum counter",
        ]
        for endpoint, m in sorted(state["metrics"].items()):
            label = endpoint.replace('"', '\\"')
            lines.append(f'adaptmem_request_total{{endpoint="{label}"}} {m["count"]}')
            lines.append(
                f'adaptmem_request_duration_seconds_sum{{endpoint="{label}"}} {m["duration_s_sum"]:.6f}'
            )
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.get("/healthz", response_model=HealthzResponse)
    def healthz() -> HealthzResponse:
        return HealthzResponse(ok=True, uptime_s=round(time.time() - _STARTED_AT, 2))

    @app.get("/version", response_model=VersionResponse)
    def version() -> VersionResponse:
        from adaptmem import __version__

        return VersionResponse(
            adaptmem=__version__,
            encoder=state["encoder_name"] or "uninitialised",
            corpora=sorted(state["corpora"].keys()),
        )

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest) -> EmbedResponse:
        engine = _ensure_encoder()
        assert engine._model is not None  # _ensure_encoder post-condition
        embeddings = engine._model.encode(
            req.texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model=state["encoder_name"],
            dim=int(embeddings.shape[1]),
        )

    @app.post("/reindex", response_model=ReindexResponse)
    def reindex(req: ReindexRequest) -> ReindexResponse:
        from adaptmem.miner import CorpusEntry
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        engine = AdaptMem(base_model=state["encoder_name"], device=state.get("device"))
        model = SentenceTransformer(state["encoder_name"], device=state.get("device") or None)
        engine._model = model
        engine._corpus = [CorpusEntry(id=d.id, text=d.text) for d in req.documents]
        engine._embeddings = model.encode(
            [d.text for d in req.documents],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        state["corpora"][req.corpus_id] = engine
        if state["engine"] is None:
            state["engine"] = engine
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return ReindexResponse(
            corpus_id=req.corpus_id, n_docs=len(req.documents), elapsed_ms=elapsed_ms
        )

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        if req.corpus_id not in state["corpora"]:
            raise HTTPException(
                status_code=404,
                detail=f"corpus '{req.corpus_id}' not indexed — POST /reindex first",
            )
        engine: AdaptMem = state["corpora"][req.corpus_id]
        t0 = time.perf_counter()
        hits = engine.search(req.query, top_k=req.top_k)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return SearchResponse(
            hits=[SearchHit(id=h.chunk_id, text=h.text, score=float(h.score)) for h in hits],
            model=state["encoder_name"],
            elapsed_ms=elapsed_ms,
        )

    return app, state


def serve(
    base_model: str = "all-MiniLM-L6-v2",
    host: str = "127.0.0.1",
    port: int = 7800,
    device: str | None = None,
    uds: str | None = None,
) -> None:
    """Start the daemon. Blocks the calling thread."""
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "adaptmem serve requires the [server] extras. Run\n"
            "    pip install \"adaptmem[server]\""
        ) from e

    app, state = _build_app()
    state["encoder_name"] = base_model
    state["device"] = device

    if uds:
        uvicorn.run(app, uds=uds, log_level="info")
    else:
        uvicorn.run(app, host=host, port=port, log_level="info")
