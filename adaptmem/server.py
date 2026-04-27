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

    class ReadyzResponse(BaseModel):
        ready: bool
        encoder_loaded: bool
        corpora_indexed: int
        uptime_s: float

    class VersionResponse(BaseModel):
        adaptmem: str
        encoder: str
        corpora: list[str]


def _build_app() -> Any:
    """Build the FastAPI app + return (app, state). Imports gated to keep `[server]` optional."""
    try:
        from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
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
        # API key (None = auth disabled; str = required Bearer token).
        "api_key": None,
    }

    def verify_api_key(authorization: str | None = Header(default=None)) -> None:
        """Bearer-token auth. No-op when state["api_key"] is None.

        Raises 401 when:
        - the daemon is configured with an api_key but the request is missing one;
        - the supplied key doesn't match.
        """
        configured = state["api_key"]
        if configured is None:
            return  # auth disabled
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing Bearer token")
        supplied = authorization.split(" ", 1)[1].strip()
        if supplied != configured:
            raise HTTPException(status_code=401, detail="invalid api key")

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
        """Liveness probe — process is up. Kubernetes-style.

        Returns 200 as long as the daemon's HTTP server is responding.
        Use this for restart loops (k8s livenessProbe).
        """
        return HealthzResponse(ok=True, uptime_s=round(time.time() - _STARTED_AT, 2))

    @app.get("/readyz", response_model=ReadyzResponse)
    def readyz() -> Any:
        """Readiness probe — daemon can serve real traffic.

        Returns 200 once the encoder model has been loaded (lazy on first
        embed/reindex). Returns 503 before that — k8s readinessProbe will
        hold the daemon out of the load-balancer rotation until it's
        actually ready.
        """
        from fastapi.responses import JSONResponse

        encoder_loaded = state["engine"] is not None and state["engine"]._model is not None
        body = ReadyzResponse(
            ready=encoder_loaded,
            encoder_loaded=encoder_loaded,
            corpora_indexed=len(state["corpora"]),
            uptime_s=round(time.time() - _STARTED_AT, 2),
        )
        if not encoder_loaded:
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    @app.get("/version", response_model=VersionResponse)
    def version() -> VersionResponse:
        from adaptmem import __version__

        return VersionResponse(
            adaptmem=__version__,
            encoder=state["encoder_name"] or "uninitialised",
            corpora=sorted(state["corpora"].keys()),
        )

    # Data endpoints live on a router so we can mount them under both
    # `/v1/` (canonical) and `/` (legacy, deprecated). Same handler,
    # two paths — clients can migrate at their own pace.
    data_router = APIRouter(dependencies=[Depends(verify_api_key)])

    @data_router.post("/embed", response_model=EmbedResponse)
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

    @data_router.post("/reindex", response_model=ReindexResponse)
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

    @data_router.post("/search", response_model=SearchResponse)
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

    # Mount the data router under both prefixes:
    #   /v1/embed    /v1/search    /v1/reindex   ← canonical
    #   /embed       /search       /reindex      ← legacy (deprecated, still works)
    app.include_router(data_router, prefix="/v1")
    app.include_router(data_router)  # legacy paths

    return app, state


def serve(
    base_model: str = "all-MiniLM-L6-v2",
    host: str = "127.0.0.1",
    port: int = 7800,
    device: str | None = None,
    uds: str | None = None,
    api_key: str | None = None,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
    ssl_ca_certs: str | None = None,
) -> None:
    """Start the daemon. Blocks the calling thread.

    Auth: when `api_key` is set (CLI flag or `ADAPTMEM_API_KEY` env),
    `/embed`, `/search`, `/reindex` require `Authorization: Bearer <key>`.
    `/healthz`, `/version`, `/metrics` stay open for health probes /
    Prometheus scrape regardless.

    TLS: pass `ssl_keyfile` and `ssl_certfile` to serve over HTTPS. For
    mTLS, also pass `ssl_ca_certs` (PEM bundle of trusted client CAs);
    uvicorn will require a verified client certificate per request.
    """
    import os

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
    # CLI flag wins; env var is the fallback. Empty strings ignored.
    resolved_key = api_key or os.environ.get("ADAPTMEM_API_KEY") or None
    state["api_key"] = resolved_key if resolved_key else None

    ssl_kwargs: dict[str, Any] = {}
    if ssl_keyfile and ssl_certfile:
        ssl_kwargs["ssl_keyfile"] = ssl_keyfile
        ssl_kwargs["ssl_certfile"] = ssl_certfile
        if ssl_ca_certs:
            import ssl as _ssl

            ssl_kwargs["ssl_ca_certs"] = ssl_ca_certs
            ssl_kwargs["ssl_cert_reqs"] = _ssl.CERT_REQUIRED  # mTLS
    elif ssl_keyfile or ssl_certfile:
        raise SystemExit(
            "TLS requires both --ssl-keyfile and --ssl-certfile (PEM paths)."
        )

    if uds:
        uvicorn.run(app, uds=uds, log_level="info", **ssl_kwargs)
    else:
        uvicorn.run(app, host=host, port=port, log_level="info", **ssl_kwargs)
