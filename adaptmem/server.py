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


def _enable_otel(app: Any) -> bool:
    """Auto-instrument FastAPI when the [telemetry] extras + OTLP endpoint are present.

    Returns True if instrumentation was attached, False otherwise (missing
    deps, no endpoint, or already instrumented). Caller can check the
    return value in tests; production just calls and forgets.

    Required env (any standard OTEL_* setup works — these are the most
    common):
        OTEL_EXPORTER_OTLP_ENDPOINT   e.g. https://api.honeycomb.io
        OTEL_EXPORTER_OTLP_HEADERS    e.g. x-honeycomb-team=KEY
        OTEL_SERVICE_NAME             defaults to "adaptmem"
    """
    import os as _os

    if not _os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover — exercised when [telemetry] absent
        return False

    service = _os.environ.get("OTEL_SERVICE_NAME", "adaptmem")
    resource = Resource.create({"service.name": service})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    return True


def _build_app() -> Any:
    """Build the FastAPI app + return (app, state). Imports gated to keep `[server]` optional."""
    try:
        from contextlib import asynccontextmanager

        from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
    except ImportError as e:  # pragma: no cover — exercised only without `[server]`
        raise SystemExit(
            "adaptmem.server requires the [server] extras. Run\n"
            "    pip install \"adaptmem[server]\""
        ) from e

    @asynccontextmanager
    async def _lifespan(_app: Any) -> Any:
        """Lifespan handler — runs at startup (before yield) and shutdown
        (after yield). Gives us a clean hook for closing the SQLite
        store on SIGTERM so WAL data is flushed and no .db-wal/-shm
        files dangle.
        """
        yield  # startup is no-op; encoder loads lazily on first request
        # --- shutdown phase ---
        store = state.get("store")
        if store is not None:
            try:
                store.close()
            except Exception:
                # Shutdown is best-effort; swallow rather than crash uvicorn.
                pass

    app = FastAPI(
        title="adaptmem",
        description="Domain-tuned retrieval daemon",
        lifespan=_lifespan,
    )

    # OpenTelemetry tracing — opt-in via the [telemetry] extras + the
    # standard OTEL_EXPORTER_OTLP_ENDPOINT env. Missing extras = silent
    # no-op (dev / test boxes don't pay the import cost).
    _enable_otel(app)

    # Rate limiting (slowapi). Default cap can be overridden via env. Per-
    # endpoint granularity is a v0.6 follow-up — keeping the wiring simple
    # avoids slowapi's signature-introspection quirks with FastAPI.
    import os as _os

    _default_rate = _os.environ.get("ADAPTMEM_RATE_LIMIT", "120/minute")

    try:
        from slowapi import Limiter
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.util import get_remote_address

        limiter = Limiter(key_func=get_remote_address, default_limits=[_default_rate])
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)

        @app.exception_handler(RateLimitExceeded)
        def _rate_limit_handler(request: Any, exc: Any) -> Any:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={"detail": f"rate limit exceeded: {exc.detail}"},
            )
    except ImportError:  # pragma: no cover
        # Older install without slowapi — rate limiting silently disabled.
        pass

    state: dict[str, Any] = {
        "encoder_name": None,
        "engine": None,        # AdaptMem instance (shared encoder)
        "corpora": {},         # corpus_id -> AdaptMem
        "device": None,
        # Prometheus-style counters: endpoint -> {"count": int, "duration_s_sum": float}
        "metrics": {},
        # API key (None = auth disabled; str = required Bearer token).
        # Backward-compat single-key field. For multi-key + RBAC, see api_keys.
        "api_key": None,
        # Multi-key map: token -> {"role": "viewer"|"admin", "tenant_id": str|None}
        # When non-empty, takes precedence over the single api_key field.
        "api_keys": {},
        # Audit log: structured JSON lines. Optional file sink in addition
        # to stdout (configured by serve()). When None, logs go to stdout
        # via the standard logger.
        "audit_log_file": None,
        # Optional CorpusStore for surviving restarts. None = in-memory only.
        "store": None,
    }

    def _resolve_key(authorization: str | None) -> dict[str, Any] | None:
        """Decode Authorization header → key metadata or None when auth disabled."""
        # Auth disabled mode: no api_key, no api_keys.
        if state["api_key"] is None and not state["api_keys"]:
            return {"role": "admin", "tenant_id": None}  # full access, no auth required
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing Bearer token")
        supplied = authorization.split(" ", 1)[1].strip()

        # Multi-key map takes precedence.
        if state["api_keys"]:
            meta = state["api_keys"].get(supplied)
            if meta is None:
                raise HTTPException(status_code=401, detail="invalid api key")
            result: dict[str, Any] = meta
            return result

        # Backward-compat single-key mode: implicit admin role.
        if supplied != state["api_key"]:
            raise HTTPException(status_code=401, detail="invalid api key")
        return {"role": "admin", "tenant_id": None}

    def verify_api_key(authorization: str | None = Header(default=None)) -> None:
        """Bearer-token auth. No-op when no api_key is configured."""
        _resolve_key(authorization)

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        """Bearer-token auth + role check. Used for write endpoints."""
        meta = _resolve_key(authorization)
        if meta is None or meta.get("role") != "admin":
            raise HTTPException(
                status_code=403, detail="admin role required for this endpoint"
            )

    def _enforce_tenant(corpus_id: str, authorization: str | None) -> None:
        """Reject requests whose corpus_id falls outside the caller's tenant.

        Convention: corpus_id starts with `<tenant_id>/`. When the API key
        has a non-null tenant_id, the corpus_id must begin with that
        prefix. Admin keys with tenant_id=None can address any corpus.
        """
        meta = _resolve_key(authorization)
        if meta is None:
            return  # auth disabled, no tenant boundaries
        tenant = meta.get("tenant_id")
        if tenant is None:
            return  # admin / unscoped key
        prefix = f"{tenant}/"
        if not corpus_id.startswith(prefix):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"corpus_id '{corpus_id}' is outside tenant '{tenant}'. "
                    f"Names must start with '{prefix}'."
                ),
            )

    def _record_metric(endpoint: str, duration_s: float) -> None:
        m = state["metrics"].setdefault(endpoint, {"count": 0, "duration_s_sum": 0.0})
        m["count"] += 1
        m["duration_s_sum"] += duration_s

    def _hash_key(api_key: str | None) -> str | None:
        """8-char SHA256 prefix of the api_key — enough to correlate
        requests in logs without leaking the secret."""
        if not api_key:
            return None
        import hashlib

        return hashlib.sha256(api_key.encode()).hexdigest()[:8]

    def _write_audit(entry: dict[str, Any]) -> None:
        """Emit one JSON line to stdout + optional file sink."""
        import json as _json
        import sys as _sys

        line = _json.dumps(entry, separators=(",", ":"))
        # stdout for container log collectors / journald.
        print(line, file=_sys.stdout, flush=True)
        # Optional file sink for shippers that prefer log-file tailing.
        path = state.get("audit_log_file")
        if path:
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                # Audit failure must not break the request path.
                pass

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
        # Skip metrics + healthz from audit + counters — high-frequency
        # probes would spam the log otherwise. Auditable endpoints are
        # the data path (embed/search/reindex) plus version/readyz.
        skip_audit = request.url.path in ("/metrics", "/healthz")
        t0 = time.perf_counter()
        # Generate a request id we can log + return in headers.
        import uuid as _uuid

        req_id = _uuid.uuid4().hex[:12]
        request.state.req_id = req_id
        response = await call_next(request)
        duration_s = time.perf_counter() - t0
        if request.url.path == "/metrics":
            return response
        _record_metric(request.url.path, duration_s)

        if not skip_audit:
            # Pull the bearer token (if any) for audit correlation.
            auth = request.headers.get("authorization", "")
            supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else None
            client = request.client.host if request.client else "unknown"
            _write_audit(
                {
                    "ts": time.time(),
                    "req_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration_s * 1000, 2),
                    "client": client,
                    "api_key_id": _hash_key(supplied),
                }
            )
        # Echo the request id so callers can correlate with logs.
        response.headers["x-request-id"] = req_id
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

    @data_router.post(
        "/reindex",
        response_model=ReindexResponse,
        dependencies=[Depends(require_admin)],
    )
    def reindex(
        req: ReindexRequest,
        authorization: str | None = Header(default=None),
    ) -> ReindexResponse:
        _enforce_tenant(req.corpus_id, authorization)

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

        # Persist to disk if a store is configured. Survives restarts.
        if state["store"] is not None:
            state["store"].save_corpus(
                corpus_id=req.corpus_id,
                model=state["encoder_name"],
                documents=[(d.id, d.text) for d in req.documents],
                embeddings=engine._embeddings,
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return ReindexResponse(
            corpus_id=req.corpus_id, n_docs=len(req.documents), elapsed_ms=elapsed_ms
        )

    @data_router.post("/search", response_model=SearchResponse)
    def search(
        req: SearchRequest,
        authorization: str | None = Header(default=None),
    ) -> SearchResponse:
        _enforce_tenant(req.corpus_id, authorization)
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
    api_keys_file: str | None = None,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
    ssl_ca_certs: str | None = None,
    audit_log_file: str | None = None,
    persist_dir: str | None = None,
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
    state["audit_log_file"] = audit_log_file or os.environ.get("ADAPTMEM_AUDIT_LOG") or None

    # Multi-key + RBAC: load JSON file shaped like
    #   [{"key": "...", "role": "viewer"|"admin", "tenant_id": "..."}]
    # When present, overrides the single api_key field. Each request's
    # Bearer token is looked up against this map; admin role is required
    # for /reindex.
    keys_path = api_keys_file or os.environ.get("ADAPTMEM_API_KEYS_FILE") or None
    if keys_path:
        import json as _json

        with open(keys_path, encoding="utf-8") as fh:
            entries = _json.load(fh)
        keymap: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if "key" not in entry or "role" not in entry:
                raise SystemExit(
                    f"--api-keys-file: each entry needs 'key' and 'role'; got {entry}"
                )
            if entry["role"] not in ("viewer", "admin"):
                raise SystemExit(
                    f"--api-keys-file: role must be viewer|admin, got '{entry['role']}'"
                )
            keymap[entry["key"]] = {
                "role": entry["role"],
                "tenant_id": entry.get("tenant_id"),
            }
        state["api_keys"] = keymap

    # Optional SQLite-backed corpus store. When set, /reindex writes both to
    # memory and to disk; on startup all stored corpora are reloaded so
    # the daemon comes up ready (with /readyz still 503 until the encoder
    # itself is loaded — that happens lazily on first request).
    persist_path = persist_dir or os.environ.get("ADAPTMEM_PERSIST_DIR") or None
    if persist_path:
        from pathlib import Path as _Path

        from adaptmem.persistence import CorpusStore

        db_path = _Path(persist_path) / "corpora.db"
        store = CorpusStore(db_path)
        state["store"] = store

        # Replay every persisted corpus into the in-memory state so /search
        # works without a fresh /reindex.
        for cid in store.list_corpora():
            loaded = store.load_corpus(cid)
            if loaded is None:
                continue
            docs, embeddings, model_name = loaded
            from adaptmem.miner import CorpusEntry as _CorpusEntry
            from sentence_transformers import SentenceTransformer as _ST

            engine = AdaptMem(base_model=model_name, device=device)
            engine._model = _ST(model_name, device=device or None)
            engine._corpus = [_CorpusEntry(id=did, text=text) for did, text in docs]
            engine._embeddings = embeddings
            state["corpora"][cid] = engine
            if state["engine"] is None:
                state["engine"] = engine

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
