"""Daemon endpoint tests — FastAPI TestClient, no real network."""
from __future__ import annotations

import importlib.util

import pytest

# Skip the whole module unless [server] extras are installed.
if importlib.util.find_spec("fastapi") is None:
    pytest.skip("adaptmem[server] extras not installed", allow_module_level=True)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["device"] = "cpu"
    return TestClient(app), state


def test_healthz(client):
    c, _ = client
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["uptime_s"] >= 0


def test_version_before_corpora(client):
    c, _ = client
    r = c.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["encoder"] == "all-MiniLM-L6-v2"
    assert body["corpora"] == []


def test_search_404_for_unknown_corpus(client):
    c, _ = client
    r = c.post("/search", json={"query": "x", "top_k": 3, "corpus_id": "nope"})
    assert r.status_code == 404
    assert "not indexed" in r.json()["detail"]


def test_reindex_then_search(client):
    c, state = client
    # Reindex with a tiny corpus.
    docs = [
        {"id": "a", "text": "PostgreSQL has native JSON since version 9.4."},
        {"id": "b", "text": "Redis stores JSON via the RedisJSON module."},
        {"id": "c", "text": "MongoDB stores documents as BSON."},
    ]
    r = c.post(
        "/reindex",
        json={"corpus_id": "test", "documents": docs},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["corpus_id"] == "test"
    assert body["n_docs"] == 3
    assert body["elapsed_ms"] >= 0

    # Now /version should list the corpus.
    v = c.get("/version").json()
    assert "test" in v["corpora"]

    # Search returns ranked hits.
    s = c.post(
        "/search",
        json={"query": "JSON in postgres", "top_k": 3, "corpus_id": "test"},
    )
    assert s.status_code == 200, s.text
    sbody = s.json()
    assert "hits" in sbody
    assert len(sbody["hits"]) == 3
    # PostgreSQL doc should rank first for a postgres-JSON query.
    assert sbody["hits"][0]["id"] == "a"
    assert sbody["hits"][0]["score"] > sbody["hits"][2]["score"]
    assert sbody["elapsed_ms"] >= 0


def test_embed(client):
    c, _ = client
    r = c.post("/embed", json={"texts": ["hello", "world"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["embeddings"]) == 2
    assert body["dim"] > 0
    assert len(body["embeddings"][0]) == body["dim"]


def test_auth_enabled_rejects_missing_token():
    """When api_key is configured, missing Authorization header → 401."""
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["api_key"] = "secret-key-xyz"
    c = TestClient(app)

    # /healthz still open
    assert c.get("/healthz").status_code == 200
    # /metrics still open
    assert c.get("/metrics").status_code == 200
    # /embed without token → 401
    r = c.post("/embed", json={"texts": ["x"]})
    assert r.status_code == 401
    assert "missing" in r.json()["detail"].lower()


def test_auth_enabled_rejects_wrong_token():
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["api_key"] = "secret-key-xyz"
    c = TestClient(app)

    r = c.post(
        "/embed",
        json={"texts": ["x"]},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401
    assert "invalid" in r.json()["detail"].lower()


def test_auth_enabled_accepts_correct_token():
    """Correct Bearer token should pass the auth check.

    We assert that the response is NOT a 401 — going further (200 with
    encoded body) requires loading the SentenceTransformer, which is
    overkill for an auth-only test and stresses Mac/Py3.14 ardışık model
    load. The auth path is the unit under test.
    """
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["api_key"] = "secret-key-xyz"
    c = TestClient(app)

    # Use /search with a missing corpus_id — auth passes, then 404 from
    # the handler. Confirms Bearer auth let the request through without
    # invoking the encoder.
    r = c.post(
        "/search",
        json={"query": "x", "top_k": 1, "corpus_id": "no-such-corpus"},
        headers={"Authorization": "Bearer secret-key-xyz"},
    )
    assert r.status_code == 404
    assert "not indexed" in r.json()["detail"]


def test_serve_rejects_unpaired_tls_flags(monkeypatch):
    """`serve(ssl_keyfile=X)` without certfile (or vice versa) → SystemExit."""
    import pytest as _pytest

    from adaptmem.server import serve

    # Stub uvicorn.run so we never actually bind a port.
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    with _pytest.raises(SystemExit, match="ssl-keyfile and --ssl-certfile"):
        serve(ssl_keyfile="/tmp/key.pem")
    with _pytest.raises(SystemExit, match="ssl-keyfile and --ssl-certfile"):
        serve(ssl_certfile="/tmp/cert.pem")


def test_serve_accepts_paired_tls_flags(monkeypatch):
    """Both keyfile + certfile present → no exception, uvicorn.run called with ssl kwargs."""
    captured: dict = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    from adaptmem.server import serve

    serve(ssl_keyfile="/tmp/key.pem", ssl_certfile="/tmp/cert.pem")
    assert captured["ssl_keyfile"] == "/tmp/key.pem"
    assert captured["ssl_certfile"] == "/tmp/cert.pem"
    # No CA → no client cert requirement
    assert "ssl_ca_certs" not in captured


def test_serve_mtls_when_ca_certs_provided(monkeypatch):
    """`ssl_ca_certs` → uvicorn requires verified client cert (mTLS)."""
    import ssl as _ssl

    captured: dict = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    from adaptmem.server import serve

    serve(
        ssl_keyfile="/tmp/key.pem",
        ssl_certfile="/tmp/cert.pem",
        ssl_ca_certs="/tmp/ca.pem",
    )
    assert captured["ssl_ca_certs"] == "/tmp/ca.pem"
    assert captured["ssl_cert_reqs"] == _ssl.CERT_REQUIRED


def test_rbac_viewer_can_search_but_not_reindex():
    """Viewer role: /search + /embed allowed, /reindex returns 403."""
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["api_keys"] = {
        "viewer-key": {"role": "viewer", "tenant_id": None},
        "admin-key": {"role": "admin", "tenant_id": None},
    }
    c = TestClient(app)

    # /search via viewer key — auth+role OK; corpus missing → 404 (not 403).
    r = c.post(
        "/v1/search",
        json={"query": "x", "top_k": 1, "corpus_id": "anything"},
        headers={"Authorization": "Bearer viewer-key"},
    )
    assert r.status_code == 404

    # /reindex via viewer key — 403 forbidden.
    r = c.post(
        "/v1/reindex",
        json={"corpus_id": "c", "documents": [{"id": "d", "text": "t"}]},
        headers={"Authorization": "Bearer viewer-key"},
    )
    assert r.status_code == 403
    assert "admin" in r.json()["detail"].lower()


def test_rbac_unknown_key_rejected():
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["api_keys"] = {"viewer-key": {"role": "viewer", "tenant_id": None}}
    c = TestClient(app)

    r = c.post(
        "/v1/search",
        json={"query": "x", "top_k": 1, "corpus_id": "c"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401


def test_audit_log_writes_json_line_per_data_request(tmp_path, capsys):
    """Audit middleware emits one JSON line per /search call (and to file)."""
    import json as _json

    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    audit_path = tmp_path / "audit.jsonl"

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["audit_log_file"] = str(audit_path)
    c = TestClient(app)

    # /search hits the audit path (404 from missing corpus, but still audited).
    r = c.post(
        "/v1/search",
        json={"query": "x", "top_k": 1, "corpus_id": "missing"},
        headers={"x-test-marker": "1"},
    )
    assert r.status_code == 404
    # x-request-id echoed back.
    assert "x-request-id" in r.headers

    # File sink received one line.
    lines = audit_path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = _json.loads(lines[0])
    assert entry["path"] == "/v1/search"
    assert entry["status"] == 404
    assert entry["method"] == "POST"
    assert entry["duration_ms"] >= 0
    assert entry["req_id"] == r.headers["x-request-id"]
    # No api key was sent → api_key_id is null.
    assert entry["api_key_id"] is None


def test_audit_log_skips_healthz_and_metrics(tmp_path):
    """High-frequency probes shouldn't spam the audit log."""
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    audit_path = tmp_path / "audit.jsonl"

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["audit_log_file"] = str(audit_path)
    c = TestClient(app)

    c.get("/healthz")
    c.get("/healthz")
    c.get("/metrics")

    # File never created (no audit lines emitted).
    assert not audit_path.exists() or audit_path.read_text() == ""


def test_audit_log_hashes_api_key():
    """When a Bearer token is present, audit logs the SHA256[:8] not the key."""
    import json as _json

    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["api_key"] = "supersecret"

    # Replace _write_audit's stdout path by capturing via the file sink.
    # (capsys is async-tricky in TestClient; use file path instead.)
    import tempfile

    f = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl")
    f.close()
    state["audit_log_file"] = f.name

    c = TestClient(app)
    c.post(
        "/v1/search",
        json={"query": "x", "top_k": 1, "corpus_id": "missing"},
        headers={"Authorization": "Bearer supersecret"},
    )

    with open(f.name) as fh:
        line = fh.read().strip().split("\n")[-1]
    entry = _json.loads(line)
    # api_key_id is the 8-char SHA256 prefix, not "supersecret".
    assert entry["api_key_id"] is not None
    assert entry["api_key_id"] != "supersecret"
    assert len(entry["api_key_id"]) == 8


def test_readyz_503_before_encoder_loaded():
    """Default state: no engine, /readyz returns 503."""
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    c = TestClient(app)

    r = c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False
    assert body["encoder_loaded"] is False
    assert body["corpora_indexed"] == 0


def test_readyz_200_after_encoder_loaded():
    """Once the engine has a loaded model, /readyz returns 200."""
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    c = TestClient(app)

    # Stub a fake engine — readyz only checks structural truthiness.
    class _FakeEngine:
        _model = object()  # truthy

    state["engine"] = _FakeEngine()
    r = c.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["encoder_loaded"] is True


def test_v1_prefix_routes_resolve():
    """`/v1/embed`, `/v1/search`, `/v1/reindex` should reach the same handlers."""
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    c = TestClient(app)

    # /v1/search 404s on a missing corpus, exactly like /search.
    legacy = c.post(
        "/search",
        json={"query": "x", "top_k": 1, "corpus_id": "nope"},
    )
    canonical = c.post(
        "/v1/search",
        json={"query": "x", "top_k": 1, "corpus_id": "nope"},
    )
    assert legacy.status_code == 404
    assert canonical.status_code == 404
    # Same JSON shape for the same handler logic.
    assert legacy.json() == canonical.json()


def test_v1_prefix_inherits_auth_dependency():
    """`/v1/embed` is also protected by api_key when configured."""
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["api_key"] = "k"
    c = TestClient(app)

    # No header → 401 on both prefixes.
    assert c.post("/embed", json={"texts": ["x"]}).status_code == 401
    assert c.post("/v1/embed", json={"texts": ["x"]}).status_code == 401


def test_metrics_prometheus_format(client):
    c, _ = client
    # Prime counters with known traffic.
    c.get("/healthz")
    c.get("/healthz")
    c.get("/version")

    r = c.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "adaptmem_uptime_seconds " in body
    assert "adaptmem_corpora_total " in body
    # Healthz hit twice, version once.
    assert 'adaptmem_request_total{endpoint="/healthz"} 2' in body
    assert 'adaptmem_request_total{endpoint="/version"} 1' in body
    # Duration counter exists for each endpoint.
    assert 'adaptmem_request_duration_seconds_sum{endpoint="/healthz"}' in body
    # /metrics itself is not counted (middleware skips it).
    assert '/metrics"' not in body
