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


def test_auth_disabled_lets_protected_endpoints_through(client):
    """Default state: api_key=None → /embed/search/reindex open."""
    c, _ = client
    # No Authorization header — should succeed (engine load required, so embed is the cheapest).
    r = c.post("/embed", json={"texts": ["hello"]})
    assert r.status_code == 200


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
    from fastapi.testclient import TestClient

    from adaptmem.server import _build_app

    app, state = _build_app()
    state["encoder_name"] = "all-MiniLM-L6-v2"
    state["api_key"] = "secret-key-xyz"
    c = TestClient(app)

    r = c.post(
        "/embed",
        json={"texts": ["x"]},
        headers={"Authorization": "Bearer secret-key-xyz"},
    )
    assert r.status_code == 200
    assert "embeddings" in r.json()


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
