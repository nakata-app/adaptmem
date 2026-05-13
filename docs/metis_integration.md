# ADR: Metis × adaptmem integration

**Status:** accepted, 2026-04-27.
**Decision driver:** atakan@nakata-app.

## Context

`metis` is a Rust agent CLI (~129 commits, 487 tests, v0.10 in flight) with
its own memory tooling under `.metis/memory/`, markdown files with YAML
frontmatter, indexed by `MEMORY.md`. The current memory tools (`save_memory`,
`list_memories`, `read_memory`, `delete_memory`) are pure file operations:
markdown grep, no semantic retrieval.

`adaptmem` is a Python research package that ships a domain-tuned
bi-encoder retrieval pipeline (R@5=0.995 on LongMemEval, +3pt over
MemPalace raw). It composes with `claimcheck` (NLI verification +
orchestration layer).

The cluster's value claim, *"domain-tuned semantic retrieval that
beats off-the-shelf RAG without an LLM judge"*, is currently a set of
benchmark JSONs. The next demonstration is to plug it into a real agent
loop. metis is that loop.

## Decision

**Run adaptmem as a local HTTP/Unix-socket daemon.** metis talks to it
via `reqwest` (already a workspace dep). One process, one model load,
multiple consumers (metis today, claimcheck tomorrow if it wants a
shared encoder).

```
┌─────────────────┐  HTTP   ┌──────────────────┐
│   metis (Rust)  │ ──────▶ │  adaptmem serve  │
│  semantic_memo  │  JSON   │   (Python +      │
│  ry_search tool │ ◀────── │   FastAPI +      │
└─────────────────┘         │   uvicorn)       │
                            └──────────────────┘
                                    │
                                    ▼
                            cached encoder + index
                            (one model load,
                            many queries)
```

## Options considered

### A) `subprocess` shell-out

`metis` spawns `python -m adaptmem search ...`, parses JSON from stdout.

**Pros:** zero linkage, easiest to ship, total process isolation.
**Cons:** every call pays 3-5s for Python startup + model load. An agent
issuing 50 retrievals per session burns 150-250 seconds *just on cold
starts*. Stateless, no embedding cache, no shared corpus state.

### B) PyO3 (Rust ↔ Python in-process)

Bind into the Python interpreter from Rust via `pyo3`.

**Pros:** fastest path, single binary, shared heap.
**Cons:** building a Rust binary that links a Python interpreter is
deployment hell, `cargo build` now depends on Python headers, the
metis binary balloons by ~15MB, distributing cross-platform (macOS arm64
+ macOS x86 + Linux + Windows) becomes a chore. Forces the Rust
ecosystem to inherit Python's packaging story.

### C) HTTP/Unix-socket daemon, *chosen*

Run `adaptmem serve` as a long-lived process. metis is a thin reqwest
client.

**Pros:**
- Cold-start cost paid once at daemon launch, not per-query.
- metis `cargo build` stays Python-free, Atakan's 487-test Rust CLI is
  not impacted by the Python toolchain.
- Reusable across consumers: claimcheck can hit the same daemon for
  its own retrieval needs.
- Process isolation, daemon crash does not take metis down.
- Localhost overhead is a few microseconds; Unix-socket option for
  zero TCP overhead.
- Daemon can be replaced (different encoder, different corpus) without
  re-shipping the metis binary.

**Cons:**
- Two processes to manage (daemon up before queries, lifecycle in
  systemd/launchd or `metis` could spawn it on demand).
- A few extra deps in Python (FastAPI, uvicorn). Wired as
  `pip install adaptmem[server]` so the core package stays minimal.

## Use case (v0.7 first hit): semantic memory search

metis already has `save_memory` / `list_memories` / `read_memory`. The
gap is **semantic retrieval over those memories**, the agent asks
"what did I learn about caching?" and the current toolset can only grep
for the literal string `caching`. Embedding retrieval over the same
markdown files is exactly the LongMemEval problem on a smaller scale.

The new tool: `semantic_memory_search` in `crates/core/src/tools/`.

```rust
// args
{"query": "caching strategies", "top_k": 5}

// behavior
1. POST http://127.0.0.1:7800/search
   {"query": "caching strategies", "top_k": 5,
    "corpus_id": "metis_memory"}
2. daemon returns [{id, text, score}, ...]
3. tool formats as a markdown bulleted list with backlinks to
   .metis/memory/<id>.md
```

The corpus is the contents of `.metis/memory/*.md`, encoded once at
daemon startup (or on `POST /reindex` when the user saves a new memory).

## Out of scope for v0.7

- **Tool-output verification** (claimcheck plug-in before metis acts on a
  result). Requires an NLI verifier + a corpus of "ground truth", open
  design question, defer.
- **Codebase semantic search** (replace grep/glob with embeddings).
  Different corpus shape, different latency budget; v0.8.
- **Daemon lifecycle in metis itself.** v0.7 expects the user to start
  the daemon manually (`adaptmem serve`); v0.8 can add an `auto-spawn`
  option.

## API contract (v1)

JSON over HTTP. All endpoints accept and return JSON.

### `POST /embed`
```json
{"texts": ["hello", "world"]}
```
→
```json
{"embeddings": [[0.12, ...], [0.34, ...]], "model": "all-MiniLM-L6-v2", "dim": 384}
```

### `POST /search`
```json
{"query": "caching strategies", "top_k": 5, "corpus_id": "metis_memory"}
```
→
```json
{
  "hits": [
    {"id": "feedback_caching.md", "text": "...", "score": 0.87},
    ...
  ],
  "model": "all-MiniLM-L6-v2",
  "elapsed_ms": 12.3
}
```

### `POST /reindex`
```json
{"corpus_id": "metis_memory",
 "documents": [{"id": "feedback_caching.md", "text": "..."}, ...]}
```
→
```json
{"corpus_id": "metis_memory", "n_docs": 47, "elapsed_ms": 1812}
```

### `GET /healthz` → `{"ok": true, "uptime_s": 12.3}`

### `GET /version`
```json
{"adaptmem": "0.5.0", "encoder": "all-MiniLM-L6-v2", "corpora": ["metis_memory"]}
```

## Compatibility matrix

| adaptmem | metis (semantic_memory_search tool) |
|---|---|
| ≥ 0.5.0 | required (server module landed in 0.5) |
| 0.6+ | breaking-change-free; daemon adds endpoints, doesn't remove |

Major version bump on adaptmem implies API rev. metis tool checks
`/version` at startup and refuses to use a daemon below the minimum
version.

## Failure modes

| Failure | Daemon behaviour | metis behaviour |
|---|---|---|
| daemon not running | n/a | tool returns `ToolError::Service { name: "adaptmem", reason: "connection refused" }`, agent surfaces "memory search unavailable" |
| corpus not indexed | 404 on `/search` | tool returns "corpus not indexed; run `adaptmem reindex --corpus metis_memory`" |
| daemon slow (>5s) | n/a | client timeout 5s, falls back to grep over `.metis/memory/` |
| version mismatch | `/version` returns | tool refuses, prints upgrade instructions |

## Test strategy

1. **Unit (adaptmem)**, mock corpus, fixture queries, assert hit IDs.
2. **Unit (metis tool)**, `mockito` HTTP server, assert request shape +
   response parsing.
3. **Integration (cross-repo)**, bash script: start daemon on port
   7800, `curl /healthz`, `curl /search`, kill daemon, assert exit codes.
4. **Smoke (manual, in this PR)**, start daemon, run `metis` REPL,
   issue `/memory search "..."`, observe response.

## Rollout

1. **adaptmem v0.5.0**, ships `adaptmem.server` module, `adaptmem serve`
   CLI subcommand, `[project.optional-dependencies] server = [...]`,
   integration tests under `tests/test_server.py`. README documents
   endpoint contract (this ADR).
2. **metis**, new `semantic_memory_search` tool registered in the
   default tool set (gated behind a config flag for the first release;
   off by default until daemon adoption is real).
3. **Discoverability**, `adaptmem -h` shows `serve` subcommand;
   metis `/help` documents the tool; both READMEs cross-link this ADR.

## Non-goals (until further notice)

- Authentication / multi-tenant, daemon is single-user-localhost-only.
- gRPC / protobuf, JSON is fine, switch later if a benchmark says so.
- GPU support in the daemon, Mac/Linux CPU is the target; users with
  CUDA can set `--device cuda` on `adaptmem serve` and get free.
