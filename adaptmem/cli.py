"""adaptmem CLI: train, search, bench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _cmd_train(args: argparse.Namespace) -> None:
    from adaptmem import AdaptMem
    from adaptmem.types import TrainConfig

    corpus = json.loads(Path(args.corpus).read_text())
    queries = json.loads(Path(args.queries).read_text())
    am = AdaptMem(
        base_model=args.base_model,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
    )
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        top_k_mine=args.top_k_mine,
    )
    stats = am.train(corpus=corpus, labelled=queries, config=cfg)
    am.save(args.out)
    print(json.dumps(stats, indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Compute Recall@k for a saved model against a labelled queries file.

    queries.json shape: list of {"query": str, "relevant_ids": [str, ...]}
    """
    from adaptmem import AdaptMem

    queries = json.loads(Path(args.queries).read_text())
    am = AdaptMem.load(args.model)
    if args.rerank:
        am.rerank_enabled = True
        if args.rerank_model:
            am.rerank_model_name = args.rerank_model
            am._rerank_model = None

    ks = sorted({1, 5, args.top_k})
    max_k = max(ks)
    hits_at = {k: 0 for k in ks}
    n = 0
    for q in queries:
        relevant = set(q["relevant_ids"])
        if not relevant:
            continue
        ranked = [h.chunk_id for h in am.search(q["query"], top_k=max_k)]
        for k in ks:
            if any(rid in ranked[:k] for rid in relevant):
                hits_at[k] += 1
        n += 1

    if n == 0:
        print(json.dumps({"error": "no labelled queries with relevant_ids"}, indent=2))
        return
    out = {"n": n, "recall": {f"@{k}": round(hits_at[k] / n, 4) for k in ks}}
    print(json.dumps(out, indent=2))


def _cmd_search(args: argparse.Namespace) -> None:
    from adaptmem import AdaptMem

    am = AdaptMem.load(args.model)
    # Allow CLI override of stored rerank state (e.g. "trained without rerank,
    # serve with rerank for an A/B").
    if args.rerank:
        am.rerank_enabled = True
        if args.rerank_model:
            am.rerank_model_name = args.rerank_model
            am._rerank_model = None  # force re-resolve on next search
    hits = am.search(args.query, top_k=args.top_k, rerank_top_k=args.rerank_top_k)
    for h in hits:
        print(f"{h.score:.4f}\t{h.chunk_id}\t{h.text[:120]}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="adaptmem")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="mine + fine-tune + save")
    t.add_argument("--corpus", required=True, help="JSON list of {id, text} or strings")
    t.add_argument("--queries", required=True, help="JSON list of {query, relevant_ids}")
    t.add_argument("--out", required=True, help="output directory")
    t.add_argument("--base-model", default="all-MiniLM-L6-v2")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--top-k-mine", type=int, default=10)
    t.add_argument(
        "--rerank",
        action="store_true",
        help="Persist rerank=True so .load() restores cross-encoder rerank.",
    )
    t.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="Cross-encoder model name (only used when --rerank is set).",
    )
    t.set_defaults(func=_cmd_train)

    s = sub.add_parser("search", help="run a query on a saved model")
    s.add_argument("--model", required=True, help="path saved by `adaptmem train`")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument(
        "--rerank",
        action="store_true",
        help="Force-enable cross-encoder rerank for this search (overrides "
             "the saved model's rerank flag).",
    )
    s.add_argument(
        "--rerank-model",
        default=None,
        help="Override the saved cross-encoder model name (requires --rerank).",
    )
    s.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help="Bi-encoder candidate set size before CE rerank (default: top-k * 3).",
    )
    s.set_defaults(func=_cmd_search)

    e = sub.add_parser("evaluate", help="recall@k against a labelled queries file")
    e.add_argument("--model", required=True, help="path saved by `adaptmem train`")
    e.add_argument(
        "--queries",
        required=True,
        help='JSON list of {"query": str, "relevant_ids": [str,...]}',
    )
    e.add_argument("--top-k", type=int, default=10, help="Largest k to compute (R@1/R@5/R@k)")
    e.add_argument("--rerank", action="store_true")
    e.add_argument("--rerank-model", default=None)
    e.set_defaults(func=_cmd_evaluate)

    sv = sub.add_parser(
        "serve",
        help="run the long-lived HTTP daemon (Metis / claimcheck consumers)",
    )
    sv.add_argument("--base-model", default="all-MiniLM-L6-v2")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=7800)
    sv.add_argument(
        "--device",
        default=None,
        help="PyTorch device override: 'cpu', 'cuda', 'mps'",
    )
    sv.add_argument(
        "--uds",
        default=None,
        help="Optional Unix-domain-socket path (overrides --host/--port).",
    )
    sv.add_argument(
        "--api-key",
        default=None,
        help="Bearer token required for /embed, /search, /reindex. "
             "Falls back to ADAPTMEM_API_KEY env. Unset = auth disabled. "
             "Implicit admin role. Use --api-keys-file for RBAC.",
    )
    sv.add_argument(
        "--api-keys-file",
        default=None,
        help="JSON file with multi-key RBAC: "
             '[{"key": "...", "role": "viewer"|"admin", "tenant_id": "..."}]. '
             "viewer = /search + /embed. admin = also /reindex. "
             "Falls back to ADAPTMEM_API_KEYS_FILE env. Overrides --api-key.",
    )
    sv.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to TLS private key (PEM). Pair with --ssl-certfile to serve HTTPS.",
    )
    sv.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to TLS certificate chain (PEM). Required when --ssl-keyfile is set.",
    )
    sv.add_argument(
        "--ssl-ca-certs",
        default=None,
        help="PEM bundle of trusted client CA certs. When set, mTLS is enforced "
             "(daemon requires a verified client certificate per request).",
    )
    sv.add_argument(
        "--audit-log-file",
        default=None,
        help="Path to a file that mirrors the JSON audit lines emitted to stdout. "
             "Falls back to ADAPTMEM_AUDIT_LOG env. Each request appends one line "
             "with timestamp, request id, endpoint, status, duration, client IP, "
             "and api_key hash.",
    )
    sv.add_argument(
        "--persist-dir",
        default=None,
        help="Directory for the SQLite corpus store (corpora.db). When set, "
             "/reindex writes to disk + memory; on startup the daemon reloads "
             "all stored corpora. Falls back to ADAPTMEM_PERSIST_DIR env. "
             "Unset = in-memory only (corpora lost on restart).",
    )
    sv.set_defaults(func=_cmd_serve)

    # `corpora` subcommand group — talks to a running daemon over HTTP.
    co = sub.add_parser("corpora", help="manage corpora on a running daemon")
    co_sub = co.add_subparsers(dest="corpora_cmd", required=True)

    def _add_daemon_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--daemon",
            default="http://127.0.0.1:7800",
            help="Daemon base URL. Defaults to http://127.0.0.1:7800.",
        )
        p.add_argument(
            "--api-key",
            default=None,
            help="Bearer token. Falls back to ADAPTMEM_API_KEY env.",
        )

    cl = co_sub.add_parser("list", help="list indexed corpora")
    _add_daemon_args(cl)
    cl.set_defaults(func=_cmd_corpora_list)

    cs = co_sub.add_parser("search", help="search a single corpus")
    _add_daemon_args(cs)
    cs.add_argument("--corpus-id", required=True)
    cs.add_argument("--query", required=True)
    cs.add_argument("--top-k", type=int, default=5)
    cs.set_defaults(func=_cmd_corpora_search)

    cr = co_sub.add_parser("reindex", help="(re)index a corpus from a JSON file")
    _add_daemon_args(cr)
    cr.add_argument("--corpus-id", required=True)
    cr.add_argument(
        "--file",
        required=True,
        help='Path to JSON list of {"id": ..., "text": ...} entries.',
    )
    cr.set_defaults(func=_cmd_corpora_reindex)

    cd = co_sub.add_parser("delete", help="drop a corpus from memory + disk (admin only)")
    _add_daemon_args(cd)
    cd.add_argument("--corpus-id", required=True)
    cd.set_defaults(func=_cmd_corpora_delete)

    # Optional: argcomplete-driven shell tab-completion. Best-effort
    # import — if argcomplete isn't installed (default), CLI works
    # exactly as before. With it installed + a one-time
    # `eval "$(register-python-argcomplete adaptmem)"` in your shell
    # rc-file, you get tab-completion for all subcommands and flags.
    try:
        import argcomplete

        argcomplete.autocomplete(ap)
    except ImportError:
        pass

    args = ap.parse_args()
    args.func(args)


def _daemon_request(
    method: str,
    daemon_url: str,
    path: str,
    api_key: str | None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Common HTTP wrapper for `corpora` subcommands. Imports requests
    lazily so the rest of the CLI works without [server] extras."""
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit(
            "`corpora` subcommands need `requests`. Run "
            '`pip install adaptmem[server]` or `pip install requests`.'
        ) from e

    import os as _os

    headers: dict[str, str] = {}
    key = api_key or _os.environ.get("ADAPTMEM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{daemon_url.rstrip('/')}{path}"
    resp = requests.request(method, url, headers=headers, json=json_body, timeout=120)
    if not resp.ok:
        raise SystemExit(f"daemon {method} {path} → {resp.status_code}: {resp.text[:200]}")
    body: dict[str, Any] = resp.json()
    return body


def _cmd_corpora_list(args: argparse.Namespace) -> None:
    body = _daemon_request("GET", args.daemon, "/version", args.api_key)
    corpora = body.get("corpora", [])
    if not corpora:
        print("(no corpora indexed)")
        return
    for cid in corpora:
        print(cid)


def _cmd_corpora_search(args: argparse.Namespace) -> None:
    body = _daemon_request(
        "POST",
        args.daemon,
        "/v1/search",
        args.api_key,
        {"query": args.query, "top_k": args.top_k, "corpus_id": args.corpus_id},
    )
    for hit in body.get("hits", []):
        print(f"{hit['score']:.4f}\t{hit['id']}\t{hit['text'][:120]}")


def _cmd_corpora_delete(args: argparse.Namespace) -> None:
    body = _daemon_request(
        "DELETE",
        args.daemon,
        f"/v1/corpora/{args.corpus_id}",
        args.api_key,
    )
    print(json.dumps(body, indent=2))


def _cmd_corpora_reindex(args: argparse.Namespace) -> None:
    docs = json.loads(Path(args.file).read_text())
    if not isinstance(docs, list) or not all("id" in d and "text" in d for d in docs):
        raise SystemExit(
            "--file must point to a JSON list of {id, text} entries."
        )
    body = _daemon_request(
        "POST",
        args.daemon,
        "/v1/reindex",
        args.api_key,
        {"corpus_id": args.corpus_id, "documents": docs},
    )
    print(json.dumps(body, indent=2))


def _cmd_serve(args: argparse.Namespace) -> None:
    from adaptmem.server import serve

    serve(
        base_model=args.base_model,
        host=args.host,
        port=args.port,
        device=args.device,
        uds=args.uds,
        api_key=args.api_key,
        api_keys_file=args.api_keys_file,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        audit_log_file=args.audit_log_file,
        persist_dir=args.persist_dir,
    )


if __name__ == "__main__":
    main()
