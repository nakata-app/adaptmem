# Contributing to adaptmem

Thanks for considering a contribution. The repo is small enough that the
review pipeline is short, keep changes focused, the bar is "honest
numbers + clear tradeoffs."

## Quickstart for a local dev loop

```bash
git clone https://github.com/nakata-app/adaptmem.git
cd adaptmem
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,server]"
pre-commit install        # ruff + standard hygiene before each commit
```

## What we run before every commit

```bash
ruff check adaptmem tests              # lint
mypy --strict adaptmem                 # type check
pytest -q                              # unit tests
```

CI runs the same three on Python 3.10 / 3.11 / 3.12. A PR that doesn't
pass them locally won't pass CI either.

## What lands easily

- Bug fixes with a regression test that fails before / passes after.
- New benchmarks. We track honest numbers, null results are valuable
  and will land. See `benchmarks/results_*.json` for the expected
  shape.
- Encoder / dataset support. Drop a new harness in `benchmarks/`
  shaped like the existing ones, ship a JSON, update the README table.
- Server endpoint additions for the `[server]` extras, with matching
  tests in `tests/test_server.py`.

## What needs a discussion first

- Anything that changes the public API surface of `AdaptMem` (search /
  train / save / load / encoder / corpus / embeddings).
- LLM-as-judge coupling, adaptmem is intentionally LLM-free. If you
  think a use case demands an LLM, open an issue first; we'll usually
  point you at claimcheck.
- New required dependencies. We try hard to keep the core install
  small; new heavy dependencies should be `[optional]` extras.

## Style

- Match the existing code. Type hints on public surfaces; no
  speculative abstractions; comments only for non-obvious WHY.
- One commit per logical change. Squash if you accumulate "fix
  comments" commits.
- Commit messages: imperative mood, short subject ("add streaming
  add_corpus"), longer body if the change is non-trivial.

## Reporting bugs

GitHub Issues. Include:
- Python version + OS.
- The minimum reproduction (corpus snippet + query is enough).
- What you expected vs what you got.
- Whether you ran on `cpu` / `mps` / `cuda`.

## Reporting security issues

See [`SECURITY.md`](SECURITY.md). Don't open a public issue for an
unpatched vulnerability.
