## What this changes

<!-- one-paragraph summary; link to a tracking issue if there is one -->

## How it was tested

<!-- pytest output, a manual repro, or a benchmark JSON -->

## Checklist

- [ ] `ruff check adaptmem tests` is clean
- [ ] `mypy --strict adaptmem` is clean
- [ ] `pytest -q` passes locally
- [ ] CHANGELOG entry added (under `[Unreleased]`)
- [ ] If this changes the public API: README + ROADMAP updated
- [ ] If this adds a benchmark: `benchmarks/results_*.json` committed +
      README table updated with honest numbers (null results welcome)
- [ ] If this adds a dependency: it's an `[optional]` extra unless
      truly core
