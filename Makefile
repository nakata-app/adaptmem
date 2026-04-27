PYTHON ?= python
DATASET ?= /Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json
# Force CPU by default — bypasses MPS deadlocks observed on Apple silicon
# during contrastive fine-tuning. Override with `make DEVICE= bench-...` to
# let PyTorch autodetect.
DEVICE ?= cpu

# Self-contained 100/400 split shipped with the repo
SPLIT_100 ?= benchmarks/data/split_ids_100_400.json
MODEL_100 ?= benchmarks/bench-model-100
RESULTS_100 ?= benchmarks/results_ft100_400.json

# External reference models from metis-pair (300 train / 200 test split)
SPLIT_300 ?= /Users/macmini/Projects/metis-pair/benchmarks/data/training_300/split_ids.json
MODEL_300 ?= /Users/macmini/Projects/metis-pair/benchmarks/models/minilm-lme-ft-300
MODEL_200 ?= /Users/macmini/Projects/metis-pair/benchmarks/models/minilm-lme-ft-200
RESULTS_300 ?= benchmarks/results_ft300_direct.json
RESULTS_200 ?= benchmarks/results_ft200_direct.json

.PHONY: bench-longmemeval bench-ft100 bench-ft300 bench-ft200 train-100 clean-bench \
        test lint typecheck verify install dev docker-build helm-lint clean help

# ---- Common developer entry points ---------------------------------------

help:
	@echo "adaptmem — common dev targets:"
	@echo "  make install      — pip install -e .[dev]"
	@echo "  make dev          — install + pre-commit hooks"
	@echo "  make test         — pytest -q"
	@echo "  make lint         — ruff check"
	@echo "  make typecheck    — mypy --strict"
	@echo "  make verify       — lint + typecheck + test (CI-equivalent)"
	@echo "  make docker-build — docker build -t adaptmem:dev ."
	@echo "  make helm-lint    — helm lint charts/adaptmem"
	@echo "  make bench-longmemeval — self-contained R@5 reproduce"
	@echo "  make clean        — strip caches and build artefacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

dev: install
	$(PYTHON) -m pip install pre-commit
	pre-commit install

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check adaptmem tests

typecheck:
	$(PYTHON) -m mypy --strict adaptmem

verify: lint typecheck test
	@echo "all gates green"

docker-build:
	docker build -t adaptmem:dev .

helm-lint:
	helm lint charts/adaptmem

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Default: self-contained reproduction (train + test on the 100/400 split).
# A stranger only needs the LongMemEval dataset; everything else is in this repo.
bench-longmemeval: bench-ft100
	@echo
	@echo "wrote: $(RESULTS_100)"
	@echo "compare to README's claimed numbers (R@5 ~ 0.985–0.99 expected)."

bench-ft100: train-100
	$(PYTHON) benchmarks/longmemeval_eval.py --mode test \
	    --dataset $(DATASET) --split-ids $(SPLIT_100) \
	    --st-model $(MODEL_100)/model --results-out $(RESULTS_100) \
	    $(if $(DEVICE),--device $(DEVICE),)

train-100:
	$(PYTHON) benchmarks/longmemeval_eval.py --mode train \
	    --dataset $(DATASET) --split-ids $(SPLIT_100) \
	    --n-train 100 --model-out $(MODEL_100) \
	    $(if $(DEVICE),--device $(DEVICE),)

# Reproduce the FT-300 / FT-200 reference numbers in the README. Requires the
# external models from metis-pair (paths overridable via MODEL_300 / MODEL_200).
bench-ft300:
	$(PYTHON) benchmarks/longmemeval_eval.py --mode test \
	    --dataset $(DATASET) --split-ids $(SPLIT_300) \
	    --st-model $(MODEL_300) --results-out $(RESULTS_300) \
	    $(if $(DEVICE),--device $(DEVICE),)

bench-ft200:
	$(PYTHON) benchmarks/longmemeval_eval.py --mode test \
	    --dataset $(DATASET) --split-ids $(SPLIT_300) \
	    --st-model $(MODEL_200) --results-out $(RESULTS_200) \
	    $(if $(DEVICE),--device $(DEVICE),)

clean-bench:
	rm -rf $(MODEL_100) $(RESULTS_100) benchmarks/*.log
