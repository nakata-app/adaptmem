PYTHON ?= python
DATASET ?= /Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json

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

.PHONY: bench-longmemeval bench-ft100 bench-ft300 bench-ft200 train-100 clean-bench

# Default: self-contained reproduction (train + test on the 100/400 split).
# A stranger only needs the LongMemEval dataset; everything else is in this repo.
bench-longmemeval: bench-ft100
	@echo
	@echo "wrote: $(RESULTS_100)"
	@echo "compare to README's claimed numbers (R@5 ~ 0.985–0.99 expected)."

bench-ft100: train-100
	$(PYTHON) benchmarks/longmemeval_eval.py --mode test \
	    --dataset $(DATASET) --split-ids $(SPLIT_100) \
	    --st-model $(MODEL_100) --results-out $(RESULTS_100)

train-100:
	$(PYTHON) benchmarks/longmemeval_eval.py --mode train \
	    --dataset $(DATASET) --split-ids $(SPLIT_100) \
	    --n-train 100 --model-out $(MODEL_100)

# Reproduce the FT-300 / FT-200 reference numbers in the README. Requires the
# external models from metis-pair (paths overridable via MODEL_300 / MODEL_200).
bench-ft300:
	$(PYTHON) benchmarks/longmemeval_eval.py --mode test \
	    --dataset $(DATASET) --split-ids $(SPLIT_300) \
	    --st-model $(MODEL_300) --results-out $(RESULTS_300)

bench-ft200:
	$(PYTHON) benchmarks/longmemeval_eval.py --mode test \
	    --dataset $(DATASET) --split-ids $(SPLIT_300) \
	    --st-model $(MODEL_200) --results-out $(RESULTS_200)

clean-bench:
	rm -rf $(MODEL_100) $(RESULTS_100) benchmarks/*.log
