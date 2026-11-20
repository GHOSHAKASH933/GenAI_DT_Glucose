PY ?= python

test:            ## unit tests and a one-minute end-to-end smoke test on synthetic data
	$(PY) -m pytest -q

quick:           ## 2-epoch run of the extended benchmark on the real data (needs data/)
	$(PY) paper1/p1_advanced.py --quick --seeds 0-0 --out results/quick_advanced.json

experiments:     ## full study: 8 seeds, about 5 hours on 2 CPU cores (seeds can be split across processes)
	$(PY) paper1/p1_experiments.py --seeds 8
	$(PY) paper1/p1_hybrid.py --seeds 8
	$(PY) paper1/hybrid_examples.py
	$(PY) paper1/p1_advanced.py --seeds 0-3 --out results/paper1_adv_a.json &
	$(PY) paper1/p1_advanced.py --seeds 4-7 --out results/paper1_adv_b.json ; wait
	$(PY) paper1/merge_advanced.py
	$(PY) paper1/latency_bench.py     # run on an otherwise idle machine

summary:         ## headline table of the extended benchmark as Markdown
	$(PY) paper1/summarise.py > results/summary.md

paper:           ## regenerate tables, figures and main.tex from results/*.json
	cd paper1/paper && $(PY) figures_p1.py && $(PY) figures_adv.py && $(PY) fill_p1.py

.PHONY: test quick experiments summary paper
