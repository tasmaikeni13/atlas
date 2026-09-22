# ATLAS: Adaptive Taylor Landscape Analysis System
# Complete reproduction pipeline on Google Cloud TPU v4.

VENV_PY := $(shell if [ -f .venv/bin/python ]; then echo ".venv/bin/python"; else echo "python3"; fi)
PY := $(VENV_PY) -W ignore
TPU_ENV := TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1"

.PHONY: all train benchmark sharpness render paper proofs example clean smoke_125m train_125m benchmark_125m smoke_vit train_vit_imagenet sweep_vit benchmark_vit phases phase-status

all: train benchmark sharpness render paper proofs phases

smoke_125m:
	$(TPU_ENV) $(PY) test_125m_smoke.py

train_125m:
	$(TPU_ENV) $(PY) experiments/05_train_125m_fineweb.py

benchmark_125m:
	$(TPU_ENV) $(PY) experiments/06_benchmark_125m_all_methods.py

smoke_vit:
	$(TPU_ENV) $(PY) test_vit_imagenet_smoke.py

train_vit_imagenet:
	$(TPU_ENV) $(PY) experiments/07_train_vit_imagenet100.py

sweep_vit:
	$(TPU_ENV) $(PY) experiments/08_vit_sweep_diagnostics.py

benchmark_vit:
	$(TPU_ENV) $(PY) experiments/09_benchmark_vit_all_methods.py

train:
	$(TPU_ENV) $(PY) experiments/01_train_vit.py
	$(TPU_ENV) $(PY) experiments/01_train_transformer.py

benchmark:
	$(TPU_ENV) $(PY) experiments/02_benchmark.py --model vit
	$(TPU_ENV) $(PY) experiments/02_benchmark.py --model transformer

sharpness:
	$(TPU_ENV) $(PY) experiments/03_sharpness_audit.py

render:
	$(TPU_ENV) $(PY) experiments/04_render_all.py

paper:
	cd paper && pdflatex -interaction=nonstopmode atlas.tex >/dev/null \
	  && bibtex atlas >/dev/null \
	  && pdflatex -interaction=nonstopmode atlas.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode atlas.tex >/dev/null
	@echo "paper/atlas.pdf generated successfully."

proofs:
	cd proofs/AtlasCert && ~/.elan/bin/lake build

example:
	$(TPU_ENV) $(PY) examples/quickstart.py

rag-index:
	$(PY) -m rag.index

rag-test:
	$(PY) -m unittest rag/tests/test_rag.py

phases:
	$(PY) phases/run_phase.py --all

phase-status:
	$(PY) phases/run_phase.py --status

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	cd paper && rm -f atlas.aux atlas.bbl atlas.blg atlas.log atlas.out


