# ATLAS: Adaptive Taylor Landscape Analysis System
# Complete reproduction pipeline on Google Cloud TPU v4.

PY := python3 -W ignore
TPU_ENV := TPU_CHIPS_PER_HOST_BOUNDS="2,2,1" TPU_HOST_BOUNDS="1,1,1"

.PHONY: all train benchmark sharpness render paper proofs example clean

all: train benchmark sharpness render paper proofs

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

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	cd paper && rm -f atlas.aux atlas.bbl atlas.blg atlas.log atlas.out
