CONFIG ?= configs/base.yaml

.PHONY: ingest infer mine train evaluate pipeline test clean

ingest:
	python -m src.ingest --config $(CONFIG)

infer:
	python -m src.infer --config $(CONFIG)

mine:
	python -m src.mine --config $(CONFIG)

train:
	python -m src.train --config $(CONFIG)

evaluate:
	python -m src.evaluate --config $(CONFIG)

pipeline: ingest infer mine train evaluate

test:
	python -m pytest tests/ -v

clean:
	rm -rf runs/ __pycache__ src/**/__pycache__ tests/__pycache__
