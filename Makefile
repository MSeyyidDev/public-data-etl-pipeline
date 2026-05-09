.PHONY: install run quality test report clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

run:
	$(PY) -m etl.cli run-all

quality:
	$(PY) -m etl.cli quality-report

test:
	$(PY) -m pytest

report:
	$(PY) -m etl.cli report

clean:
	rm -rf output/*.db output/*.parquet output/*.csv output/*.json output/*.html
	rm -rf .pytest_cache __pycache__ etl/__pycache__ tests/__pycache__
