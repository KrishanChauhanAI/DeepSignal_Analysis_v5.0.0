.PHONY: install dev test lint typecheck check demo clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy

check: lint typecheck test

# End-to-end demonstration on SYNTHETIC data (never a real measurement).
# Synthetic data is generated only when missing: the generator never overwrites files.
demo: data/SYNTHETIC_demo_metadata.csv
	$(PYTHON) -m trapsig dry-run --config configs/demo_synthetic.yaml
	$(PYTHON) -m trapsig prepare --config configs/demo_synthetic.yaml
	$(PYTHON) -m trapsig events --config configs/demo_synthetic.yaml --rate-hz 1000

data/SYNTHETIC_demo_metadata.csv:
	$(PYTHON) -m trapsig.synthetic_cli --out data/SYNTHETIC_demo

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
