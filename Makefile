PY ?= python3
PIP ?= $(PY) -m pip

.PHONY: help install install-ml install-all test lint demo clean

help:
	@echo "make install     - Install core deps (offline-friendly heuristic mode works)"
	@echo "make install-ml  - Install ML extras (torch, transformers, sentence-transformers)"
	@echo "make install-all - Install everything (ml, retrieval, docs, mcp, dev)"
	@echo "make test        - Run pytest"
	@echo "make lint        - Run ruff"
	@echo "make demo        - Run end-to-end CLI demo on bundled fixture"
	@echo "make mcp         - Start the Layer 3 MCP retrieval server"
	@echo "make clean       - Remove caches and build artifacts"

install:
	$(PIP) install -e .

install-ml:
	$(PIP) install -e ".[ml]"

install-all:
	$(PIP) install -e ".[all]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

demo:
	$(PY) -m halludetect.cli demo

mcp:
	$(PY) -m halludetect.retrieval.mcp_server

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
