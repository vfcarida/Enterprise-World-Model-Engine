# Enterprise World Model Engine (EWM Engine)
# Canonical Makefile for developer workflows, CI verification, and artifact reproduction.

.PHONY: help reproduce test lint format typecheck docs clean

help:
	@echo "Available make targets:"
	@echo "  reproduce   - Regenerate proof-hero benchmark graphics and scientific benchmark results"
	@echo "  test        - Run unit and contract test suites"
	@echo "  lint        - Run ruff linter and style checks"
	@echo "  format      - Auto-format codebase using ruff"
	@echo "  typecheck   - Run mypy strict type checker"
	@echo "  docs        - Build documentation site with strict link checking"
	@echo "  clean       - Remove build artifacts, caches, and temporary files"

reproduce:
	python scripts/generate_hero_proof.py
	python scripts/reproduce_results.py

test:
	pytest tests/unit/ tests/contract/

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src/ewm_engine

docs:
	mkdocs build --strict

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache site/
