.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck test check notebook data data-list process queries features diagnostics comparisons bootstrap permutation

# Extra arguments for the acquisition command, e.g.
#   make data DATA_ARGS="--competition-season 11/27 --limit-matches 5"
DATA_ARGS ?=

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync the environment and install git hooks
	uv sync
	uv run pre-commit install

format: ## Apply formatting and safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Check formatting and lint rules without modifying files
	uv run ruff format --check .
	uv run ruff check .

typecheck: ## Run static type checking
	uv run mypy

test: ## Run the test suite with coverage
	uv run pytest

check: lint typecheck test ## Run all quality gates

notebook: ## Launch JupyterLab
	uv run jupyter lab

data: ## Download the StatsBomb development dataset into data/raw/statsbomb
	uv run python -m football_intelligence.data.statsbomb $(DATA_ARGS)

data-list: ## List the competition seasons available in StatsBomb Open Data
	uv run python -m football_intelligence.data.statsbomb --list

process: ## Rebuild the processed Parquet tables from raw JSON
	uv run python -m football_intelligence.data.storage

queries: ## Run the analytical DuckDB queries over the processed tables
	uv run python -m football_intelligence.data.queries

features: ## Build the canonical shot modelling dataset
	uv run python -m football_intelligence.features.shots

diagnostics: ## Describe shot distance and angle with sample diagnostics
	uv run python -m football_intelligence.statistics.diagnostics

comparisons: ## Run the two-sample and contingency test demonstrations
	uv run python -m football_intelligence.statistics.tests

bootstrap: ## Bootstrap football quantities under different resampling units
	uv run python -m football_intelligence.statistics.bootstrap

permutation: ## Permutation tests under row-level and cluster-level exchangeability
	uv run python -m football_intelligence.statistics.permutation
