.PHONY: install install-dev run run-verbose test lint format typecheck check clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install all dependencies including dev tools
	uv sync
	uv run pre-commit install

run: ## Run the agent in performance mode (no visualizations)
	uv run python run.py

run-verbose: ## Run the agent in verbose mode (with rendering and plotting)
	uv run python run.py --verbose

test: ## Run unit tests
	uv run pytest .

test-verbose: ## Run unit tests with verbose output
	uv run pytest . -v

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with black and isort
	uv run black .
	uv run isort .

typecheck: ## Run ty type checker
	uv run ty check

check: ## Run all checks (lint, typecheck, test)
	uv run pre-commit run --all-files
	uv run pytest .

clean: ## Remove build artifacts and cache files
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf __pycache__
	rm -rf cartpole/__pycache__
	rm -rf tests/__pycache__
	rm -rf .venv
	rm -rf experiment-results
