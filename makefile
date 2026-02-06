# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: help docs install install-rust install-uv install-dependencies test lint format type-check check clean

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies using uv
	@echo "Installing the 'DoD' project..."
	@$(MAKE) --quiet install-rust
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet install-commit-cli
	@echo "Installed the 'DoD' project."

install-rust: ## Install Rust if not present (used by uv for certain wheels)
	@if [ "$(shell which rustup)" = "" ]; then \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		echo "Installed Rust."; \
	fi

install-uv: ## Install uv if not already installed
	@if [ "$(shell which uv)" = "" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
        echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update; \
	fi

install-dependencies: ## Set up Python and sync dependencies
	@uv python install 3.11
	@uv sync --all-extras --python 3.11
	@if [ ! -d .git ]; then \
		echo "🔧 Initializing Git repository..."; \
		git init && git add . && git commit -m 'Initial commit'; \
	fi
	@uv run pre-commit install

install-commit-cli: ## Install semantic-git-commit-cli (sgc)
	@if [ "$(shell which sgc)" = "" ]; then \
		echo "Installing semantic-git-commit-cli..."; \
		npm install -g semantic-git-commit-cli; \
	else \
		echo "semantic-git-commit-cli already installed."; \
	fi

docs: ## View documentation locally
	@echo "Viewing documentation - run 'make publish-docs' to publish the documentation website."
	@uv run mkdocs serve

test: ## Run tests
	@uv run pytest && uv run readme-cov && rm .coverage*

lint: ## Lint the project
	uv run ruff check . --fix

format: ## Format the project
	uv run ruff format .

type-check: ## Type-check the project
	@uv run ty check

check: lint format type-check ## Lint, format, and type-check the code

clean: ## Remove outputs and Python/cache artifacts
	@echo "Cleaning outputs and Python caches..."
	@rm -rf outputs htmlcov .pytest_cache .mypy_cache .ruff_cache .hypothesis .tox .nox .pytype
	@rm -f .coverage .coverage.*
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	@echo "Clean complete."
