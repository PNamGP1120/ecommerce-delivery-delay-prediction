.PHONY: help install notebook test lint clean

help:
	@echo "Available commands:"
	@echo "  make install   Install dependencies"
	@echo "  make notebook  Start Jupyter Lab"
	@echo "  make test      Run tests"
	@echo "  make lint      Run Ruff"
	@echo "  make clean     Remove generated caches"

install:
	uv sync

notebook:
	uv run jupyter lab

test:
	uv run pytest -v

lint:
	uv run ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +  
