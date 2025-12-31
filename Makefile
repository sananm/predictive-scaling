.PHONY: help install install-dev test lint format typecheck clean docker-up docker-down docker-build migrate run dev

# Default target
help:
	@echo "Predictive Scaler - Available Commands"
	@echo "======================================="
	@echo "install       Install production dependencies"
	@echo "install-dev   Install development dependencies"
	@echo "test          Run all tests"
	@echo "test-unit     Run unit tests only"
	@echo "test-cov      Run tests with coverage"
	@echo "lint          Run linter (ruff)"
	@echo "format        Format code (black + ruff)"
	@echo "typecheck     Run type checker (mypy)"
	@echo "clean         Remove build artifacts"
	@echo "docker-up     Start all Docker services"
	@echo "docker-down   Stop all Docker services"
	@echo "docker-build  Build Docker image"
	@echo "migrate       Run database migrations"
	@echo "run           Run the application"
	@echo "dev           Run in development mode with reload"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code Quality
lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose down && docker-compose up -d

# Database
migrate:
	alembic upgrade head

migrate-new:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

migrate-down:
	alembic downgrade -1

# Running
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Training
train-models:
	python scripts/train_models.py

generate-data:
	python scripts/generate_synthetic_data.py

# Demo
demo:
	python scripts/demo.py
