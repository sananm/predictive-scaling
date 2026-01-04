.PHONY: help install install-dev test lint format typecheck clean docker-up docker-down docker-build migrate run dev

# Default target
help:
	@echo "Predictive Scaler - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-cov      Run tests with coverage"
	@echo "  test-imports  Quick smoke test (imports only)"
	@echo "  test-phase1   Test Phase 1 (config, logging, db)"
	@echo "  test-phase2   Test Phase 2 (collectors)"
	@echo "  test-phase3   Test Phase 3 (streaming)"
	@echo "  test-phase4   Test Phase 4 (features)"
	@echo "  test-phase5   Test Phase 5 (models)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          Run linter (ruff)"
	@echo "  format        Format code (black + ruff)"
	@echo "  typecheck     Run type checker (mypy)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up     Start all Docker services"
	@echo "  docker-down   Stop all Docker services"
	@echo "  docker-build  Build Docker image"
	@echo ""
	@echo "Database:"
	@echo "  migrate       Run database migrations"
	@echo ""
	@echo "Running:"
	@echo "  run           Run the application"
	@echo "  dev           Run in development mode with reload"
	@echo ""
	@echo "Other:"
	@echo "  clean         Remove build artifacts"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Test individual phases
test-phase1:
	pytest tests/unit/test_phase1_config.py -v

test-phase2:
	pytest tests/unit/test_phase2_collectors.py -v

test-phase3:
	pytest tests/unit/test_phase3_streaming.py -v

test-phase4:
	pytest tests/unit/test_phase4_features.py -v

test-phase5:
	pytest tests/unit/test_phase5_models.py -v

# Quick smoke test (imports only)
test-imports:
	python -c "from config.settings import get_settings; print('Phase 1: OK')"
	python -c "from src.collectors import PrometheusCollector; print('Phase 2: OK')"
	python -c "from src.streaming import MetricsProducer; print('Phase 3: OK')"
	python -c "from src.features import FeatureEngineer; print('Phase 4: OK')"
	python -c "from src.models import EnsembleCombiner; print('Phase 5: OK')"

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
