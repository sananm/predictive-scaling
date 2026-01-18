"""
Health check endpoints.
"""

from pathlib import Path

import redis.asyncio as redis
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from src.storage.database import get_db
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness check response with component status."""

    status: str
    database: str
    kafka: str
    redis: str
    models_loaded: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(status="healthy")


@router.get("/health/live", response_model=HealthResponse)
async def liveness_check() -> HealthResponse:
    """Liveness probe for Kubernetes."""
    return HealthResponse(status="alive")


async def check_redis() -> str:
    """Check Redis connectivity."""
    try:
        redis_url = settings.redis.url
        client = redis.from_url(redis_url, decode_responses=True)
        await client.ping()
        await client.aclose()
        return "healthy"
    except Exception as e:
        logger.debug("Redis health check failed", error=str(e))
        return "unhealthy"


async def check_kafka() -> str:
    """Check Kafka connectivity."""
    try:
        from aiokafka import AIOKafkaProducer

        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka.bootstrap_servers,
            request_timeout_ms=5000,
        )
        await producer.start()
        await producer.stop()
        return "healthy"
    except Exception as e:
        logger.debug("Kafka health check failed", error=str(e))
        return "not_available"


def check_models_loaded() -> bool:
    """Check if ML models are loaded."""
    model_dir = settings.model.dir
    if not model_dir.exists():
        return False

    # Check for at least one model file
    model_patterns = ["*.pt", "*.joblib", "*.pkl"]
    for pattern in model_patterns:
        if list(model_dir.glob(pattern)):
            return True

    return False


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> ReadinessResponse:
    """
    Readiness probe for Kubernetes.

    Checks all dependencies are available:
    - Database (PostgreSQL/TimescaleDB)
    - Redis (caching and rate limiting)
    - Kafka (message streaming)
    - ML Models (prediction models)
    """
    # Check database
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning("Database health check failed", error=str(e))
        db_status = "unhealthy"

    # Check Redis
    redis_status = await check_redis()

    # Check Kafka
    kafka_status = await check_kafka()

    # Check if models are loaded
    models_loaded = check_models_loaded()

    # Determine overall status
    # Database is required, others are optional
    if db_status == "unhealthy":
        overall_status = "not_ready"
    elif redis_status == "unhealthy" or kafka_status == "unhealthy":
        overall_status = "degraded"
    else:
        overall_status = "ready"

    return ReadinessResponse(
        status=overall_status,
        database=db_status,
        kafka=kafka_status,
        redis=redis_status,
        models_loaded=models_loaded,
    )
