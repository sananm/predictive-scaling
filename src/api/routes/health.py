"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.database import get_db

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


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> ReadinessResponse:
    """
    Readiness probe for Kubernetes.

    Checks all dependencies are available.
    """
    # Check database
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unhealthy"

    # TODO: Check Kafka
    kafka_status = "not_configured"

    # TODO: Check Redis
    redis_status = "not_configured"

    # TODO: Check if models are loaded
    models_loaded = False

    overall_status = "ready" if db_status == "healthy" else "not_ready"

    return ReadinessResponse(
        status=overall_status,
        database=db_status,
        kafka=kafka_status,
        redis=redis_status,
        models_loaded=models_loaded,
    )
