"""
FastAPI application for the Predictive Scaler.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from config.settings import get_settings
from src.storage.database import close_db, init_db
from src.utils.logging import get_logger, setup_logging

from .middleware import RequestIdMiddleware, request_timing_middleware
from .routes import events, health, metrics, predictions, scaling

# Initialize logging
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting Predictive Scaler", env=settings.env)

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

    # TODO: Initialize Kafka connections
    # TODO: Load ML models
    # TODO: Start background schedulers

    logger.info("Predictive Scaler started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Predictive Scaler")
    await close_db()
    logger.info("Predictive Scaler shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Predictive Scaler",
    description="Production-grade predictive infrastructure scaling system",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)
app.middleware("http")(request_timing_middleware)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.exception("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Prometheus metrics endpoint
@app.get("/metrics")
async def prometheus_metrics() -> bytes:
    """Prometheus metrics endpoint."""
    return generate_latest()


# Include routers
app.include_router(health.router)
app.include_router(metrics.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(scaling.router, prefix="/api/v1")
app.include_router(events.router, prefix="/api/v1")


def main() -> None:
    """Entry point for running the application."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
    )


if __name__ == "__main__":
    main()
