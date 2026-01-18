"""
FastAPI application for the Predictive Scaler.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest

from config.settings import get_settings
from src.storage.database import close_db, init_db
from src.streaming import MetricsProducer
from src.utils.logging import get_logger, setup_logging

from .middleware import RequestIdMiddleware, request_timing_middleware
from .routes import config, events, health, metrics, predictions, scaling

# Initialize logging
setup_logging()
logger = get_logger(__name__)
settings = get_settings()

# Global state for application components
app_state: dict[str, Any] = {
    "kafka_producer": None,
    "models": {},
    "scheduler": None,
}


async def init_kafka() -> MetricsProducer | None:
    """Initialize Kafka producer connection."""
    try:
        producer = MetricsProducer()
        await producer.start()
        logger.info("Kafka producer initialized")
        return producer
    except Exception as e:
        logger.warning(
            "Failed to initialize Kafka producer - streaming features disabled",
            error=str(e),
        )
        return None


async def stop_kafka(producer: MetricsProducer | None) -> None:
    """Stop Kafka producer connection."""
    if producer:
        try:
            await producer.stop()
            logger.info("Kafka producer stopped")
        except Exception as e:
            logger.error("Error stopping Kafka producer", error=str(e))


def load_models() -> dict[str, Any]:
    """Load ML models from disk if available."""
    models: dict[str, Any] = {}
    model_dir = settings.model.dir

    if not model_dir.exists():
        logger.info("Model directory does not exist, skipping model loading")
        return models

    # Try to load each model type
    model_files = {
        "short_term": model_dir / "short_term_model.pt",
        "medium_term": model_dir / "medium_term_model.joblib",
        "long_term": model_dir / "long_term_model.pkl",
    }

    for model_name, model_path in model_files.items():
        if model_path.exists():
            try:
                # Models are loaded on-demand by the prediction service
                # Here we just verify they exist
                models[model_name] = {"path": str(model_path), "loaded": False}
                logger.info(f"Model file found: {model_name}", path=str(model_path))
            except Exception as e:
                logger.warning(f"Failed to verify model: {model_name}", error=str(e))
        else:
            logger.info(f"Model file not found: {model_name}", path=str(model_path))

    return models


def create_scheduler() -> AsyncIOScheduler:
    """Create and configure the background task scheduler."""
    scheduler = AsyncIOScheduler(
        timezone="UTC",
        job_defaults={
            "coalesce": True,  # Combine missed runs
            "max_instances": 1,  # Only one instance of each job
            "misfire_grace_time": 60,  # Allow 60s grace for misfires
        },
    )
    return scheduler


async def scheduled_collect_metrics() -> None:
    """Scheduled task: Collect metrics every minute."""
    logger.debug("Running scheduled metrics collection")
    # Actual collection is handled by collectors when running


async def scheduled_short_term_predictions() -> None:
    """Scheduled task: Run short-term predictions every 5 minutes."""
    logger.debug("Running scheduled short-term predictions")
    # Actual prediction is handled by prediction service


async def scheduled_medium_term_predictions() -> None:
    """Scheduled task: Run medium-term predictions and evaluate scaling decisions."""
    logger.debug("Running scheduled medium-term predictions and scaling evaluation")
    # Actual prediction and scaling is handled by respective services


async def scheduled_long_term_predictions() -> None:
    """Scheduled task: Run long-term predictions and calculate cost savings."""
    logger.debug("Running scheduled long-term predictions")
    # Actual prediction is handled by prediction service


async def scheduled_daily_maintenance() -> None:
    """Scheduled task: Daily model retraining and report generation."""
    logger.info("Running daily maintenance tasks")
    # Model retraining and reports handled by respective services


def setup_scheduled_jobs(scheduler: AsyncIOScheduler) -> None:
    """Configure all scheduled jobs."""
    # Every 1 minute: Collect metrics
    scheduler.add_job(
        scheduled_collect_metrics,
        "interval",
        minutes=1,
        id="collect_metrics",
        name="Collect metrics from sources",
    )

    # Every 5 minutes: Short-term predictions
    scheduler.add_job(
        scheduled_short_term_predictions,
        "interval",
        minutes=5,
        id="short_term_predictions",
        name="Run short-term predictions",
    )

    # Every 15 minutes: Medium-term predictions and scaling evaluation
    scheduler.add_job(
        scheduled_medium_term_predictions,
        "interval",
        minutes=15,
        id="medium_term_predictions",
        name="Run medium-term predictions and evaluate scaling",
    )

    # Every 1 hour: Long-term predictions
    scheduler.add_job(
        scheduled_long_term_predictions,
        "interval",
        hours=1,
        id="long_term_predictions",
        name="Run long-term predictions",
    )

    # Daily at 2 AM UTC: Maintenance tasks
    scheduler.add_job(
        scheduled_daily_maintenance,
        "cron",
        hour=2,
        minute=0,
        id="daily_maintenance",
        name="Daily maintenance and retraining",
    )

    logger.info(
        "Scheduled jobs configured",
        job_count=len(scheduler.get_jobs()),
    )


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

    # Initialize Kafka connections (optional - graceful degradation)
    app_state["kafka_producer"] = await init_kafka()

    # Load ML models (optional - models may not exist yet)
    app_state["models"] = load_models()
    logger.info("Model loading complete", models_found=len(app_state["models"]))

    # Initialize and start background scheduler
    scheduler = create_scheduler()
    setup_scheduled_jobs(scheduler)
    scheduler.start()
    app_state["scheduler"] = scheduler
    logger.info("Background scheduler started")

    logger.info("Predictive Scaler started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Predictive Scaler")

    # Stop scheduler
    if app_state["scheduler"]:
        app_state["scheduler"].shutdown(wait=False)
        logger.info("Scheduler stopped")

    # Stop Kafka
    await stop_kafka(app_state["kafka_producer"])

    # Close database
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
app.include_router(config.router, prefix="/api/v1")


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
