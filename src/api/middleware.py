"""
FastAPI middleware for request handling.
"""

import time
import uuid
from collections import deque
from collections.abc import Callable
from threading import Lock

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RequestRateTracker:
    """Track request rate over a sliding window."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.timestamps: deque[float] = deque()
        self._lock = Lock()

    def record_request(self) -> None:
        """Record a request timestamp."""
        now = time.time()
        with self._lock:
            self.timestamps.append(now)
            self._cleanup(now)

    def _cleanup(self, now: float) -> None:
        """Remove timestamps outside the window."""
        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def get_rps(self) -> float:
        """Get current requests per second."""
        now = time.time()
        with self._lock:
            self._cleanup(now)
            count = len(self.timestamps)
            if count == 0:
                return 0.0
            # Calculate RPS over the actual time span
            if count == 1:
                return 1.0 / self.window_seconds
            time_span = now - self.timestamps[0]
            if time_span <= 0:
                return float(count)
            return count / time_span


# Global request rate tracker
request_rate_tracker = RequestRateTracker(window_seconds=60.0)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add a unique request ID to each request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


async def request_timing_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to log request timing."""
    start_time = time.perf_counter()

    # Track request for RPS calculation
    request_rate_tracker.record_request()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Log request (skip health checks to reduce noise)
    if not request.url.path.startswith("/health"):
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            request_id=getattr(request.state, "request_id", None),
        )

    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    return response


def get_current_rps() -> float:
    """Get current requests per second."""
    return request_rate_tracker.get_rps()
