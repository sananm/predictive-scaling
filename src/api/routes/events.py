"""
Business Events API routes.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.database import get_db
from src.storage.repositories import BusinessEventsRepository

router = APIRouter(prefix="/events", tags=["Events"])


class EventInput(BaseModel):
    """Input model for creating a business event."""

    event_type: str = Field(..., description="Type of event (e.g., 'marketing_campaign', 'product_launch')")
    name: str = Field(..., description="Name of the event")
    description: str | None = None
    start_time: datetime
    end_time: datetime
    expected_impact_multiplier: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Expected traffic multiplier (1.0 = no change)",
    )
    source: str = Field(default="manual", description="Source of the event")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventResponse(BaseModel):
    """Response model for a business event."""

    id: UUID
    created_at: datetime
    event_type: str
    name: str
    description: str | None
    start_time: datetime
    end_time: datetime
    expected_impact_multiplier: float
    actual_impact_multiplier: float | None
    source: str
    metadata: dict[str, Any]
    is_active: bool


class EventListResponse(BaseModel):
    """Response for list of events."""

    events: list[EventResponse]
    count: int


def _event_to_response(event: Any) -> EventResponse:
    """Convert event model to response."""
    return EventResponse(
        id=event.id,
        created_at=event.created_at,
        event_type=event.event_type,
        name=event.name,
        description=event.description,
        start_time=event.start_time,
        end_time=event.end_time,
        expected_impact_multiplier=event.expected_impact_multiplier,
        actual_impact_multiplier=event.actual_impact_multiplier,
        source=event.source,
        metadata=event.metadata,
        is_active=event.is_active,
    )


@router.get("", response_model=EventListResponse)
async def list_events(
    active_only: bool = False,
    hours_ahead: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> EventListResponse:
    """List business events."""
    repo = BusinessEventsRepository(db)

    if active_only:
        events = await repo.get_active_events()
    elif hours_ahead is not None:
        events = await repo.get_upcoming_events(hours_ahead)
    else:
        # Get all recent events
        events = await repo.get_upcoming_events(hours_ahead=168)  # 1 week

    return EventListResponse(
        events=[_event_to_response(e) for e in events],
        count=len(events),
    )


@router.get("/active", response_model=EventListResponse)
async def get_active_events(
    db: AsyncSession = Depends(get_db),
) -> EventListResponse:
    """Get currently active events."""
    repo = BusinessEventsRepository(db)
    events = await repo.get_active_events()

    return EventListResponse(
        events=[_event_to_response(e) for e in events],
        count=len(events),
    )


@router.get("/upcoming", response_model=EventListResponse)
async def get_upcoming_events(
    hours_ahead: int = 24,
    db: AsyncSession = Depends(get_db),
) -> EventListResponse:
    """Get upcoming events within the specified hours."""
    repo = BusinessEventsRepository(db)
    events = await repo.get_upcoming_events(hours_ahead)

    return EventListResponse(
        events=[_event_to_response(e) for e in events],
        count=len(events),
    )


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> EventResponse:
    """Get a specific event by ID."""
    repo = BusinessEventsRepository(db)
    event = await repo.get_by_id(event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return _event_to_response(event)


@router.post("", response_model=EventResponse, status_code=201)
async def create_event(
    event: EventInput,
    db: AsyncSession = Depends(get_db),
) -> EventResponse:
    """Create a new business event."""
    repo = BusinessEventsRepository(db)
    created = await repo.create(
        event_type=event.event_type,
        name=event.name,
        description=event.description,
        start_time=event.start_time,
        end_time=event.end_time,
        expected_impact_multiplier=event.expected_impact_multiplier,
        source=event.source,
        metadata=event.metadata,
        is_active=True,
    )

    return _event_to_response(created)


@router.put("/{event_id}", response_model=EventResponse)
async def update_event(
    event_id: UUID,
    event: EventInput,
    db: AsyncSession = Depends(get_db),
) -> EventResponse:
    """Update an existing event."""
    repo = BusinessEventsRepository(db)
    updated = await repo.update(
        event_id,
        event_type=event.event_type,
        name=event.name,
        description=event.description,
        start_time=event.start_time,
        end_time=event.end_time,
        expected_impact_multiplier=event.expected_impact_multiplier,
        source=event.source,
        metadata=event.metadata,
    )

    if not updated:
        raise HTTPException(status_code=404, detail="Event not found")

    return _event_to_response(updated)


@router.delete("/{event_id}", status_code=204)
async def delete_event(
    event_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete an event."""
    repo = BusinessEventsRepository(db)
    deleted = await repo.delete(event_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Event not found")
