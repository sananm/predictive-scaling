# Predictive Infrastructure Scaling System - API Documentation

## Base URL

```
http://localhost:8000
```

Production: Configure via `API_BASE_URL` environment variable.

## Authentication

All API endpoints (except `/health`) require JWT authentication.

```
Authorization: Bearer <token>
```

## Response Format

All responses follow this structure:

```json
{
  "data": { ... },
  "meta": {
    "timestamp": "2025-01-09T12:00:00Z",
    "request_id": "uuid"
  }
}
```

Error responses:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": { ... }
  }
}
```

---

## Endpoints

### Health Check

#### GET /health

Check system health and component status.

**Response**

```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "kafka": "healthy"
  },
  "version": "1.0.0",
  "timestamp": "2025-01-09T12:00:00Z"
}
```

**Status Codes**
- `200`: System healthy
- `503`: System unhealthy

---

### Predictions

#### GET /api/v1/predictions/current

Get current predictions for all monitored services.

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| service | string | No | Filter by service name |
| horizon | string | No | Filter by horizon (short, medium, long) |

**Response**

```json
{
  "predictions": [
    {
      "service_name": "api",
      "horizon_minutes": 15,
      "horizon_type": "short",
      "prediction_p50": 150.5,
      "prediction_p90": 180.2,
      "prediction_p99": 210.8,
      "confidence_score": 0.85,
      "timestamp": "2025-01-09T12:00:00Z",
      "features_used": ["cpu_utilization", "requests_per_second", "hour_of_day"],
      "model_version": "v2025.01.09"
    },
    {
      "service_name": "api",
      "horizon_minutes": 60,
      "horizon_type": "medium",
      "prediction_p50": 200.0,
      "prediction_p90": 250.5,
      "prediction_p99": 300.2,
      "confidence_score": 0.78,
      "timestamp": "2025-01-09T12:00:00Z"
    }
  ],
  "generated_at": "2025-01-09T12:00:00Z"
}
```

#### GET /api/v1/predictions/history

Get historical predictions with actual outcomes.

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| service | string | No | Filter by service name |
| start_time | datetime | No | Start of time range (ISO 8601) |
| end_time | datetime | No | End of time range (ISO 8601) |
| limit | int | No | Max results (default: 100, max: 1000) |

**Response**

```json
{
  "predictions": [
    {
      "id": "uuid",
      "service_name": "api",
      "horizon_minutes": 15,
      "predicted_value": 150.5,
      "actual_value": 148.2,
      "error_percentage": 1.55,
      "timestamp": "2025-01-09T11:45:00Z"
    }
  ],
  "total_count": 500,
  "accuracy_metrics": {
    "mape": 8.5,
    "rmse": 12.3
  }
}
```

---

### Scaling

#### GET /api/v1/scaling/status

Get current scaling status for all services.

**Response**

```json
{
  "service_name": "api",
  "current_instances": 5,
  "desired_instances": 5,
  "min_instances": 2,
  "max_instances": 20,
  "cpu_utilization": 0.65,
  "memory_utilization": 0.45,
  "status": "stable",
  "last_scaling_action": {
    "type": "scale_up",
    "from_instances": 4,
    "to_instances": 5,
    "timestamp": "2025-01-09T11:30:00Z"
  }
}
```

#### GET /api/v1/scaling/decisions

Get recent scaling decisions.

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| service | string | No | Filter by service name |
| status | string | No | Filter by status (pending, executed, failed, rolled_back) |
| limit | int | No | Max results (default: 50) |

**Response**

```json
{
  "decisions": [
    {
      "id": "uuid",
      "service_name": "api",
      "decision_type": "scale_up",
      "current_instances": 4,
      "target_instances": 6,
      "status": "executed",
      "confidence_score": 0.92,
      "reasoning": "Predicted traffic increase of 50% in next 15 minutes based on historical patterns and active marketing campaign",
      "triggered_by": {
        "prediction_value": 200.5,
        "threshold": 150.0,
        "business_events": ["marketing_campaign_123"]
      },
      "created_at": "2025-01-09T12:00:00Z",
      "executed_at": "2025-01-09T12:00:15Z",
      "verification": {
        "status": "success",
        "verified_at": "2025-01-09T12:02:00Z"
      }
    }
  ],
  "count": 1
}
```

#### POST /api/v1/scaling/manual

Trigger a manual scaling operation (requires admin role).

**Request Body**

```json
{
  "service_name": "api",
  "target_instances": 10,
  "reason": "Preparing for scheduled maintenance"
}
```

**Response**

```json
{
  "decision_id": "uuid",
  "status": "pending",
  "message": "Manual scaling request submitted"
}
```

---

### Events

#### POST /api/v1/events

Register a business event that may affect traffic.

**Request Body**

```json
{
  "event_type": "marketing_campaign",
  "name": "Summer Sale 2025",
  "start_time": "2025-01-10T00:00:00Z",
  "end_time": "2025-01-12T23:59:59Z",
  "expected_impact_multiplier": 2.5,
  "source": "marketing_api",
  "metadata": {
    "campaign_id": "camp_123",
    "channels": ["email", "social"]
  }
}
```

**Event Types**

| Type | Description | Typical Impact |
|------|-------------|----------------|
| marketing_campaign | Marketing campaigns | 1.5x - 3x |
| product_launch | New product releases | 2x - 5x |
| flash_sale | Limited time sales | 3x - 10x |
| scheduled_maintenance | Planned maintenance | 0.1x - 0.5x |
| holiday | Public holidays | 0.5x - 0.8x |
| partnership_announcement | Partner announcements | 1.3x - 2x |

**Response**

```json
{
  "id": "uuid",
  "event_type": "marketing_campaign",
  "name": "Summer Sale 2025",
  "status": "active",
  "created_at": "2025-01-09T12:00:00Z"
}
```

#### GET /api/v1/events/active

Get currently active business events.

**Response**

```json
{
  "events": [
    {
      "id": "uuid",
      "event_type": "marketing_campaign",
      "name": "Summer Sale 2025",
      "start_time": "2025-01-10T00:00:00Z",
      "end_time": "2025-01-12T23:59:59Z",
      "expected_impact_multiplier": 2.5,
      "status": "active"
    }
  ],
  "count": 1
}
```

#### DELETE /api/v1/events/{event_id}

Cancel/deactivate a business event.

**Response**

```json
{
  "id": "uuid",
  "status": "cancelled",
  "message": "Event cancelled successfully"
}
```

---

### Metrics

#### GET /api/v1/metrics

Get current system metrics (Prometheus format).

**Response**

```
# HELP predictions_generated_total Total predictions generated
# TYPE predictions_generated_total counter
predictions_generated_total{service="api",horizon="short"} 15234

# HELP scaling_decisions_total Total scaling decisions made
# TYPE scaling_decisions_total counter
scaling_decisions_total{service="api",type="scale_up"} 42
scaling_decisions_total{service="api",type="scale_down"} 38

# HELP prediction_latency_seconds Prediction generation latency
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.05"} 12000
prediction_latency_seconds_bucket{le="0.1"} 14500
prediction_latency_seconds_bucket{le="0.5"} 15200
```

#### GET /api/v1/metrics/accuracy

Get model accuracy metrics.

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| window | string | No | Time window (1h, 24h, 7d, 30d) |

**Response**

```json
{
  "accuracy": {
    "short_term": {
      "mape": 8.5,
      "rmse": 12.3,
      "mae": 9.8,
      "r2": 0.92
    },
    "medium_term": {
      "mape": 12.1,
      "rmse": 18.5,
      "mae": 14.2,
      "r2": 0.85
    },
    "long_term": {
      "mape": 18.5,
      "rmse": 25.3,
      "mae": 20.1,
      "r2": 0.75
    }
  },
  "window": "24h",
  "sample_count": 2880
}
```

---

### Models

#### GET /api/v1/models

List deployed models.

**Response**

```json
{
  "models": [
    {
      "name": "short_term",
      "version": "v2025.01.09",
      "algorithm": "gradient_boosting",
      "deployed_at": "2025-01-09T00:00:00Z",
      "metrics": {
        "mape": 8.5,
        "rmse": 12.3
      },
      "status": "active"
    }
  ]
}
```

#### POST /api/v1/models/retrain

Trigger model retraining (requires admin role).

**Request Body**

```json
{
  "model_name": "short_term",
  "training_days": 30
}
```

**Response**

```json
{
  "job_id": "uuid",
  "status": "queued",
  "estimated_duration_minutes": 15
}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| UNAUTHORIZED | 401 | Missing or invalid authentication |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| VALIDATION_ERROR | 422 | Invalid request parameters |
| RATE_LIMITED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Internal server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| GET endpoints | 100 requests/minute |
| POST endpoints | 20 requests/minute |
| /api/v1/metrics | 10 requests/minute |

Rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704801600
```

---

## Webhooks

Configure webhooks to receive notifications for scaling events.

### Webhook Payload

```json
{
  "event_type": "scaling.executed",
  "timestamp": "2025-01-09T12:00:00Z",
  "data": {
    "decision_id": "uuid",
    "service_name": "api",
    "from_instances": 4,
    "to_instances": 6,
    "status": "success"
  }
}
```

### Event Types

- `scaling.pending`: Scaling decision created
- `scaling.executed`: Scaling completed successfully
- `scaling.failed`: Scaling operation failed
- `scaling.rolled_back`: Scaling was rolled back
- `prediction.anomaly`: Anomalous prediction detected
- `model.retrained`: Model retraining completed

---

## SDKs

### Python

```python
from predictive_scaling import Client

client = Client(api_key="your-api-key")

# Get current predictions
predictions = client.predictions.current(service="api")

# Register an event
event = client.events.create(
    event_type="flash_sale",
    name="Black Friday",
    start_time="2025-11-29T00:00:00Z",
    end_time="2025-11-29T23:59:59Z",
    impact_multiplier=5.0
)

# Get scaling status
status = client.scaling.status()
```

---

## OpenAPI Specification

The full OpenAPI 3.0 specification is available at:

```
GET /openapi.json
```

Interactive API documentation (Swagger UI):

```
GET /docs
```

Alternative documentation (ReDoc):

```
GET /redoc
```
