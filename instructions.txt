# Predictive Infrastructure Scaling System - Build Specification

## Project Overview

Build a production-grade predictive infrastructure scaling system that forecasts application load 15 minutes to 7 days in advance and automatically provisions/deprovisions cloud infrastructure before demand changes occur.

**Project Name:** `predictive-scaler`

---

## Tech Stack Requirements

- **Language:** Python 3.11+
- **API Framework:** FastAPI with async support
- **Database:** PostgreSQL 15 with TimescaleDB extension for time-series data
- **Message Queue:** Apache Kafka for streaming metrics
- **Cache:** Redis for feature caching and rate limiting
- **ML Frameworks:** PyTorch (transformers), XGBoost, LightGBM, CatBoost, Prophet
- **Infrastructure:** Kubernetes Python client, Terraform (optional), Boto3 for AWS
- **Monitoring:** Prometheus client, Grafana dashboards
- **Testing:** pytest with async support, hypothesis for property-based testing

---

## Project Structure

Create a modular project with the following top-level directories:

1. **config/** - Configuration management using pydantic-settings
2. **src/api/** - FastAPI routes and middleware
3. **src/collectors/** - Data collection from various sources
4. **src/features/** - Feature engineering pipeline
5. **src/models/** - ML models (short-term, medium-term, long-term)
6. **src/prediction/** - Prediction orchestration and uncertainty quantification
7. **src/decision/** - Scaling decision engine and optimization
8. **src/execution/** - Infrastructure scaling execution (K8s, Terraform, AWS)
9. **src/streaming/** - Kafka producers and consumers
10. **src/storage/** - Database models, repositories, and feature store
11. **src/monitoring/** - Metrics, alerts, and dashboards
12. **src/services/** - Business logic services
13. **src/utils/** - Shared utilities
14. **tests/** - Unit, integration, and e2e tests
15. **scripts/** - Setup, training, and demo scripts
16. **notebooks/** - Jupyter notebooks for exploration
17. **terraform/** - Infrastructure as code (optional)
18. **kubernetes/** - K8s manifests
19. **grafana/** - Dashboard JSON definitions
20. **docs/** - Documentation

---

## Phase 1: Project Foundation

### 1.1 Project Setup
- Initialize project with pyproject.toml using hatchling build system
- Configure dependencies for all required libraries
- Set up dev dependencies (pytest, black, ruff, mypy, pre-commit)
- Create Dockerfile with multi-stage build
- Create docker-compose.yml with all services (app, postgres, redis, kafka, zookeeper, prometheus, grafana)
- Create Makefile with common commands (install, test, lint, run, docker-up, docker-down)
- Set up .env.example with all configuration variables
- Configure logging with structlog in JSON format

### 1.2 Configuration System
Create a hierarchical configuration system using pydantic-settings with the following config groups:
- **DatabaseSettings:** host, port, name, user, password, connection pool settings
- **KafkaSettings:** bootstrap servers, topics (metrics, features, predictions), consumer group
- **PrometheusSettings:** URL, scrape interval
- **KubernetesSettings:** namespace, deployment name, HPA name, in-cluster flag, config path
- **AWSSettings:** region, credentials (optional)
- **ModelSettings:** horizons for each model type, context window, model directory, retrain interval
- **ScalingSettings:** min/max instances, target utilization, headroom factor, cooldown, rate limits, optimization weights

Use environment variables with sensible defaults. Implement a cached settings getter function.

### 1.3 Database Setup
- Create Alembic configuration for migrations
- Design TimescaleDB hypertables for time-series data with appropriate retention policies
- Create the following database tables:
  - **metrics:** timestamp, service_name, metric_name, value, labels (JSONB)
  - **features:** timestamp, service_name, feature_set_version, features (JSONB)
  - **predictions:** model info, horizon, timestamps, p10/p50/p90 values, metadata
  - **scaling_decisions:** strategy, status, target config, cost estimates, execution tracking, rollback info
  - **business_events:** event type, timing, expected/actual impact, source, metadata
  - **model_performance:** model info, accuracy metrics (MAE, MAPE, RMSE), coverage stats
  - **cost_tracking:** actual vs simulated costs, savings calculations
  - **alert_logs:** alert type, severity, message, acknowledgment tracking

Create appropriate indexes for common query patterns. Use UUID primary keys and JSONB for flexible metadata.

### 1.4 Repository Layer
Create async repository classes for each entity with methods for:
- Single and batch inserts
- Queries by time range, service name, and other filters
- Aggregation queries using TimescaleDB functions (time_bucket)
- DataFrame conversion for ML pipelines
- Latest value retrieval

---

## Phase 2: Data Collection Layer

### 2.1 Base Collector
Create an abstract base collector class with:
- Configurable collection interval
- Async start/stop methods
- Background collection loop with error handling
- Abstract collect() method returning list of metric dictionaries
- Publish method for sending to Kafka

### 2.2 Prometheus Collector
Implement a collector that queries Prometheus for application metrics:
- Define default PromQL queries for: requests_per_second, latency percentiles (p50, p95, p99), error_rate, cpu_utilization, memory_utilization, active_connections, queue_depth
- Support custom queries per service
- Implement both instant queries and range queries
- Parse Prometheus response format and extract values with labels
- Handle connection errors gracefully

### 2.3 Kubernetes Collector
Implement a collector for Kubernetes metrics:
- Connect using in-cluster config or kubeconfig file
- Collect: pod count, replica status, resource requests/limits, HPA status, node capacity
- Watch for deployment events (scale up/down, rollouts)
- Support multiple namespaces and deployments

### 2.4 Business Context Collector
Implement a collector for business events that impact traffic:
- Google Calendar integration for scheduled events (product launches, marketing campaigns)
- Marketing platform API integration (placeholder for HubSpot, Mailchimp, etc.)
- CI/CD system integration for scheduled deployments
- Event impact estimation based on historical data and event characteristics
- Event caching with TTL

### 2.5 External Signals Collector
Implement a collector for external signals:
- Social media monitoring for brand mentions and trending
- News monitoring for company coverage
- Competitor activity tracking (optional)
- Signal confidence scoring based on source reliability

---

## Phase 3: Streaming Pipeline

### 3.1 Kafka Producer
Create an async Kafka producer with:
- Connection management and reconnection logic
- Message serialization (JSON with timestamp handling)
- Batching for efficiency
- Error handling and dead letter queue support
- Metrics for throughput and errors

### 3.2 Kafka Consumer
Create an async Kafka consumer with:
- Consumer group management
- Offset management (auto-commit with configurable interval)
- Message deserialization
- Processing callback registration
- Graceful shutdown handling

### 3.3 Stream Processors
Implement processors for:
- **Metrics Processor:** Validate, normalize, and store raw metrics; compute basic aggregations
- **Feature Processor:** Trigger feature engineering on new metrics; cache computed features
- **Prediction Trigger:** Trigger predictions when sufficient new data arrives

---

## Phase 4: Feature Engineering

### 4.1 Feature Configuration
Define a configuration class for feature engineering with:
- Data granularity (default: 1 minute)
- Lag window sizes (5, 10, 15, 30, 60, 120, 360, 720, 1440 minutes)
- Rolling window sizes (same as lag windows)
- Fourier periods for seasonality (1440 min = daily, 10080 min = weekly)
- Feature group toggles (time, lag, rolling, fourier, business)

### 4.2 Feature Engineer Orchestrator
Create a main feature engineering class that:
- Orchestrates all feature extractors
- Accepts a DataFrame with timestamp index and metric columns
- Returns a DataFrame with all engineered features
- Tracks feature names for model input
- Computes feature hash for versioning/caching

### 4.3 Time Feature Extractor
Extract calendar and time-based features:
- Cyclical encoding (sin/cos) for: hour of day, day of week, day of month, month of year, week of year
- Boolean flags: is_weekend, is_monday, is_friday, is_business_hours
- Time of day categories: is_morning, is_afternoon, is_evening, is_night
- Month boundary indicators: is_month_start, is_month_end
- Quarter number

### 4.4 Lag Feature Extractor
Extract lag features from the target time series:
- Simple lags at all configured windows
- Same-time-yesterday and same-time-last-week lags
- Difference from lagged values (absolute change)
- Ratio to lagged values (relative change)

### 4.5 Rolling Feature Extractor
Extract rolling window statistics:
- Basic stats: mean, std, min, max
- Percentiles: p25, p75
- Range and IQR
- Coefficient of variation
- Z-score (current value relative to rolling distribution)
- Trend indicator (slope of rolling window)

### 4.6 Fourier Feature Extractor
Extract frequency-domain features for seasonality:
- Sine and cosine components for daily seasonality (first 3 harmonics)
- Sine and cosine components for weekly seasonality (first 2 harmonics)
- Support for custom periods

### 4.7 Business Feature Extractor
Extract features from business events:
- Active campaign indicator
- Hours until campaign end
- Days until next product launch
- Event impact multiplier (decaying over event duration)
- Cumulative event impact (when multiple events overlap)

### 4.8 Derivative Features
Compute rate-of-change features:
- First derivative (velocity) at multiple scales (1m, 5m, 15m)
- Second derivative (acceleration)
- Smoothed derivatives using rolling mean

---

## Phase 5: Prediction Models

### 5.1 Model Base Class
Create an abstract base class for all prediction models with:
- train() method accepting features DataFrame and target Series
- predict() method returning predictions with uncertainty
- save() and load() methods for model persistence
- Model versioning and metadata tracking
- Quantile prediction support (p10, p50, p90)

### 5.2 Short-Term Model (Transformer)
Build a PyTorch Transformer model for 5-15 minute predictions:
- **Architecture:**
  - Input projection layer (features → d_model)
  - Positional encoding (sinusoidal)
  - Transformer encoder (4-6 layers, 8 heads, d_model=128)
  - Layer normalization
  - Separate output heads for each quantile (p10, p50, p90)
- **Configuration:** input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_length, prediction_horizon, quantiles
- **Loss function:** Quantile loss (pinball loss) averaged across quantiles
- **Training:** Adam optimizer, learning rate scheduler, early stopping, gradient clipping

### 5.3 Short-Term Model Trainer
Create a trainer class for the Transformer model:
- Dataset class for time series with sliding windows
- DataLoader with proper shuffling for training
- Training loop with validation
- Learning rate scheduling (cosine annealing or reduce on plateau)
- Early stopping based on validation loss
- Model checkpointing
- Training metrics logging

### 5.4 Medium-Term Model (Gradient Boosting Ensemble)
Build an ensemble of gradient boosting models for 1-24 hour predictions:
- **Base models:** XGBoost, LightGBM, CatBoost
- **Ensemble strategy:** Weighted average with weights optimized via cross-validation
- Train separate models for each prediction horizon (1, 2, 4, 6, 12, 24 hours)
- Use Optuna for hyperparameter optimization
- Time series cross-validation (expanding window or sliding window)

### 5.5 Medium-Term Model Trainer
Create a trainer for the ensemble:
- Feature preparation with proper target shifting for each horizon
- Cross-validation using TimeSeriesSplit
- Hyperparameter tuning with Optuna (50-100 trials)
- Weight optimization for ensemble combination
- Feature importance tracking

### 5.6 Long-Term Model (Prophet + Adjustments)
Build a Prophet-based model for 1-7 day predictions:
- Configure Prophet with: yearly, weekly, daily seasonality (multiplicative mode)
- Add custom monthly seasonality
- Add business event regressors (one per event type)
- Add US holidays
- Train a secondary model (LightGBM) to learn residual adjustments during business events
- Combine Prophet forecast with learned adjustments

### 5.7 Long-Term Model Trainer
Create a trainer for the Prophet model:
- Data preparation in Prophet format (ds, y columns)
- Event regressor creation with decay functions
- Prophet fitting with configured seasonalities
- Residual model training for business event adjustment
- Cross-validation for hyperparameter selection

### 5.8 Ensemble Combiner
Create a meta-model that combines predictions from all three model types:
- Weight predictions based on model performance at different horizons
- Handle overlapping prediction windows
- Propagate uncertainty correctly
- Support dynamic model selection based on recent accuracy

---

## Phase 6: Prediction Service

### 6.1 Predictor Orchestrator
Create a main prediction service that:
- Loads all trained models
- Fetches recent metrics and computes features
- Runs predictions across all models
- Combines results into unified prediction output
- Stores predictions in database
- Publishes to Kafka for downstream consumers

### 6.2 Uncertainty Quantification
Implement uncertainty handling:
- Combine prediction intervals from multiple models
- Compute model agreement as confidence indicator
- Calibrate prediction intervals using historical accuracy
- Flag predictions with high uncertainty for human review

### 6.3 Prediction Calibration
Implement calibration for prediction intervals:
- Track empirical coverage (% of actuals within predicted intervals)
- Adjust interval widths to achieve target coverage (e.g., 80%, 90%)
- Separate calibration by horizon and model
- Periodic recalibration based on recent performance

---

## Phase 7: Decision Engine

### 7.1 Cost Model
Create a cloud cost model that:
- Stores pricing for different instance types (on-demand and spot)
- Calculates hourly/monthly costs for a given configuration
- Estimates costs for spot instance usage with interruption risk
- Supports multiple cloud providers (AWS, GCP, Azure) via strategy pattern
- Updates pricing periodically from cloud APIs

### 7.2 Capacity Model
Create a capacity model that:
- Maps instance types to capacity (requests per second)
- Accounts for warm-up time when scaling up
- Models capacity degradation under high load
- Supports capacity testing/benchmarking to calibrate estimates

### 7.3 Risk Model
Create a risk assessment model that:
- Estimates spot instance interruption probability
- Calculates risk of under-provisioning based on prediction uncertainty
- Assesses stability risk from rapid scaling
- Produces overall risk score (0-1) for scaling decisions

### 7.4 Decision Engine
Create the main decision engine that:
- Takes current infrastructure state and predictions as input
- Calculates required capacity at different confidence levels (p10, p50, p90)
- Generates candidate scaling configurations
- Scores candidates using multi-objective optimization:
  - Cost score (lower cost = higher score)
  - Performance score (adequate headroom = higher score)
  - Stability score (fewer changes = higher score)
  - Risk score (lower risk = higher score)
  - Transition score (can complete before demand spike = higher score)
- Selects best candidate based on weighted scoring
- Determines scaling strategy (gradual_ramp, preemptive_burst, emergency_scale, scale_down)
- Generates human-readable reasoning
- Creates rollback plan
- Defines verification criteria

### 7.5 Scaling Strategies
Implement different scaling strategies:
- **Gradual Ramp:** Slowly increase/decrease capacity over time
- **Preemptive Burst:** Quickly scale up before predicted spike
- **Emergency Scale:** Immediate scaling for unexpected load
- **Scale Down:** Carefully reduce capacity during low-traffic periods

### 7.6 Candidate Generation
Implement candidate configuration generation:
- Iterate over available instance types
- Calculate instances needed for different capacity targets
- Apply constraints (min/max instances)
- Generate variants with different spot percentages (0%, 30%, 50%, 70%)
- Filter infeasible candidates

---

## Phase 8: Execution Layer

### 8.1 Executor Base Class
Create an abstract executor base with:
- scale() method to apply scaling decision
- rollback() method to revert changes
- verify() method to confirm scaling succeeded
- get_current_state() method to read infrastructure state

### 8.2 Kubernetes Executor
Implement Kubernetes scaling:
- Connect to cluster (in-cluster or via kubeconfig)
- Scale deployments by updating replica count
- Adjust HPA min/max replicas
- Suspend HPA for preemptive scaling (restore after stabilization)
- Wait for rollout completion
- Verify pod readiness
- Support for multiple deployments

### 8.3 Terraform Executor (Optional)
Implement Terraform-based scaling:
- Generate tfvars from scaling decision
- Run terraform plan and parse output
- Safety check for destructive changes
- Run terraform apply with auto-approve (or manual approval workflow)
- Parse terraform output for new resource IDs

### 8.4 AWS Executor (Optional)
Implement direct AWS scaling:
- Auto Scaling Group updates (desired capacity, min/max)
- Instance type changes via launch template updates
- Spot fleet management
- ECS service scaling

### 8.5 Verification System
Implement scaling verification:
- Wait for target replica count
- Check pod/instance health
- Verify capacity via synthetic load or metrics
- Check latency remains within SLO
- Timeout handling with automatic rollback trigger

### 8.6 Rollback System
Implement rollback capabilities:
- Store previous state before each scaling action
- Automatic rollback on verification failure
- Manual rollback trigger via API
- Rollback verification
- Alert on rollback events

---

## Phase 9: API Layer

### 9.1 FastAPI Application
Create the main FastAPI application with:
- Lifespan handler for startup/shutdown (DB connections, Kafka, model loading)
- Exception handlers for custom error types
- Request ID middleware for tracing
- CORS middleware
- Prometheus metrics middleware

### 9.2 Health Routes
- GET /health - Basic health check
- GET /health/ready - Readiness check (DB, Kafka, models loaded)
- GET /health/live - Liveness check

### 9.3 Metrics Routes
- GET /metrics - Prometheus metrics endpoint
- GET /api/v1/metrics - Query stored metrics with filters
- POST /api/v1/metrics - Ingest metrics (for testing)

### 9.4 Predictions Routes
- GET /api/v1/predictions/current - Get latest predictions for all horizons
- GET /api/v1/predictions/{horizon} - Get predictions for specific horizon
- GET /api/v1/predictions/history - Query historical predictions
- POST /api/v1/predictions/trigger - Manually trigger prediction run

### 9.5 Scaling Routes
- GET /api/v1/scaling/status - Current infrastructure state
- GET /api/v1/scaling/decisions - List recent scaling decisions
- GET /api/v1/scaling/decisions/{id} - Get specific decision details
- POST /api/v1/scaling/decisions/{id}/approve - Approve pending decision
- POST /api/v1/scaling/decisions/{id}/reject - Reject pending decision
- POST /api/v1/scaling/rollback - Trigger manual rollback

### 9.6 Events Routes
- GET /api/v1/events - List business events
- POST /api/v1/events - Create business event manually
- PUT /api/v1/events/{id} - Update event
- DELETE /api/v1/events/{id} - Delete event

### 9.7 Config Routes
- GET /api/v1/config - Get current configuration
- PUT /api/v1/config/scaling - Update scaling parameters
- PUT /api/v1/config/models - Update model parameters

---

## Phase 10: Monitoring & Observability

### 10.1 Prometheus Metrics
Define custom metrics:
- **prediction_accuracy:** Gauge for model accuracy by horizon
- **prediction_latency_seconds:** Histogram for prediction time
- **scaling_actions_total:** Counter for scaling actions by strategy and outcome
- **scaling_decision_confidence:** Gauge for decision confidence
- **cost_savings_hourly:** Gauge for estimated savings
- **model_inference_duration_seconds:** Histogram for model inference time
- **feature_engineering_duration_seconds:** Histogram for feature computation time
- **kafka_messages_processed_total:** Counter for stream processing

### 10.2 Accuracy Tracker
Create a service that:
- Compares predictions to actual values after the fact
- Calculates MAE, MAPE, RMSE for each model and horizon
- Calculates coverage (% within prediction intervals)
- Stores performance metrics in database
- Triggers alerts when accuracy degrades
- Generates daily/weekly accuracy reports

### 10.3 Cost Tracker
Create a service that:
- Records actual infrastructure costs
- Simulates what reactive scaling would have cost
- Calculates savings from predictive scaling
- Tracks savings over time
- Generates cost reports

### 10.4 Alert Manager
Create an alerting system:
- Define alert rules (accuracy degradation, failed scaling, high cost, etc.)
- Support multiple alert channels (Slack, email, PagerDuty)
- Alert acknowledgment and resolution tracking
- Alert deduplication and grouping
- Escalation policies

### 10.5 Grafana Dashboards
Create dashboard definitions for:
- **Main Dashboard:** Current load, capacity, utilization, recent scaling actions
- **Predictions Dashboard:** Predictions vs actuals, model accuracy over time
- **Cost Dashboard:** Hourly costs, savings, cost breakdown by instance type
- **Model Performance Dashboard:** Per-model metrics, feature importance
- **Alerts Dashboard:** Active alerts, alert history

---

## Phase 11: Background Services

### 11.1 Scheduler Service
Create a scheduler using APScheduler:
- **Every 1 minute:** Collect metrics, compute features
- **Every 5 minutes:** Run short-term predictions
- **Every 15 minutes:** Run medium-term predictions, evaluate scaling decisions
- **Every 1 hour:** Run long-term predictions, calculate cost savings
- **Every 24 hours:** Retrain models, recalibrate predictions, generate reports

### 11.2 Prediction Service
Background service that:
- Listens for new feature data
- Runs predictions when enough new data arrives
- Publishes predictions to Kafka
- Stores predictions in database

### 11.3 Scaling Service
Background service that:
- Monitors predictions and current state
- Triggers decision engine when action needed
- Executes approved decisions
- Monitors verification and triggers rollback if needed

### 11.4 Model Training Service
Background service for model maintenance:
- Monitors model performance
- Triggers retraining when accuracy degrades
- Handles model versioning
- Blue-green model deployment (train new, verify, swap)

---

## Phase 12: Testing

### 12.1 Unit Tests
Write unit tests for:
- All feature extractors (verify output shape and values)
- Model forward pass (verify output shapes)
- Decision engine scoring (verify calculations)
- Cost/capacity models (verify estimates)
- Repositories (using test database)

### 12.2 Integration Tests
Write integration tests for:
- Full prediction pipeline (metrics → features → prediction)
- Scaling pipeline (prediction → decision → execution)
- API endpoints (using TestClient)
- Kafka message flow

### 12.3 E2E Tests
Write end-to-end tests that:
- Simulate traffic patterns
- Verify predictions are generated
- Verify scaling decisions are made
- Verify infrastructure is scaled (using mock executor)

### 12.4 Load Tests
Create Locust load tests:
- Simulate realistic traffic patterns
- Verify system handles load
- Measure API latency under load

---

## Phase 13: Scripts & Utilities

### 13.1 Database Setup Script
Script that:
- Creates database if not exists
- Runs migrations
- Creates TimescaleDB hypertables
- Sets up retention policies
- Seeds initial configuration

### 13.2 Model Training Script
Script that:
- Loads historical data
- Runs feature engineering
- Trains all models
- Evaluates on holdout set
- Saves models with versioning

### 13.3 Synthetic Data Generator
Script that:
- Generates realistic traffic patterns with:
  - Daily seasonality (peak during business hours)
  - Weekly seasonality (lower on weekends)
  - Random noise
  - Occasional spikes (simulating events)
  - Gradual trends
- Outputs data in format compatible with the system

### 13.4 Load Generator
Script using Locust that:
- Generates configurable load patterns
- Supports gradual ramps and spikes
- Can replay historical patterns
- Useful for demos and testing

### 13.5 Demo Script
Script that:
- Starts all services
- Generates synthetic traffic
- Shows predictions in real-time
- Triggers business events
- Demonstrates scaling decisions

---

## Phase 14: Documentation

### 14.1 Architecture Documentation
Document:
- System architecture diagram
- Data flow diagram
- Component descriptions
- Technology choices and rationale

### 14.2 API Documentation
- OpenAPI/Swagger auto-generated docs
- Example requests and responses
- Authentication (if any)
- Rate limiting

### 14.3 Deployment Guide
Document:
- Prerequisites
- Environment variables
- Docker deployment
- Kubernetes deployment
- Monitoring setup

### 14.4 Runbook
Document:
- Common operational tasks
- Troubleshooting guide
- Alert response procedures
- Rollback procedures
- Disaster recovery

---

## Implementation Order

Build the system in this order to have working functionality at each step:

1. **Week 1:** Project setup, configuration, database, basic API
2. **Week 2:** Prometheus collector, Kafka streaming, feature engineering
3. **Week 3:** Short-term Transformer model, training pipeline
4. **Week 4:** Medium-term ensemble model, long-term Prophet model
5. **Week 5:** Prediction service, accuracy tracking
6. **Week 6:** Decision engine, cost/capacity/risk models
7. **Week 7:** Kubernetes executor, verification, rollback
8. **Week 8:** API completion, business event handling
9. **Week 9:** Monitoring, Grafana dashboards, alerting
10. **Week 10:** Testing, documentation, demo script
11. **Week 11-12:** Polish, optimization, edge case handling

---

## Key Technical Requirements

1. **All database operations must be async** using SQLAlchemy async
2. **All external API calls must be async** using httpx
3. **Use dependency injection** for testability
4. **Use Pydantic models** for all API request/response schemas
5. **Use structured logging** (structlog) with JSON output
6. **Use type hints** throughout the codebase
7. **Handle errors gracefully** with proper error types and messages
8. **Make all services idempotent** where possible
9. **Use database transactions** appropriately
10. **Implement circuit breakers** for external service calls

---

## Success Criteria

The system is complete when:

1. Metrics are collected every 15 seconds from Prometheus
2. Features are computed and stored in real-time
3. Short-term predictions (15 min) are generated every minute with <5% MAPE
4. Medium-term predictions (24 hr) are generated every 15 minutes with <10% MAPE
5. Long-term predictions (7 day) incorporate business events
6. Scaling decisions are made automatically with clear reasoning
7. Kubernetes deployments are scaled based on predictions
8. Rollback triggers automatically on verification failure
9. Grafana dashboards show predictions vs actuals in real-time
10. Cost savings are tracked and reported
11. All components have >80% test coverage
12. Documentation is complete for deployment and operations