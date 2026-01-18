# Predictive Infrastructure Scaling System - Operations Runbook

## Overview

This runbook provides procedures for operating, monitoring, and troubleshooting the Predictive Infrastructure Scaling System.

---

## Quick Reference

| Service | Port | Health Check | Logs |
|---------|------|--------------|------|
| API | 8000 | GET /health | `docker-compose logs api` |
| PostgreSQL | 5432 | `pg_isready` | `docker-compose logs postgres` |
| Redis | 6379 | `redis-cli ping` | `docker-compose logs redis` |
| Kafka | 29092 | Broker API | `docker-compose logs kafka` |
| Prometheus | 9090 | GET /-/healthy | `docker-compose logs prometheus` |
| Grafana | 3000 | GET /api/health | `docker-compose logs grafana` |

---

## Daily Operations

### Morning Health Check

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Predictive Scaling Health Check ==="

# 1. API Health
echo -n "API: "
curl -s http://localhost:8000/health | jq -r '.status'

# 2. Database
echo -n "Database: "
docker-compose exec -T postgres pg_isready -q && echo "healthy" || echo "unhealthy"

# 3. Redis
echo -n "Redis: "
docker-compose exec -T redis redis-cli ping

# 4. Kafka
echo -n "Kafka: "
docker-compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1 && echo "healthy" || echo "unhealthy"

# 5. Recent scaling decisions
echo ""
echo "=== Last 5 Scaling Decisions ==="
curl -s http://localhost:8000/api/v1/scaling/decisions?limit=5 | jq '.decisions[] | {time: .created_at, service: .service_name, action: .decision_type, status: .status}'

# 6. Model accuracy
echo ""
echo "=== Model Accuracy (24h) ==="
curl -s http://localhost:8000/api/v1/metrics/accuracy?window=24h | jq '.accuracy'
```

### Key Metrics to Monitor

| Metric | Warning Threshold | Critical Threshold | Action |
|--------|-------------------|-------------------|--------|
| Prediction MAPE | > 15% | > 25% | Retrain models |
| API Latency (p99) | > 200ms | > 500ms | Scale API, check DB |
| Scaling Success Rate | < 95% | < 90% | Check cloud API |
| Kafka Consumer Lag | > 1000 | > 5000 | Scale consumers |
| DB Connection Pool | > 80% | > 95% | Increase pool size |

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| SEV1 | System down | 15 minutes | API unresponsive, database down |
| SEV2 | Major degradation | 1 hour | Predictions failing, scaling stuck |
| SEV3 | Minor issues | 4 hours | High latency, accuracy degraded |
| SEV4 | Informational | Next business day | Non-critical warnings |

---

## Runbook Procedures

### RB-001: API Unresponsive

**Symptoms**: Health check failing, 5xx errors

**Investigation Steps**:

```bash
# 1. Check if container is running
docker-compose ps api

# 2. Check container logs
docker-compose logs --tail=100 api

# 3. Check resource usage
docker stats predictive-scaling-api-1

# 4. Check database connectivity
docker-compose exec api python -c "
from src.storage.database import engine
import asyncio
async def test():
    async with engine.connect() as conn:
        await conn.execute('SELECT 1')
        print('DB: OK')
asyncio.run(test())
"

# 5. Check Redis connectivity
docker-compose exec api python -c "
import redis
r = redis.from_url('redis://redis:6379/0')
print('Redis:', r.ping())
"
```

**Resolution Steps**:

```bash
# Option 1: Restart API
docker-compose restart api

# Option 2: Scale up replicas (if load issue)
docker-compose up -d --scale api=3

# Option 3: Check for memory issues
docker-compose exec api cat /proc/meminfo

# Option 4: If database issue, see RB-002
```

---

### RB-002: Database Connection Issues

**Symptoms**: "Connection refused", "Too many connections", slow queries

**Investigation Steps**:

```bash
# 1. Check PostgreSQL status
docker-compose exec postgres pg_isready

# 2. Check active connections
docker-compose exec postgres psql -U postgres -c "
SELECT count(*) as total,
       state,
       wait_event_type
FROM pg_stat_activity
GROUP BY state, wait_event_type;
"

# 3. Check for long-running queries
docker-compose exec postgres psql -U postgres -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC
LIMIT 5;
"

# 4. Check disk space
docker-compose exec postgres df -h /var/lib/postgresql/data
```

**Resolution Steps**:

```bash
# Option 1: Kill long-running queries
docker-compose exec postgres psql -U postgres -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE duration > interval '5 minutes'
  AND state != 'idle';
"

# Option 2: Restart PostgreSQL (last resort)
docker-compose restart postgres

# Option 3: Increase max connections
# Edit postgresql.conf: max_connections = 200
docker-compose restart postgres

# Option 4: Clear old data (if disk full)
docker-compose exec postgres psql -U postgres -d predictive_scaling -c "
SELECT drop_chunks('metrics_raw', older_than => interval '30 days');
"
```

---

### RB-003: Predictions Failing

**Symptoms**: Empty predictions, high error rate, stale data

**Investigation Steps**:

```bash
# 1. Check prediction service logs
docker-compose logs --tail=200 api | grep -i "predict"

# 2. Verify model files exist
ls -la models/

# 3. Test prediction manually
curl -s http://localhost:8000/api/v1/predictions/current | jq

# 4. Check Kafka consumer lag
docker-compose exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe --group metrics-processor

# 5. Check feature store
docker-compose exec api python -c "
from src.features import FeatureStore
import asyncio
async def test():
    store = FeatureStore()
    features = await store.get_latest('api')
    print(f'Latest features: {features}')
asyncio.run(test())
"
```

**Resolution Steps**:

```bash
# Option 1: Restart prediction service
docker-compose restart api

# Option 2: Reload models
curl -X POST http://localhost:8000/api/v1/models/reload

# Option 3: Retrain models if accuracy degraded
make train-models

# Option 4: Check data pipeline
docker-compose logs kafka | grep -i error
docker-compose restart kafka
```

---

### RB-004: Scaling Operations Failing

**Symptoms**: Decisions created but not executed, verification failures

**Investigation Steps**:

```bash
# 1. Check recent failed decisions
curl -s "http://localhost:8000/api/v1/scaling/decisions?status=failed&limit=10" | jq

# 2. Check cloud provider connectivity
# AWS
aws sts get-caller-identity
aws autoscaling describe-auto-scaling-groups

# 3. Check execution logs
docker-compose logs api | grep -i "scaling\|executor"

# 4. Verify IAM permissions
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::ACCOUNT:role/predictive-scaling \
  --action-names autoscaling:SetDesiredCapacity
```

**Resolution Steps**:

```bash
# Option 1: Check AWS credentials
aws configure list

# Option 2: Verify scaling limits
curl -s http://localhost:8000/api/v1/scaling/status | jq

# Option 3: Clear stuck decisions
curl -X DELETE http://localhost:8000/api/v1/scaling/decisions/stuck

# Option 4: Manual scaling (emergency)
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name my-asg \
  --desired-capacity 10
```

---

### RB-005: High Prediction Latency

**Symptoms**: Prediction latency > 200ms, timeouts

**Investigation Steps**:

```bash
# 1. Check Prometheus metrics
curl -s http://localhost:9090/api/v1/query?query=prediction_latency_seconds_bucket | jq

# 2. Profile prediction
curl -w "@curl-format.txt" -s http://localhost:8000/api/v1/predictions/current > /dev/null

# 3. Check database query times
docker-compose exec postgres psql -U postgres -d predictive_scaling -c "
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
"

# 4. Check Redis cache hit rate
docker-compose exec redis redis-cli INFO stats | grep hit
```

**Resolution Steps**:

```bash
# Option 1: Clear Redis cache
docker-compose exec redis redis-cli FLUSHDB

# Option 2: Add database indexes
docker-compose exec postgres psql -U postgres -d predictive_scaling -c "
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_service_time
ON predictions (service_name, created_at DESC);
"

# Option 3: Increase API replicas
docker-compose up -d --scale api=5

# Option 4: Optimize models (reduce features)
# Edit config/settings.py: FEATURE_COUNT = 10
```

---

### RB-006: Kafka Issues

**Symptoms**: Consumer lag increasing, messages not processing

**Investigation Steps**:

```bash
# 1. Check Kafka broker status
docker-compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# 2. List topics
docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# 3. Check consumer groups
docker-compose exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --list

# 4. Check consumer lag
docker-compose exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe --group metrics-processor

# 5. Check topic partitions
docker-compose exec kafka kafka-topics \
  --describe --topic metrics.raw \
  --bootstrap-server localhost:9092
```

**Resolution Steps**:

```bash
# Option 1: Restart consumers
docker-compose restart api

# Option 2: Reset consumer offset (data loss!)
docker-compose exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group metrics-processor \
  --reset-offsets --to-latest \
  --topic metrics.raw \
  --execute

# Option 3: Add partitions (more parallelism)
docker-compose exec kafka kafka-topics \
  --alter --topic metrics.raw \
  --partitions 6 \
  --bootstrap-server localhost:9092

# Option 4: Restart Kafka (last resort)
docker-compose restart kafka
```

---

## Maintenance Procedures

### MP-001: Database Maintenance

**Weekly**: Vacuum and analyze

```bash
docker-compose exec postgres psql -U postgres -d predictive_scaling -c "
VACUUM ANALYZE;
"
```

**Monthly**: Reindex

```bash
docker-compose exec postgres psql -U postgres -d predictive_scaling -c "
REINDEX DATABASE predictive_scaling;
"
```

**Quarterly**: Check table bloat

```bash
docker-compose exec postgres psql -U postgres -d predictive_scaling -c "
SELECT schemaname, relname,
       pg_size_pretty(pg_total_relation_size(schemaname || '.' || relname)) as size
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname || '.' || relname) DESC
LIMIT 10;
"
```

---

### MP-002: Model Retraining

**When to retrain**:
- MAPE increases > 5% from baseline
- Significant traffic pattern change
- New business event types

**Procedure**:

```bash
# 1. Generate new training data
make generate-data --days 60

# 2. Train models
make train-models

# 3. Verify accuracy improvement
curl -s http://localhost:8000/api/v1/metrics/accuracy | jq

# 4. Deploy (automatic - models hot-reloaded)
```

---

### MP-003: Log Rotation

**Configure logrotate**:

```
# /etc/logrotate.d/predictive-scaling
/var/log/predictive-scaling/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 app app
}
```

---

### MP-004: Backup and Restore

**Backup database**:

```bash
# Full backup
docker-compose exec postgres pg_dump -U postgres predictive_scaling | gzip > backup_$(date +%Y%m%d).sql.gz

# TimescaleDB backup (preserves hypertables)
docker-compose exec postgres pg_dump -U postgres -Fc predictive_scaling > backup_$(date +%Y%m%d).dump
```

**Restore database**:

```bash
# From SQL dump
gunzip -c backup_20250109.sql.gz | docker-compose exec -T postgres psql -U postgres predictive_scaling

# From custom format
docker-compose exec -T postgres pg_restore -U postgres -d predictive_scaling < backup_20250109.dump
```

---

## Alerting Rules

### Prometheus Alert Rules

```yaml
# config/alerts.yml
groups:
  - name: predictive-scaling
    rules:
      - alert: APIDown
        expr: up{job="predictive-scaling-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API is down"

      - alert: HighPredictionLatency
        expr: histogram_quantile(0.99, prediction_latency_seconds_bucket) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"

      - alert: PoorPredictionAccuracy
        expr: prediction_mape > 0.20
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Prediction accuracy degraded"

      - alert: ScalingFailures
        expr: rate(scaling_failures_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Scaling operations failing"

      - alert: KafkaConsumerLag
        expr: kafka_consumer_lag > 5000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Kafka consumer lag high"

      - alert: DatabaseConnectionsHigh
        expr: pg_stat_activity_count / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connections near limit"
```

---

## Escalation Matrix

| Severity | Primary | Secondary | Management |
|----------|---------|-----------|------------|
| SEV1 | On-call engineer | Team lead | VP Engineering |
| SEV2 | On-call engineer | Team lead | - |
| SEV3 | On-call engineer | - | - |
| SEV4 | Next available engineer | - | - |

---

## Contact Information

| Role | Contact | Hours |
|------|---------|-------|
| On-call | PagerDuty | 24/7 |
| Team Lead | team-lead@example.com | Business hours |
| Database Admin | dba@example.com | Business hours |
| Cloud Support | cloud-support@example.com | 24/7 |

---

## Appendix

### Useful Commands

```bash
# Quick status check
make demo-check

# Tail all logs
docker-compose logs -f

# Enter container shell
docker-compose exec api /bin/bash

# Run database query
docker-compose exec postgres psql -U postgres -d predictive_scaling

# Clear all data (DANGER)
docker-compose down -v

# Rebuild everything
docker-compose build --no-cache && docker-compose up -d
```

### Common Queries

```sql
-- Recent predictions
SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;

-- Scaling decisions today
SELECT * FROM scaling_decisions
WHERE created_at >= CURRENT_DATE
ORDER BY created_at DESC;

-- Model performance
SELECT model_name, AVG(mape) as avg_mape
FROM model_performance
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY model_name;

-- Active events
SELECT * FROM business_events
WHERE start_time <= NOW() AND end_time >= NOW();
```
