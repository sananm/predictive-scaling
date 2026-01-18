# Predictive Infrastructure Scaling System - Deployment Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- PostgreSQL 15+ with TimescaleDB extension
- Redis 7+
- Apache Kafka 3.0+
- Kubernetes cluster (for production)

## Quick Start (Local Development)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd predictive-scaling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev
```

### 2. Start Infrastructure Services

```bash
# Start all services (PostgreSQL, Redis, Kafka, Prometheus, Grafana)
make docker-up

# Verify services are running
make demo-check
```

### 3. Initialize Database

```bash
# Run database setup
make setup-db

# Or manually run migrations
alembic upgrade head
```

### 4. Train Initial Models

```bash
# Generate synthetic data and train models
make generate-data
make train-models
```

### 5. Start the Application

```bash
# Development mode with hot reload
make dev

# Production mode
make run
```

### 6. Verify Installation

```bash
# Check health
curl http://localhost:8000/health

# Run demo
make demo
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/predictive_scaling

# Redis
REDIS_URL=redis://localhost:6379/0

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:29092

# Prometheus
PROMETHEUS_URL=http://localhost:9090

# Cloud Provider (AWS example)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Model Settings
MODEL_DIR=models
PREDICTION_CONFIDENCE_THRESHOLD=0.7

# Scaling Settings
SCALING_MIN_INSTANCES=2
SCALING_MAX_INSTANCES=50
SCALING_COOLDOWN_SECONDS=300
```

### Configuration Files

- `config/settings.py`: Application settings
- `docker-compose.yml`: Local development services
- `alembic.ini`: Database migration settings

---

## Docker Deployment

### Build the Application Image

```bash
# Build image
docker build -t predictive-scaling:latest .

# Or use docker-compose
docker-compose build
```

### Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/predictive_scaling
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - redis
      - kafka

  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: predictive_scaling
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "29092:29092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
    depends_on:
      - zookeeper

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  grafana_data:
```

### Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or local)
- kubectl configured
- Helm 3.x

### Namespace Setup

```bash
kubectl create namespace predictive-scaling
kubectl config set-context --current --namespace=predictive-scaling
```

### Deploy with Helm

```bash
# Add Helm repository (if published)
helm repo add predictive-scaling https://charts.example.com

# Install
helm install predictive-scaling predictive-scaling/predictive-scaling \
  --namespace predictive-scaling \
  --values values-production.yaml
```

### Kubernetes Manifests

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: predictive-scaling-api
  labels:
    app: predictive-scaling
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: predictive-scaling
      component: api
  template:
    metadata:
      labels:
        app: predictive-scaling
        component: api
    spec:
      containers:
      - name: api
        image: predictive-scaling:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: predictive-scaling-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: predictive-scaling-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: predictive-scaling-api
spec:
  selector:
    app: predictive-scaling
    component: api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: predictive-scaling-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - scaling.example.com
    secretName: predictive-scaling-tls
  rules:
  - host: scaling.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: predictive-scaling-api
            port:
              number: 80
```

#### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: predictive-scaling-secrets
type: Opaque
stringData:
  database-url: postgresql+asyncpg://user:pass@postgres:5432/predictive_scaling
  redis-url: redis://redis:6379/0
  secret-key: your-secret-key-here
```

#### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: predictive-scaling-config
data:
  LOG_LEVEL: "INFO"
  SCALING_MIN_INSTANCES: "2"
  SCALING_MAX_INSTANCES: "50"
  PREDICTION_CONFIDENCE_THRESHOLD: "0.7"
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/predictive-scaling-api
```

---

## AWS Deployment

### Using AWS CDK

```python
# infrastructure/aws_stack.py
from aws_cdk import (
    Stack,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_rds as rds,
    aws_elasticache as elasticache,
)

class PredictiveScalingStack(Stack):
    def __init__(self, scope, id, **kwargs):
        super().__init__(scope, id, **kwargs)

        # VPC
        vpc = ec2.Vpc(self, "VPC", max_azs=3)

        # ECS Cluster
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        # RDS PostgreSQL
        database = rds.DatabaseInstance(
            self, "Database",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_15
            ),
            vpc=vpc,
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM
            ),
        )

        # ElastiCache Redis
        redis = elasticache.CfnCacheCluster(
            self, "Redis",
            cache_node_type="cache.t3.micro",
            engine="redis",
            num_cache_nodes=1,
        )
```

### Using Terraform

```hcl
# infrastructure/main.tf
provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  name   = "predictive-scaling-vpc"
  cidr   = "10.0.0.0/16"
}

module "ecs" {
  source       = "terraform-aws-modules/ecs/aws"
  cluster_name = "predictive-scaling"
}

resource "aws_rds_cluster" "postgres" {
  cluster_identifier = "predictive-scaling-db"
  engine             = "aurora-postgresql"
  engine_version     = "15.4"
  database_name      = "predictive_scaling"
  master_username    = var.db_username
  master_password    = var.db_password
}
```

---

## Production Checklist

### Security

- [ ] Enable HTTPS/TLS
- [ ] Configure authentication (JWT)
- [ ] Set up secrets management (AWS Secrets Manager, Vault)
- [ ] Enable audit logging
- [ ] Configure firewall rules / security groups
- [ ] Enable database encryption at rest
- [ ] Set up IAM roles with least privilege

### High Availability

- [ ] Deploy across multiple availability zones
- [ ] Configure database replication
- [ ] Set up Redis cluster mode
- [ ] Configure Kafka replication factor
- [ ] Set up load balancer health checks
- [ ] Configure auto-scaling for API servers

### Monitoring

- [ ] Set up Prometheus metrics collection
- [ ] Configure Grafana dashboards
- [ ] Set up alerting (PagerDuty, Slack)
- [ ] Enable distributed tracing (Jaeger)
- [ ] Configure log aggregation (ELK, Loki)

### Backup & Recovery

- [ ] Configure database backups (daily)
- [ ] Set up point-in-time recovery
- [ ] Test disaster recovery procedures
- [ ] Document rollback procedures

### Performance

- [ ] Enable connection pooling
- [ ] Configure caching strategies
- [ ] Set up CDN for static assets
- [ ] Optimize database queries
- [ ] Configure resource limits

---

## Upgrading

### Database Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1
```

### Application Updates

```bash
# Rolling update in Kubernetes
kubectl set image deployment/predictive-scaling-api \
  api=predictive-scaling:v2.0.0

# Check rollout status
kubectl rollout status deployment/predictive-scaling-api

# Rollback if needed
kubectl rollout undo deployment/predictive-scaling-api
```

### Model Updates

```bash
# Retrain models
make train-models

# Deploy new models (zero-downtime)
# Models are loaded dynamically, no restart needed
```

---

## Troubleshooting

### Common Issues

**Database connection failed**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Test connection
psql postgresql://postgres:postgres@localhost:5432/predictive_scaling
```

**Kafka connection failed**
```bash
# Check Kafka is running
docker-compose ps kafka

# List topics
docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

**Migrations failed**
```bash
# Check current migration state
alembic current

# View migration history
alembic history

# Stamp to specific version (careful!)
alembic stamp head
```

### Logs

```bash
# Docker Compose
docker-compose logs -f api

# Kubernetes
kubectl logs -f deployment/predictive-scaling-api

# Systemd
journalctl -u predictive-scaling -f
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Database health
psql -c "SELECT 1" postgresql://localhost:5432/predictive_scaling

# Redis health
redis-cli ping

# Kafka health
docker-compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```
