-- Initialize database with TimescaleDB extension
-- This script runs automatically when the PostgreSQL container starts

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
