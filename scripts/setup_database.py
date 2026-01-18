#!/usr/bin/env python3
"""
Database setup script for the Predictive Scaling system.

This script:
- Creates the database if it doesn't exist
- Runs migrations
- Creates TimescaleDB hypertables
- Sets up retention policies
- Seeds initial configuration
"""

import asyncio
import sys
from pathlib import Path

import asyncpg
from alembic import command
from alembic.config import Config

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings


async def create_database_if_not_exists(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> bool:
    """Create the database if it doesn't exist."""
    # Connect to postgres database to create our database
    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres",
        )

        # Check if database exists
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)",
            database,
        )

        if not exists:
            # Create database
            await conn.execute(f'CREATE DATABASE "{database}"')
            print(f"Created database: {database}")
            await conn.close()
            return True
        else:
            print(f"Database already exists: {database}")
            await conn.close()
            return False

    except Exception as e:
        print(f"Error creating database: {e}")
        raise


async def setup_timescaledb(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> None:
    """Set up TimescaleDB extension and hypertables."""
    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )

    try:
        # Create TimescaleDB extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        print("TimescaleDB extension enabled")

    except Exception as e:
        print(f"Note: TimescaleDB setup: {e}")

    finally:
        await conn.close()


async def setup_retention_policies(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> None:
    """Set up data retention policies for hypertables."""
    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )

    try:
        # Set retention policy for metrics (keep 30 days)
        await conn.execute("""
            SELECT add_retention_policy('metrics', INTERVAL '30 days', if_not_exists => true)
        """)
        print("Retention policy set for metrics: 30 days")

        # Set retention policy for features (keep 30 days)
        await conn.execute("""
            SELECT add_retention_policy('features', INTERVAL '30 days', if_not_exists => true)
        """)
        print("Retention policy set for features: 30 days")

        # Set retention policy for predictions (keep 90 days)
        await conn.execute("""
            SELECT add_retention_policy('predictions', INTERVAL '90 days', if_not_exists => true)
        """)
        print("Retention policy set for predictions: 90 days")

    except Exception as e:
        print(f"Note: Retention policies: {e}")

    finally:
        await conn.close()


async def seed_initial_config(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> None:
    """Seed initial configuration data."""
    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )

    try:
        # Check if we have any services configured
        count = await conn.fetchval("SELECT COUNT(*) FROM scaling_decisions")

        if count == 0:
            print("Database is empty - ready for data")
        else:
            print(f"Database has {count} scaling decisions")

    except Exception as e:
        print(f"Note: Seeding check: {e}")

    finally:
        await conn.close()


def run_migrations() -> None:
    """Run Alembic migrations."""
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    print("Migrations completed")


def parse_database_url(url: str) -> dict:
    """Parse database URL into components."""
    # postgresql+asyncpg://user:password@host:port/database
    from urllib.parse import urlparse

    parsed = urlparse(url.replace("+asyncpg", ""))
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "postgres",
        "password": parsed.password or "postgres",
        "database": parsed.path.lstrip("/") or "predictive_scaler",
    }


async def main() -> None:
    """Main setup function."""
    print("=" * 60)
    print("Predictive Scaling - Database Setup")
    print("=" * 60)

    settings = get_settings()
    db_config = parse_database_url(settings.database.url)

    print(f"\nDatabase: {db_config['database']}@{db_config['host']}:{db_config['port']}")
    print()

    # Step 1: Create database
    print("Step 1: Creating database...")
    await create_database_if_not_exists(**db_config)

    # Step 2: Set up TimescaleDB
    print("\nStep 2: Setting up TimescaleDB...")
    await setup_timescaledb(**db_config)

    # Step 3: Run migrations
    print("\nStep 3: Running migrations...")
    run_migrations()

    # Step 4: Set up retention policies
    print("\nStep 4: Setting up retention policies...")
    await setup_retention_policies(**db_config)

    # Step 5: Seed initial config
    print("\nStep 5: Checking initial data...")
    await seed_initial_config(**db_config)

    print("\n" + "=" * 60)
    print("Database setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
