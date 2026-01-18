#!/usr/bin/env python3
"""
Demo script for the Predictive Scaling system.

This script:
- Starts all services (or verifies they're running)
- Generates synthetic traffic
- Shows predictions in real-time
- Triggers business events
- Demonstrates scaling decisions
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DemoRunner:
    """Run interactive demo of the predictive scaling system."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def check_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except Exception:
            return False

    async def get_predictions(self) -> dict:
        """Get current predictions."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/predictions/current"
            ) as response:
                return await response.json()

    async def get_scaling_status(self) -> dict:
        """Get current scaling status."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/scaling/status"
            ) as response:
                return await response.json()

    async def get_scaling_decisions(self) -> dict:
        """Get recent scaling decisions."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/scaling/decisions"
            ) as response:
                return await response.json()

    async def create_event(
        self,
        event_type: str,
        name: str,
        impact_multiplier: float,
        duration_hours: int = 2,
    ) -> dict:
        """Create a business event."""
        now = datetime.now(timezone.utc)
        event_data = {
            "event_type": event_type,
            "name": name,
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(hours=duration_hours)).isoformat(),
            "expected_impact_multiplier": impact_multiplier,
            "source": "demo",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/events",
                json=event_data,
            ) as response:
                return await response.json()

    async def get_events(self) -> dict:
        """Get active events."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/events/active"
            ) as response:
                return await response.json()

    def print_header(self, title: str):
        """Print a section header."""
        print()
        print("=" * 60)
        print(f" {title}")
        print("=" * 60)

    def print_status(self, status: dict):
        """Print infrastructure status."""
        print(f"  Service:      {status.get('service_name', 'N/A')}")
        print(f"  Instances:    {status.get('current_instances', 'N/A')} / {status.get('desired_instances', 'N/A')}")
        print(f"  CPU Usage:    {status.get('cpu_utilization', 0) * 100:.1f}%")
        print(f"  Memory:       {status.get('memory_utilization', 0) * 100:.1f}%")
        print(f"  Status:       {status.get('status', 'N/A')}")

    def print_predictions(self, predictions: dict):
        """Print predictions."""
        if not predictions.get("predictions"):
            print("  No predictions available yet")
            return

        for pred in predictions["predictions"]:
            horizon = pred.get("horizon_minutes", 0)
            print(f"  {horizon} min ahead:")
            print(f"    P50: {pred.get('prediction_p50', 0):.1f}")
            print(f"    P90: {pred.get('prediction_p90', 0):.1f}")

    def print_decisions(self, decisions: dict):
        """Print scaling decisions."""
        if not decisions.get("decisions"):
            print("  No scaling decisions yet")
            return

        for decision in decisions["decisions"][:5]:
            status = decision.get("status", "unknown")
            current = decision.get("current_instances", 0)
            target = decision.get("target_instances", 0)
            print(f"  [{status}] {current} -> {target} instances")
            print(f"    Reason: {decision.get('reasoning', 'N/A')[:50]}...")

    async def run_demo(self, duration_minutes: int = 5):
        """Run the interactive demo."""
        print("\n" + "=" * 60)
        print(" PREDICTIVE SCALING SYSTEM - DEMO")
        print("=" * 60)

        # Check health
        self.print_header("Checking System Health")
        if await self.check_health():
            print("  API is healthy and running")
        else:
            print("  ERROR: API is not responding!")
            print("  Please start the server: make dev")
            return

        # Show current status
        self.print_header("Current Infrastructure Status")
        try:
            status = await self.get_scaling_status()
            self.print_status(status)
        except Exception as e:
            print(f"  Could not get status: {e}")

        # Show predictions
        self.print_header("Current Predictions")
        try:
            predictions = await self.get_predictions()
            self.print_predictions(predictions)
        except Exception as e:
            print(f"  Could not get predictions: {e}")

        # Create a demo event
        self.print_header("Creating Demo Event: Flash Sale")
        try:
            event = await self.create_event(
                event_type="flash_sale",
                name="Demo Flash Sale",
                impact_multiplier=2.5,
                duration_hours=1,
            )
            print(f"  Created event: {event.get('name', 'N/A')}")
            print(f"  Impact multiplier: {event.get('expected_impact_multiplier', 0)}x")
        except Exception as e:
            print(f"  Could not create event: {e}")

        # Monitor for a while
        self.print_header(f"Monitoring for {duration_minutes} minutes...")
        print("  (Press Ctrl+C to stop)")
        print()

        start_time = time.time()
        iteration = 0

        try:
            while (time.time() - start_time) < (duration_minutes * 60):
                iteration += 1
                elapsed = int(time.time() - start_time)

                # Get current state
                try:
                    status = await self.get_scaling_status()
                    predictions = await self.get_predictions()
                    decisions = await self.get_scaling_decisions()

                    # Print update
                    print(f"\r  [{elapsed:3d}s] ", end="")
                    print(
                        f"Instances: {status.get('current_instances', '?')} | "
                        f"CPU: {status.get('cpu_utilization', 0) * 100:.0f}% | "
                        f"Decisions: {decisions.get('count', 0)}",
                        end="",
                        flush=True,
                    )

                except Exception as e:
                    print(f"\r  [{elapsed:3d}s] Error: {e}", end="", flush=True)

                await asyncio.sleep(5)

        except KeyboardInterrupt:
            print("\n\n  Demo stopped by user")

        # Final summary
        self.print_header("Demo Complete - Final State")

        try:
            status = await self.get_scaling_status()
            self.print_status(status)
        except Exception:
            pass

        self.print_header("Recent Scaling Decisions")
        try:
            decisions = await self.get_scaling_decisions()
            self.print_decisions(decisions)
        except Exception:
            pass

        print()
        print("=" * 60)
        print(" Demo finished!")
        print(" Check Grafana for visualizations: http://localhost:3000")
        print("=" * 60)


def check_services_running() -> dict:
    """Check which services are running."""
    services = {
        "api": ("localhost", 8000),
        "postgres": ("localhost", 5432),
        "redis": ("localhost", 6379),
        "kafka": ("localhost", 29092),
        "prometheus": ("localhost", 9090),
        "grafana": ("localhost", 3000),
    }

    status = {}
    import socket

    for name, (host, port) in services.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            status[name] = result == 0
            sock.close()
        except Exception:
            status[name] = False

    return status


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run predictive scaling demo")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API URL")
    parser.add_argument("--duration", type=int, default=5, help="Demo duration in minutes")
    parser.add_argument("--check-services", action="store_true", help="Check service status only")
    args = parser.parse_args()

    if args.check_services:
        print("\nChecking services...")
        status = check_services_running()
        for service, running in status.items():
            icon = "✓" if running else "✗"
            print(f"  [{icon}] {service}")
        return

    demo = DemoRunner(base_url=args.url)
    asyncio.run(demo.run_demo(duration_minutes=args.duration))


if __name__ == "__main__":
    main()
