#!/usr/bin/env python3
"""
Load generator for the Predictive Scaling system.

Generates configurable load patterns:
- Constant load
- Gradual ramps
- Spike patterns
- Replay historical patterns
- Sine wave patterns

Can be used for demos and testing.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LoadPattern:
    """Base class for load patterns."""

    def get_rate(self, elapsed_seconds: float) -> float:
        """Get requests per second at given time."""
        raise NotImplementedError


class ConstantLoad(LoadPattern):
    """Constant load pattern."""

    def __init__(self, rps: float):
        self.rps = rps

    def get_rate(self, elapsed_seconds: float) -> float:
        return self.rps


class RampLoad(LoadPattern):
    """Gradual ramp up/down pattern."""

    def __init__(
        self,
        start_rps: float,
        end_rps: float,
        duration_seconds: float,
    ):
        self.start_rps = start_rps
        self.end_rps = end_rps
        self.duration_seconds = duration_seconds

    def get_rate(self, elapsed_seconds: float) -> float:
        progress = min(1.0, elapsed_seconds / self.duration_seconds)
        return self.start_rps + (self.end_rps - self.start_rps) * progress


class SpikeLoad(LoadPattern):
    """Spike pattern with sudden increases."""

    def __init__(
        self,
        base_rps: float,
        spike_rps: float,
        spike_interval: float,
        spike_duration: float,
    ):
        self.base_rps = base_rps
        self.spike_rps = spike_rps
        self.spike_interval = spike_interval
        self.spike_duration = spike_duration

    def get_rate(self, elapsed_seconds: float) -> float:
        cycle_position = elapsed_seconds % self.spike_interval
        if cycle_position < self.spike_duration:
            return self.spike_rps
        return self.base_rps


class SineWaveLoad(LoadPattern):
    """Sine wave pattern for gradual oscillation."""

    def __init__(
        self,
        base_rps: float,
        amplitude: float,
        period_seconds: float,
    ):
        self.base_rps = base_rps
        self.amplitude = amplitude
        self.period_seconds = period_seconds

    def get_rate(self, elapsed_seconds: float) -> float:
        return self.base_rps + self.amplitude * np.sin(
            2 * np.pi * elapsed_seconds / self.period_seconds
        )


class LoadGenerator:
    """Generate HTTP load against the API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        pattern: LoadPattern = None,
    ):
        self.base_url = base_url
        self.pattern = pattern or ConstantLoad(10)
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "total_latency": 0.0,
        }

    async def make_request(self, session: aiohttp.ClientSession, endpoint: str) -> float:
        """Make a single request and return latency."""
        url = f"{self.base_url}{endpoint}"
        start = time.time()

        try:
            async with session.get(url) as response:
                await response.read()
                latency = time.time() - start

                self.stats["requests"] += 1
                self.stats["total_latency"] += latency

                if response.status == 200:
                    self.stats["successes"] += 1
                else:
                    self.stats["failures"] += 1

                return latency

        except Exception as e:
            self.stats["requests"] += 1
            self.stats["failures"] += 1
            return time.time() - start

    async def run(
        self,
        duration_seconds: float,
        endpoints: list[str] | None = None,
    ) -> dict:
        """Run load generation for specified duration."""
        if endpoints is None:
            endpoints = [
                "/health",
                "/api/v1/predictions/current",
                "/api/v1/scaling/status",
                "/api/v1/events",
            ]

        print(f"Starting load generation against {self.base_url}")
        print(f"Duration: {duration_seconds}s")
        print(f"Endpoints: {endpoints}")
        print()

        start_time = time.time()
        self.stats = {"requests": 0, "successes": 0, "failures": 0, "total_latency": 0.0}

        async with aiohttp.ClientSession() as session:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    break

                # Get current target rate
                target_rps = self.pattern.get_rate(elapsed)
                interval = 1.0 / max(0.1, target_rps)

                # Pick random endpoint
                endpoint = endpoints[np.random.randint(0, len(endpoints))]

                # Make request
                await self.make_request(session, endpoint)

                # Print progress every 10 seconds
                if int(elapsed) % 10 == 0 and elapsed > 0:
                    avg_latency = (
                        self.stats["total_latency"] / max(1, self.stats["requests"])
                    )
                    print(
                        f"[{int(elapsed):3d}s] "
                        f"RPS: {target_rps:.1f} | "
                        f"Requests: {self.stats['requests']} | "
                        f"Success: {self.stats['successes']} | "
                        f"Avg Latency: {avg_latency*1000:.1f}ms"
                    )

                # Wait for next request
                await asyncio.sleep(interval)

        # Final stats
        elapsed = time.time() - start_time
        actual_rps = self.stats["requests"] / elapsed
        avg_latency = self.stats["total_latency"] / max(1, self.stats["requests"])
        success_rate = self.stats["successes"] / max(1, self.stats["requests"]) * 100

        results = {
            "duration_seconds": elapsed,
            "total_requests": self.stats["requests"],
            "successful_requests": self.stats["successes"],
            "failed_requests": self.stats["failures"],
            "actual_rps": actual_rps,
            "avg_latency_ms": avg_latency * 1000,
            "success_rate_percent": success_rate,
        }

        return results


async def run_scenario(scenario: str, base_url: str, duration: int) -> dict:
    """Run a predefined load scenario."""
    scenarios = {
        "constant": ConstantLoad(rps=50),
        "ramp-up": RampLoad(start_rps=10, end_rps=100, duration_seconds=duration),
        "ramp-down": RampLoad(start_rps=100, end_rps=10, duration_seconds=duration),
        "spike": SpikeLoad(base_rps=30, spike_rps=150, spike_interval=60, spike_duration=10),
        "wave": SineWaveLoad(base_rps=50, amplitude=30, period_seconds=120),
    }

    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}")

    pattern = scenarios[scenario]
    generator = LoadGenerator(base_url=base_url, pattern=pattern)

    return await generator.run(duration_seconds=duration)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate load for testing")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument(
        "--scenario",
        type=str,
        default="constant",
        choices=["constant", "ramp-up", "ramp-down", "spike", "wave"],
        help="Load scenario",
    )
    parser.add_argument("--rps", type=float, default=50, help="Requests per second (for constant)")
    args = parser.parse_args()

    print("=" * 60)
    print("Predictive Scaling - Load Generator")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Target URL: {args.url}")
    print(f"Duration: {args.duration}s")
    print("=" * 60)
    print()

    results = asyncio.run(run_scenario(args.scenario, args.url, args.duration))

    print()
    print("=" * 60)
    print("Load Generation Complete")
    print("=" * 60)
    print(f"Duration:        {results['duration_seconds']:.1f}s")
    print(f"Total Requests:  {results['total_requests']}")
    print(f"Successful:      {results['successful_requests']}")
    print(f"Failed:          {results['failed_requests']}")
    print(f"Actual RPS:      {results['actual_rps']:.1f}")
    print(f"Avg Latency:     {results['avg_latency_ms']:.1f}ms")
    print(f"Success Rate:    {results['success_rate_percent']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
