#!/usr/bin/env python3
"""
Synthetic data generator for the Predictive Scaling system.

Generates realistic traffic patterns with:
- Daily seasonality (peak during business hours)
- Weekly seasonality (lower on weekends)
- Random noise
- Occasional spikes (simulating events)
- Gradual trends
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TrafficPatternGenerator:
    """Generate realistic traffic patterns."""

    def __init__(
        self,
        base_load: float = 100.0,
        daily_amplitude: float = 0.5,
        weekly_amplitude: float = 0.3,
        noise_level: float = 0.1,
        spike_probability: float = 0.02,
        spike_multiplier_range: tuple = (1.5, 3.0),
        trend_per_day: float = 0.01,
        seed: int | None = None,
    ):
        self.base_load = base_load
        self.daily_amplitude = daily_amplitude
        self.weekly_amplitude = weekly_amplitude
        self.noise_level = noise_level
        self.spike_probability = spike_probability
        self.spike_multiplier_range = spike_multiplier_range
        self.trend_per_day = trend_per_day

        if seed is not None:
            np.random.seed(seed)

    def daily_pattern(self, hour: int) -> float:
        """Daily pattern: peaks at 2pm, lowest at 4am."""
        # Shift so peak is at 14:00 (2pm)
        return 1.0 + self.daily_amplitude * np.sin((hour - 6) * np.pi / 12)

    def weekly_pattern(self, day_of_week: int) -> float:
        """Weekly pattern: lower on weekends."""
        if day_of_week >= 5:  # Saturday, Sunday
            return 1.0 - self.weekly_amplitude
        return 1.0

    def generate_load(
        self,
        timestamp: datetime,
        day_number: int = 0,
    ) -> float:
        """Generate load for a specific timestamp."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        # Base components
        load = self.base_load
        load *= self.daily_pattern(hour)
        load *= self.weekly_pattern(day_of_week)

        # Gradual trend
        load *= (1.0 + self.trend_per_day * day_number)

        # Random noise
        load *= (1.0 + np.random.normal(0, self.noise_level))

        # Occasional spikes
        if np.random.random() < self.spike_probability:
            load *= np.random.uniform(*self.spike_multiplier_range)

        return max(0, load)

    def generate_metrics(
        self,
        timestamp: datetime,
        service_name: str,
        load: float,
    ) -> dict:
        """Generate full metrics based on load."""
        # CPU correlates with load
        cpu = 0.2 + (load / (self.base_load * 3)) * 0.7
        cpu += np.random.normal(0, 0.05)
        cpu = max(0.05, min(0.98, cpu))

        # Memory is more stable
        memory = 0.4 + np.random.normal(0, 0.08)
        memory = max(0.2, min(0.9, memory))

        # Latency increases with load
        latency_p50 = 30 + (load / self.base_load) * 50 + np.random.exponential(10)
        latency_p99 = latency_p50 * (2.0 + np.random.exponential(0.5))

        # Error rate increases with high load
        error_rate = 0.001
        if cpu > 0.8:
            error_rate += (cpu - 0.8) * 0.1
        error_rate += np.random.exponential(0.002)
        error_rate = max(0, min(0.1, error_rate))

        return {
            "timestamp": timestamp.isoformat(),
            "service_name": service_name,
            "requests_per_second": round(load, 2),
            "cpu_utilization": round(cpu, 4),
            "memory_utilization": round(memory, 4),
            "latency_p50_ms": round(latency_p50, 2),
            "latency_p99_ms": round(latency_p99, 2),
            "error_rate": round(error_rate, 6),
            "active_connections": int(load * 0.5 + np.random.normal(0, 10)),
            "bytes_in_per_second": int(load * 1024 * np.random.uniform(0.8, 1.2)),
            "bytes_out_per_second": int(load * 4096 * np.random.uniform(0.8, 1.2)),
        }


class BusinessEventGenerator:
    """Generate business events that affect traffic."""

    EVENT_TYPES = [
        ("marketing_campaign", 1.5, 2.5),
        ("product_launch", 2.0, 4.0),
        ("flash_sale", 3.0, 5.0),
        ("scheduled_maintenance", 0.1, 0.3),
        ("holiday", 0.5, 0.7),
        ("partnership_announcement", 1.3, 2.0),
    ]

    def __init__(self, events_per_week: float = 2.0, seed: int | None = None):
        self.events_per_week = events_per_week
        if seed is not None:
            np.random.seed(seed)

    def generate_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Generate random business events."""
        events = []
        days = (end_date - start_date).days

        num_events = int(days / 7 * self.events_per_week)

        for _ in range(num_events):
            event_type, min_mult, max_mult = self.EVENT_TYPES[
                np.random.randint(0, len(self.EVENT_TYPES))
            ]

            # Random start time
            offset_days = np.random.randint(0, days)
            offset_hours = np.random.randint(8, 20)  # Business hours
            event_start = start_date + timedelta(days=offset_days, hours=offset_hours)

            # Duration: 1-48 hours
            duration_hours = np.random.randint(1, 49)
            event_end = event_start + timedelta(hours=duration_hours)

            events.append({
                "event_type": event_type,
                "name": f"{event_type.replace('_', ' ').title()} #{len(events) + 1}",
                "start_time": event_start.isoformat(),
                "end_time": event_end.isoformat(),
                "expected_impact_multiplier": round(
                    np.random.uniform(min_mult, max_mult), 2
                ),
            })

        return sorted(events, key=lambda x: x["start_time"])


def generate_data(
    days: int = 30,
    interval_minutes: int = 5,
    services: list[str] | None = None,
    output_dir: Path | None = None,
    include_events: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Generate synthetic data for the specified period."""
    if services is None:
        services = ["api", "worker", "web", "cache"]

    if output_dir is None:
        output_dir = Path("data/synthetic")

    output_dir.mkdir(parents=True, exist_ok=True)

    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)

    print(f"Generating data from {start_time} to {end_time}")
    print(f"Services: {services}")
    print(f"Interval: {interval_minutes} minutes")

    # Generate business events
    events = []
    if include_events:
        event_gen = BusinessEventGenerator(seed=42)
        events = event_gen.generate_events(start_time, end_time)
        print(f"Generated {len(events)} business events")

    # Generate metrics
    metrics = []
    generators = {
        service: TrafficPatternGenerator(
            base_load=100 * (1 + 0.2 * i),  # Different base loads per service
            seed=42 + i,
        )
        for i, service in enumerate(services)
    }

    current_time = start_time
    day_number = 0
    prev_day = start_time.day

    while current_time <= end_time:
        # Track day number for trend
        if current_time.day != prev_day:
            day_number += 1
            prev_day = current_time.day

        for service in services:
            gen = generators[service]

            # Check if any event is active
            event_multiplier = 1.0
            for event in events:
                event_start = datetime.fromisoformat(event["start_time"])
                event_end = datetime.fromisoformat(event["end_time"])
                if event_start <= current_time <= event_end:
                    event_multiplier *= event["expected_impact_multiplier"]

            # Generate base load
            load = gen.generate_load(current_time, day_number)
            load *= event_multiplier

            # Generate full metrics
            metric = gen.generate_metrics(current_time, service, load)
            metrics.append(metric)

        current_time += timedelta(minutes=interval_minutes)

    print(f"Generated {len(metrics)} metric records")

    # Save to files
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    if events:
        events_file = output_dir / "events.json"
        with open(events_file, "w") as f:
            json.dump(events, f, indent=2)
        print(f"Saved events to: {events_file}")

    # Also save as CSV for easier analysis
    import csv

    csv_file = output_dir / "metrics.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved CSV to: {csv_file}")

    return metrics, events


def stream_to_kafka(metrics: list[dict], topic: str = "metrics"):
    """Stream metrics to Kafka (if available)."""
    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=["localhost:29092"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        for metric in metrics:
            producer.send(topic, metric)

        producer.flush()
        print(f"Streamed {len(metrics)} records to Kafka topic: {topic}")

    except Exception as e:
        print(f"Could not stream to Kafka: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--days", type=int, default=30, help="Days of data to generate")
    parser.add_argument("--interval", type=int, default=5, help="Interval in minutes")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--services", nargs="+", default=["api", "worker", "web"], help="Services")
    parser.add_argument("--no-events", action="store_true", help="Skip business events")
    parser.add_argument("--stream-kafka", action="store_true", help="Stream to Kafka")
    args = parser.parse_args()

    print("=" * 60)
    print("Predictive Scaling - Synthetic Data Generator")
    print("=" * 60)

    metrics, events = generate_data(
        days=args.days,
        interval_minutes=args.interval,
        services=args.services,
        output_dir=Path(args.output_dir),
        include_events=not args.no_events,
    )

    if args.stream_kafka:
        stream_to_kafka(metrics)

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
