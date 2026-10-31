"""Run the documented CLI with small workloads, without Postgres or APIs."""

import json
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_script(*args):
    # Gateway environment overrides must not change this mock-only workload.
    environment = {"MOCK_MODEL": "true", "MODEL_PRICES": "{}",
                   "DATABASE_URL": "postgresql://unused.invalid/unused"}
    return subprocess.run(
        [sys.executable, "scripts/loadtest.py", *args],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=10,
    )


def test_small_loadtest_runs_quickly_and_reports_gateway_behavior():
    started = time.perf_counter()
    result = run_script("--requests", "20", "--concurrency", "3", "--latency-scale", "0.01")
    assert result.returncode == 0, result.stderr
    assert time.perf_counter() - started < 5
    report = json.loads(result.stdout)
    assert report["requests"] == 20
    assert report["concurrency"] == 3
    assert report["cache_hits"] == report["unique_prompts"] == 10
    assert report["cache_hit_rate"] == 0.5
    assert report["cheap_responses"] == 16
    assert report["fallback_responses"] == 4
    assert report["cheap_route_rate"] == 0.8
    assert report["fallback_route_rate"] == 0.2
    # Ten cold cheap calls, two quality fallbacks, ten cached responses.
    assert report["upstream_attempts"] == {"cheap": 10, "fallback": 2}
    assert 0 < report["p50_latency_ms"] <= report["p99_latency_ms"]
    assert report["p99_latency_ms"] >= 8.3  # Scaled cheap + fallback delays.
    # Includes the rejected cheap outputs (one billed token each).
    assert Decimal(report["estimated_cost_usd"]) == Decimal("0.0104572")
    # Same prompt/output usage, expensive model on all 20 requests, no cache.
    assert Decimal(report["always_expensive_usd"]) == Decimal("0.1024")
    assert Decimal(report["saved_usd"]) == Decimal("0.0919428")
    assert report["saved_percent"] == 89.79


def test_single_request_counts_both_attempts_and_can_have_negative_savings():
    result = run_script("--requests", "1", "--concurrency", "10", "--latency-scale", "0")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["cache_hit_rate"] == report["cheap_route_rate"] == 0
    assert report["fallback_route_rate"] == 1
    assert report["upstream_attempts"] == {"cheap": 1, "fallback": 1}
    assert report["p50_latency_ms"] == report["p99_latency_ms"]
    assert Decimal(report["saved_usd"]) == Decimal("-0.0001206")


@pytest.mark.parametrize("args", [
    ("--requests", "0"), ("--concurrency", "0"),
    ("--latency-scale", "-1"), ("--latency-scale", "nan"),
])
def test_invalid_workload_is_rejected(args):
    result = run_script(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert result.stdout == ""
