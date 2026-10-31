"""Behavior of dispatch order, token shares, and the mock worker comparison."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from infergate.scheduling import FIFOQueue, FairQueue, ScheduledRequest


ROOT = Path(__file__).resolve().parents[1]
SIMULATE = runpy.run_path(str(ROOT / "scripts/fair_loadtest.py"))["simulate"]


def enqueue(queue, tenant, costs):
    requests = [ScheduledRequest(f"{tenant}-{index}", tenant, cost)
                for index, cost in enumerate(costs)]
    for request in requests:
        queue.push(request)
    return requests


def drain(queue):
    result = []
    while queue:
        result.append(queue.pop())
    return result


@pytest.mark.parametrize("queue_type", [FIFOQueue, FairQueue])
def test_empty_queue_and_reuse(queue_type):
    queue = queue_type()
    assert not queue
    with pytest.raises(IndexError):
        queue.pop()
    request = enqueue(queue, "a", [100])[0]
    assert queue
    assert queue.pop() == request
    assert not queue
    assert drain(queue) == []
    assert drain_after_enqueue(queue) == ["a-0", "a-1"]


def drain_after_enqueue(queue):
    enqueue(queue, "a", [100, 100])
    return [request.request_id for request in drain(queue)]


def test_fifo_keeps_global_arrival_order():
    queue = FIFOQueue()
    expected = enqueue(queue, "noisy", [200] * 1000)
    expected += enqueue(queue, "small", [50] * 20)
    assert drain(queue) == expected


def test_fair_serves_small_tenant_before_draining_burst():
    queue = FairQueue()
    noisy = enqueue(queue, "noisy", [100] * 1000)
    small = enqueue(queue, "small", [100] * 20)
    completed = drain(queue)
    assert completed[:4] == [noisy[0], small[0], noisy[1], small[1]]
    assert completed.index(small[-1]) == 39
    assert [request for request in completed if request.tenant == "noisy"] == noisy
    assert [request for request in completed if request.tenant == "small"] == small
    assert len({request.request_id for request in completed}) == 1020


def test_weights_allocate_token_service_in_two_to_one_ratio():
    queue = FairQueue(quantum=100, weights={"a": 2})
    enqueue(queue, "a", [50] * 100)
    enqueue(queue, "b", [100] * 100)
    first = [queue.pop() for _ in range(50)]
    tokens_a = sum(request.estimated_tokens for request in first if request.tenant == "a")
    tokens_b = sum(request.estimated_tokens for request in first if request.tenant == "b")
    assert tokens_a == 2 * tokens_b == 2000


def test_equal_weights_balance_tokens_rather_than_request_counts():
    queue = FairQueue(quantum=100)
    enqueue(queue, "cheap", [25] * 100)
    enqueue(queue, "expensive", [100] * 100)
    first = [queue.pop() for _ in range(50)]
    assert sum(request.tenant == "cheap" for request in first) == 40
    assert sum(request.estimated_tokens for request in first if request.tenant == "cheap") == 1000
    assert sum(request.estimated_tokens for request in first if request.tenant == "expensive") == 1000


def test_large_head_accumulates_credit_without_reordering_or_starving():
    queue = FairQueue(quantum=100)
    large = enqueue(queue, "large", [350, 10])
    small = enqueue(queue, "small", [100] * 10)
    result = drain(queue)
    assert result[:5] == [*small[:3], *large]
    assert len(result) == 12


def test_idle_tenant_drops_credit_and_rejoins_at_end():
    queue = FairQueue(quantum=100)
    enqueue(queue, "a", [10])
    b = enqueue(queue, "b", [100] * 10)
    assert queue.pop().tenant == "a"
    a = enqueue(queue, "a", [150])
    # A cannot reuse the 90 tokens left over from its previous active period.
    assert [queue.pop(), queue.pop(), queue.pop()] == [b[0], b[1], a[0]]


def test_new_tenant_joins_while_another_tenant_has_an_active_visit():
    queue = FairQueue(quantum=100)
    a = enqueue(queue, "a", [50] * 4)
    assert queue.pop() == a[0]
    b = enqueue(queue, "b", [100])
    assert drain(queue) == [a[1], b[0], a[2], a[3]]


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan"), True])
def test_invalid_costs_quantum_and_weights_are_rejected(value):
    with pytest.raises(ValueError):
        ScheduledRequest("a", "a", value)
    with pytest.raises(ValueError):
        FairQueue(quantum=value)
    with pytest.raises(ValueError):
        FairQueue(weights={"a": value})


def test_mock_workers_measure_completion_including_wait_and_stay_busy():
    requests = [ScheduledRequest(str(index), "a", 100) for index in range(5)]
    report = SIMULATE(requests, FIFOQueue(), workers=2, tokens_per_second=1000)
    assert report["requests"] == 5
    assert report["simulated_seconds"] == 0.3
    assert report["throughput_requests_per_second"] == 16.667
    assert report["tenants"]["a"] == {
        "requests": 5, "p50_latency_ms": 200, "p95_latency_ms": 300,
    }


def test_empty_workload_has_zero_throughput():
    report = SIMULATE([], FairQueue())
    assert report["requests"] == report["simulated_seconds"] == 0
    assert report["throughput_requests_per_second"] == 0
    assert report["tenants"] == {}


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"tokens_per_second": 0}])
def test_invalid_worker_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SIMULATE([], FIFOQueue(), **kwargs)


def test_simulation_rejects_existing_work():
    queue = FairQueue()
    enqueue(queue, "a", [100])
    with pytest.raises(ValueError, match="empty queue"):
        SIMULATE([], queue)


def run_script(*args):
    return subprocess.run(
        [sys.executable, "scripts/fair_loadtest.py", *args], cwd=ROOT,
        env={**os.environ, "MOCK_MODEL": "false", "UPSTREAM_BASE_URL": "https://unused.invalid"},
        capture_output=True, text=True, timeout=10,
    )


def test_documented_cli_reports_burst_isolation_and_same_completed_work():
    result = run_script()
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["workers"] == 4
    for mode in ("fifo", "fair"):
        assert report[mode]["requests"] == 1060
        assert report[mode]["tenants"]["noisy"]["requests"] == 1000
        for tenant in ("small-1", "small-2", "small-3"):
            stats = report[mode]["tenants"][tenant]
            assert stats["requests"] == 20
            assert 0 < stats["p50_latency_ms"] <= stats["p95_latency_ms"]
    for tenant in ("small-1", "small-2", "small-3"):
        assert report["fair"]["tenants"][tenant]["p95_latency_ms"] < report["fifo"]["tenants"][tenant]["p95_latency_ms"] / 10
    assert report["fair"]["throughput_requests_per_second"] >= report["fifo"]["throughput_requests_per_second"] * 0.99
    assert result.stdout == run_script().stdout


def test_cli_rejects_zero_workers():
    result = run_script("--workers", "0")
    assert result.returncode == 2
    assert "--workers must be positive" in result.stderr
    assert not result.stdout
