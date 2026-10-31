"""Compare FIFO and token-weighted DRR with a discrete-event mock worker pool.

No gateway, database, environment configuration, network, or model APIs are used.
All requests arrive at simulated time zero, with the noisy tenant queued first.
"""

import argparse
import heapq
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from infergate.scheduling import FIFOQueue, FairQueue, ScheduledRequest, positive_integer


def workload() -> list[ScheduledRequest]:
    return [
        ScheduledRequest(f"{tenant}-{index}", tenant, (50, 100, 150, 200)[index % 4])
        for tenant, count in (("noisy", 1000), ("small-1", 20), ("small-2", 20), ("small-3", 20))
        for index in range(count)
    ]


def simulate(
    requests: Iterable[ScheduledRequest],
    queue: FIFOQueue | FairQueue,
    workers: int = 4,
    tokens_per_second: int = 1000,
) -> dict[str, Any]:
    """Dispatch only when a worker is free; latency includes queue wait + service.

    Service time equals estimated tokens / worker token rate, with no overhead.
    Worker finish times are integer token ticks, avoiding clock noise.
    """
    positive_integer(workers, "workers")
    positive_integer(tokens_per_second, "tokens_per_second")
    if queue:
        raise ValueError("simulation requires an empty queue")
    for request in requests:
        queue.push(request)
    available = [(0, worker) for worker in range(workers)]
    heapq.heapify(available)
    latencies: dict[str, list[float]] = defaultdict(list)
    elapsed_ticks = 0
    while queue:
        started, worker = heapq.heappop(available)
        request = queue.pop()
        finished = started + request.estimated_tokens
        heapq.heappush(available, (finished, worker))
        elapsed_ticks = max(elapsed_ticks, finished)
        latencies[request.tenant].append(finished / tokens_per_second * 1000)
    count = sum(len(values) for values in latencies.values())
    elapsed = elapsed_ticks / tokens_per_second

    def percentile(values: list[float], fraction: float) -> float:
        return round(sorted(values)[math.ceil(len(values) * fraction) - 1], 3)

    return {
        "requests": count,
        "simulated_seconds": elapsed,
        "throughput_requests_per_second": round(count / elapsed, 3) if elapsed else 0,
        "tenants": {
            tenant: {"requests": len(values), "p50_latency_ms": percentile(values, 0.5),
                     "p95_latency_ms": percentile(values, 0.95)}
            for tenant, values in sorted(latencies.items())
        },
    }


def run_comparison(workers: int = 4, weights: Mapping[str, int] | None = None) -> dict[str, Any]:
    requests = workload()
    return {
        "mode": "mock discrete-event simulation (no network, $0)",
        "workers": workers,
        "tokens_per_second_per_worker": 1000,
        "arrival": "all at time zero, noisy tenant first",
        "estimated_tokens": [50, 100, 150, 200],
        "quantum_tokens": 100,
        "weights": {tenant: (weights or {}).get(tenant, 1)
                    for tenant in ("noisy", "small-1", "small-2", "small-3")},
        "fifo": simulate(requests, FIFOQueue(), workers),
        "fair": simulate(requests, FairQueue(quantum=100, weights=weights), workers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    print(json.dumps(run_comparison(args.workers), indent=2))


if __name__ == "__main__":
    main()
