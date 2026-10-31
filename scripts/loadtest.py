"""Exercise the gateway with priced, delayed fakes. No network or Postgres needed.

Run from the repo root: uv run --no-env-file python scripts/loadtest.py
Prices and token usage are illustrative, not live model measurements.
"""

import argparse
import asyncio
import json
import math
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx

from infergate.app import create_app
from infergate.config import ModelPrice, ModelRoute, Settings
from infergate.database import CachedResponse, Database


CHEAP = "gpt-4o-mini"
EXPENSIVE = "gpt-4o"
PRICES = {
    CHEAP: ModelPrice(Decimal("0.15"), Decimal("0.60")),
    EXPENSIVE: ModelPrice(Decimal("2.50"), Decimal("10.00")),
}


class MemoryDatabase(Database):
    """Ephemeral cache and real gateway accounting; spend caps are disabled."""

    def __init__(self) -> None:
        self.cache: dict[str, tuple[CachedResponse, datetime]] = {}
        self.logs: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        pass

    def connection(self):
        raise RuntimeError("The mock load test cannot connect to a database")

    async def get_cached(self, key: str, now: datetime) -> CachedResponse | None:
        cached = self.cache.get(key)
        return cached[0] if cached and cached[1] > now else None

    async def put_cached(self, key: str, response: CachedResponse, expires_at: datetime) -> None:
        self.cache[key] = response, expires_at

    async def log_request(self, **row: Any) -> None:
        self.logs.append(row)


def tokens(prompt_id: int) -> tuple[int, int]:
    return 800 + 200 * (prompt_id % 5), 160 + 40 * (prompt_id % 4)


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
    price = PRICES[model]
    return (prompt_tokens * price.prompt + completion_tokens * price.completion) / Decimal(1_000_000)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, including all client requests."""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


async def run_loadtest(
    requests: int = 200, concurrency: int = 10, latency_scale: float = 1.0,
) -> dict[str, Any]:
    if requests < 1 or concurrency < 1:
        raise ValueError("requests and concurrency must be positive")
    if not math.isfinite(latency_scale) or latency_scale < 0:
        raise ValueError("latency_scale must be finite and non-negative")

    database = MemoryDatabase()
    upstream_calls = {CHEAP: 0, EXPENSIVE: 0}

    async def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        prompt_id = int(body["messages"][0]["content"].rsplit(" ", 1)[1])
        upstream_calls[model] += 1
        # Vary latency deterministically. Every fifth cheap answer is too short.
        delay_ms = (180 + 20 * (prompt_id % 5)) if model == CHEAP else (650 + 100 * (prompt_id % 4))
        await asyncio.sleep(delay_ms / 1000 * latency_scale)
        prompt_tokens, completion_tokens = tokens(prompt_id)
        rejected = model == CHEAP and prompt_id % 5 == 0
        if rejected:
            completion_tokens = 1
        return httpx.Response(200, json={
            "id": f"chatcmpl-loadtest-{prompt_id}-{model}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "?" if rejected else "A useful mock answer."},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

    # Explicit settings ignore gateway env config. MockTransport is the only
    # upstream path; mock_model=False exercises the gateway's USD accounting.
    settings = Settings(
        database_url="unused",
        upstream_base_url="https://upstream.invalid/v1",
        model_routes={"auto": ModelRoute(CHEAP, EXPENSIVE, min_output_chars=8)},
        model_prices=PRICES,
    )
    app = create_app(settings, upstream_transport=httpx.MockTransport(upstream), database=database)
    latencies: list[float] = []
    served = {CHEAP: 0, EXPENSIVE: 0}
    unique = (requests + 1) // 2
    prompt_ids = [index % unique for index in range(requests)]
    baseline = sum((cost(EXPENSIVE, *tokens(index)) for index in prompt_ids), Decimal(0))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid",
        ) as client:
            async def send(prompt_id: int) -> None:
                started = time.perf_counter()
                response = await client.post("/v1/chat/completions", json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": f"Summarize support ticket {prompt_id}"}],
                    "max_tokens": 320,
                })
                response.raise_for_status()
                served[response.json()["model"]] += 1
                latencies.append((time.perf_counter() - started) * 1000)

            async def phase(ids: list[int]) -> None:
                # Fixed workers bound both active requests and task allocation.
                pending = iter(ids)

                async def worker() -> None:
                    for prompt_id in pending:
                        await send(prompt_id)

                async with asyncio.TaskGroup() as group:
                    for _ in range(min(concurrency, len(ids))):
                        group.create_task(worker())

            started = time.perf_counter()
            # Finish unique prompts before replay so repeats cannot race misses.
            # Both phases count toward every reported metric; no hidden warmup.
            await phase(prompt_ids[:unique])
            await phase(prompt_ids[unique:])
            elapsed = time.perf_counter() - started

    hits = sum(row["cache_hit"] for row in database.logs)
    actual = sum((row["cost_usd"] for row in database.logs), Decimal(0))
    saved = baseline - actual
    return {
        "mode": "mock (ASGI, in-memory store, no network)",
        "requests": requests,
        "concurrency": concurrency,
        "unique_prompts": unique,
        "latency_scale": latency_scale,
        "elapsed_seconds": round(elapsed, 3),
        "cache_hits": hits,
        "cache_hit_rate": hits / requests,
        # Final response models, including hits, rather than attempt counts.
        "cheap_responses": served[CHEAP],
        "fallback_responses": served[EXPENSIVE],
        "cheap_route_rate": served[CHEAP] / requests,
        "fallback_route_rate": served[EXPENSIVE] / requests,
        "upstream_attempts": {"cheap": upstream_calls[CHEAP], "fallback": upstream_calls[EXPENSIVE]},
        "p50_latency_ms": round(percentile(latencies, 0.50), 3),
        "p99_latency_ms": round(percentile(latencies, 0.99), 3),
        "estimated_cost_usd": str(actual),
        "always_expensive_usd": str(baseline),
        "saved_usd": str(saved),
        "saved_percent": round(float(saved / baseline * 100), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--latency-scale", type=float, default=1.0,
                        help="Multiply fake delays; use 0 for a fast smoke run")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")
    if not math.isfinite(args.latency_scale) or args.latency_scale < 0:
        parser.error("--latency-scale must be finite and non-negative")
    print(json.dumps(asyncio.run(run_loadtest(args.requests, args.concurrency, args.latency_scale)), indent=2))


if __name__ == "__main__":
    main()
