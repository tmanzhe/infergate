"""Non-streaming OpenAI-compatible chat proxy with Postgres cache and accounting."""

import json
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from infergate.cache import cache_key
from infergate.concurrency import ConcurrencyLimiter
from infergate.config import Settings
from infergate.database import CachedResponse, Database
from infergate.errors import error_response
from infergate.identity import bearer_key, key_hash
from infergate.retries import Attempt, send_attempt, with_retries
from infergate.routing import fails_quality, should_fallback
from infergate.spend import SpendBudget
from infergate.upstream import mock_completion


def utc_now() -> datetime:
    return datetime.now(UTC)


def usage_tokens(body: bytes) -> tuple[int, int]:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return 0, 0
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return 0, 0

    def count(name: str) -> int:
        value = usage.get(name, 0)
        return value if type(value) is int and value >= 0 else 0

    return count("prompt_tokens"), count("completion_tokens")


def create_app(
    settings: Settings | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> FastAPI:
    settings = settings or Settings.from_env()
    database = Database(settings.database_url, settings.database_schema)
    limiter = ConcurrencyLimiter(settings.max_in_flight_per_key)
    model_limiters = {
        model: ConcurrencyLimiter(maximum)
        for model, maximum in settings.max_in_flight_per_model.items()
    }
    unlimited = ConcurrencyLimiter(0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.initialize()
        headers = {"Authorization": f"Bearer {settings.upstream_api_key}"} if settings.upstream_api_key else {}
        async with httpx.AsyncClient(
            headers=headers,
            timeout=settings.upstream_timeout_seconds,
            transport=upstream_transport,
        ) as client:
            app.state.upstream = client
            yield

    app = FastAPI(title="infergate", lifespan=lifespan)
    app.state.database = database
    app.state.settings = settings

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        started = time.perf_counter()
        timestamp = clock()
        model = ""
        api_key = bearer_key(request.headers.get("authorization", ""))
        api_key_hash = key_hash(api_key)

        async def finish(
            response: Response,
            *,
            cache_hit: bool = False,
            log_timestamp: datetime = timestamp,
            log_started: float = started,
        ) -> Response:
            prompt_tokens, completion_tokens = usage_tokens(response.body)
            cost = Decimal(0)
            price = settings.model_prices.get(model)
            if price is not None and not cache_hit and not settings.mock_model:
                cost = (
                    prompt_tokens * price.prompt + completion_tokens * price.completion
                ) / Decimal(1_000_000)
            await database.log_request(
                timestamp=log_timestamp,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_ms=(time.perf_counter() - log_started) * 1000,
                cache_hit=cache_hit,
                status=response.status_code,
                api_key_hash=api_key_hash,
            )
            return response

        raw_body = await request.body()
        try:
            body: Any = json.loads(raw_body)
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            model = body.get("model") if isinstance(body.get("model"), str) else ""
            if body.get("stream"):
                return await finish(error_response("Streaming is not supported; set stream=false.", 400))
            route = settings.model_routes.get(model)
            key = cache_key(body)
            if route:
                key = "route:" + cache_key({"request_key": key, "route": asdict(route)})
        except (ValueError, UnicodeDecodeError):
            return await finish(error_response("Request body must be a valid JSON object.", 400))

        with limiter.acquire(api_key_hash) as admitted:
            if not admitted:
                response = error_response("API key concurrency limit reached. Try again shortly.", 429)
                response.headers["Retry-After"] = "1"
                return await finish(response)

            budget = SpendBudget(database, api_key_hash, settings.spend_caps_usd.get(api_key))
            async with budget.guard():
                if await budget.exhausted():
                    return await finish(error_response("API key USD spend cap reached.", 402))

                if settings.cache_ttl_seconds > 0:
                    cached = await database.get_cached(key, clock())
                    if cached is not None:
                        model = cached.model or model
                        headers = {"content-type": cached.content_type} if cached.content_type else {}
                        return await finish(
                            Response(cached.body, status_code=cached.status, headers=headers), cache_hit=True
                        )

                async def run_model(target: str) -> tuple[Response, str | None]:
                    nonlocal model
                    model = target
                    if await budget.exhausted():
                        return await finish(error_response("API key USD spend cap reached.", 402)), "caller_cap"
                    with model_limiters.get(model, unlimited).acquire(model) as model_admitted:
                        if not model_admitted:
                            limited = error_response("Model concurrency limit reached. Try again shortly.", 429)
                            limited.headers["Retry-After"] = "1"
                            return await finish(limited), "model_limit"
                        model_budget = SpendBudget(
                            database, api_key_hash, settings.model_spend_caps_usd.get(model), model=model
                        )
                        async with model_budget.guard():
                            reason = None
                            routed_body = {**body, "model": model} if route else body
                            routed_raw = json.dumps(routed_body).encode() if route else raw_body

                            async def attempt() -> Attempt:
                                nonlocal reason
                                # Each committed attempt can exhaust either budget.
                                if await budget.exhausted():
                                    reason = "caller_cap"
                                    return Attempt(await finish(error_response("API key USD spend cap reached.", 402)))
                                if await model_budget.exhausted():
                                    reason = "model_limit"
                                    return Attempt(await finish(error_response("Model USD spend cap reached.", 402)))
                                attempt_started = time.perf_counter()
                                attempt_timestamp = clock()
                                if settings.mock_model:
                                    result = Attempt(JSONResponse(mock_completion(routed_body)))
                                else:
                                    result = await send_attempt(
                                        app.state.upstream,
                                        f"{settings.upstream_base_url.rstrip('/')}/chat/completions",
                                        routed_raw,
                                    )
                                await finish(
                                    result.response,
                                    log_timestamp=attempt_timestamp,
                                    log_started=attempt_started,
                                )
                                return result

                            response = await with_retries(attempt, settings)
                            return response, reason

                response, reason = await run_model(route.cheap_model if route else model)
                if route and should_fallback(response, route, reason):
                    # Resolve exactly once; target names are never recursively routed.
                    response, reason = await run_model(route.strong_model)

                acceptable = not (route and route.fallback_on_quality and fails_quality(response, route))
                if settings.cache_ttl_seconds > 0 and 200 <= response.status_code < 300 and acceptable:
                    await database.put_cached(
                        key,
                        CachedResponse(response.body, response.status_code, response.headers.get("content-type"), model),
                        clock() + timedelta(seconds=settings.cache_ttl_seconds),
                    )
                return response

    return app


app = create_app()
