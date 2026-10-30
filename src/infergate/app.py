"""Non-streaming OpenAI-compatible chat proxy with Postgres cache and accounting."""

import json
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from infergate.cache import cache_key
from infergate.config import Settings
from infergate.database import CachedResponse, Database
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


def error_response(message: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "gateway_error", "code": None}},
        status_code=status,
    )


def create_app(
    settings: Settings | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> FastAPI:
    settings = settings or Settings.from_env()
    database = Database(settings.database_url, settings.database_schema)

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

        async def finish(response: Response, *, cache_hit: bool = False) -> Response:
            prompt_tokens, completion_tokens = usage_tokens(response.body)
            cost = Decimal(0)
            price = settings.model_prices.get(model)
            if price is not None and not cache_hit and not settings.mock_model:
                cost = (
                    prompt_tokens * price.prompt + completion_tokens * price.completion
                ) / Decimal(1_000_000)
            await database.log_request(
                timestamp=timestamp,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_ms=(time.perf_counter() - started) * 1000,
                cache_hit=cache_hit,
                status=response.status_code,
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
            key = cache_key(body)
        except (ValueError, UnicodeDecodeError):
            return await finish(error_response("Request body must be a valid JSON object.", 400))

        if settings.cache_ttl_seconds > 0:
            cached = await database.get_cached(key, clock())
            if cached is not None:
                headers = {"content-type": cached.content_type} if cached.content_type else {}
                return await finish(
                    Response(cached.body, status_code=cached.status, headers=headers), cache_hit=True
                )

        if settings.mock_model:
            response = JSONResponse(mock_completion(body))
        else:
            try:
                upstream = await app.state.upstream.post(
                    f"{settings.upstream_base_url.rstrip('/')}/chat/completions",
                    content=raw_body,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.TimeoutException:
                return await finish(error_response("Upstream request timed out.", 504))
            except httpx.RequestError:
                return await finish(error_response("Unable to reach upstream API.", 502))
            headers = {"content-type": upstream.headers["content-type"]} if "content-type" in upstream.headers else {}
            response = Response(upstream.content, status_code=upstream.status_code, headers=headers)

        if settings.cache_ttl_seconds > 0 and 200 <= response.status_code < 300:
            await database.put_cached(
                key,
                CachedResponse(response.body, response.status_code, response.headers.get("content-type")),
                clock() + timedelta(seconds=settings.cache_ttl_seconds),
            )
        return await finish(response)

    return app


app = create_app()
