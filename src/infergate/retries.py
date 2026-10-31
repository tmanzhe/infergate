"""Retry only upstream throttling, server errors, and timeouts."""

from asyncio import sleep
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from random import uniform

import httpx
from fastapi.responses import Response

from infergate.config import Settings
from infergate.errors import error_response


@dataclass(frozen=True)
class Attempt:
    response: Response
    retryable: bool = False


async def send_attempt(client: httpx.AsyncClient, url: str, body: bytes) -> Attempt:
    try:
        upstream = await client.post(url, content=body, headers={"Content-Type": "application/json"})
    except httpx.TimeoutException:
        return Attempt(error_response("Upstream request timed out.", 504), retryable=True)
    except httpx.RequestError:
        return Attempt(error_response("Unable to reach upstream API.", 502))
    headers = {
        name: upstream.headers[name]
        for name in ("content-type", "retry-after")
        if name in upstream.headers
    }
    return Attempt(
        Response(upstream.content, status_code=upstream.status_code, headers=headers),
        retryable=upstream.status_code == 429 or 500 <= upstream.status_code < 600,
    )


async def with_retries(attempt: Callable[[], Awaitable[Attempt]], settings: Settings) -> Response:
    for number in range(settings.upstream_max_attempts):
        result = await attempt()
        if not result.retryable or number + 1 == settings.upstream_max_attempts:
            return result.response
        ceiling = min(
            settings.upstream_retry_max_seconds,
            settings.upstream_retry_base_seconds * 2**number,
        )
        await sleep(uniform(0, ceiling))
    raise AssertionError("Validated max attempts must be positive")
