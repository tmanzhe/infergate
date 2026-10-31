"""SSE delivery with admission and accounting scoped to the response lifetime."""

import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from typing import Any

import anyio
import httpx
from fastapi.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from infergate.concurrency import ConcurrencyLimiter
from infergate.config import ModelRoute, Settings
from infergate.database import Database
from infergate.errors import error_response
from infergate.retries import Attempt, with_retries
from infergate.routing import should_fallback
from infergate.spend import SpendBudget
from infergate.upstream import mock_stream


class SSEUsage:
    """Observe complete SSE events without changing or buffering delivery."""

    def __init__(self) -> None:
        self.pending = b""
        self.data: list[bytes] = []
        self.usage: dict[str, Any] = {}
        self.error = False

    def feed(self, chunk: bytes) -> None:
        self.pending += chunk
        while match := re.search(rb"\r\n|\r|\n", self.pending):
            # A trailing CR may be half of a CRLF split across transport chunks.
            if match.group() == b"\r" and match.end() == len(self.pending):
                break
            line = self.pending[:match.start()]
            self.pending = self.pending[match.end():]
            if not line:
                self.event()
            elif line.startswith(b"data:"):
                self.data.append(line[5:].removeprefix(b" "))

    def event(self) -> None:
        data, self.data = b"\n".join(self.data), []
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return
        if isinstance(payload, dict):
            if isinstance(payload.get("usage"), dict):
                # Usage events are cumulative snapshots, never additive deltas.
                self.usage = payload["usage"]
            if payload.get("error"):
                self.error = True


class DeferredStream(StreamingResponse):
    """Monitor disconnects during upstream setup as well as body delivery."""

    def __init__(self, deliver: Callable[[Send], Awaitable[None]]) -> None:
        super().__init__(iter(()))
        self.deliver = deliver

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with anyio.create_task_group() as group:
            async def run() -> None:
                try:
                    await self.deliver(send)
                except OSError:
                    # ASGI servers can signal a closed socket through send().
                    pass
                finally:
                    group.cancel_scope.cancel()

            group.start_soon(run)
            await self.listen_for_disconnect(receive)
            group.cancel_scope.cancel()


class OpenStream(Response):
    """A prepared upstream; ownership transfers to deliver after the first read."""

    def __init__(
        self, upstream: httpx.Response, chunks: AsyncIterator[bytes], first: bytes,
        observer: SSEUsage, log: Callable[[Response], Awaitable[Response]],
    ) -> None:
        self.upstream = upstream
        self.chunks = chunks
        self.first = first
        self.observer = observer
        self.log = log
        super().__init__(status_code=upstream.status_code, headers={
            "content-type": upstream.headers.get("content-type", "text/event-stream"),
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        })
        del self.headers["content-length"]

    async def deliver(self, send: Send) -> None:
        status = 499
        try:
            await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
            if self.first:
                self.observer.feed(self.first)
                await send({"type": "http.response.body", "body": self.first, "more_body": True})
            async for chunk in self.chunks:
                self.observer.feed(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            if self.observer.pending.endswith(b"\r"):
                self.observer.feed(b"\n")
            status = 502 if self.observer.error else self.status_code
        except httpx.TimeoutException:
            status = 504
        except httpx.RequestError:
            status = 502
        except OSError:
            status = 499
            raise
        finally:
            # Starlette/AnyIO cancels delivery on disconnect. Persist before
            # releasing either concurrency slot or either spend-budget lock.
            with anyio.CancelScope(shield=True):
                try:
                    await self.upstream.aclose()
                finally:
                    await self.log(Response(json.dumps({"usage": self.observer.usage}), status_code=status))


@asynccontextmanager
async def stream_budget(budget: SpendBudget) -> AsyncIterator[None]:
    stack = AsyncExitStack()
    try:
        await stack.enter_async_context(budget.guard())
        yield
    finally:
        # Releasing Postgres advisory locks also awaits I/O on disconnect.
        with anyio.CancelScope(shield=True):
            await stack.aclose()


async def open_attempt(
    client: httpx.AsyncClient, settings: Settings, body: dict[str, Any], raw_body: bytes,
    check_empty: bool, log: Callable[[Response], Awaitable[Response]],
) -> Attempt:
    upstream = None
    transferred = False
    response = error_response("Client disconnected.", 499)
    retryable = False
    try:
        if settings.mock_model:
            upstream = httpx.Response(200, content=mock_stream(body), headers={"content-type": "text/event-stream"})
        else:
            upstream = await client.send(client.build_request(
                "POST", f"{settings.upstream_base_url.rstrip('/')}/chat/completions",
                content=raw_body, headers={"Content-Type": "application/json"},
            ), stream=True)
        headers = {name: upstream.headers[name] for name in ("content-type", "retry-after")
                   if name in upstream.headers}
        if not 200 <= upstream.status_code < 300:
            response = Response(await upstream.aread(), status_code=upstream.status_code, headers=headers)
            retryable = upstream.status_code == 429 or 500 <= upstream.status_code < 600
        else:
            chunks = upstream.aiter_bytes()
            first = b""
            async for chunk in chunks:
                if chunk:
                    first = chunk
                    break
            if not first and check_empty:
                response = Response(b"", status_code=upstream.status_code, headers=headers)
            else:
                result = OpenStream(upstream, chunks, first, SSEUsage(), log)
                transferred = True
                return Attempt(result)
    except httpx.TimeoutException:
        response = error_response("Upstream request timed out.", 504)
        retryable = True
    except httpx.RequestError:
        response = error_response("Unable to reach upstream API.", 502)
    finally:
        if not transferred:
            with anyio.CancelScope(shield=True):
                try:
                    if upstream is not None:
                        await upstream.aclose()
                finally:
                    await log(response)
    return Attempt(response, retryable)


def streaming_response(
    *, settings: Settings, database: Database, client: httpx.AsyncClient,
    limiter: ConcurrencyLimiter, model_limiters: dict[str, ConcurrencyLimiter],
    unlimited: ConcurrencyLimiter, api_key: str, api_key_hash: str,
    body: dict[str, Any], raw_body: bytes, model: str, route: ModelRoute | None,
    finish: Callable[..., Awaitable[Response]], clock: Callable[[], datetime],
) -> Response:
    async def send_response(response: Response, send: Send) -> None:
        await send({"type": "http.response.start", "status": response.status_code, "headers": response.raw_headers})
        await send({"type": "http.response.body", "body": response.body})

    async def deliver(send: Send) -> None:
        completed_stream = False
        with limiter.acquire(api_key_hash) as admitted:
            if not admitted:
                limited = error_response("API key concurrency limit reached. Try again shortly.", 429)
                limited.headers["Retry-After"] = "1"
                await send_response(await finish(limited), send)
                return
            budget = SpendBudget(database, api_key_hash, settings.spend_caps_usd.get(api_key))
            async with stream_budget(budget):
                if await budget.exhausted():
                    await send_response(await finish(error_response("API key USD spend cap reached.", 402)), send)
                    return
                async def run_model(target: str) -> tuple[Response, str | None]:
                    reason = None

                    async def log(response: Response, **kwargs) -> Response:
                        return await finish(response, log_model=target, **kwargs)

                    if await budget.exhausted():
                        return await log(error_response("API key USD spend cap reached.", 402)), "caller_cap"
                    with model_limiters.get(target, unlimited).acquire(target) as model_admitted:
                        if not model_admitted:
                            limited = error_response("Model concurrency limit reached. Try again shortly.", 429)
                            limited.headers["Retry-After"] = "1"
                            return await log(limited), "model_limit"
                        model_budget = SpendBudget(
                            database, api_key_hash, settings.model_spend_caps_usd.get(target), model=target
                        )
                        async with stream_budget(model_budget):
                            async def attempt() -> Attempt:
                                nonlocal reason
                                if await budget.exhausted():
                                    reason = "caller_cap"
                                    return Attempt(await log(error_response("API key USD spend cap reached.", 402)))
                                if await model_budget.exhausted():
                                    reason = "model_limit"
                                    return Attempt(await log(error_response("Model USD spend cap reached.", 402)))
                                attempt_started, attempt_timestamp = time.perf_counter(), clock()

                                async def log_attempt(response: Response) -> Response:
                                    return await log(response, log_started=attempt_started, log_timestamp=attempt_timestamp)

                                routed_body = {**body, "model": target} if route else body
                                routed_raw = json.dumps(routed_body).encode() if route else raw_body
                                return await open_attempt(
                                    client, settings, routed_body, routed_raw,
                                    bool(route and route.fallback_on_quality), log_attempt,
                                )

                            response = await with_retries(attempt, settings)
                            if isinstance(response, OpenStream):
                                await response.deliver(send)
                            return response, reason

                response, reason = await run_model(route.cheap_model if route else model)
                if not isinstance(response, OpenStream) and route and should_fallback(response, route, reason):
                    # No bytes or headers have been sent by any failed attempt.
                    response, reason = await run_model(route.strong_model)
                if isinstance(response, OpenStream):
                    completed_stream = True
                else:
                    await send_response(response, send)
                    return
        if completed_stream:
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    return DeferredStream(deliver)
