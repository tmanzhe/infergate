"""Real local HTTP streaming with controlled, free upstream byte streams."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest
import uvicorn

from infergate.app import create_app
from infergate.config import ModelRoute
from infergate.identity import key_hash


PATH = "/v1/chat/completions"
CHEAP, STRONG = "gpt-4o-mini", "gpt-4o"
ROUTES = {"auto": ModelRoute(CHEAP, STRONG, min_output_chars=100)}
FIRST = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
USAGE = b'data: {"usage":{"prompt_tokens":100,"completion_tokens":20}}\n\n'
DONE = b"data: [DONE]\n\n"


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, produce):
        self.produce = produce
        self.closed = asyncio.Event()

    async def __aiter__(self):
        async for chunk in self.produce():
            yield chunk

    async def aclose(self):
        self.closed.set()


def upstream_response(produce):
    stream = ByteStream(produce)
    return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"}), stream


@pytest.fixture
def live_client(settings):
    @asynccontextmanager
    async def make(handler=None, **overrides):
        def unexpected(request):
            pytest.fail(f"Unexpected upstream request: {request.url}")

        app = create_app(replace(settings, **overrides), upstream_transport=httpx.MockTransport(handler or unexpected))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", timeout_graceful_shutdown=1))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            task = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                async with asyncio.timeout(5):
                    while not server.started:
                        if task.done():
                            await task
                            raise AssertionError("Server failed to start")
                        await asyncio.sleep(0.001)
                async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=5) as client:
                    yield client
            finally:
                server.should_exit = True
                async with asyncio.timeout(5):
                    await task

    return make


async def wait_for_logs(rows, count):
    async with asyncio.timeout(5):
        while len(rows()) < count:
            await asyncio.sleep(0.005)
    assert len(rows()) == count
    return rows()


@pytest.mark.parametrize("limit", ["caller", "model"])
async def test_chunks_arrive_immediately_and_slots_last_until_end(
    live_client, rows, chat_request, limit
):
    release = asyncio.Event()
    streams = []
    calls = []
    body = {**chat_request, "stream": True, "stream_options": {"include_usage": True}, "vendor": [1, 2]}
    raw = json.dumps(body, indent=2).encode()

    async def produce():
        yield FIRST
        await release.wait()
        yield USAGE
        yield DONE

    def upstream(request):
        calls.append(request)
        assert request.content == raw
        assert request.headers["authorization"] == "Bearer test-secret"
        assert str(request.url) == "https://upstream.test/custom/v1/chat/completions"
        response, stream = upstream_response(produce)
        streams.append(stream)
        return response

    limits = {"max_in_flight_per_key": 1} if limit == "caller" else {"max_in_flight_per_model": {CHEAP: 1}}
    auth = {"Authorization": "Bearer original"}
    competing = auth if limit == "caller" else {"Authorization": "Bearer competitor"}
    async with live_client(upstream, **limits) as client, asyncio.timeout(5):
        try:
            async with client.stream("POST", PATH, content=raw, headers=auth) as response:
                chunks = response.aiter_raw()
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"
                assert "content-length" not in response.headers
                assert await anext(chunks) == FIRST
                assert rows() == []
                assert not streams[0].closed.is_set()
                rejected = await client.post(PATH, content=raw, headers=competing)
                assert rejected.status_code == 429
                assert rejected.headers["retry-after"] == "1"
                assert len(calls) == 1
                release.set()
                assert b"".join([chunk async for chunk in chunks]) == USAGE + DONE
            assert (await client.post(PATH, content=raw, headers=competing)).status_code == 200
        finally:
            release.set()
        logs = await wait_for_logs(rows, 3)
    assert [row["status"] for row in logs] == [429, 200, 200]
    assert logs[1]["api_key_hash"] == key_hash("original")
    assert (logs[1]["prompt_tokens"], logs[1]["completion_tokens"]) == (100, 20)
    assert logs[1]["cost_usd"] == Decimal("0.000027")
    assert all(stream.closed.is_set() for stream in streams)
    assert rows("response_cache") == []


@pytest.mark.parametrize("end,status", [("disconnect", 499), ("read_error", 502), ("timeout", 504), ("error_event", 502)])
async def test_terminal_paths_log_usage_once_close_upstream_and_release_slots(
    live_client, rows, chat_request, end, status
):
    resume = asyncio.Event()
    streams = []
    calls = []

    async def produce():
        yield FIRST + USAGE
        await resume.wait()
        if end == "timeout":
            raise httpx.ReadTimeout("fake mid-stream timeout")
        if end == "read_error":
            raise httpx.ReadError("fake mid-stream disconnect")
        if end == "error_event":
            yield b'data: {"error":{"message":"fake failure"}}\n\n'
        else:
            yield DONE

    def upstream(request):
        calls.append(json.loads(request.content)["model"])
        response, stream = upstream_response(produce)
        streams.append(stream)
        return response

    body = {**chat_request, "model": "auto", "stream": True}
    async with live_client(
        upstream, model_routes=ROUTES, upstream_max_attempts=3,
        max_in_flight_per_key=1, max_in_flight_per_model={CHEAP: 1},
        spend_caps_usd={"": Decimal(1)}, model_spend_caps_usd={CHEAP: Decimal(1)},
    ) as client, asyncio.timeout(5):
        async with client.stream("POST", PATH, json=body) as response:
            chunks = response.aiter_raw()
            assert await anext(chunks) == FIRST + USAGE
            assert rows() == []
            if end != "disconnect":
                resume.set()
                rest = b"".join([chunk async for chunk in chunks])
                assert rest == (b'data: {"error":{"message":"fake failure"}}\n\n' if end == "error_event" else b"")
        logs = await wait_for_logs(rows, 1)
        assert streams[0].closed.is_set()
        assert logs[0]["status"] == status
        assert logs[0]["model"] == CHEAP
        assert logs[0]["cost_usd"] == Decimal("0.000027")
        assert (logs[0]["prompt_tokens"], logs[0]["completion_tokens"]) == (100, 20)
        assert calls == [CHEAP]  # No retry, quality fallback, or error fallback after bytes.
        resume.set()
        assert (await client.post(PATH, json=body)).status_code == 200
        await wait_for_logs(rows, 2)
    assert calls == [CHEAP, CHEAP]
    assert rows("response_cache") == []


@pytest.mark.parametrize("wire,expected", [
    (b": keepalive\r\n\r\ndata: invalid\r\n\r\ndata: [DONE]\r\n\r\n", (0, 0)),
    (b'data: {"usage":null}\n\ndata: {"usage":{"prompt_tokens":-1,"completion_tokens":true}}\n\n', (0, 0)),
    (b'data: {"usage":{"prompt_tokens":2,"completion_tokens":1}}\r\n\r\n'
     b'data: {"usage":\r\ndata: {"prompt_tokens":100,"completion_tokens":20}}\r\n\r\n' + DONE, (100, 20)),
    (b'data: {"usage":{"prompt_tokens":100,"completion_tokens":20}}\r\r', (100, 20)),
])
async def test_sse_parsing_across_every_byte_preserves_wire_and_uses_last_usage(
    live_client, rows, chat_request, wire, expected
):
    async def produce():
        for byte in wire:
            yield bytes([byte])

    response, stream = upstream_response(produce)
    async with live_client(lambda request: response) as client:
        result = await client.post(PATH, json={**chat_request, "stream": True})
        logs = await wait_for_logs(rows, 1)
    assert result.content == wire
    assert stream.closed.is_set()
    assert (logs[0]["prompt_tokens"], logs[0]["completion_tokens"]) == expected


@pytest.mark.parametrize("failure", [503, "timeout", "connect", "empty"])
async def test_fallback_before_headers_only_and_actual_model_accounting(
    live_client, rows, chat_request, failure, retry_timing
):
    calls = []
    streams = []

    async def produce():
        yield FIRST
        yield USAGE
        yield DONE

    async def fail_first_read():
        raise httpx.ReadTimeout("fake pre-byte timeout")
        yield b""  # Make this an async generator.

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        assert json.loads(request.content)["vendor_option"] == [1, 2]
        if model == CHEAP:
            if failure == 503:
                return httpx.Response(503, json={"error": {"message": "fake failure"}})
            if failure == "connect":
                raise httpx.ConnectError("fake connection error")
            if failure == "empty":
                return httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})
            response, stream = upstream_response(fail_first_read)
        else:
            expected = 2 if failure in (503, "timeout") else 1
            assert len(rows()) == expected
            response, stream = upstream_response(produce)
        streams.append(stream)
        return response

    async with live_client(upstream, model_routes=ROUTES, upstream_max_attempts=2) as client:
        result = await client.post(PATH, json={**chat_request, "stream": True, "model": "auto", "vendor_option": [1, 2]})
        expected = 2 if failure in (503, "timeout") else 1
        logs = await wait_for_logs(rows, expected + 1)
    assert result.status_code == 200
    assert result.content == FIRST + USAGE + DONE
    assert calls == [CHEAP] * expected + [STRONG]
    assert all(stream.closed.is_set() for stream in streams)
    assert [row["model"] for row in logs] == [CHEAP] * expected + [STRONG]
    assert logs[-1]["cost_usd"] == Decimal("0.000450")
    assert rows("response_cache") == []


@pytest.mark.parametrize("cap", ["caller", "model"])
async def test_spend_cap_checked_before_stream_start(live_client, rows, chat_request, cap):
    async def produce():
        yield FIRST + USAGE + DONE

    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return upstream_response(produce)[0]

    overrides = {"spend_caps_usd": {"": Decimal("0.000027")}} if cap == "caller" else {
        "model_spend_caps_usd": {CHEAP: Decimal("0.000027")}
    }
    async with live_client(upstream, **overrides) as client:
        assert (await client.post(PATH, json={**chat_request, "stream": True})).status_code == 200
        assert (await client.post(PATH, json={**chat_request, "stream": True})).status_code == 402
        logs = await wait_for_logs(rows, 2)
    assert calls == 1
    assert [row["cost_usd"] for row in logs] == [Decimal("0.000027"), 0]


async def test_mock_stream_is_deterministic_free_and_never_cached(live_client, rows, chat_request):
    async with live_client(mock_model=True) as client:
        first = await client.post(PATH, json={**chat_request, "stream": True})
        second = await client.post(PATH, json={**chat_request, "stream": True})
        logs = await wait_for_logs(rows, 2)
    assert first.content == second.content
    assert first.content.endswith(DONE)
    assert all(row["prompt_tokens"] > 0 and row["completion_tokens"] > 0 for row in logs)
    assert all(row["cost_usd"] == 0 and not row["cache_hit"] for row in logs)
    assert rows("response_cache") == []


async def test_disconnect_before_first_byte_logs_once_and_releases_slots_and_budget_locks(
    live_client, rows, chat_request
):
    entered = asyncio.Event()
    streams = []
    calls = []

    async def blocked():
        entered.set()
        await asyncio.Event().wait()
        yield b""

    async def success():
        yield FIRST + USAGE + DONE

    def upstream(request):
        calls.append(json.loads(request.content)["model"])
        response, stream = upstream_response(blocked if len(calls) == 1 else success)
        streams.append(stream)
        return response

    async with live_client(
        upstream, model_routes=ROUTES, upstream_max_attempts=3,
        max_in_flight_per_key=1, max_in_flight_per_model={CHEAP: 1},
        spend_caps_usd={"": Decimal(1)}, model_spend_caps_usd={CHEAP: Decimal(1)},
    ) as client, asyncio.timeout(5):
        body = {**chat_request, "model": "auto", "stream": True}
        task = asyncio.create_task(client.post(PATH, json=body))
        try:
            await entered.wait()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        logs = await wait_for_logs(rows, 1)
        assert logs[0]["status"] == 499
        assert logs[0]["prompt_tokens"] == logs[0]["completion_tokens"] == logs[0]["cost_usd"] == 0
        assert streams[0].closed.is_set()
        assert (await client.post(PATH, json=body)).status_code == 200
        await wait_for_logs(rows, 2)
    assert calls == [CHEAP, CHEAP]


@pytest.mark.parametrize("fallback", [True, False])
async def test_model_cap_fallback_respects_route_limit_flag(
    live_client, rows, chat_request, fallback
):
    calls = []

    async def produce():
        yield FIRST + USAGE + DONE

    def upstream(request):
        calls.append(json.loads(request.content)["model"])
        return upstream_response(produce)[0]

    routes = {"auto": replace(ROUTES["auto"], fallback_on_limit=fallback)}
    async with live_client(upstream, model_routes=routes, model_spend_caps_usd={CHEAP: Decimal(0)}) as client:
        result = await client.post(PATH, json={**chat_request, "model": "auto", "stream": True})
        logs = await wait_for_logs(rows, 2 if fallback else 1)
    assert result.status_code == (200 if fallback else 402)
    assert calls == ([STRONG] if fallback else [])
    assert [row["model"] for row in logs] == ([CHEAP, STRONG] if fallback else [CHEAP])


@pytest.mark.parametrize("fallback", [True, False])
async def test_caller_cap_after_failed_attempt_stops_retries_and_fallback(
    live_client, rows, chat_request, retry_timing, fallback
):
    calls = []

    def upstream(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(503, json={"usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    routes = {"auto": replace(ROUTES["auto"], fallback_on_error=fallback)}
    async with live_client(
        upstream, model_routes=routes, spend_caps_usd={"": Decimal("0.000027")}, upstream_max_attempts=3,
    ) as client:
        result = await client.post(PATH, json={**chat_request, "model": "auto", "stream": True})
        logs = await wait_for_logs(rows, 2)
    assert result.status_code == 402
    assert calls == [CHEAP]
    assert [row["status"] for row in logs] == [503, 402]
    assert [row["cost_usd"] for row in logs] == [Decimal("0.000027"), 0]


async def test_exhausted_retries_preserve_upstream_error_body_status_and_headers(
    live_client, rows, chat_request, retry_timing
):
    wire = b'{ "error": {"message": "fake throttling"}, "usage": {"prompt_tokens": 7, "completion_tokens": 3} }\n'
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        assert len(rows()) == calls - 1
        return httpx.Response(429, content=wire, headers={"content-type": "application/json", "retry-after": "7"})

    async with live_client(upstream, upstream_max_attempts=2) as client:
        result = await client.post(PATH, json={**chat_request, "stream": True})
        logs = await wait_for_logs(rows, 2)
    assert result.status_code == 429
    assert result.content == wire
    assert result.headers["retry-after"] == "7"
    assert result.headers["content-type"] == "application/json"
    assert calls == 2
    assert all(row["prompt_tokens"] == 7 and row["completion_tokens"] == 3 for row in logs)
