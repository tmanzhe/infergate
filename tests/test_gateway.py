import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from infergate.cache import cache_key
from infergate.config import ModelPrice


PATH = "/v1/chat/completions"


async def test_cache_miss_then_hit_logs_cost(client_factory, rows, chat_request, completion):
    calls = []

    async def upstream(request):
        calls.append(request)
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=completion)

    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    async with client_factory(upstream, clock=lambda: timestamp) as client:
        first = await client.post(PATH, json=chat_request)
        second = await client.post(PATH, json=chat_request)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == completion
    assert first.content == second.content
    assert len(calls) == 1
    miss, hit = rows()
    for row in (miss, hit):
        assert row["timestamp"] == timestamp
        assert row["model"] == "gpt-4o-mini"
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 20
        assert row["status"] == 200
        assert row["latency_ms"] >= 0
    assert miss["latency_ms"] >= 10
    assert miss["cost_usd"] == Decimal("0.000027")
    assert miss["cache_hit"] is False
    assert hit["cost_usd"] == 0
    assert hit["cache_hit"] is True
    cached, = rows("response_cache")
    assert cached["cache_key"] == cache_key(chat_request)
    assert cached["expires_at"] == timestamp + timedelta(hours=1)


async def test_ttl_expiry(client_factory, rows, chat_request, completion):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={**completion, "id": f"response-{calls}"})

    async with client_factory(upstream, clock=lambda: now, cache_ttl_seconds=10) as client:
        first = await client.post(PATH, json=chat_request)
        now += timedelta(seconds=9)
        assert (await client.post(PATH, json=chat_request)).content == first.content
        now += timedelta(seconds=1)  # Exactly at expires_at must miss.
        expired = await client.post(PATH, json=chat_request)
        assert expired.json()["id"] == "response-2"
        assert (await client.post(PATH, json=chat_request)).content == expired.content

    assert calls == 2
    assert [row["cache_hit"] for row in rows()] == [False, True, False, True]
    assert len(rows("response_cache")) == 1


async def test_proxy_passes_raw_request_and_response(client_factory, rows, chat_request):
    request_body = json.dumps({**chat_request, "user": "tracking-id", "vendor_option": [1, 2]}, indent=2).encode()
    response_body = b'{ "usage": {"prompt_tokens": 7, "completion_tokens": 3}, "vendor": true }\n'

    def upstream(request):
        assert str(request.url) == "https://upstream.test/custom/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-secret"
        assert request.headers["content-type"] == "application/json"
        assert request.content == request_body
        return httpx.Response(201, content=response_body, headers={"content-type": "application/json; charset=utf-8"})

    async with client_factory(upstream) as client:
        first = await client.post(PATH, content=request_body)
        cached = await client.post(PATH, content=request_body)
    for response in (first, cached):
        assert response.status_code == 201
        assert response.content == response_body
        assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert [row["cache_hit"] for row in rows()] == [False, True]


@pytest.mark.parametrize("status,response_body,content_type", [
    (429, b'{ "error": {"message": "slow down", "code": "rate_limit"} }\n', "application/json"),
    (503, b"temporarily unavailable\n", "text/plain"),
])
async def test_upstream_errors_are_passed_through_and_not_cached(
    client_factory, rows, chat_request, status, response_body, content_type
):
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, content=response_body, headers={"content-type": content_type})

    async with client_factory(upstream) as client:
        for _ in range(2):
            response = await client.post(PATH, json=chat_request)
            assert response.status_code == status
            assert response.content == response_body
            assert response.headers["content-type"] == content_type
    assert calls == 2
    assert rows("response_cache") == []
    assert len(rows()) == 2
    for row in rows():
        assert row["status"] == status
        assert row["cache_hit"] is False
        assert row["cost_usd"] == row["prompt_tokens"] == row["completion_tokens"] == 0


async def test_equivalent_requests_share_cached_response(client_factory, rows, chat_request, completion):
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion)

    first_body = {**chat_request, "user": "alice", "stream": False}
    second_body = {
        "max_tokens": 64,
        "temperature": 0.2,
        "messages": [{"content": "Hello", "role": "user"}],
        "model": "gpt-4o-mini",
        "user": "bob",
        "metadata": {"trace": "123"},
    }
    async with client_factory(upstream) as client:
        assert (await client.post(PATH, json=first_body)).status_code == 200
        assert (await client.post(PATH, json=second_body)).status_code == 200
        assert (await client.post(PATH, json={**second_body, "temperature": 0.9})).status_code == 200
    assert calls == 2
    assert [row["cache_hit"] for row in rows()] == [False, True, False]


async def test_streaming_bypasses_cached_json(client_factory, rows, chat_request):
    async with client_factory(mock_model=True) as client:
        assert (await client.post(PATH, json=chat_request)).status_code == 200
        response = await client.post(PATH, json={**chat_request, "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert rows()[-1]["status"] == 200
    assert rows()[-1]["cache_hit"] is False
    assert len(rows("response_cache")) == 1


async def test_mock_mode_is_deterministic_and_free(client_factory, rows, chat_request):
    async with client_factory(mock_model=True, cache_ttl_seconds=0) as client:
        first = await client.post(PATH, json=chat_request)
        second = await client.post(PATH, json=chat_request)
        changed = await client.post(PATH, json={**chat_request, "seed": 42})
    assert first.status_code == second.status_code == changed.status_code == 200
    assert first.content == second.content
    assert first.content != changed.content
    body = first.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == chat_request["model"]
    assert body["choices"][0]["message"]["role"] == "assistant"
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert len(rows()) == 3
    assert all(row["cost_usd"] == 0 and not row["cache_hit"] for row in rows())
    assert rows("response_cache") == []


async def test_custom_price_table(client_factory, rows, chat_request, completion):
    prices = {"gpt-4o-mini": ModelPrice(Decimal("2"), Decimal("5"))}
    async with client_factory(lambda r: httpx.Response(200, json=completion), model_prices=prices) as client:
        await client.post(PATH, json=chat_request)
    assert rows()[0]["cost_usd"] == Decimal("0.0003")


async def test_unpriced_model_logs_zero(client_factory, rows, chat_request, completion):
    async with client_factory(lambda r: httpx.Response(200, json=completion), model_prices={}) as client:
        await client.post(PATH, json=chat_request)
    assert rows()[0]["cost_usd"] == 0
    assert rows()[0]["prompt_tokens"] == 100


@pytest.mark.parametrize("exception,status", [
    (httpx.ConnectError("offline"), 502),
    (httpx.ReadTimeout("too slow"), 504),
])
async def test_transport_errors_are_logged(client_factory, rows, chat_request, exception, status):
    def upstream(request):
        raise exception

    async with client_factory(upstream) as client:
        assert (await client.post(PATH, json=chat_request)).status_code == status
    assert rows()[0]["status"] == status
    assert rows()[0]["cost_usd"] == 0
    assert rows("response_cache") == []


@pytest.mark.parametrize("body", [b"invalid", b"[]", b'{"model":"x","temperature":NaN}'])
async def test_invalid_json_rejected_and_logged(client_factory, rows, body):
    async with client_factory(mock_model=True) as client:
        assert (await client.post(PATH, content=body)).status_code == 400
    assert rows()[0]["status"] == 400
    assert rows("response_cache") == []


async def test_schema_initialization_and_cache_survive_restart(client_factory, rows, chat_request):
    async with client_factory(mock_model=True) as client:
        first = await client.post(PATH, json=chat_request)
    async with client_factory(mock_model=True) as client:
        second = await client.post(PATH, json=chat_request)
    assert first.content == second.content
    assert [row["cache_hit"] for row in rows()] == [False, True]
