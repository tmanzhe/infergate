import asyncio
from collections import Counter

import httpx
import pytest

from infergate import retries
from infergate.identity import key_hash


PATH = "/v1/chat/completions"


@pytest.mark.parametrize("limit", [1, 2])
async def test_excess_in_flight_requests_are_rejected_per_key(
    client_factory, rows, chat_request, completion, limit
):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def upstream(request):
        nonlocal calls
        calls += 1
        if calls == limit:
            entered.set()
        if calls <= limit:
            await release.wait()
        return httpx.Response(200, json=completion)

    alpha = {"Authorization": "Bearer alpha"}
    beta = {"Authorization": "Bearer beta"}
    async with client_factory(
        upstream, max_in_flight_per_key=limit, cache_ttl_seconds=0
    ) as client, asyncio.timeout(5), asyncio.TaskGroup() as group:
        pending = [
            group.create_task(client.post(PATH, json=chat_request, headers=alpha))
            for _ in range(limit)
        ]
        try:
            await entered.wait()
            excess = await client.post(PATH, json=chat_request, headers=alpha)
            assert excess.status_code == 429
            assert int(excess.headers["Retry-After"]) >= 1
            assert "concurrency" in excess.json()["error"]["message"].lower()
            assert calls == limit
            assert all(not task.done() for task in pending)

            # A separate caller still gets its own allowance while alpha is blocked.
            other_key = await client.post(PATH, json=chat_request, headers=beta)
            assert other_key.status_code == 200
            assert calls == limit + 1
        finally:
            release.set()
        assert all(response.status_code == 200 for response in await asyncio.gather(*pending))
        assert (await client.post(PATH, json=chat_request, headers=alpha)).status_code == 200

    logs = rows()
    assert Counter(row["status"] for row in logs) == {200: limit + 2, 429: 1}
    rejected, = [row for row in logs if row["status"] == 429]
    assert rejected["api_key_hash"] == key_hash("alpha")
    assert rejected["cost_usd"] == 0
    assert not rejected["cache_hit"]
    assert Counter(row["api_key_hash"] for row in logs) == {
        key_hash("alpha"): limit + 2, key_hash("beta"): 1
    }


async def test_zero_concurrency_limit_allows_overlapping_requests(
    client_factory, rows, chat_request, completion
):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def upstream(request):
        nonlocal calls
        calls += 1
        if calls == 3:
            entered.set()
        await release.wait()
        return httpx.Response(200, json=completion)

    async with client_factory(
        upstream, max_in_flight_per_key=0, cache_ttl_seconds=0
    ) as client, asyncio.timeout(5), asyncio.TaskGroup() as group:
        pending = [group.create_task(client.post(PATH, json=chat_request)) for _ in range(3)]
        try:
            await entered.wait()
            assert all(not task.done() for task in pending)
        finally:
            release.set()
        assert all(response.status_code == 200 for response in await asyncio.gather(*pending))
    assert [row["status"] for row in rows()] == [200, 200, 200]


@pytest.mark.parametrize("failure", ["http", "timeout", "cancel"])
async def test_concurrency_slot_is_released_after_failure_or_cancellation(
    client_factory, rows, chat_request, completion, failure
):
    entered = asyncio.Event()
    calls = 0

    async def upstream(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == "cancel":
                entered.set()
                await asyncio.Event().wait()
            if failure == "timeout":
                raise httpx.ReadTimeout("fake timeout", request=request)
            return httpx.Response(503, json={"error": {"message": "fake failure"}})
        return httpx.Response(200, json=completion)

    async with client_factory(
        upstream, max_in_flight_per_key=1, cache_ttl_seconds=0
    ) as client, asyncio.timeout(5):
        if failure == "cancel":
            task = asyncio.create_task(client.post(PATH, json=chat_request))
            try:
                await entered.wait()
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        else:
            response = await client.post(PATH, json=chat_request)
            assert response.status_code == (504 if failure == "timeout" else 503)
        assert (await client.post(PATH, json=chat_request)).status_code == 200
    assert calls == 2
    assert [row["status"] for row in rows()] == {
        "http": [503, 200], "timeout": [504, 200], "cancel": [200]
    }[failure]


async def test_concurrency_slot_is_held_during_retry_backoff(
    client_factory, rows, chat_request, completion, monkeypatch
):
    backing_off = asyncio.Event()
    resume = asyncio.Event()
    calls = 0

    async def sleep(delay):
        backing_off.set()
        await resume.wait()

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503) if calls == 1 else httpx.Response(200, json=completion)

    monkeypatch.setattr(retries, "sleep", sleep)
    monkeypatch.setattr(retries, "uniform", lambda lower, upper: upper / 2)
    async with client_factory(
        upstream, max_in_flight_per_key=1, upstream_max_attempts=2, cache_ttl_seconds=0
    ) as client, asyncio.timeout(5), asyncio.TaskGroup() as group:
        original = group.create_task(client.post(PATH, json=chat_request))
        try:
            await backing_off.wait()
            assert [row["status"] for row in rows()] == [503]
            excess = await client.post(PATH, json=chat_request)
            assert excess.status_code == 429
            assert int(excess.headers["Retry-After"]) >= 1
            assert calls == 1
        finally:
            resume.set()
        assert (await original).status_code == 200
    assert [row["status"] for row in rows()] == [503, 429, 200]
