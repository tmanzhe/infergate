from decimal import Decimal

import httpx
import pytest

from infergate.identity import key_hash


PATH = "/v1/chat/completions"
TIMEOUTS = [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout]


def fail_upstream(request, failure):
    if isinstance(failure, int):
        # Failed upstream attempts may still report billable usage.
        return httpx.Response(
            failure,
            json={
                "error": {"message": "fake transient failure"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
            headers={"Retry-After": "7"},
        )
    raise failure("fake transport failure", request=request)


@pytest.mark.parametrize("failure", [429, 500, 503, 599, *TIMEOUTS])
async def test_retryable_failures_use_capped_exponential_full_jitter_and_log_every_attempt(
    client_factory, rows, chat_request, completion, retry_timing, failure
):
    windows, delays = retry_timing
    calls = []
    status = failure if isinstance(failure, int) else 504

    def upstream(request):
        # Each failed attempt must already be committed before the next one starts.
        assert [row["status"] for row in rows()] == [status] * len(calls)
        assert len(delays) == len(calls)
        calls.append(request)
        if len(calls) < 5:
            return fail_upstream(request, failure)
        return httpx.Response(200, json=completion)

    async with client_factory(
        upstream,
        cache_ttl_seconds=0,
        upstream_max_attempts=5,
        upstream_retry_base_seconds=2,
        upstream_retry_max_seconds=5,
    ) as client:
        response = await client.post(
            PATH, json=chat_request, headers={"Authorization": "Bearer retry-caller"}
        )

    assert response.status_code == 200
    assert response.json() == completion
    assert len(calls) == 5
    assert windows == [(0, 2), (0, 4), (0, 5), (0, 5)]
    assert delays == [0.5, 3, 1.25, 3.75]
    assert all(request.content == calls[0].content for request in calls)
    assert all(request.headers["authorization"] == "Bearer test-secret" for request in calls)
    assert all(str(request.url) == "https://upstream.test/custom/v1/chat/completions" for request in calls)
    logs = rows()
    assert [row["status"] for row in logs] == [status] * 4 + [200]
    for row in logs:
        assert row["api_key_hash"] == key_hash("retry-caller")
        assert row["model"] == chat_request["model"]
        assert row["latency_ms"] >= 0
        assert not row["cache_hit"]
    for row in logs[:-1]:
        assert row["prompt_tokens"] == (10 if isinstance(failure, int) else 0)
        assert row["completion_tokens"] == (2 if isinstance(failure, int) else 0)
        assert row["cost_usd"] == (Decimal("0.0000027") if isinstance(failure, int) else 0)
    assert logs[-1]["cost_usd"] == Decimal("0.000027")


@pytest.mark.parametrize("failure", [429, 500, 503, httpx.ReadTimeout])
@pytest.mark.parametrize("maximum", [1, 3])
async def test_retry_attempts_are_bounded_and_last_failure_is_returned(
    client_factory, rows, chat_request, retry_timing, failure, maximum
):
    windows, delays = retry_timing
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return fail_upstream(request, failure)

    async with client_factory(upstream, upstream_max_attempts=maximum) as client:
        response = await client.post(PATH, json=chat_request)

    status = failure if isinstance(failure, int) else 504
    assert response.status_code == status
    if isinstance(failure, int):
        assert response.json()["error"]["message"] == "fake transient failure"
        assert response.headers["Retry-After"] == "7"
    else:
        assert "timed out" in response.json()["error"]["message"].lower()
    assert calls == maximum
    assert len(delays) == len(windows) == maximum - 1
    assert [row["status"] for row in rows()] == [status] * maximum
    assert rows("response_cache") == []


@pytest.mark.parametrize("failure", [400, 401, 403, 404, 408, 422, httpx.ConnectError])
async def test_other_failures_do_not_retry(
    client_factory, rows, chat_request, retry_timing, failure
):
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return fail_upstream(request, failure)

    async with client_factory(upstream, upstream_max_attempts=5) as client:
        response = await client.post(PATH, json=chat_request)
    status = failure if isinstance(failure, int) else 502
    assert response.status_code == status
    assert calls == 1
    assert retry_timing == ([], [])
    assert [row["status"] for row in rows()] == [status]
    assert rows("response_cache") == []


async def test_only_successful_retry_is_cached(
    client_factory, rows, chat_request, completion, retry_timing
):
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="fake failure")
        return httpx.Response(200, json=completion)

    async with client_factory(upstream, upstream_max_attempts=3) as client:
        response = await client.post(PATH, json=chat_request)
        cached = await client.post(PATH, json=chat_request)
    assert response.status_code == cached.status_code == 200
    assert response.content == cached.content
    assert calls == 2
    assert len(retry_timing[1]) == 1
    assert [row["status"] for row in rows()] == [503, 200, 200]
    assert [row["cache_hit"] for row in rows()] == [False, False, True]
    assert [row["cost_usd"] for row in rows()] == [0, Decimal("0.000027"), 0]
    assert len(rows("response_cache")) == 1
