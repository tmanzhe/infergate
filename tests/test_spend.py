import asyncio
from decimal import Decimal

import httpx
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from infergate.identity import key_hash


PATH = "/v1/chat/completions"
COST = Decimal("0.000027")  # Fixture usage at the default gpt-4o-mini prices.
AUTH = {"Authorization": "Bearer capped-caller"}


@pytest.mark.parametrize("cap,allowed", [
    (Decimal(0), 0),
    (COST, 1),
    (COST + Decimal("0.000001"), 2),
    (COST * 2, 2),
])
async def test_cap_rejects_once_logged_spend_reaches_or_exceeds_limit(
    client_factory, rows, chat_request, completion, cap, allowed
):
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion)

    async with client_factory(
        upstream, cache_ttl_seconds=0, spend_caps_usd={"capped-caller": cap}
    ) as client:
        for _ in range(allowed):
            assert (await client.post(PATH, json=chat_request, headers=AUTH)).status_code == 200
        for _ in range(2):
            rejected = await client.post(PATH, json=chat_request, headers=AUTH)
            assert rejected.status_code == 402
            assert "spend cap" in rejected.json()["error"]["message"].lower()

    assert calls == allowed
    logs = rows()
    assert [row["status"] for row in logs] == [200] * allowed + [402, 402]
    assert [row["cost_usd"] for row in logs] == [COST] * allowed + [0, 0]
    assert all(row["api_key_hash"] == key_hash("capped-caller") for row in logs)
    assert all(not row["cache_hit"] for row in logs)


async def test_spend_is_isolated_per_caller_and_uncapped_callers_remain_allowed(
    client_factory, rows, chat_request, completion
):
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion)

    async with client_factory(
        upstream,
        cache_ttl_seconds=0,
        spend_caps_usd={"capped-caller": COST, "second-caller": COST * 2},
    ) as client:
        for caller, expected in [
            ("capped-caller", 200), ("capped-caller", 402),
            ("second-caller", 200), ("second-caller", 200), ("second-caller", 402),
            ("uncapped-caller", 200), ("uncapped-caller", 200), ("uncapped-caller", 200),
        ]:
            response = await client.post(
                PATH, json=chat_request, headers={"Authorization": f"Bearer {caller}"}
            )
            assert response.status_code == expected

    assert calls == 6
    assert {
        caller: sum(row["cost_usd"] for row in rows() if row["api_key_hash"] == key_hash(caller))
        for caller in ("capped-caller", "second-caller", "uncapped-caller")
    } == {"capped-caller": COST, "second-caller": COST * 2, "uncapped-caller": COST * 3}


async def test_spend_cap_and_cached_response_survive_app_restart(
    client_factory, rows, chat_request, completion
):
    # Record spend before the cap is even configured, then create a fresh app.
    async with client_factory(lambda request: httpx.Response(200, json=completion)) as client:
        assert (await client.post(PATH, json=chat_request, headers=AUTH)).status_code == 200
    assert len(rows("response_cache")) == 1

    # No handler: the factory fails the test if a rejected request touches upstream.
    async with client_factory(spend_caps_usd={"capped-caller": COST}) as restarted:
        same_body = await restarted.post(PATH, json=chat_request, headers=AUTH)
        cache_miss = await restarted.post(PATH, json={**chat_request, "seed": 42}, headers=AUTH)
    assert same_body.status_code == cache_miss.status_code == 402
    assert [row["status"] for row in rows()] == [200, 402, 402]
    assert [row["cost_usd"] for row in rows()] == [COST, 0, 0]
    assert all(not row["cache_hit"] for row in rows())


async def test_budget_reads_committed_postgres_cost_log_on_each_request(
    client_factory, settings, rows, chat_request, completion
):
    calls = 0
    cap = COST * 10

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion)

    async with client_factory(
        upstream, cache_ttl_seconds=0, spend_caps_usd={"capped-caller": cap}
    ) as client:
        assert (await client.post(PATH, json=chat_request, headers=AUTH)).status_code == 200
        # Simulate another process committing spend, without touching app state.
        with psycopg.connect(settings.database_url) as conn:
            conn.execute(
                sql.SQL("UPDATE {} SET cost_usd = %s WHERE api_key_hash = %s").format(
                    sql.Identifier(settings.database_schema, "request_logs")
                ),
                (cap, key_hash("capped-caller")),
            )
        assert (await client.post(PATH, json=chat_request, headers=AUTH)).status_code == 402
    assert calls == 1
    assert [row["cost_usd"] for row in rows()] == [cap, 0]


async def test_failed_attempts_consume_budget_and_stop_further_retries(
    client_factory, rows, chat_request, completion, retry_timing
):
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={**completion, "error": {"message": "fake failure"}})

    async with client_factory(
        upstream,
        cache_ttl_seconds=0,
        upstream_max_attempts=5,
        spend_caps_usd={"capped-caller": COST * 2},
    ) as client:
        rejected = await client.post(PATH, json=chat_request, headers=AUTH)
        assert rejected.status_code == 402
        assert "spend cap" in rejected.json()["error"]["message"].lower()
        assert (await client.post(PATH, json=chat_request, headers=AUTH)).status_code == 402
    assert calls == 2
    assert [row["status"] for row in rows()] == [503, 503, 402, 402]
    assert [row["cost_usd"] for row in rows()] == [COST, COST, 0, 0]
    assert rows("response_cache") == []


async def test_cache_hits_are_free_but_cap_is_checked_before_serving_cache(
    client_factory, rows, chat_request, completion
):
    calls = 0

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion)

    async with client_factory(upstream, spend_caps_usd={"capped-caller": COST * 2}) as client:
        for _ in range(3):
            assert (await client.post(PATH, json=chat_request, headers=AUTH)).status_code == 200
        assert (await client.post(PATH, json={**chat_request, "seed": 1}, headers=AUTH)).status_code == 200
        assert (await client.post(PATH, json=chat_request, headers=AUTH)).status_code == 402
    assert calls == 2
    assert [row["cost_usd"] for row in rows()] == [COST, 0, 0, COST, 0]
    assert [row["cache_hit"] for row in rows()] == [False, True, True, False, False]


async def test_mock_requests_do_not_consume_positive_spend_cap(client_factory, rows, chat_request):
    async with client_factory(
        mock_model=True, cache_ttl_seconds=0, spend_caps_usd={"capped-caller": Decimal("0.000001")}
    ) as client:
        for _ in range(3):
            assert (await client.post(PATH, json=chat_request, headers=AUTH)).status_code == 200
    assert len(rows()) == 3
    assert all(row["cost_usd"] == 0 for row in rows())


async def test_budget_serializes_same_key_across_app_instances(
    client_factory, settings, rows, chat_request, completion
):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def upstream(request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return httpx.Response(200, json=completion)

    overrides = {
        "cache_ttl_seconds": 0,
        "spend_caps_usd": {"capped-caller": COST},
        # Identify this test's connections when observing the actual database wait.
        "database_url": make_conninfo(settings.database_url, application_name=settings.database_schema),
    }
    async with client_factory(upstream, **overrides) as first, client_factory(
        upstream, **overrides
    ) as second, asyncio.timeout(5), asyncio.TaskGroup() as group:
        original = group.create_task(first.post(PATH, json=chat_request, headers=AUTH))
        try:
            await entered.wait()
            competing = group.create_task(second.post(PATH, json=chat_request, headers=AUTH))
            # Wait for evidence that the other app is blocked on the budget lock.
            # This avoids releasing the first request before its competitor runs.
            async with await psycopg.AsyncConnection.connect(
                settings.database_url, autocommit=True
            ) as observer:
                while True:
                    cursor = await observer.execute(
                        """SELECT EXISTS (
                            SELECT 1 FROM pg_stat_activity
                            WHERE application_name = %s AND wait_event_type = 'Lock'
                              AND wait_event = 'advisory'
                        )""",
                        (settings.database_schema,),
                    )
                    if (await cursor.fetchone())[0]:
                        break
                    await asyncio.sleep(0.01)
            assert calls == 1
            assert not competing.done()
        finally:
            release.set()
        assert (await original).status_code == 200
        assert (await competing).status_code == 402
    assert calls == 1
    assert [row["status"] for row in rows()] == [200, 402]
    assert [row["cost_usd"] for row in rows()] == [COST, 0]
