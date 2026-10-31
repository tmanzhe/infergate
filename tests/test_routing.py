"""HTTP routing behavior using fake upstreams only; no network or paid APIs."""

import asyncio
import json
from dataclasses import replace
from decimal import Decimal

import httpx
import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from infergate.config import ModelRoute
from infergate.identity import key_hash


PATH = "/v1/chat/completions"
CHEAP = "gpt-4o-mini"
STRONG = "gpt-4o"
ROUTE = ModelRoute(CHEAP, STRONG, min_output_chars=8)
ROUTES = {"auto": ROUTE}
CHEAP_COST = Decimal("0.000027")
STRONG_COST = Decimal("0.000450")


def request_body(chat_request):
    return {**chat_request, "model": "auto", "vendor_option": {"keep": True}}


def answer(completion, model, content="A useful complete answer."):
    return {
        **completion,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


async def test_cheap_success_preserves_options_and_is_cached_with_actual_model(
    client_factory, rows, chat_request, completion
):
    calls = []
    body = request_body(chat_request)

    def upstream(request):
        calls.append(json.loads(request.content))
        assert calls[-1] == {**body, "model": CHEAP}
        return httpx.Response(200, json=answer(completion, CHEAP))

    async with client_factory(upstream, model_routes=ROUTES) as client:
        first = await client.post(PATH, json=body)
        cached = await client.post(PATH, json={**body, "user": "tracking-only", "stream": False,
                                              "metadata": {"trace": "ignored"}})
    assert first.status_code == cached.status_code == 200
    assert first.json() == cached.json() == answer(completion, CHEAP)
    assert len(calls) == 1
    assert [(row["model"], row["cost_usd"], row["cache_hit"]) for row in rows()] == [
        (CHEAP, CHEAP_COST, False), (CHEAP, 0, True)
    ]


@pytest.mark.parametrize("failure", [400, 429, 503, httpx.ConnectError, httpx.ReadTimeout])
async def test_upstream_failure_falls_back_and_logs_each_attempt_before_next(
    client_factory, rows, chat_request, completion, failure
):
    calls = []
    status = failure if isinstance(failure, int) else (504 if failure is httpx.ReadTimeout else 502)

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == CHEAP:
            if isinstance(failure, int):
                return httpx.Response(failure, json=answer(completion, CHEAP))
            raise failure("fake failure", request=request)
        assert [(row["model"], row["status"]) for row in rows()] == [(CHEAP, status)]
        return httpx.Response(200, json=answer(completion, STRONG))

    async with client_factory(upstream, model_routes=ROUTES) as client:
        first = await client.post(PATH, json=request_body(chat_request), headers={"Authorization": "Bearer caller"})
        cached = await client.post(PATH, json=request_body(chat_request), headers={"Authorization": "Bearer caller"})
    assert first.status_code == cached.status_code == 200
    assert first.json() == cached.json() == answer(completion, STRONG)
    assert calls == [CHEAP, STRONG]
    logs = rows()
    assert [row["model"] for row in logs] == [CHEAP, STRONG, STRONG]
    assert [row["status"] for row in logs] == [status, 200, 200]
    assert [row["cost_usd"] for row in logs] == [CHEAP_COST if isinstance(failure, int) else 0, STRONG_COST, 0]
    assert [row["cache_hit"] for row in logs] == [False, False, True]
    assert all(row["api_key_hash"] == key_hash("caller") for row in logs)


@pytest.mark.parametrize("bad", [
    {"choices": []}, {"choices": [None]}, {"choices": [{"message": None}]}, {},
    {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": "  \n "}}]},
    {"choices": [{"message": {"content": "short"}}]},
    {"choices": [{"message": {"content": "  I CAN'T help with that request."}}]},
    {"choices": [{"message": {"content": "I’m sorry, I cannot do that."}}]},
    {"choices": [{"message": {"content": "", "refusal": "No thanks."}}]},
    {"choices": [{"message": {"content": "Long enough answer", "refusal": "No thanks."}}]},
    "invalid JSON",
])
async def test_quality_failures_fall_back(client_factory, rows, chat_request, completion, bad):
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == CHEAP:
            return httpx.Response(200, text=bad) if isinstance(bad, str) else httpx.Response(200, json=bad)
        return httpx.Response(200, json=answer(completion, STRONG))

    async with client_factory(upstream, model_routes=ROUTES) as client:
        response = await client.post(PATH, json=request_body(chat_request))
    assert response.json() == answer(completion, STRONG)
    assert calls == [CHEAP, STRONG]
    assert [row["model"] for row in rows()] == [CHEAP, STRONG]
    assert rows("response_cache")[0]["model"] == STRONG


async def test_tool_call_without_text_is_valid_output(client_factory, rows, chat_request, completion):
    tool_answer = {**completion, "choices": [{"message": {"content": None, "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]}}]}
    calls = []

    def upstream(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=tool_answer)

    async with client_factory(upstream, model_routes=ROUTES) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).json() == tool_answer
    assert calls == [CHEAP]
    assert len(rows()) == 1


@pytest.mark.parametrize("rule,status,content,limits", [
    ("fallback_on_error", 503, "error", {}),
    ("fallback_on_quality", 200, "", {}),
    ("fallback_on_limit", 402, "unused", {"model_spend_caps_usd": {CHEAP: Decimal(0)}}),
])
async def test_each_fallback_rule_can_be_disabled(
    client_factory, rows, chat_request, completion, rule, status, content, limits
):
    calls = []

    def upstream(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(status, json=answer(completion, CHEAP, content))

    route = replace(ROUTE, **{rule: False})
    async with client_factory(upstream, model_routes={"auto": route}, **limits) as client:
        response = await client.post(PATH, json=request_body(chat_request))
    assert response.status_code == status
    assert calls == ([] if limits else [CHEAP])
    assert [row["model"] for row in rows()] == [CHEAP]


async def test_quality_threshold_and_refusal_prefixes_are_configurable(
    client_factory, rows, chat_request, completion
):
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(200, json=answer(completion, model, "declined" if model == CHEAP else "ok"))

    async with client_factory(upstream, model_routes={
        "auto": replace(ROUTE, min_output_chars=2, refusal_prefixes=("declined",))
    }) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).json()["model"] == STRONG
    assert calls == [CHEAP, STRONG]
    assert len(rows()) == 2


@pytest.mark.parametrize("strong_status", [200, 503])
async def test_fallback_is_bounded_and_unsatisfactory_answers_are_not_cached(
    client_factory, rows, chat_request, completion, strong_status
):
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(200 if model == CHEAP else strong_status, json=answer(completion, model, ""))

    # Even if the strong target has a route of its own, it must not recurse.
    async with client_factory(upstream, model_routes={**ROUTES, STRONG: ModelRoute("other", "last")}) as client:
        for _ in range(2):
            response = await client.post(PATH, json=request_body(chat_request))
            assert response.status_code == strong_status
            assert response.json()["model"] == STRONG
    assert calls == [CHEAP, STRONG, CHEAP, STRONG]
    assert len(rows()) == 4
    assert rows("response_cache") == []


async def test_retries_finish_before_fallback_and_all_models_are_accounted(
    client_factory, rows, chat_request, completion, retry_timing
):
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        status = 503 if len(calls) <= 3 else 200
        return httpx.Response(status, json=answer(completion, model))

    async with client_factory(upstream, model_routes=ROUTES, upstream_max_attempts=2) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).status_code == 200
    assert calls == [CHEAP, CHEAP, STRONG, STRONG]
    assert [row["model"] for row in rows()] == calls
    assert [row["cost_usd"] for row in rows()] == [CHEAP_COST, CHEAP_COST, STRONG_COST, STRONG_COST]
    assert len(retry_timing[1]) == 2


async def test_model_cap_uses_persisted_spend_across_callers_and_app_restarts(
    client_factory, rows, chat_request, completion
):
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(200, json=answer(completion, model))

    overrides = dict(model_routes=ROUTES, model_spend_caps_usd={CHEAP: CHEAP_COST}, cache_ttl_seconds=0)
    async with client_factory(upstream, **overrides) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).json()["model"] == CHEAP
    async with client_factory(upstream, **overrides) as client:
        assert (await client.post(PATH, json=request_body(chat_request),
                                  headers={"Authorization": "Bearer another-caller"})).json()["model"] == STRONG
    assert calls == [CHEAP, STRONG]
    assert [(row["model"], row["status"], row["cost_usd"]) for row in rows()] == [
        (CHEAP, 200, CHEAP_COST), (CHEAP, 402, 0), (STRONG, 200, STRONG_COST)
    ]


async def test_model_cap_stops_retries_then_falls_back(
    client_factory, rows, chat_request, completion, retry_timing
):
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(503 if model == CHEAP else 200, json=answer(completion, model))

    async with client_factory(upstream, model_routes=ROUTES, upstream_max_attempts=3,
                              model_spend_caps_usd={CHEAP: CHEAP_COST}) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).json()["model"] == STRONG
    assert calls == [CHEAP, STRONG]
    assert [(row["model"], row["status"]) for row in rows()] == [(CHEAP, 503), (CHEAP, 402), (STRONG, 200)]


@pytest.mark.parametrize("cap", [Decimal(0), CHEAP_COST])
async def test_caller_spend_cap_cannot_be_bypassed_by_fallback(
    client_factory, rows, chat_request, completion, cap
):
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(200, json=answer(completion, model, ""))

    async with client_factory(upstream, model_routes=ROUTES, spend_caps_usd={"caller": cap}) as client:
        response = await client.post(PATH, json=request_body(chat_request), headers={"Authorization": "Bearer caller"})
    assert response.status_code == 402
    assert "API key" in response.json()["error"]["message"]
    assert calls == ([] if cap == 0 else [CHEAP])
    assert sum(row["cost_usd"] for row in rows()) == cap
    assert rows("response_cache") == []


async def test_strong_model_also_obeys_its_own_cap(client_factory, rows, chat_request):
    async with client_factory(model_routes=ROUTES, model_spend_caps_usd={CHEAP: Decimal(0), STRONG: Decimal(0)}) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).status_code == 402
    assert [(row["model"], row["status"]) for row in rows()] == [(CHEAP, 402), (STRONG, 402)]


async def test_model_concurrency_falls_back_across_callers_and_releases_slot(
    client_factory, rows, chat_request, completion
):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == CHEAP and len(calls) == 1:
            entered.set()
            await release.wait()
        return httpx.Response(200, json=answer(completion, model))

    async with client_factory(upstream, model_routes=ROUTES, max_in_flight_per_model={CHEAP: 1},
                              cache_ttl_seconds=0) as client, asyncio.timeout(5), asyncio.TaskGroup() as group:
        first = group.create_task(client.post(PATH, json=request_body(chat_request)))
        try:
            await entered.wait()
            competing = await client.post(PATH, json=request_body(chat_request), headers={"Authorization": "Bearer other"})
            assert competing.json()["model"] == STRONG
            assert calls == [CHEAP, STRONG]
            assert [(row["model"], row["status"]) for row in rows()] == [(CHEAP, 429), (STRONG, 200)]
        finally:
            release.set()
        assert (await first).json()["model"] == CHEAP
        assert (await client.post(PATH, json=request_body(chat_request))).json()["model"] == CHEAP
    assert calls == [CHEAP, STRONG, CHEAP]


async def test_model_concurrency_slot_is_released_on_cancellation(
    client_factory, rows, chat_request, completion
):
    entered = asyncio.Event()
    calls = 0

    async def upstream(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.Event().wait()
        return httpx.Response(200, json=answer(completion, CHEAP))

    async with client_factory(upstream, model_routes=ROUTES, max_in_flight_per_model={CHEAP: 1}) as client, asyncio.timeout(5):
        pending = asyncio.create_task(client.post(PATH, json=request_body(chat_request)))
        try:
            await entered.wait()
        finally:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert (await client.post(PATH, json=request_body(chat_request))).json()["model"] == CHEAP
    assert calls == 2
    assert [row["model"] for row in rows()] == [CHEAP]


async def test_model_budget_serializes_different_callers_across_apps(
    client_factory, settings, rows, chat_request, completion
):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == CHEAP:
            entered.set()
            await release.wait()
        return httpx.Response(200, json=answer(completion, model))

    overrides = dict(model_routes=ROUTES, model_spend_caps_usd={CHEAP: CHEAP_COST}, cache_ttl_seconds=0,
                     database_url=make_conninfo(settings.database_url, application_name=settings.database_schema))
    async with client_factory(upstream, **overrides) as first, client_factory(upstream, **overrides) as second, \
            asyncio.timeout(5), asyncio.TaskGroup() as group:
        original = group.create_task(first.post(PATH, json=request_body(chat_request)))
        try:
            await entered.wait()
            competing = group.create_task(second.post(PATH, json=request_body(chat_request),
                                                      headers={"Authorization": "Bearer other"}))
            async with await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True) as observer:
                while True:
                    cursor = await observer.execute(
                        """SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE application_name = %s
                           AND wait_event_type = 'Lock' AND wait_event = 'advisory')""",
                        (settings.database_schema,),
                    )
                    if (await cursor.fetchone())[0]:
                        break
                    await asyncio.sleep(0.01)
            assert calls == [CHEAP]
        finally:
            release.set()
        assert (await original).json()["model"] == CHEAP
        assert (await competing).json()["model"] == STRONG
    assert calls == [CHEAP, STRONG]
    assert [(row["model"], row["status"]) for row in rows()] == [(CHEAP, 200), (CHEAP, 402), (STRONG, 200)]


async def test_route_policy_changes_invalidate_cache_and_restart_preserves_winner(
    client_factory, rows, chat_request, completion
):
    calls = []

    def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return httpx.Response(200, json=answer(completion, model))

    async with client_factory(upstream, model_routes=ROUTES) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).json()["model"] == CHEAP
    changed = {"auto": replace(ROUTE, min_output_chars=100)}
    async with client_factory(upstream, model_routes=changed) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).json()["model"] == STRONG
    async with client_factory(model_routes=ROUTES) as client:
        assert (await client.post(PATH, json=request_body(chat_request))).json()["model"] == CHEAP
    assert calls == [CHEAP, CHEAP, STRONG]
    assert rows()[-1]["cache_hit"]
    assert rows()[-1]["model"] == CHEAP


async def test_mock_routing_is_network_free_and_logs_zero_cost(
    client_factory, rows, chat_request, monkeypatch
):
    from infergate import app as app_module

    original = app_module.mock_completion

    def mock_completion(body):
        result = original(body)
        if body["model"] == CHEAP:
            result["choices"][0]["message"]["content"] = ""
        return result

    monkeypatch.setattr(app_module, "mock_completion", mock_completion)
    async with client_factory(mock_model=True, model_routes=ROUTES) as client:
        first = await client.post(PATH, json=request_body(chat_request))
        cached = await client.post(PATH, json=request_body(chat_request))
        streaming = await client.post(PATH, json={**request_body(chat_request), "stream": True})
    assert first.json()["model"] == cached.json()["model"] == STRONG
    assert streaming.status_code == 400
    assert [row["model"] for row in rows()] == [CHEAP, STRONG, STRONG, "auto"]
    assert all(row["cost_usd"] == 0 for row in rows())
