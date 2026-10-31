import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from infergate import retries
from infergate.app import create_app
from infergate.config import Settings


@pytest.fixture(scope="session")
def database_url():
    url = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/infergate")
    try:
        with psycopg.connect(url, connect_timeout=3):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres is unreachable: {exc}")
    return url


@pytest.fixture
def settings(database_url):
    schema = f"test_infergate_{uuid.uuid4().hex}"
    yield Settings(
        database_url=database_url,
        database_schema=schema,
        upstream_base_url="https://upstream.test/custom/v1/",
        upstream_api_key="test-secret",
    )
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def client_factory(settings):
    @asynccontextmanager
    async def make_client(handler=None, *, clock=None, **overrides):
        def unexpected_request(request):
            pytest.fail(f"Unexpected network request: {request.url}")

        transport = httpx.MockTransport(handler or unexpected_request)
        kwargs = {"clock": clock} if clock is not None else {}
        app = create_app(replace(settings, **overrides), upstream_transport=transport, **kwargs)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://infergate.test"
            ) as client:
                yield client

    return make_client


@pytest.fixture
def rows(settings):
    def read(table="request_logs"):
        with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
            query = sql.SQL("SELECT * FROM {} ORDER BY {}").format(
                sql.Identifier(settings.database_schema, table),
                sql.Identifier("id" if table == "request_logs" else "cache_key"),
            )
            return conn.execute(query).fetchall()

    return read


@pytest.fixture
def chat_request():
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
        "max_tokens": 64,
    }


@pytest.fixture
def completion():
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello back!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


@pytest.fixture
def retry_timing(monkeypatch):
    """Observe full-jitter backoff without sleeping or changing global asyncio."""
    windows = []
    delays = []

    def jitter(lower, upper):
        windows.append((lower, upper))
        fraction = (0.25, 0.75)[(len(windows) - 1) % 2]
        return lower + (upper - lower) * fraction

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(retries, "uniform", jitter)
    monkeypatch.setattr(retries, "sleep", sleep)
    return windows, delays
