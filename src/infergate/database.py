"""Postgres persistence; identifiers are quoted, values are bound parameters."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import AsyncIterator

import psycopg
from psycopg import sql


@dataclass(frozen=True)
class CachedResponse:
    body: bytes
    status: int
    content_type: str | None
    model: str | None = None


class Database:
    def __init__(self, url: str, schema: str = "public") -> None:
        self.url = url
        self.schema = schema

    def table(self, name: str) -> sql.Identifier:
        return sql.Identifier(self.schema, name)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection]:
        async with await psycopg.AsyncConnection.connect(self.url, connect_timeout=5) as conn:
            yield conn

    async def initialize(self) -> None:
        async with self.connection() as conn:
            await conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
            )
            await conn.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens BIGINT NOT NULL,
                    completion_tokens BIGINT NOT NULL,
                    cost_usd NUMERIC(24, 12) NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL,
                    cache_hit BOOLEAN NOT NULL,
                    status INTEGER NOT NULL
                )
            """).format(self.table("request_logs")))
            # Existing installations retain their unattributed historical rows.
            await conn.execute(sql.SQL(
                "ALTER TABLE {} ADD COLUMN IF NOT EXISTS api_key_hash TEXT"
            ).format(self.table("request_logs")))
            await conn.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS request_logs_api_key_hash_idx ON {} (api_key_hash)"
            ).format(self.table("request_logs")))
            await conn.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS request_logs_model_idx ON {} (model)"
            ).format(self.table("request_logs")))
            await conn.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    cache_key TEXT PRIMARY KEY,
                    response_body BYTEA NOT NULL,
                    status INTEGER NOT NULL,
                    content_type TEXT,
                    expires_at TIMESTAMPTZ NOT NULL
                )
            """).format(self.table("response_cache")))
            await conn.execute(sql.SQL(
                "ALTER TABLE {} ADD COLUMN IF NOT EXISTS model TEXT"
            ).format(self.table("response_cache")))

    async def get_cached(self, key: str, now: datetime) -> CachedResponse | None:
        async with self.connection() as conn:
            cursor = await conn.execute(
                sql.SQL("""
                    SELECT response_body, status, content_type, model FROM {}
                    WHERE cache_key = %s AND expires_at > %s
                """).format(self.table("response_cache")),
                (key, now),
            )
            row = await cursor.fetchone()
        return CachedResponse(*row) if row else None

    async def put_cached(self, key: str, response: CachedResponse, expires_at: datetime) -> None:
        async with self.connection() as conn:
            await conn.execute(
                sql.SQL("""
                    INSERT INTO {} (cache_key, response_body, status, content_type, expires_at, model)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        response_body = EXCLUDED.response_body,
                        status = EXCLUDED.status,
                        content_type = EXCLUDED.content_type,
                        expires_at = EXCLUDED.expires_at,
                        model = EXCLUDED.model
                """).format(self.table("response_cache")),
                (key, response.body, response.status, response.content_type, expires_at, response.model),
            )

    async def log_request(
        self,
        *,
        timestamp: datetime,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: Decimal,
        latency_ms: float,
        cache_hit: bool,
        status: int,
        api_key_hash: str | None = None,
    ) -> None:
        async with self.connection() as conn:
            await conn.execute(
                sql.SQL("""
                    INSERT INTO {} (timestamp, model, prompt_tokens, completion_tokens,
                                    cost_usd, latency_ms, cache_hit, status, api_key_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """).format(self.table("request_logs")),
                (timestamp, model, prompt_tokens, completion_tokens,
                 cost_usd, latency_ms, cache_hit, status, api_key_hash),
            )
