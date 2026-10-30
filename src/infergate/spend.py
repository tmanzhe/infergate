"""Lifetime budgets from committed Postgres cost logs, serialized per capped key."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal

import psycopg
from psycopg import sql

from infergate.database import Database


@dataclass
class SpendBudget:
    database: Database
    api_key_hash: str
    cap: Decimal | None
    connection: psycopg.AsyncConnection | None = None

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        if self.cap is None:
            yield
            return
        async with self.database.connection() as conn:
            # Transaction-scoped lock also coordinates other gateway processes.
            lock_id = int.from_bytes(bytes.fromhex(self.api_key_hash)[:8], signed=True)
            await conn.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))
            self.connection = conn
            try:
                yield
            finally:
                self.connection = None

    async def exhausted(self) -> bool:
        if self.cap is None:
            return False
        assert self.connection is not None, "Check budgets inside guard()"
        cursor = await self.connection.execute(
            sql.SQL("SELECT COALESCE(SUM(cost_usd), 0) FROM {} WHERE api_key_hash = %s").format(
                self.database.table("request_logs")
            ),
            (self.api_key_hash,),
        )
        row = await cursor.fetchone()
        return row[0] >= self.cap
