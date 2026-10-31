"""Lifetime budgets from committed Postgres cost logs, serialized per capped key."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal

import psycopg
from psycopg import sql

from infergate.database import Database
from infergate.identity import key_hash


@dataclass
class SpendBudget:
    database: Database
    api_key_hash: str
    cap: Decimal | None
    model: str | None = None
    connection: psycopg.AsyncConnection | None = None

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        if self.cap is None:
            yield
            return
        async with self.database.connection() as conn:
            # Transaction-scoped lock also coordinates other gateway processes.
            identity = key_hash(f"infergate:model:{self.model}") if self.model is not None else self.api_key_hash
            lock_id = int.from_bytes(bytes.fromhex(identity)[:8], signed=True)
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
        column = "model" if self.model is not None else "api_key_hash"
        value = self.model if self.model is not None else self.api_key_hash
        cursor = await self.connection.execute(
            sql.SQL("SELECT COALESCE(SUM(cost_usd), 0) FROM {} WHERE {} = %s").format(
                self.database.table("request_logs"), sql.Identifier(column)
            ),
            (value,),
        )
        row = await cursor.fetchone()
        return row[0] >= self.cap
