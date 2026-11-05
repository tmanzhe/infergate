"""Async admission and response lifetime around the vendored DRR queue."""

import asyncio
from dataclasses import dataclass

from fastapi.responses import Response
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from infergate._vendor.fairq import FairQueue, Request


class QueueFull(Exception):
    pass


@dataclass
class Slot:
    gate: "FairGate"
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.gate.active -= 1
            self.gate._wake()


class FairGate:
    def __init__(self, *, weights: dict[str, int], slots: int, max_depth: int) -> None:
        self.weights = dict(weights)
        self.slots = slots
        self.max_depth = max_depth
        self.depth = 0
        self.active = 0
        self._queue: FairQueue[asyncio.Future[Slot]] = FairQueue(quantum=100, weights=weights)
        self._scheduled = False

    def _wake(self) -> None:
        if not self._scheduled:
            self._scheduled = True
            asyncio.get_running_loop().call_soon(self._dispatch)

    def _dispatch(self) -> None:
        self._scheduled = False
        # Cancelled entries are tombstones: depth counts only live waiters.
        while self._queue and (self.active < self.slots or self.depth == 0):
            request: Request[asyncio.Future[Slot]] = self._queue.dispatch()
            if request.item.cancelled():
                continue
            self.depth -= 1
            self.active += 1
            request.item.set_result(Slot(self))

    async def acquire(self, tenant: str, receive: Receive) -> Slot:
        if self.depth >= self.max_depth:
            raise QueueFull
        future: asyncio.Future[Slot] = asyncio.get_running_loop().create_future()
        tenant = tenant if tenant in self.weights else "default"
        # Equal admission cost: one request spends one base quantum.
        self._queue.enqueue(tenant, future, cost=100)
        self.depth += 1
        self._wake()

        async def disconnected() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        watcher = asyncio.create_task(disconnected())
        try:
            try:
                done, _ = await asyncio.wait((future, watcher), return_when=asyncio.FIRST_COMPLETED)
                if watcher in done:
                    watcher.result()
                    raise ClientDisconnect
                return future.result()
            finally:
                # Hand receive back to the streaming response before returning.
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
        except BaseException:
            if future.done() and not future.cancelled():
                future.result().release()
            else:
                future.cancel()
                self.depth -= 1
                self._wake()
            raise


class GatedResponse(Response):
    def __init__(self, response: Response, slot: Slot) -> None:
        super().__init__(status_code=response.status_code)
        self.raw_headers = response.raw_headers
        self.response = response
        self.slot = slot

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self.response(scope, receive, send)
        finally:
            self.slot.release()
