"""Adapted from infergate's MIT-licensed scheduling.py by Truman Chan."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _tenant_name(tenant: str) -> None:
    if not isinstance(tenant, str) or not tenant.strip():
        raise ValueError("tenant must be a nonempty string")


@dataclass(frozen=True, slots=True)
class Request[T]:
    """An opaque payload and its tenant and estimated token cost."""

    tenant: str
    item: T
    cost: int

    def __post_init__(self) -> None:
        _tenant_name(self.tenant)
        _positive_integer(self.cost, "cost")


class FIFOQueue[T]:
    """Global arrival order. Accepts weight for API parity, but ignores it."""

    def __init__(self) -> None:
        self._requests: deque[Request[T]] = deque()

    def enqueue(
        self, tenant: str, item: T, *, cost: int, weight: int | None = None
    ) -> Request[T]:
        """Append a request, validating cost and any supplied weight."""
        request = Request(tenant, item, cost)
        if weight is not None:
            _positive_integer(weight, "weight")
        self._requests.append(request)
        return request

    def dispatch(self) -> Request[T]:
        """Remove the oldest request; raise IndexError if empty."""
        if not self._requests:
            raise IndexError("dispatch from an empty FIFO queue")
        return self._requests.popleft()

    def __len__(self) -> int:
        return len(self._requests)


class FairQueue[T]:
    """Weighted deficit round robin, FIFO within each tenant.

    A visit grants quantum * weight tokens and dispatches affordable requests
    before rotating. Unspent credit carries between rounds. Empty tenants lose
    credit and leave the ring; new tenants join at the end. Running requests
    cannot be preempted. Instances belong to one dispatcher, without locking.
    """

    def __init__(
        self, quantum: int = 100, *, weights: Mapping[str, int] | None = None
    ) -> None:
        _positive_integer(quantum, "quantum")
        self._quantum = quantum
        self._weights = dict(weights) if weights is not None else {}
        for tenant, weight in self._weights.items():
            _tenant_name(tenant)
            _positive_integer(weight, "weight")
        self._queues: dict[str, deque[Request[T]]] = {}
        self._deficits: dict[str, int] = {}
        self._active: deque[str] = deque()
        self._visiting = False
        self._size = 0

    @property
    def quantum(self) -> int:
        """Token allowance per visit before applying tenant weight."""
        return self._quantum

    def enqueue(
        self, tenant: str, item: T, *, cost: int, weight: int | None = None
    ) -> Request[T]:
        """Queue work and optionally set this tenant's persistent weight.

        Weights default to one. Changing a weight while that tenant has queued
        work raises ValueError; repeating its current weight is allowed.
        Invalid input leaves the queue unchanged.
        """
        request = Request(tenant, item, cost)
        if weight is not None:
            _positive_integer(weight, "weight")
            if tenant in self._queues and weight != self._weights.get(tenant, 1):
                raise ValueError("cannot change weight while tenant has queued work")
            self._weights[tenant] = weight
        if tenant not in self._queues:
            self._queues[tenant] = deque()
            self._deficits[tenant] = 0
            self._active.append(tenant)
        self._queues[tenant].append(request)
        self._size += 1
        return request

    def dispatch(self) -> Request[T]:
        """Select and remove one request; raise IndexError if empty.

        Requests larger than a visit's allowance accumulate credit over rounds
        until affordable, without letting later requests pass their tenant head.
        """
        while self._active:
            tenant = self._active[0]
            queue = self._queues[tenant]
            if not self._visiting:
                self._deficits[tenant] += self._quantum * self._weights.get(tenant, 1)
                self._visiting = True
            if queue[0].cost <= self._deficits[tenant]:
                request = queue.popleft()
                self._deficits[tenant] -= request.cost
                self._size -= 1
                if not queue:
                    self._active.popleft()
                    del self._queues[tenant]
                    del self._deficits[tenant]
                    self._visiting = False
                return request
            self._active.rotate(-1)
            self._visiting = False
        raise IndexError("dispatch from an empty fair queue")

    def __len__(self) -> int:
        return self._size
