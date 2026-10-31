"""In-memory queues for the mock worker experiment, owned by one dispatcher."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass


def positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    tenant: str
    estimated_tokens: int

    def __post_init__(self) -> None:
        positive_integer(self.estimated_tokens, "estimated_tokens")


class FIFOQueue:
    def __init__(self) -> None:
        self._requests: deque[ScheduledRequest] = deque()

    def push(self, request: ScheduledRequest) -> None:
        self._requests.append(request)

    def pop(self) -> ScheduledRequest:
        return self._requests.popleft()

    def __bool__(self) -> bool:
        return bool(self._requests)


class FairQueue:
    """Weighted deficit round robin; each visit grants quantum * tenant weight.

    Requests stay FIFO within each tenant. Unspent credit carries to the next
    round, so requests larger than a quantum eventually run. Empty tenants lose
    their credit and leave the active ring. Unknown tenants have weight one.
    This queue selects whole requests; running work is never preempted.
    """

    def __init__(self, quantum: int = 100, weights: Mapping[str, int] | None = None) -> None:
        positive_integer(quantum, "quantum")
        self.quantum = quantum
        self.weights = dict(weights or {})
        for weight in self.weights.values():
            positive_integer(weight, "weight")
        self._queues: dict[str, deque[ScheduledRequest]] = {}
        self._deficits: dict[str, int] = {}
        self._active: deque[str] = deque()
        self._visiting = False

    def push(self, request: ScheduledRequest) -> None:
        if request.tenant not in self._queues:
            self._queues[request.tenant] = deque()
            self._deficits[request.tenant] = 0
            self._active.append(request.tenant)
        self._queues[request.tenant].append(request)

    def pop(self) -> ScheduledRequest:
        while self._active:
            tenant = self._active[0]
            queue = self._queues[tenant]
            if not self._visiting:
                self._deficits[tenant] += self.quantum * self.weights.get(tenant, 1)
                self._visiting = True
            if queue[0].estimated_tokens <= self._deficits[tenant]:
                request = queue.popleft()
                self._deficits[tenant] -= request.estimated_tokens
                if not queue:
                    self._active.popleft()
                    del self._queues[tenant]
                    del self._deficits[tenant]
                    self._visiting = False
                return request
            self._active.rotate(-1)
            self._visiting = False
        raise IndexError("pop from an empty fair queue")

    def __bool__(self) -> bool:
        return bool(self._active)
