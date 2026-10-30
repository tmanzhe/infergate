"""Non-blocking per-caller admission within one gateway process."""

from collections import Counter
from contextlib import contextmanager
from collections.abc import Iterator


class ConcurrencyLimiter:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.in_flight: Counter[str] = Counter()

    @contextmanager
    def acquire(self, key: str) -> Iterator[bool]:
        # No await between checking and incrementing: atomic on the app's event loop.
        if self.maximum == 0:
            yield True
            return
        if self.in_flight[key] >= self.maximum:
            yield False
            return
        self.in_flight[key] += 1
        try:
            yield True
        finally:
            self.in_flight[key] -= 1
            if self.in_flight[key] == 0:
                del self.in_flight[key]
