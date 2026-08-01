"""Request pacing and failure containment for source adapters.

Two independent concerns, deliberately kept separate:

* `RateLimiter` — never exceed a steady request rate. Always on.
* `CircuitBreaker` — stop entirely after repeated failures. Only trips on error.

The endpoints we call are unauthenticated and unofficial. Sustained abuse is
how such endpoints get locked down, which would break this project and everyone
else's. Restraint here is self-interested, not just polite.

Time is injected (`clock` / `sleep`) so the tests can verify pacing and cooldown
behaviour without actually sleeping.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from sources.base import CircuitOpen

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Enforces a minimum interval between requests, with jitter.

    Jitter matters: a request landing on an exact metronome tick is itself a
    machine signature, and it synchronises badly with server-side windows.
    """

    min_interval: float = 2.0
    jitter: float = 0.4
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last_request: float | None = field(default=None, init=False, repr=False)

    def acquire(self) -> None:
        """Block until it is acceptable to issue the next request."""
        now = self.clock()
        if self._last_request is not None:
            target = self.min_interval + random.uniform(0.0, self.jitter)
            elapsed = now - self._last_request
            if elapsed < target:
                delay = target - elapsed
                logger.debug("rate limiter sleeping %.2fs", delay)
                self.sleep(delay)
        self._last_request = self.clock()


@dataclass
class CircuitBreaker:
    """Halts a run after `threshold` consecutive failures.

    A single failure is noise — a removed listing, a blip. A *run* of them means
    something systemic (rate limiting, an endpoint change, no network), and every
    further request is both wasted and harmful. The counter resets on any
    success, so intermittent errors never accumulate into a false trip.
    """

    threshold: int = 3
    cooldown: float = 900.0
    clock: Callable[[], float] = time.monotonic
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _retry_after: float | None = field(default=None, init=False)

    @property
    def is_open(self) -> bool:
        """True while the breaker is refusing requests."""
        if self._opened_at is None:
            return False
        wait = self._retry_after if self._retry_after is not None else self.cooldown
        if self.clock() - self._opened_at >= wait:
            # Cooldown elapsed: half-open. Let the next request try.
            logger.info("circuit breaker cooldown elapsed, resuming")
            self._reset()
            return False
        return True

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def check(self) -> None:
        """Raise `CircuitOpen` if requests are currently suspended."""
        if self.is_open:
            assert self._opened_at is not None
            wait = self._retry_after if self._retry_after is not None else self.cooldown
            remaining = wait - (self.clock() - self._opened_at)
            raise CircuitOpen(
                f"circuit open after {self._consecutive_failures} consecutive "
                f"failures; {remaining:.0f}s of cooldown remaining"
            )

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self, retry_after: float | None = None) -> None:
        """Note a failure, tripping the breaker once the threshold is reached.

        `retry_after` (from the server's header) overrides the default cooldown —
        the remote telling us when to return beats us guessing.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.threshold:
            self._opened_at = self.clock()
            self._retry_after = retry_after
            wait = retry_after if retry_after is not None else self.cooldown
            logger.warning(
                "circuit breaker TRIPPED after %d consecutive failures; "
                "pausing for %.0fs",
                self._consecutive_failures,
                wait,
            )

    def _reset(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None
        self._retry_after = None
