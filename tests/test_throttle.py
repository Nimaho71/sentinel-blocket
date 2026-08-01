"""Pacing and failure-containment tests.

Clock and sleep are injected, so these assert real timing behaviour without
spending real time.
"""

from __future__ import annotations

import pytest

from sources.base import CircuitOpen
from sources.throttle import CircuitBreaker, RateLimiter


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ------------------------------------------------------------------ rate limiter


def test_first_request_does_not_sleep():
    clock = FakeClock()
    slept: list[float] = []
    limiter = RateLimiter(min_interval=2.0, jitter=0.0, clock=clock, sleep=slept.append)

    limiter.acquire()

    assert slept == [], "the first request should go out immediately"


def test_back_to_back_requests_are_spaced():
    clock = FakeClock()
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(seconds)

    limiter = RateLimiter(min_interval=2.0, jitter=0.0, clock=clock, sleep=sleep)
    limiter.acquire()
    limiter.acquire()

    assert slept == pytest.approx([2.0]), "second request must wait the full interval"


def test_no_sleep_when_enough_time_already_passed():
    clock = FakeClock()
    slept: list[float] = []
    limiter = RateLimiter(min_interval=2.0, jitter=0.0, clock=clock, sleep=slept.append)

    limiter.acquire()
    clock.advance(5.0)
    limiter.acquire()

    assert slept == [], "a naturally slow caller should never be delayed further"


def test_jitter_stays_within_bounds():
    clock = FakeClock()
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(seconds)

    limiter = RateLimiter(min_interval=2.0, jitter=0.5, clock=clock, sleep=sleep)
    for _ in range(20):
        limiter.acquire()

    assert slept, "expected throttling to occur"
    assert all(2.0 <= s <= 2.5 for s in slept), f"jitter escaped its bounds: {slept}"
    assert len(set(slept)) > 1, "jitter should vary between requests, not be constant"


# --------------------------------------------------------------- circuit breaker


def test_stays_closed_below_threshold():
    breaker = CircuitBreaker(threshold=3, clock=FakeClock())

    breaker.record_failure()
    breaker.record_failure()

    breaker.check()  # must not raise: two failures is noise, not a pattern
    assert not breaker.is_open


def test_trips_on_threshold_consecutive_failures():
    breaker = CircuitBreaker(threshold=3, clock=FakeClock())

    for _ in range(3):
        breaker.record_failure()

    assert breaker.is_open
    with pytest.raises(CircuitOpen, match="3 consecutive"):
        breaker.check()


def test_success_resets_the_counter():
    """Intermittent failures must never accumulate into a false trip."""
    breaker = CircuitBreaker(threshold=3, clock=FakeClock())

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    breaker.check()  # 2 consecutive, not 4 total
    assert not breaker.is_open


def test_resumes_after_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown=900.0, clock=clock)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open

    clock.advance(899.0)
    assert breaker.is_open, "must not resume early"

    clock.advance(2.0)
    breaker.check()  # cooldown elapsed
    assert breaker.consecutive_failures == 0, "resuming should reset the counter"


def test_retry_after_overrides_default_cooldown():
    """The server telling us when to return beats our guess."""
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown=900.0, clock=clock)

    breaker.record_failure(retry_after=30.0)
    assert breaker.is_open

    clock.advance(31.0)
    breaker.check()  # honoured 30s, not the 900s default
    assert not breaker.is_open
