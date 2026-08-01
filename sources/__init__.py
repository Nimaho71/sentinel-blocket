"""Marketplace source adapters.

Adding a marketplace means adding a module here that satisfies `Source`; nothing
downstream changes.
"""

from sources.base import (
    AdRecord,
    AdSummary,
    CircuitOpen,
    RateLimited,
    Source,
    SourceError,
)
from sources.blocket import BlocketSource
from sources.throttle import CircuitBreaker, RateLimiter

__all__ = [
    "AdRecord",
    "AdSummary",
    "BlocketSource",
    "CircuitBreaker",
    "CircuitOpen",
    "RateLimited",
    "RateLimiter",
    "Source",
    "SourceError",
]
