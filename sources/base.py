"""Marketplace source abstraction.

Every marketplace we pull from implements `Source`. The pipeline downstream of
this module never sees vendor-specific payloads — only `AdSummary` / `AdRecord`.

Two invariants this module is responsible for, both load-bearing:

1. **PII never enters the system.** Stripping happens here, at the boundary,
   not at publish time. Blocket's detail payload carries GPS coordinates
   (`location.position.lat/lng`) and a seller id (`meta.ownerId`); neither has
   any field on `AdRecord` to land in. A redaction bug downstream therefore
   cannot leak what was never ingested.
2. **Sources are swappable.** Adding Tradera means adding a module here and
   nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, Sequence


class SourceError(Exception):
    """Base class for source-layer failures."""


class RateLimited(SourceError):
    """The remote asked us to slow down (429/503, or an explicit Retry-After).

    Carries the server-advertised cooldown when one was supplied, so callers can
    honour it instead of guessing.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class CircuitOpen(SourceError):
    """Too many consecutive failures; the source has stopped making requests.

    Raised instead of issuing a request. Hammering a failing unauthenticated
    endpoint is how it gets locked down for everyone, so we stop early and
    resume on a later run.
    """


@dataclass(frozen=True, slots=True)
class AdSummary:
    """A listing as it appears in search results.

    Cheap to obtain — one request returns ~50 of these. Deliberately carries no
    description: that requires a second request per ad, so the pipeline
    pre-filters on this before paying for detail.
    """

    source: str
    external_id: str
    url: str
    title: str
    price_sek: int
    city: str | None = None
    listed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.external_id:
            raise ValueError("external_id is required")
        if self.price_sek < 0:
            raise ValueError(f"negative price for {self.external_id}: {self.price_sek}")


@dataclass(frozen=True, slots=True)
class AdRecord:
    """A fully-detailed listing, normalized and PII-free.

    `attributes` holds the marketplace's own structured key/value pairs (Blocket
    calls these "extras"). These are worth far more than they look: `condition`
    and component type arrive pre-labelled, so they never need to reach an LLM.
    """

    source: str
    external_id: str
    url: str
    title: str
    description: str
    price_sek: int
    category_path: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)
    city: str | None = None
    listed_at: datetime | None = None
    image_count: int = 0

    def __post_init__(self) -> None:
        if not self.external_id:
            raise ValueError("external_id is required")

    @property
    def condition(self) -> str | None:
        """Seller-declared condition, when the marketplace captures it structurally."""
        return self.attributes.get("condition")

    @property
    def searchable_text(self) -> str:
        """Title plus description — the text specs are extracted from."""
        return f"{self.title}\n\n{self.description}".strip()


class Source(Protocol):
    """A marketplace we can pull listings from."""

    name: str

    def search(
        self,
        query: str,
        *,
        page: int = 1,
        price_from: int | None = None,
        price_to: int | None = None,
    ) -> Sequence[AdSummary]:
        """Return one page of search results.

        Raises `RateLimited` or `CircuitOpen` rather than returning empty, so
        that "no results" and "we were blocked" are never confused — conflating
        them is how a scraper silently stops working.

        Price bounds are a *hint*. A source may apply them server-side, halve
        them, or ignore them entirely; the caller re-checks locally either way.
        Making them advisory keeps the protocol implementable by a marketplace
        whose API has no price filter, instead of forcing every source to
        pretend it has one.
        """
        ...

    def fetch_detail(self, summary: AdSummary) -> AdRecord | None:
        """Fetch the full record for one listing.

        Returns None when the listing has been removed. Raises on transport
        failure.
        """
        ...
