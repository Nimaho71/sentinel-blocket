"""Blocket source adapter, built on the `blocket-api` package.

Replaces the original Playwright + stealth crawler (now in `legacy/`). The JSON
endpoints are lighter on Blocket's infrastructure than rendering full pages,
faster by roughly two orders of magnitude, and have no CAPTCHA surface.

Two-call shape, forced by the API and worth understanding:

    search()      -> ~50 summaries per request, NO description field
    fetch_detail() -> one request per ad, returns description + extras

Detail is therefore the expensive half. The pipeline pre-filters on summaries so
we never pay for the detail of an ad that is out of budget anyway.

Payload shapes verified live against the API (2026-07). Both are defensively
parsed: this is an unofficial API and it will drift.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from blocket_api import BlocketAPI, RecommerceAd

from sources.base import AdRecord, AdSummary, SourceError
from sources.throttle import CircuitBreaker, RateLimiter

logger = logging.getLogger(__name__)

SOURCE_NAME = "blocket"

# Blocket's own subcategory id for "Datorer" (computers), which contains the
# "Datorkomponenter" leaf we care about. Confirmed against live category data.
SUBCATEGORY_COMPUTERS = "1.93.3215"


class BlocketSource:
    """Reads PC listings from Blocket, dropping personal data at ingest.

    The detail payload contains GPS coordinates and a seller id. Neither is
    extracted here, so no downstream bug can leak them — `AdRecord` has nowhere
    to put them. Location is kept only at the coarse city level that search
    results already provide.
    """

    name = SOURCE_NAME

    def __init__(
        self,
        api: Any | None = None,
        rate_limiter: RateLimiter | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._api = api if api is not None else BlocketAPI()
        # 1.5–2.0 s between requests, ~1.75 s on average: far below anything
        # that would burden Blocket, while still allowing a few thousand ads
        # per session.
        #
        # This figure is published in ETHICS.md and the two must be changed
        # together. A stated politeness measure that no longer matches the code
        # is worse than never having claimed one, because it is read as a
        # commitment by anyone checking whether this project is well behaved.
        self._limiter = rate_limiter or RateLimiter(min_interval=1.5, jitter=0.5)
        self._breaker = breaker or CircuitBreaker(threshold=3, cooldown=900.0)

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: str,
        *,
        page: int = 1,
        price_from: int | None = None,
        price_to: int | None = None,
    ) -> Sequence[AdSummary]:
        """One page of search results (~50 ads), without descriptions.

        Price bounds are applied **server-side**, which is worth more than it
        first appears. A filtered page comes back *full* of in-budget ads rather
        than partly full — measured: `price_from=3000` still returned 53 results,
        but the cheapest was 3 100 kr instead of 900 kr. The same single request
        yields more usable ads, so fewer requests are needed for the same
        coverage. Faster for us and lighter on Blocket, in the same change.

        The client-side check in `pipeline.ingest.is_candidate` stays too, and is
        not redundant. These are undocumented parameters on an unofficial
        endpoint, and the failure mode found while testing them is the dangerous
        one: `price_max` — an entirely plausible guess — is **silently ignored**
        and returns everything, unfiltered, with a 200. An unsupported filter
        here fails *open*, so the local bound is what makes a silent API change
        a non-event.
        """
        payload = self._request(
            lambda: self._search_raw(query, page, price_from, price_to),
            what=f"search({query!r}, page={page})",
        )
        docs = payload.get("docs") or []
        summaries: list[AdSummary] = []
        for doc in docs:
            summary = self._parse_summary(doc)
            if summary is not None:
                summaries.append(summary)

        skipped = len(docs) - len(summaries)
        if skipped:
            # Never silent: an unofficial API that changes shape shows up here
            # first, and a quiet drop would look identical to "no results".
            logger.warning(
                "search(%r) page %d: skipped %d/%d unparseable results",
                query, page, skipped, len(docs),
            )
        logger.info("search(%r) page %d -> %d ads", query, page, len(summaries))
        return summaries

    def _search_raw(
        self, query: str, page: int, price_from: int | None, price_to: int | None
    ) -> dict[str, Any]:
        """Call the wrapper, adding price bounds it does not expose itself.

        `blocket_api.search()` has no price parameters — only `search_car`,
        `search_boat` and `search_mc` do. The underlying recommerce endpoint
        accepts `price_from` / `price_to` regardless; verified live. When no
        bounds are given this is exactly the wrapper's own call, so the
        supported path stays the default.
        """
        if price_from is None and price_to is None:
            return self._api.search(query, page=page)

        from blocket_api.blocket import SEARCH_URL
        from blocket_api.blocket import _request as _api_request

        params: list[tuple[str, Any]] = [("q", query), ("page", page)]
        if price_from is not None:
            params.append(("price_from", price_from))
        if price_to is not None:
            params.append(("price_to", price_to))
        return _api_request(url=SEARCH_URL, params=params).json()

    def _parse_summary(self, doc: Mapping[str, Any]) -> AdSummary | None:
        try:
            ad_id = str(doc.get("ad_id") or doc.get("id") or "").strip()
            if not ad_id:
                return None

            # price is {"amount": int, "currency_code": "SEK", ...}
            price_obj = doc.get("price") or {}
            amount = price_obj.get("amount") if isinstance(price_obj, Mapping) else None
            if amount is None:
                return None

            return AdSummary(
                source=SOURCE_NAME,
                external_id=ad_id,
                url=doc.get("canonical_url")
                or f"https://www.blocket.se/recommerce/forsale/item/{ad_id}",
                title=(doc.get("heading") or "").strip(),
                price_sek=int(amount),
                # Coarse place name only ("Älvsjö"). The detail payload's GPS
                # coordinates are deliberately never read.
                city=(doc.get("location") or None),
                listed_at=_epoch_ms_to_dt(doc.get("timestamp")),
            )
        except (TypeError, ValueError) as exc:
            logger.debug("unparseable search result: %s", exc)
            return None

    # ------------------------------------------------------------------ detail

    def fetch_detail(self, summary: AdSummary) -> AdRecord | None:
        """Fetch description and structured attributes for one ad.

        Returns None if the listing is gone or the payload is unusable.
        """
        payload = self._request(
            lambda: self._api.get_ad(RecommerceAd(int(summary.external_id))),
            what=f"get_ad({summary.external_id})",
        )

        item = _dig(payload, "loaderData", "item-recommerce", "itemData")
        if not isinstance(item, Mapping):
            logger.warning("ad %s: no itemData in payload", summary.external_id)
            return None
        if item.get("disposed"):
            logger.info("ad %s: listing disposed/removed", summary.external_id)
            return None

        price = item.get("price")
        images = item.get("images")

        return AdRecord(
            source=SOURCE_NAME,
            external_id=summary.external_id,
            url=summary.url,
            title=(item.get("title") or summary.title or "").strip(),
            description=(item.get("description") or "").strip(),
            price_sek=int(price) if isinstance(price, (int, float)) else summary.price_sek,
            category_path=_category_path(item.get("category")),
            attributes=_extras_to_dict(item.get("extras")),
            # From the summary, not the detail payload's GPS position.
            city=summary.city,
            listed_at=summary.listed_at,
            image_count=len(images) if isinstance(images, list) else 0,
        )

    # ----------------------------------------------------------------- plumbing

    def _request(self, call, *, what: str):
        """Run one API call under the rate limiter and circuit breaker."""
        self._breaker.check()
        self._limiter.acquire()
        try:
            result = call()
        except Exception as exc:
            retry_after = _retry_after_from(exc)
            self._breaker.record_failure(retry_after=retry_after)
            logger.warning(
                "%s failed (%d consecutive): %s",
                what, self._breaker.consecutive_failures, exc,
            )
            raise SourceError(f"{what} failed: {exc}") from exc

        self._breaker.record_success()
        if not isinstance(result, Mapping):
            raise SourceError(f"{what}: expected a mapping, got {type(result).__name__}")
        return result


# ---------------------------------------------------------------- parse helpers


def _dig(payload: Any, *keys: str) -> Any:
    """Walk nested mappings, returning None at the first missing key."""
    current = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _extras_to_dict(extras: Any) -> dict[str, str]:
    """Flatten Blocket's `extras` list into {id: value}.

    Shape: [{"id": "condition", "label": "Skick", "value": "Mycket bra skick"}, ...]

    Worth more than it appears — `condition` and `computer_components_type`
    arrive already labelled, so they are extracted for free rather than costing
    an LLM call.
    """
    if not isinstance(extras, list):
        return {}
    out: dict[str, str] = {}
    for entry in extras:
        if not isinstance(entry, Mapping):
            continue
        key = entry.get("id")
        value = entry.get("value")
        if isinstance(key, str) and isinstance(value, (str, int, float)):
            out[key] = str(value)
    return out


def _category_path(category: Any) -> tuple[str, ...]:
    """Flatten Blocket's nested category into a root-first tuple.

    Shape: {"value": "Datorkomponenter", "parent": {"value": "Datorer", ...}}
    yields ("Datorer", "Datorkomponenter").
    """
    path: list[str] = []
    current = category
    depth = 0
    while isinstance(current, Mapping) and depth < 10:  # guard against cycles
        value = current.get("value")
        if isinstance(value, str) and value:
            path.append(value)
        current = current.get("parent")
        depth += 1
    return tuple(reversed(path))


def _epoch_ms_to_dt(timestamp: Any) -> datetime | None:
    if not isinstance(timestamp, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _retry_after_from(exc: Exception) -> float | None:
    """Pull a Retry-After hint off an httpx error, if the server sent one.

    The remote telling us when to come back always beats us guessing.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
