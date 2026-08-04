"""Janitor: verify ads are still live, and enforce retention.

### Why the throttle matters here more than anywhere else

The previous version built one task per active ad and handed the whole list to
`asyncio.gather`. At 474 active ads that is 474 simultaneous requests to
blocket.se from one residential IP — a burst indistinguishable from a small
denial-of-service, and far ruder than the crawler it was cleaning up after. The
scrape stage was carefully rate-limited and then the *cleanup* stage undid it.

The politeness budget is the same one `sources/throttle.py` applies to ingest:
one request at a time, spaced, with the run abandoned rather than pushed through
if the far end starts refusing.

### Why deleting ads does not lose data

Retention is aggressive — sold ads are purged, and description text is dropped a
week after extraction. That is only affordable because price observations are
recorded against a canonical part, never against an ad (Phase 4). The listing
disappears; the data point it contributed does not. Privacy and accuracy point
the same way here, which is unusual and worth having deliberately.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Collection, Awaitable, Callable

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from db.crud import drop_stale_descriptions
from db.models import Ad, StatusEnum

logger = logging.getLogger(__name__)

# One honest User-Agent, identifying the tool. Rotating identities to look like
# many different browsers would be the opposite of what this project claims to
# be. See ETHICS.md.
USER_AGENT = "sentinel-blocket/1.0 (personal price-tracking tool; contact via GitHub)"

# Matches the ingest budget. Liveness checking is not urgent — it can take as
# long as it takes.
MIN_INTERVAL_S = 2.0
JITTER_S = 0.5
REQUEST_TIMEOUT_S = 10.0

# Consecutive failures before the run gives up. A far end that has stopped
# answering is not a problem to solve by continuing to ask.
FAILURE_THRESHOLD = 5

SOLD_RETENTION_DAYS = 90
DESCRIPTION_RETENTION_DAYS = 7


@dataclass
class JanitorStats:
    checked: int = 0
    marked_sold: int = 0
    purged: int = 0
    descriptions_dropped: int = 0
    # Ads reached, but whose status could not be read. Counted separately from
    # `checked`: a page we no longer recognise is not evidence that an ad is
    # live, and treating it as such is how a board fossilises while every run
    # reports success.
    inconclusive: int = 0
    halted_early: bool = False

    def summary(self) -> str:
        text = (
            f"{self.checked} ads checked -> {self.marked_sold} sold, "
            f"{self.purged} purged, {self.descriptions_dropped} descriptions dropped"
        )
        if self.inconclusive:
            text += f", {self.inconclusive} status unreadable"
        if self.halted_early:
            text += " [HALTED EARLY]"
        return text


# Blocket embeds its page state as escaped JSON inside a <script> tag, so the
# flag arrives as \"disposed\":true. Both forms are accepted in case the
# escaping ever changes.
_DISPOSED = re.compile(r'\\?"disposed\\?"\s*:\s*(true|false)')


async def _is_gone(client: httpx.AsyncClient, url: str) -> bool | None:
    """Whether a listing is no longer for sale.

    A **GET**, not a HEAD, and that is the whole point. A sold listing stays at
    its URL and answers 200 — only a *deleted* one 404s, and sellers rarely
    delete. Checked against live data: an ad plainly marked "Såld" returned 200
    to a HEAD and was therefore treated as still for sale, so it would have sat
    on the published board indefinitely. Its status is in the page body, and
    the same `disposed` flag `sources/blocket.py` already reads at ingest says
    so exactly.

    Returns `None` for "could not tell" — a timeout, a server error, or a page
    whose shape we no longer recognise. Treating any of those as SOLD would
    silently delete live listings whenever the connection wobbled.
    """
    response = await client.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_S,
        follow_redirects=True,
    )
    if response.status_code == 404:
        return True
    # Blocket redirects removed listings to the site root.
    if response.url.host.endswith("blocket.se") and response.url.path == "/":
        return True
    if response.status_code >= 500:
        return None

    match = _DISPOSED.search(response.text)
    if match is None:
        # Loud on purpose. If Blocket restyles this page the marker disappears,
        # every ad becomes inconclusive, nothing is ever retired, and the board
        # quietly fossilises while every run still reports success. That is the
        # exact failure this project keeps meeting, so it must not be silent.
        return None
    return match.group(1) == "true"


async def check_liveness(
    session_factory: async_sessionmaker,
    *,
    limit: int = 500,
    published_ids: Collection[int] = (),
    client: httpx.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> JanitorStats:
    """Re-check active ads one at a time, marking removed ones SOLD.

    `published_ids` are verified first, whatever their last-checked time. **They
    are the only ads a visitor can see**, so staleness there is the only
    staleness anyone actually experiences.

    Oldest-first alone is the fair order and the wrong one. Found live: 4 083
    active ads against a budget of 500 a night is a full pass every eight
    nights, and the corpus grows faster than the janitor retires it, so the
    backlog never closes. Two listings on the published board had already gone
    — one 404, one marked *Såld* — and would have stayed up for another week
    while the janitor diligently re-checked thousands of ads nobody could see.

    The published set is roughly a hundred ads, so covering all of it costs a
    few minutes and leaves most of the budget for the rotation behind it.
    Nothing is skipped; the order just follows where being wrong is visible.

    `client` and `sleep` are injectable for the same reason they are in
    `sources/throttle.py`: the behaviour worth testing is the pacing and the
    halt, and a test that actually waited two seconds per ad would never be run.
    """
    stats = JanitorStats()

    async with session_factory() as session:
        active = (
            select(Ad.id, Ad.url)
            .where(Ad.status == StatusEnum.ACTIVE)
            .order_by(Ad.last_checked_at.asc())  # Oldest checks first.
        )
        targets: list = []
        if published_ids:
            targets = list(
                (
                    await session.execute(
                        active.where(Ad.id.in_(published_ids)).limit(limit)
                    )
                ).all()
            )
            logger.info(
                "Verifying %d published ads before the rotation.", len(targets)
            )

        # Backfill the remaining budget with the ordinary rotation, skipping
        # anything already queued above.
        remaining = limit - len(targets)
        if remaining > 0:
            seen = {row.id for row in targets}
            rest = (await session.execute(active.limit(limit + len(seen)))).all()
            targets += [row for row in rest if row.id not in seen][:remaining]

    if not targets:
        logger.info("No active ads to verify.")
        return stats

    logger.info("Verifying %d active ads at ~%.1fs each...", len(targets), MIN_INTERVAL_S)

    sold_ids: list[int] = []
    checked_ids: list[int] = []
    consecutive_failures = 0

    owned_client = client is None
    client = client or httpx.AsyncClient()
    try:
        for ad_id, url in targets:
            # Sequential, spaced, jittered. Not a burst.
            await sleep(MIN_INTERVAL_S + random.uniform(0, JITTER_S))
            try:
                gone = await _is_gone(client, url)
                consecutive_failures = 0
            except httpx.RequestError as exc:
                consecutive_failures += 1
                logger.warning("Could not check %s: %s", url, exc)
                if consecutive_failures >= FAILURE_THRESHOLD:
                    stats.halted_early = True
                    logger.error(
                        "Halting janitor after %d consecutive failures.", consecutive_failures
                    )
                    break
                continue

            stats.checked += 1
            checked_ids.append(ad_id)
            if gone is None:
                stats.inconclusive += 1
            elif gone:
                sold_ids.append(ad_id)
    finally:
        if owned_client:
            await client.aclose()

    if stats.inconclusive and stats.inconclusive > stats.checked // 2:
        logger.error(
            "Could not read a status for %d of %d ads. If Blocket changed the "
            "page, nothing will ever be retired again — check `_DISPOSED`.",
            stats.inconclusive, stats.checked,
        )

    now = datetime.now(timezone.utc)
    async with session_factory() as session:
        # Only ads we actually reached. Stamping the rest would push them to the
        # back of the oldest-first queue without anyone having looked at them —
        # so a run that halts early would starve exactly the ads it skipped.
        for ad_id in checked_ids:
            ad = await session.get(Ad, ad_id)
            if ad is None:
                continue
            ad.last_checked_at = now
            if ad_id in sold_ids:
                ad.status = StatusEnum.SOLD
        await session.commit()

    stats.marked_sold = len(sold_ids)
    logger.info(stats.summary())
    return stats


async def enforce_retention(
    session_factory: async_sessionmaker,
    *,
    sold_retention_days: int = SOLD_RETENTION_DAYS,
    description_retention_days: int = DESCRIPTION_RETENTION_DAYS,
) -> JanitorStats:
    """Purge long-sold ads and drop description text past its useful life."""
    stats = JanitorStats()
    cutoff = datetime.now(timezone.utc) - timedelta(days=sold_retention_days)

    async with session_factory() as session:
        result = await session.execute(
            delete(Ad)
            .where(Ad.status == StatusEnum.SOLD)
            .where(Ad.last_checked_at < cutoff)
        )
        stats.purged = result.rowcount or 0
        stats.descriptions_dropped = await drop_stale_descriptions(
            session, older_than_days=description_retention_days
        )
        await session.commit()

    logger.info(
        "Retention: purged %d sold ads, dropped %d descriptions.",
        stats.purged,
        stats.descriptions_dropped,
    )
    return stats


async def run_janitor(session_factory: async_sessionmaker, *, limit: int = 500) -> JanitorStats:
    """Full maintenance pass: liveness first, then retention."""
    stats = await check_liveness(session_factory, limit=limit)
    retention = await enforce_retention(session_factory)
    stats.purged = retention.purged
    stats.descriptions_dropped = retention.descriptions_dropped
    return stats
