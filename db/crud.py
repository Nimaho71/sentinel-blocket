"""Database access.

One rule runs through this module: **functions here never commit.** They stage
work on the session they are handed and let the caller decide the transaction
boundary. Committing inside a helper is how the old code ended up with a partial
write it could not roll back — `upsert_ad` committed per ad, so a batch that
failed halfway left half of itself behind.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Ad, AiEvaluation, AiFailure, CategoryEnum, StatusEnum
from sources.base import AdRecord


def content_hash(title: str, description: str, price_sek: int) -> str:
    """Stable fingerprint of the parts of an ad the AI reads.

    Anything not in this tuple can change without triggering re-analysis. Price
    *is* in it: a seller dropping the price is exactly the event worth spending
    quota to re-evaluate.
    """
    payload = f"{title.strip()}\x00{description.strip()}\x00{price_sek}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Blocket's own leaf categories, as returned by the API:
#   ('Elektronik och vitvaror', 'Datorer', 'Stationär dator')
#   ('Elektronik och vitvaror', 'Datorer', 'Datorkomponenter')
# Matched on the leaf, exactly. An earlier version tested `"dator" in path`,
# which is true of *both* — "Datorkomponenter" contains "dator" — so every
# graphics card was filed as a complete computer.
_WHOLE_MACHINE_LEAVES = frozenset({"stationär dator", "bärbar dator", "allt-i-ett dator"})


def _category_for(record: AdRecord) -> CategoryEnum:
    """Coarse split from the source's own category path — no LLM needed.

    Deliberately crude, and deliberately biased towards COMPONENT: an unknown
    leaf is far more likely to be a part than a whole machine, and misfiling a
    part as a build would let a 900 kr graphics card be valued as a computer.
    The fine-grained call (single component vs. bundle vs. full build) is
    `ItemTypeEnum`, and that one does go to the AI.
    """
    leaf = record.category_path[-1].lower().strip() if record.category_path else ""
    return CategoryEnum.STATIONARY if leaf in _WHOLE_MACHINE_LEAVES else CategoryEnum.COMPONENT


# Swedish for "buying" / "wanted". Blocket files want-to-buy ads in the same
# categories as items for sale, and their price means the opposite of a sale
# price: it is what someone is *offering*, and it is deliberately low.
#
# Found on the published leaderboard, presented as the fourth-best bargain in
# the country: **"KÖPER RTX 3090 — 8 000 kr"**, against an observed median of
# 10 999 kr, reported as 2 999 kr below market. It is not a card for sale at
# 8 000 kr. It is a stranger hoping to pay that.
#
# These do double damage, and the second kind is worse. A wanted ad on the
# leaderboard is merely embarrassing; a wanted ad in `price_observations`
# silently drags the median of that part *down* for every future valuation, and
# nothing downstream can tell it apart from a real sale.
_WANTED_MARKERS = (
    "köper", "koper", "köpes", "kopes", "sökes", "sokes",
    "önskas köpa", "onskas kopa", "wanted", "letar efter",
)


def is_wanted_ad(title: str) -> bool:
    """Whether a listing is someone wanting to *buy*, not sell.

    Matched on the title only. Descriptions routinely say things like "jag
    köpte den 2021", which is a sentence about the past and not an offer, and
    scanning the body for these words flags a large share of honest sales.
    """
    lowered = f" {title.lower()} "
    return any(f" {marker} " in lowered or lowered.startswith(f" {marker}")
               for marker in _WANTED_MARKERS)


def _status_for(record: AdRecord) -> StatusEnum:
    """Where a want-to-buy ad gets filtered out, once, for the whole pipeline.

    Marked at ingest rather than at each place that would need to skip it. Every
    downstream query — valuation, the index, basket mode, the publisher —
    already filters on `status == ACTIVE`, so one assignment here excludes these
    from all of them. The alternative was a check in four modules, which is the
    shape of bug this project keeps finding: a guard that exists and that one
    call site forgot to consult.
    """
    return StatusEnum.IGNORED if is_wanted_ad(record.title) else StatusEnum.ACTIVE


async def known_external_ids(session: AsyncSession, source: str) -> set[str]:
    """Every id already stored for a source, fetched in one query.

    The ingest loop needs a membership test per search hit. Doing that as a
    query per ad would be hundreds of round trips against a local SQLite file
    to answer a question one `SELECT` can answer.
    """
    stmt = select(Ad.external_id).where(Ad.source == source)
    return set((await session.execute(stmt)).scalars().all())


async def upsert_ad(session: AsyncSession, record: AdRecord) -> Ad:
    """Insert or refresh one ad. Staged, not committed."""
    stmt = select(Ad).where(
        tuple_(Ad.source, Ad.external_id) == (record.source, record.external_id)
    )
    existing = (await session.execute(stmt)).scalars().first()

    now = datetime.now(timezone.utc)
    digest = content_hash(record.title, record.description, record.price_sek)

    if existing is not None:
        existing.title = record.title
        existing.price_sek = record.price_sek
        existing.city = record.city
        existing.condition = record.condition
        existing.image_count = record.image_count
        existing.content_hash = digest
        existing.last_checked_at = now
        # Seeing it in search results is proof it is live; a prior janitor pass
        # may have marked it SOLD on a transient network error. Re-derived from
        # the current title rather than simply set to ACTIVE, so a want-to-buy
        # ad stays IGNORED — it is live, and still not a thing for sale.
        existing.status = _status_for(record)
        # Only refill the description if we still hold one. Once it has been
        # dropped for retention, re-scraping must not quietly restore it.
        if existing.description_dropped_at is None:
            existing.raw_description = record.description
        return existing

    ad = Ad(
        source=record.source,
        external_id=record.external_id,
        url=record.url,
        title=record.title,
        raw_description=record.description,
        price_sek=record.price_sek,
        city=record.city,
        condition=record.condition,
        image_count=record.image_count,
        content_hash=digest,
        category=_category_for(record),
        status=_status_for(record),
        listed_at=record.listed_at,
        scraped_at=now,
        last_checked_at=now,
    )
    session.add(ad)
    return ad


async def ads_needing_evaluation(
    session: AsyncSession, *, limit: int | None = 200, max_attempts: int = 3
) -> list[Ad]:
    """Active ads with no current verdict, skipping ones that keep failing.

    "No current verdict" means no evaluation *at the ad's present content hash*
    — a re-listed ad at a new price comes back into scope. The old query asked
    only whether any evaluation row existed, so a price cut was never noticed.

    `max_attempts` is what stops a permanently malformed ad from consuming a
    slot on every run forever.

    `limit=None` returns the whole backlog. The caller wants that when the
    number it can actually afford is not known until *after* the free
    deterministic pass has run — see `ai.batch_processor.process_unscored_ads`.
    """
    has_current_verdict = (
        select(AiEvaluation.id)
        .where(AiEvaluation.ad_id == Ad.id)
        .where(AiEvaluation.content_hash == Ad.content_hash)
        .exists()
    )
    exhausted = select(AiFailure.ad_id).where(AiFailure.attempts >= max_attempts)

    stmt = (
        select(Ad)
        .where(Ad.status == StatusEnum.ACTIVE)
        .where(~has_current_verdict)
        .where(Ad.id.not_in(exhausted))
        .order_by(Ad.scraped_at.desc())
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return list((await session.execute(stmt)).scalars().all())


async def record_ai_failure(session: AsyncSession, ad_id: int, error: str) -> None:
    """Mark an ad as having failed evaluation, incrementing its attempt count."""
    stmt = select(AiFailure).where(AiFailure.ad_id == ad_id)
    failure = (await session.execute(stmt)).scalars().first()

    if failure is None:
        session.add(AiFailure(ad_id=ad_id, attempts=1, last_error=error[:2000]))
    else:
        failure.attempts += 1
        failure.last_error = error[:2000]
        failure.last_attempt_at = datetime.now(timezone.utc)


async def clear_ai_failure(session: AsyncSession, ad_id: int) -> None:
    """Drop the dead-letter row after a successful evaluation."""
    await session.execute(delete(AiFailure).where(AiFailure.ad_id == ad_id))


async def drop_stale_descriptions(session: AsyncSession, *, older_than_days: int = 7) -> int:
    """Delete raw description text once extraction has had its use of it.

    GDPR data minimisation, and the cheapest kind: the pipeline needs specs, and
    specs have already been extracted into `ExtractedComponent` by the time this
    runs. The text is the only field that can hold a phone number a seller typed
    into the body, and after this it is gone.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    stmt = (
        select(Ad)
        .where(Ad.raw_description.is_not(None))
        .where(Ad.scraped_at < cutoff)
        .where(Ad.ai_evaluation.has())
    )
    stale = list((await session.execute(stmt)).scalars().all())

    now = datetime.now(timezone.utc)
    for ad in stale:
        ad.raw_description = None
        ad.description_dropped_at = now
    return len(stale)
