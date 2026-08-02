"""AI evaluation stage: batch ads to Gemini, validate, persist.

### The bug this module was rewritten around

The database held 474 ads and 120 evaluations. Nothing reported the other 354 as
missing, because three things went wrong together:

1. `analyze_ads_batch` returned `[]` on *any* error, indistinguishable from a
   legitimately empty result.
2. The caller responded with a bare `continue` — the batch was dropped, with no
   record that it had ever been attempted.
3. The next run re-selected the same ads (the query only asked whether an
   evaluation row existed) and failed the same way, forever.

A permanent failure that reports itself as ordinary progress is the hardest kind
to notice, and none of the three parts looked wrong on its own.

The fix is structural, not a patch: failures raise, every failure is recorded
against its ad, attempts are counted, and the run reports how many ads it could
not evaluate. A stall is now loud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import selectinload

from ai.gemini_client import analyze_ads_batch, build_payload
from ai.schemas import AdVerdict
from pipeline.canonical import canonicalize, is_laptop_listing
from db.crud import ads_needing_evaluation, clear_ai_failure, record_ai_failure
from db.models import Ad, AiEvaluation, ExtractedComponent
from pipeline.extract import extract
from pipeline.index import value_ad

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 10

# Fraction of the day's remaining quota one run will not touch. Evaluation is
# not the only claimant on it: a failed batch is retried, and escalation draws
# on the same key. A run that spends down to the last request leaves the next
# failure with nowhere to go.
QUOTA_RESERVE = 0.2


def affordable_today(batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """How many ads today's remaining model quota can still cover.

    Replaces a hardcoded 200, which was not derived from anything. On a
    measured run that cap evaluated 200 ads using **18 of 1 000** available
    requests — 1.8% of the free-tier quota — while 2 235 already-scraped ads
    sat unevaluated. Since the price index is built from `canonical_name`, and
    canonical names come from evaluation, those ads were invisible to every
    valuation on the site despite being fully paid for at ingest.

    A cap has to come from the quota or it is a guess, and a guess here throttles
    the whole pipeline silently — the run reports success either way.
    """
    from ai.gemini_client import get_ledger

    requests = int(get_ledger().remaining_requests_today() * (1.0 - QUOTA_RESERVE))
    return max(0, requests) * batch_size

# Recorded as the "model" for locally-resolved ads, so provenance stays
# queryable: "which verdicts did no model ever see?" is one WHERE clause.
DETERMINISTIC_MODEL = "deterministic"

# Below 1.0 deliberately. Pattern matching found no risk language, but nothing
# *read* the listing for the subtler problems a model would catch, so these must
# not outrank an ad a model looked at and approved. Ranking multiplies the price
# delta by this, so the discount is applied, not merely recorded.
DETERMINISTIC_CONFIDENCE = 0.75

_EAGER = (selectinload(Ad.extracted_components), selectinload(Ad.ai_evaluation))


@dataclass
class EvaluationStats:
    considered: int = 0
    evaluated: int = 0
    resolved_locally: int = 0
    failed: int = 0
    # Ads that needed a model and did not get one because the day's quota ran
    # out. Distinct from `failed`: nothing went wrong, the work simply has not
    # happened yet. Kept apart so a backlog cannot be read as a fault, nor a
    # fault dismissed as a backlog.
    deferred: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        text = (
            f"{self.considered} ads considered -> "
            f"{self.evaluated} evaluated "
            f"({self.resolved_locally} without a model call), "
            f"{self.failed} failed"
        )
        if self.deferred:
            text += f", {self.deferred} deferred to the next run (quota)"
        if self.failed:
            # Surfaced in the run summary rather than buried in a log line. The
            # old pipeline's silence about this gap is the whole reason it went
            # unnoticed for 354 ads.
            text += f" [see ai_failures; first error: {self.errors[0]}]"
        return text


def chunk_list(data: Sequence, chunk_size: int) -> Iterable[Sequence]:
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]


def _apply_verdict(session, ad: Ad, verdict: AdVerdict, model: str) -> None:
    """Write one verdict onto its ad. Staged; the caller commits."""
    ad.item_type = verdict.item_type

    # Re-evaluation replaces rather than accumulates. Without this, an ad
    # re-analysed after a price change would collect a second set of component
    # rows and be double-counted by the market index.
    ad.extracted_components.clear()
    # Decided once for the whole ad, from the title and the complete parts
    # list. A single component cannot answer it: the GPU of a gaming laptop is
    # written "RTX 5090", exactly as the desktop card is, and what gives it away
    # is the mobile CPU sitting next to it in the same listing.
    laptop = is_laptop_listing(
        ad.title, *(c.component_name for c in verdict.components)
    )
    for component in verdict.components:
        ad.extracted_components.append(
            ExtractedComponent(
                component_type=component.component_type,
                component_name=component.component_name,
                # Written at insert time, not computed at query time, so the
                # market index can match on an indexed column.
                canonical_name=canonicalize(component.component_name, laptop=laptop),
            )
        )

    evaluation = ad.ai_evaluation or AiEvaluation(ad_id=ad.id)
    evaluation.item_type = verdict.item_type
    evaluation.confidence = verdict.confidence
    evaluation.risk_flags = "\n".join(verdict.risk_flags)
    evaluation.ai_reasoning = verdict.reasoning
    # Stamped with the hash the verdict was formed from, so the ad only becomes
    # eligible again if its content actually changes.
    evaluation.content_hash = ad.content_hash
    evaluation.model = model
    ad.ai_evaluation = evaluation


async def _resolve_deterministically(
    session_factory: async_sessionmaker,
    pending_ids: list[int],
    stats: EvaluationStats,
) -> list[int]:
    """Settle the unambiguous ads locally; return the ids still needing a model.

    Only ads `pipeline.extract` is confident about are settled here: parts
    recognised, item type unambiguous, and no risk language anywhere in the
    text. Everything else — which on the real corpus is about two thirds — goes
    on to the model untouched.

    These verdicts are marked with `model="deterministic"` and a confidence
    below 1.0. Both are honest bookkeeping rather than decoration: nothing
    *read* these listings, so they should not outrank an ad a model looked at
    and approved, and the provenance needs to stay queryable if the rules here
    later turn out to be too lax.
    """
    settled: list[int] = []
    remaining: list[int] = []

    async with session_factory() as session:
        for ad_id in pending_ids:
            ad = await session.get(Ad, ad_id, options=_EAGER)
            if ad is None:
                continue

            extraction = extract(ad.title, ad.raw_description)
            if not extraction.is_confident:
                remaining.append(ad_id)
                continue

            # A clean parse is not enough when the *price* is the implausible
            # part. `pipeline.extract` reads text and only text, so it cannot
            # see that an i9 / RTX 4070 Ti / 64 GB DDR5 machine asking 395 kr is
            # not a machine asking 395 kr. There is no risk *word* in that
            # listing — which is exactly how it reached the fast path and got
            # published as the cheapest qualifying workstation on the market.
            #
            # Escalated to the model rather than discarded: the arithmetic
            # establishes that the number is wrong, and reading the text is what
            # might say why. The usual division — math owns value, the model
            # owns risk.
            valuation = await value_ad(session, ad_id)
            if valuation is not None and valuation.is_implausibly_cheap:
                logger.info(
                    "ad %s: %d kr against ~%s kr of parts — escalating to the model.",
                    ad_id,
                    ad.price_sek,
                    valuation.estimated_parts_sek,
                )
                remaining.append(ad_id)
                continue

            _apply_verdict(
                session,
                ad,
                AdVerdict(
                    ad_id=ad.id,
                    item_type=extraction.item_type,
                    confidence=DETERMINISTIC_CONFIDENCE,
                    risk_flags=(),
                    reasoning=(
                        "Specs read directly from the listing text by pattern "
                        "matching; no risk language found. Not reviewed by a model."
                    ),
                    components=extraction.components,
                ),
                DETERMINISTIC_MODEL,
            )
            settled.append(ad_id)

        await session.commit()

    stats.evaluated += len(settled)
    stats.resolved_locally = len(settled)
    logger.info(
        "Resolved %d/%d ads deterministically; %d need a model.",
        len(settled), len(pending_ids), len(remaining),
    )
    return remaining


async def process_unscored_ads(
    session_factory: async_sessionmaker,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    model: str | None = None,
    use_deterministic: bool = True,
) -> EvaluationStats:
    """Evaluate ads that have no verdict at their current content hash.

    Takes a session *factory*, not a session: each batch runs in its own
    transaction, so one bad batch cannot roll back the work of the batches
    before it.

    `limit=None` — the default — means "the whole backlog, as far as today's
    quota reaches". The two halves of this stage cost very different things and
    so are bounded separately: the deterministic pass is local regex work that
    spends no quota at all and is never rationed, while only the ads that still
    need a model are trimmed to what `affordable_today` allows. Capping the
    *fetch* instead, as the old fixed limit did, rationed the free half too.
    """
    stats = EvaluationStats()

    async with session_factory() as session:
        pending = await ads_needing_evaluation(session, limit=limit)
        # Detach the ids and let the session go. Holding a session open across
        # the whole run — minutes of it, given the throttle — would pin a SQLite
        # connection for the duration for no benefit.
        pending_ids = [ad.id for ad in pending]

    stats.considered = len(pending_ids)
    if not pending_ids:
        logger.info("No ads awaiting evaluation.")
        return stats

    if use_deterministic:
        pending_ids = await _resolve_deterministically(session_factory, pending_ids, stats)
        if not pending_ids:
            logger.info(stats.summary())
            return stats

    affordable = affordable_today(batch_size) if limit is None else limit
    if len(pending_ids) > affordable:
        # Said out loud, with the number. A stage that quietly evaluates a
        # fraction of its backlog and logs "done" is the exact shape of the bug
        # this module was rewritten around.
        stats.deferred = len(pending_ids) - affordable
        logger.warning(
            "Quota covers %d of %d ads needing a model; deferring %d to the "
            "next run.",
            affordable, len(pending_ids), stats.deferred,
        )
        pending_ids = pending_ids[:affordable]

    if not pending_ids:
        logger.warning("No quota left today — %d ads still await a model.", stats.deferred)
        logger.info(stats.summary())
        return stats

    logger.info(
        "Evaluating %d ads in batches of %d...", len(pending_ids), batch_size
    )

    for batch_ids in chunk_list(pending_ids, batch_size):
        async with session_factory() as session:
            # Relationships are eager-loaded: `_apply_verdict` touches both, and
            # a lazy load on an AsyncSession raises MissingGreenlet rather than
            # quietly working.
            ads = []
            for ad_id in batch_ids:
                ad = await session.get(Ad, ad_id, options=_EAGER)
                if ad is not None:
                    ads.append(ad)
            if not ads:
                continue

            try:
                result = await analyze_ads_batch(build_payload(ads), model=model)
            except Exception as exc:  # noqa: BLE001 — every failure mode is recorded, not raised
                # The batch failed as a unit, so every ad in it is charged an
                # attempt. Three strikes and the ad stops being retried — which
                # is what keeps a permanently unparseable listing from consuming
                # a slot on every future run.
                message = f"{type(exc).__name__}: {exc}"
                for ad in ads:
                    await record_ai_failure(session, ad.id, message)
                await session.commit()
                stats.failed += len(ads)
                stats.errors.append(message)
                logger.error("Batch of %d ads failed evaluation: %s", len(ads), message)
                continue

            verdicts = result.verdicts
            by_id = {ad.id: ad for ad in ads}
            applied = 0
            for verdict in verdicts:
                ad = by_id.get(verdict.ad_id)
                if ad is None:
                    # parse_ad_verdicts already rejects ids outside the batch,
                    # so this is unreachable defensively rather than expected.
                    continue
                _apply_verdict(session, ad, verdict, result.model)
                await clear_ai_failure(session, ad.id)
                applied += 1

            # An ad the model simply omitted is a failure too. Silently leaving
            # it unevaluated is precisely the original bug in miniature.
            missing = set(by_id) - {v.ad_id for v in verdicts}
            for ad_id in missing:
                await record_ai_failure(session, ad_id, "model omitted this ad from its reply")

            try:
                await session.commit()
            except Exception as exc:  # noqa: BLE001
                await session.rollback()
                stats.failed += len(ads)
                stats.errors.append(f"commit failed: {exc}")
                logger.error("Failed to save evaluations for batch: %s", exc)
                continue

            stats.evaluated += applied
            stats.failed += len(missing)
            if missing:
                stats.errors.append(f"{len(missing)} ads omitted from a reply")

    logger.info(stats.summary())
    return stats
