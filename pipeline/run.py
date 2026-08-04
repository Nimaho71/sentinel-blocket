"""Pipeline entrypoint. Replaces `legacy/orchestrator.py`.

    python -m pipeline.run              # full nightly pass
    python -m pipeline.run --stage ingest --limit 40
    python -m pipeline.run --dry-run    # scrape and report, write nothing
    python -m pipeline.run --stage basket --budget 8000

Structurally different from the orchestrator in two ways, both consequences of
Phase 2:

**Stages are sequential and separately runnable.** The old version ran five
scrape tasks under `asyncio.gather`, sharing one `AsyncSession` between them.
Concurrency bought little — the rate limiter is the bottleneck, not the CPU —
and cost correctness, since `AsyncSession` is not safe to share. Being able to
run one stage in isolation also makes a failure diagnosable.

**Every stage reports counts, and a failed stage stops the run.** The
orchestrator logged "done" whether or not anything had happened. A pipeline that
cannot distinguish success from silence is one that stalls for months without
anyone noticing — which is exactly what it did.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ai.batch_processor import process_unscored_ads
from db.connection import AsyncSessionLocal, init_db
from pipeline.basket import build_or_buy
from pipeline.index import rebuild_index
from pipeline.ingest import ingest, pc_search_targets
from pipeline.janitor import check_liveness, enforce_retention
from pipeline.persist import persist_records
from pipeline.profiles import PROFILES
from pipeline.publish import DEFAULT_OUTPUT, RedactionError, publish
from sources.blocket import BlocketSource

logger = logging.getLogger(__name__)

STAGES = ("ingest", "evaluate", "index", "basket", "publish", "janitor", "all")

# The budget this is being run to spend. Applied at *presentation* only — see
# `stage_basket` and `api/routes.py` for why it must never reach ingest.
DEFAULT_BUDGET_SEK = 10_000

# How many records to hold before writing them. Matches `persist_records`'s own
# batch size, so a flush is exactly one transaction. Small enough that a crash
# costs at most this many detail fetches; large enough to avoid an fsync per ad.
PERSIST_EVERY = 50


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Settings shared by every stage.

    A dataclass rather than a widening list of keyword arguments, because the
    two settings here mean genuinely different things and were briefly sharing
    one flag. `--limit` caps *work* — ads, requests, targets — and `--budget`
    caps *kronor*. Overloading one integer for both reads fine on the day it is
    written and is a trap by the time anyone else runs it.
    """

    limit: int | None = None
    dry_run: bool = False
    budget_sek: int = DEFAULT_BUDGET_SEK
    # Wall-clock ceiling for the sweep, in minutes. `None` means "run the whole
    # list however long it takes", which is right for the first deep sweep and
    # wrong for a scheduled nightly job that must be finished before morning.
    max_duration_min: float | None = None
    # Where in the target list to start. See `pipeline.ingest.rotate_targets`:
    # without this a time-capped run sweeps the same prefix every night.
    rotate: int = 0


async def stage_ingest(opts: RunOptions) -> bool:
    """Sweep Blocket and persist what comes back."""
    from db.crud import known_external_ids

    async with AsyncSessionLocal() as session:
        known = await known_external_ids(session, BlocketSource.name)

    targets = pc_search_targets()
    if opts.limit:
        targets = targets[:opts.limit]

    # Persisted as the sweep runs, not after it. `ingest` accumulates records in
    # memory and `persist_records` used to be called once at the end, so a power
    # cut at minute 100 of 120 discarded every detail fetch — hours of Blocket's
    # bandwidth, re-fetched from scratch on the next run. A circuit-breaker halt
    # was always handled (the records are returned and written); an abrupt death
    # was not, and that is the case a scheduled overnight job actually meets.
    #
    # `ingest` is synchronous and blocking, so it runs on a worker thread and
    # hands each batch back to the event loop to write. The alternative — a
    # second, synchronous SQLite writer — would mean a duplicate copy of the
    # upsert logic, which is how the canonical-name bug happened.
    loop = asyncio.get_running_loop()
    buffered: list = []

    def flush() -> None:
        if not buffered:
            return
        batch, buffered[:] = list(buffered), []
        asyncio.run_coroutine_threadsafe(
            persist_records(AsyncSessionLocal, batch), loop
        ).result()

    def on_record(record) -> None:
        buffered.append(record)
        if len(buffered) >= PERSIST_EVERY:
            flush()

    records, stats = await asyncio.to_thread(
        ingest,
        BlocketSource(),
        targets,
        is_known=known.__contains__,
        # A dry run must leave nothing behind, so it keeps the old shape:
        # collect everything, write none of it.
        on_record=None if opts.dry_run else on_record,
        max_duration_s=(
            opts.max_duration_min * 60 if opts.max_duration_min else None
        ),
        rotate=opts.rotate,
    )
    flush()
    logger.info(stats.summary())
    if stats.stopped_at_deadline:
        # Said plainly, because a partial sweep that reports success is how a
        # starved index looks from the outside: every run "worked", and the
        # queries at the end of the list were simply never reached.
        logger.warning(
            "Sweep was cut short by the time budget — %d of %d queries ran. "
            "The next run rotates past them.",
            stats.targets_swept,
            stats.targets_total,
        )

    if opts.dry_run:
        logger.info("[dry run] %d records not written", len(records))
    else:
        logger.info("Persisted %d records incrementally during the sweep.", len(records))
    # A halted sweep is a partial sweep. Say so, and do not let later stages
    # report success on top of it.
    return not stats.halted_early


async def stage_evaluate(opts: RunOptions) -> bool:
    if opts.dry_run:
        logger.info("[dry run] skipping AI evaluation (it costs quota)")
        return True

    # `opts.limit` unset means "as much as today's quota affords", not 200. The
    # old default was the pipeline's real bottleneck: ingest was bringing in
    # thousands of ads a night and evaluation was looking at two hundred of
    # them, using under 2% of the available quota, so most of the database
    # stayed invisible to the price index.
    stats = await process_unscored_ads(AsyncSessionLocal, limit=opts.limit)
    logger.info(stats.summary())
    # Some failures are expected and bounded; a stage where *everything* failed
    # is a broken stage, not a slow day.
    return not (stats.considered and stats.evaluated == 0)


async def stage_index(opts: RunOptions) -> bool:
    """Snapshot current component listings into the price index.

    Runs after evaluation because it reads `canonical_name`, which the
    evaluation stage writes. Cheap, local, and idempotent — no network, no
    quota, and re-running adds nothing.
    """
    if opts.dry_run:
        logger.info("[dry run] skipping index rebuild")
        return True

    async with AsyncSessionLocal() as session:
        added = await rebuild_index(session)
    logger.info("Index: %d new price observations recorded.", added)
    return True


async def stage_basket(opts: RunOptions) -> bool:
    """Report build-versus-buy for every profile, within budget.

    Read-only and side-effect free, so `--dry-run` runs it like any other pass:
    there is nothing to withhold. It is also the only stage whose output is
    meant to be read by a person rather than by the next stage.

    The budget ceiling is applied *here* and not at ingest. Filtering the scrape
    to 10 000 kr would bias the market index — a median computed only over cheap
    listings sits low, so every machine would look overpriced against it, and
    the filter would corrupt the number used to judge. Ingest stays broad;
    affordability is a question asked at the end.
    """
    async with AsyncSessionLocal() as session:
        for profile in PROFILES:
            result = await build_or_buy(session, profile, max_price_sek=opts.budget_sek)
            logger.info("[%s] %s", profile.label, result.describe())
            for item in result.basket.items:
                logger.info(
                    "    %-8s %6d kr  %s  (%s)",
                    item.slot.value,
                    item.price_sek,
                    item.canonical_name,
                    item.city or "unknown location",
                )
    return True


async def stage_publish(opts: RunOptions) -> bool:
    """Write the redacted public snapshot.

    The only stage whose output leaves this machine, and the only one where a
    failure is worth aborting the whole run over. `publish` raises rather than
    writing anything it could not verify — a half-redacted file on disk is one
    `git add .` away from being permanent.

    `--dry-run` builds and checks the snapshot but does not write it, which
    makes the redaction check runnable against live data without touching the
    committed artifact.
    """
    output = DEFAULT_OUTPUT
    if opts.dry_run:
        # Same code path, discarded destination: verifying against a temporary
        # file is what makes the dry run meaningful rather than decorative.
        import tempfile

        output = Path(tempfile.mkdtemp()) / "deals.json"

    try:
        snapshot = await publish(
            AsyncSessionLocal, output=output, budget_sek=opts.budget_sek
        )
    except RedactionError:
        logger.exception("Refusing to publish: the snapshot failed its redaction check")
        return False

    if opts.dry_run:
        output.unlink(missing_ok=True)
        logger.info("[dry run] %d deals verified, not written", len(snapshot["deals"]))
    return True


async def stage_janitor(opts: RunOptions) -> bool:
    if opts.dry_run:
        logger.info("[dry run] skipping janitor (it mutates ad status)")
        return True

    liveness = await check_liveness(AsyncSessionLocal, limit=opts.limit or 500)
    logger.info(liveness.summary())
    retention = await enforce_retention(AsyncSessionLocal)
    logger.info(retention.summary())
    return not liveness.halted_early


async def run(stage: str, opts: RunOptions) -> int:
    await init_db()

    stages = {
        "ingest": stage_ingest,
        "evaluate": stage_evaluate,
        "index": stage_index,
        "basket": stage_basket,
        "publish": stage_publish,
        "janitor": stage_janitor,
    }
    selected = list(stages) if stage == "all" else [stage]

    for name in selected:
        logger.info("--- stage: %s ---", name)
        try:
            ok = await stages[name](opts)
        except Exception:
            logger.exception("Stage %s failed", name)
            return 1
        if not ok:
            logger.error("Stage %s did not complete cleanly; stopping.", name)
            return 1

    logger.info("Pipeline finished.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentinel Blocket pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--limit", type=int, default=None, help="cap work per stage (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="scrape and report without writing"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET_SEK,
        help="kronor ceiling for the build-versus-buy report (presentation only)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="MINUTES",
        help=(
            "wall-clock ceiling for the sweep; the run stops cleanly and the "
            "next one rotates past whatever it did not reach"
        ),
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=None,
        metavar="N",
        help=(
            "start the sweep N queries into the list (default: derived from the "
            "day of the year, so scheduled runs cover the whole list over time)"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    options = RunOptions(
        limit=args.limit,
        dry_run=args.dry_run,
        budget_sek=args.budget,
        max_duration_min=args.max_duration,
        # Day-of-year rather than persisted state. A cursor in the database
        # would be one more thing to migrate and to get wrong; the date already
        # advances once a day on its own, which is exactly the cadence wanted.
        rotate=(
            args.rotate
            if args.rotate is not None
            else datetime.now().timetuple().tm_yday
        ),
    )
    return asyncio.run(run(args.stage, options))


if __name__ == "__main__":
    sys.exit(main())
