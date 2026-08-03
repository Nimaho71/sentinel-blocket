"""Ingest stage: search a source, pre-filter, fetch detail, persist.

Replaces the scrape phase of the old `legacy/orchestrator.py`, which drove a
Playwright browser. The important structural change beyond speed:

**Pre-filtering happens before the expensive call.** Search returns ~50 ads for
one request; detail costs one request *each*. Deciding on the cheap payload
whether an ad is worth the expensive one is what keeps a full sweep to a few
hundred requests instead of a few thousand.

The other change is that a blocked run now looks different from an empty one.
The old code returned `[]` on failure, so "Blocket is refusing us" and "no ads
matched" were indistinguishable — which is how the pipeline silently stopped
producing data. Here, failures propagate and the circuit breaker halts the run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from sources.base import AdRecord, AdSummary, CircuitOpen, Source, SourceError

logger = logging.getLogger(__name__)

# Ads outside this range are noise: below it sits broken/parts-only junk and
# mispriced listings, above it sits workstation gear irrelevant to the budget.
DEFAULT_MIN_PRICE_SEK = 300
DEFAULT_MAX_PRICE_SEK = 40_000


@dataclass(frozen=True, slots=True)
class SearchTarget:
    """One query to sweep.

    `component_only` marks queries that exist to feed the market index rather
    than to find a machine to buy. Those results are the price observations that
    make valuation possible, so they are swept even when nothing is for sale
    that we'd want.
    """

    query: str
    component_only: bool = False
    # A ceiling, not a quota. Paging stops as soon as a page comes back short,
    # so this only bounds the queries that genuinely have more to give.
    max_pages: int = 1


@dataclass
class IngestStats:
    """What a run actually did. Logged at the end so silent stalls are visible."""

    searched: int = 0
    found: int = 0
    filtered_out: int = 0
    already_known: int = 0
    detail_fetched: int = 0
    detail_missing: int = 0
    # Ads this run already fetched under a different query. Counted rather than
    # merely avoided: the number is a direct measure of how much the query list
    # overlaps itself, which is the only way to tell whether adding a query
    # bought new coverage or just re-bought the old.
    duplicates_skipped: int = 0
    failures: int = 0
    halted_early: bool = False
    # A run that ran out of time rather than out of work. Distinct from
    # `halted_early`, which means the circuit breaker tripped: one is a budget
    # being spent as designed, the other is the source refusing us, and a run
    # summary that conflated them would report a block as a normal night.
    stopped_at_deadline: bool = False
    targets_swept: int = 0
    targets_total: int = 0

    errors: list[str] = field(default_factory=list)

    @property
    def is_complete_sweep(self) -> bool:
        return self.targets_swept >= self.targets_total and not self.halted_early

    def summary(self) -> str:
        return (
            f"{self.searched} searches -> {self.found} ads found, "
            f"{self.filtered_out} filtered, {self.already_known} known, "
            f"{self.duplicates_skipped} dup, "
            f"{self.detail_fetched} fetched, {self.detail_missing} gone, "
            f"{self.failures} failed"
            + (
                f", swept {self.targets_swept}/{self.targets_total} queries"
                if self.targets_total
                else ""
            )
            + (" [OUT OF TIME]" if self.stopped_at_deadline else "")
            + (" [HALTED EARLY]" if self.halted_early else "")
        )


def is_candidate(
    summary: AdSummary,
    *,
    min_price: int = DEFAULT_MIN_PRICE_SEK,
    max_price: int = DEFAULT_MAX_PRICE_SEK,
) -> bool:
    """Decide, from cheap search data, whether an ad earns a detail request.

    Deliberately permissive: a false positive costs one request, while a false
    negative silently hides a deal. Only clearly-useless listings are dropped.
    """
    if not summary.title:
        return False
    if summary.price_sek < min_price or summary.price_sek > max_price:
        return False
    return True


def rotate_targets(
    targets: Sequence[SearchTarget], offset: int
) -> list[SearchTarget]:
    """The same sweep, started from a different point in the list.

    Required by `max_duration_s`, and the non-obvious half of it. A time-capped
    run walks the list in order and stops when the budget is gone, so without
    rotation it sweeps the same prefix every night and the tail is *never*
    reached — "grafikkort", the Intel sockets and the PSU queries would sit
    permanently unswept while every run reported success.

    That is the failure this project keeps meeting in different costumes: work
    that is systematically partial while presenting itself as complete, like the
    coverage bug in `pipeline/index.py`. Rotating the start point converts
    "these queries never run" into "every query runs every few nights", which is
    a property that can be stated honestly.
    """
    if not targets:
        return []
    offset %= len(targets)
    return list(targets[offset:]) + list(targets[:offset])


def ingest(
    source: Source,
    targets: Sequence[SearchTarget],
    *,
    is_known: Callable[[str], bool] | None = None,
    on_record: Callable[[AdRecord], None] | None = None,
    min_price: int = DEFAULT_MIN_PRICE_SEK,
    max_price: int = DEFAULT_MAX_PRICE_SEK,
    max_duration_s: float | None = None,
    rotate: int = 0,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[AdRecord], IngestStats]:
    """Sweep `targets` against `source`, returning fully-detailed new records.

    `is_known` skips ads already stored — the single biggest saving available,
    since re-fetching detail for an unchanged ad is pure waste.

    `on_record` is invoked per record as it arrives, so a caller can persist
    incrementally and keep partial progress if the run halts.

    ### Why the budget is wall-clock and not a request count

    `max_duration_s` bounds how long the sweep runs. Time is the unit that
    matters and the only one that can be predicted: a request count cannot be
    turned into a duration without knowing how many ads will turn out to be new,
    which is the thing nobody knows in advance. "Stop after ninety minutes" is a
    promise this can keep; "stop after 2 000 requests" is not.

    Checked before each request rather than between targets, because the run is
    dominated by one-ad-at-a-time detail fetches — a coarser check could
    overshoot the budget by an hour. Nothing is interrupted mid-request: the
    deadline decides whether to *start* work, so a capped run ends on a whole
    record and never on half of one.

    Partial by design, and therefore partial *visibly*: `stats.targets_swept`
    and `is_complete_sweep` say how much of the list was reached. Pair with
    `rotate` so consecutive runs begin at different points — see
    `rotate_targets`.
    """
    stats = IngestStats()
    records: list[AdRecord] = []

    # `is_known` is a snapshot of the database taken before the sweep starts, so
    # it cannot know about an ad this run has *just* fetched. Without this set,
    # a genuinely new ad returned by two queries fails the known check both
    # times and is fetched twice at full price.
    #
    # Measured mid-run: 927 detail requests for 759 distinct ads — 18% wasted,
    # and that was before the sweep reached the GPU block, where "grafikkort"
    # overlaps all 36 model-specific queries and "RTX 3060" overlaps "RTX 3060
    # Ti". Detail is the expensive half, one request per ad.
    seen: set[str] = set()

    ordered = rotate_targets(targets, rotate)
    stats.targets_total = len(ordered)
    deadline = None if max_duration_s is None else clock() + max_duration_s

    def out_of_time() -> bool:
        return deadline is not None and clock() >= deadline

    for target in ordered:
        if out_of_time():
            stats.stopped_at_deadline = True
            logger.info(
                "time budget spent: swept %d/%d queries this run.",
                stats.targets_swept, stats.targets_total,
            )
            break
        # The widest full page this query has returned, which is the API's page
        # size as observed rather than as assumed. A page narrower than that is
        # the last one. Measured rather than hardcoded to 53: pinning the
        # number means a silent halving of coverage on the day Blocket changes
        # it, and every run would still report success.
        page_size = 0
        for page in range(1, target.max_pages + 1):
            if out_of_time():
                stats.stopped_at_deadline = True
                break
            try:
                # Pushed server-side where the source supports it, so a page
                # comes back full of in-budget ads instead of partly full.
                # `is_candidate` below still re-checks locally — on an
                # unofficial API, an unsupported filter fails *open*.
                summaries = source.search(
                    target.query,
                    page=page,
                    price_from=min_price,
                    price_to=max_price,
                )
                stats.searched += 1
            except CircuitOpen as exc:
                logger.error("halting: %s", exc)
                stats.halted_early = True
                stats.errors.append(str(exc))
                return records, stats
            except SourceError as exc:
                logger.warning("search %r page %d failed: %s", target.query, page, exc)
                stats.failures += 1
                stats.errors.append(str(exc))
                continue

            stats.found += len(summaries)
            for summary in summaries:
                if not is_candidate(summary, min_price=min_price, max_price=max_price):
                    stats.filtered_out += 1
                    continue
                if is_known is not None and is_known(summary.external_id):
                    stats.already_known += 1
                    continue
                if summary.external_id in seen:
                    stats.duplicates_skipped += 1
                    continue
                # The dominant cost in the whole run: one request per new ad.
                # Checked here so a capped sweep stops on a whole record.
                if out_of_time():
                    stats.stopped_at_deadline = True
                    break

                # Marked before the request, not after. A detail fetch that
                # fails or returns a removed listing has still been paid for,
                # and retrying it under the next query would spend the same
                # request to get the same nothing.
                seen.add(summary.external_id)
                try:
                    record = source.fetch_detail(summary)
                except CircuitOpen as exc:
                    logger.error("halting: %s", exc)
                    stats.halted_early = True
                    stats.errors.append(str(exc))
                    return records, stats
                except SourceError as exc:
                    logger.warning("detail %s failed: %s", summary.external_id, exc)
                    stats.failures += 1
                    stats.errors.append(str(exc))
                    continue

                if record is None:
                    stats.detail_missing += 1
                    continue

                stats.detail_fetched += 1
                records.append(record)
                if on_record is not None:
                    on_record(record)

            # Stop as soon as the results run out, rather than walking to the
            # ceiling regardless. Measured on a real sweep: 14 of 41 search
            # requests — 34% — returned zero ads, because "moderkort AM4" has
            # 27 listings and was being asked for four pages of them. Those are
            # requests spent on Blocket for nothing.
            #
            # A page narrower than the widest one seen is the last page: the
            # corpus bears this out ("nätaggregat 650W" gave 50 then 0,
            # "nätaggregat 750W" gave 53, 20, then 0).
            if not summaries or len(summaries) < page_size:
                break
            page_size = max(page_size, len(summaries))

        # Counted only once every page of this query has been walked, so a
        # query abandoned halfway through its budget is not reported as swept.
        if not stats.stopped_at_deadline:
            stats.targets_swept += 1

    logger.info(
        "ingest %s: %s",
        "complete" if stats.is_complete_sweep else "partial",
        stats.summary(),
    )
    return records, stats


# How deep to page each kind of query.
#
# Component queries feed the market index and were the shallowest part of the
# sweep — one page each — which is backwards. A build query finds one machine to
# look at; a component query is a *price sample*, and every valuation the site
# publishes is only as trustworthy as the number of them behind it.
#
# The corpus under the old settings held 214 observations across 95 distinct
# parts: 71 of those parts had exactly one observation, and only 11 cleared
# `MIN_CONFIDENT_SAMPLES`. A median of one is one stranger's asking price, and
# 43 of the 90 published deals rested on that.
#
# This is affordable because the binding constraint is not the obvious one.
# Gemini's free tier is two 500-RPD pools batched ten ads to a request — roughly
# 10 000 ads a day against a corpus in the low thousands, and two thirds never
# reach a model at all. The real ceiling is the 2.4-second request spacing,
# which is an *ethical commitment* (ETHICS.md), not a technical limit. So the
# right response to needing more data is a longer run, never a faster one.
# These became *ceilings* once paging learned to stop on a short page, which
# changes what a large number costs. Under fixed paging, 4 was simultaneously
# too many and too few: "moderkort AM4" has 27 listings and was asked for four
# pages of them, while "SSD 1TB", "NVMe 1TB", "DDR5 32GB" and "moderkort" were
# all still returning full pages at page 4 and being cut off mid-stream. No
# single number was right, because the queries are not alike.
#
# Now a thin query costs one request and a deep one runs until it is genuinely
# exhausted, so this only needs to be high enough not to truncate the broadest
# queries. It bounds a pathological case; it does not ration.
COMPONENT_PAGES = 12
BUILD_PAGES = 12


def pc_search_targets() -> list[SearchTarget]:
    """The default sweep.

    Component queries dominate deliberately. Each one is a price sample, and the
    market index is only as good as its sample count — so breadth here is what
    makes every later valuation defensible rather than guesswork.
    """
    components = [
        # GPUs — the dominant term in any build's value.
        #
        # The 20-series and GTX 16-series are here because this sweep is aimed
        # at a *used* market, where a card Nvidia stopped selling in 2020 is not
        # obsolete stock but the median listing.
        "GTX 1060", "GTX 1070", "GTX 1080", "GTX 1650", "GTX 1660",
        "RTX 2060", "RTX 2070", "RTX 2080",
        "RTX 3050", "RTX 3060", "RTX 3060 Ti", "RTX 3070", "RTX 3070 Ti",
        "RTX 3080", "RTX 3080 Ti", "RTX 3090",
        "RTX 4060", "RTX 4060 Ti", "RTX 4070", "RTX 4070 Ti", "RTX 4070 Super",
        "RTX 4080", "RTX 4090",
        "RTX 5070", "RTX 5080",
        "RX 6600", "RX 6600 XT", "RX 6650 XT", "RX 6700 XT", "RX 6800",
        "RX 6800 XT", "RX 7600", "RX 7700 XT", "RX 7800 XT", "RX 7900 XTX",
        # A generic term catches the long tail no model-specific query names.
        "grafikkort",
        # CPUs. The old list had four Ryzen and four Intel parts and no i9 or
        # Ryzen 9 at all, so the top of both stacks was invisible to the index.
        "Ryzen 5 3600", "Ryzen 5 5600", "Ryzen 5 5600X", "Ryzen 7 5700X",
        "Ryzen 7 5800X3D", "Ryzen 9 5900X", "Ryzen 5 7600", "Ryzen 7 7800X3D",
        "Ryzen 9 7900X",
        "i5 10400", "i5 11400", "i5 12400", "i5 13400", "i5 13600K",
        "i7 10700K", "i7 11700K", "i7 12700K", "i7 13700K",
        "i9 12900K", "i9 13900K",
        "processor CPU",
        # Supporting parts.
        "DDR4 16GB", "DDR4 32GB", "DDR5 16GB", "DDR5 32GB",
        "SSD 1TB", "NVMe 1TB", "NVMe 2TB",
        # Intel sockets were missing entirely; only AM4/AM5 were swept.
        "moderkort AM4", "moderkort AM5", "moderkort LGA1700", "moderkort",
        "nätaggregat 650W", "nätaggregat 750W", "nätaggregat 850W",
    ]
    builds = [
        "gaming dator", "speldator", "stationär dator", "gaming PC",
        "gamingdator", "dator komplett",
    ]

    return [
        SearchTarget(q, component_only=True, max_pages=COMPONENT_PAGES)
        for q in components
    ] + [
        SearchTarget(q, component_only=False, max_pages=BUILD_PAGES) for q in builds
    ]
