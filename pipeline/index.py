"""Market index: what a part is actually worth, from what people actually charge.

### Why this is arithmetic and not a language model

Asking an LLM for the price of a used RTX 3080 in Sweden returns a fluent number
with no date, no source and no way to check it. It cannot know today's Swedish
market and it cannot say how confident it is. Every part of that is fixed by
counting listings instead: the number is reproducible, it improves as data
accumulates, and it arrives with a sample size attached.

### Median, weighted by recency

Two decisions, both load-bearing.

**Median, not mean.** Used-hardware prices have a long right tail — a seller
asking 30 000 kr "OBO" for a card that trades at 3 000 sits in the data
alongside honest listings. One such row moves a mean of a dozen by hundreds; it
moves the median not at all. Half the market would have to be delusional in the
same direction before the median follows, and at that point it *is* the market.

**Weighted by recency.** GPU prices fall. A six-month-old observation is
evidence, but weaker evidence than yesterday's, and a plain median over a long
window lags the market downward — making everything look overpriced. Weights
decay with a half-life, so old data fades rather than either dominating or being
thrown away.

### Honesty about thin data

Below `MIN_CONFIDENT_SAMPLES` the index still returns a number, but flags it as
low confidence and always reports `n`. It never fabricates and never silently
presents a median of two as a market rate. During the cold-start period the
sample counts are visible in the UI, so the user can see the system's
uncertainty rather than infer it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    Ad,
    ComponentTypeEnum,
    ExtractedComponent,
    ItemTypeEnum,
    PriceObservation,
    StatusEnum,
)

from pipeline.canonical import canonicalize  # noqa: F401  (re-exported for callers)

logger = logging.getLogger(__name__)

# Below this, say so. A median of three is a rumour with a decimal point.
MIN_CONFIDENT_SAMPLES = 5

# Fraction of a build's identified parts that must have a price before a
# parts-versus-asking-price delta means anything. Unpriced parts contribute
# zero, so low coverage does not add noise — it adds a one-directional bias that
# makes every build look overpriced. 60% is a judgement call: high enough that
# the GPU alone (typically the single most valuable part, but rarely more than
# half the value) cannot carry a verdict on its own.
MIN_COVERAGE = 0.6

# What a gaming PC's value actually consists of. Case, PSU and motherboard are
# real costs but minor ones, and ads rarely itemise them — requiring those would
# withhold a verdict on almost every listing. These three are the ones that move
# the price.
ESSENTIAL_BUILD_PARTS = frozenset(
    {ComponentTypeEnum.CPU, ComponentTypeEnum.GPU, ComponentTypeEnum.RAM}
)

# A listing asking less than this fraction of its parts' observed value is not
# a bargain, it is a data point about something other than a sale.
#
# Found in live output: "High-End Gaming: i9-13900 | RTX 4070 Ti | 64GB DDR5 |
# 4TB NVMe" asking **395 kr**. Every spec parsed cleanly, no risk language
# anywhere, so the deterministic path settled it without a model call and it was
# announced as the cheapest qualifying AI workstation on the market. In reality
# it is a monthly financing figure, a starting bid, or bait.
#
# This is the failure mode a deal-finder is *most* exposed to, because the thing
# it searches for and the thing that characterises a scam are the same thing:
# an unusually low price. And the fast path is exactly where it lands, since a
# fraudulent advert is written cleanly on purpose.
#
# 20% is deliberately far below any real bargain. This is not a deal filter —
# genuine 40%-under listings exist and are the whole point of the project. It
# is an absurdity filter, and it is set where no honest sale has ever been.
MIN_PLAUSIBLE_PRICE_RATIO = 0.20

# Evidence needed before calling a price implausible — deliberately *lower* than
# `MIN_CONFIDENT_SAMPLES`, because the two thresholds guard opposite risks.
#
# `MIN_CONFIDENT_SAMPLES` governs whether to publish a number: there, a thin
# sample is dangerous because the number could be off by 20% and nobody would
# know. Here the question is whether a listing is off by *five times*, and a
# median would have to be catastrophically wrong for that gap to be an artefact
# of small n rather than a fact about the listing.
#
# Reusing the stricter threshold was the first attempt and it failed on the very
# ad that motivated the check: the 395 kr "RTX 4070 Ti" build had n=2 on its
# graphics card, so the guard suppressed a 16× discrepancy on the grounds that
# two observations are not many. Two observations are ample to establish that a
# card does not sell for 395 kr.
MIN_SAMPLES_FOR_IMPLAUSIBILITY = 2

# Slots a single computer can only hold one of, and how many distinct models an
# ad may name before it stops being an advert for a computer.
#
# The listing that forced this: *"RTX, GTX, Intel I7, Intel i5, RX 7900 GRE,
# Ryzen, 32gb ram"* at 999 kr — "Bud på kompletta datorer", six graphics cards
# and five processors. It is a dealer's inventory, and every part in it was
# summed against the price of one.
#
# This is the same error as the multi-card `SINGLE_COMPONENT` case below, one
# level up: there, several parts in an ad claiming to sell one. Here, several
# *machines* in an ad claiming to sell one. Two is allowed because dual-GPU
# builds exist and sellers legitimately write "i5 eller i7"; three of one kind
# is a catalogue.
MAX_DISTINCT_PER_EXCLUSIVE_SLOT = 2
EXCLUSIVE_SLOTS = frozenset({ComponentTypeEnum.CPU, ComponentTypeEnum.GPU})

# Observations older than this are ignored entirely. Beyond about six months the
# used-GPU market has moved enough that old prices mislead more than they inform.
DEFAULT_WINDOW_DAYS = 180

# Weight halves every this many days. At 60 days a three-month-old listing counts
# roughly half as much as today's — enough to track a falling market without
# discarding the history that makes the sample large enough to trust.
RECENCY_HALF_LIFE_DAYS = 60.0


@dataclass(frozen=True, slots=True)
class PriceIndex:
    """A part's market price, with the evidence behind it."""

    canonical_name: str
    median_sek: int | None
    sample_count: int
    spread_sek: int | None = None
    newest_observation: datetime | None = None

    @property
    def is_confident(self) -> bool:
        return self.sample_count >= MIN_CONFIDENT_SAMPLES

    def describe(self) -> str:
        if self.median_sek is None:
            return f"{self.canonical_name}: no data"
        note = "" if self.is_confident else " (low confidence)"
        return f"{self.canonical_name}: {self.median_sek} kr (n={self.sample_count}){note}"


def weighted_median(values: list[tuple[float, float]]) -> float | None:
    """Median of `(value, weight)` pairs.

    Walks the sorted values accumulating weight and returns the value at which
    half the total weight has been passed. The weighting changes *which*
    observation sits at the halfway point, never the guarantee that outliers
    cannot drag it.

    When the weight splits exactly evenly — which is the ordinary case for an
    even number of equally-weighted observations — the two middle values are
    averaged, matching `statistics.median`. Returning the lower one instead
    would be a small but systematic downward bias, and a valuation index that
    leans low makes every listing look worse than it is.
    """
    if not values:
        return None

    ordered = sorted(values)
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return None

    midpoint = total / 2.0
    running = 0.0
    for position, (value, weight) in enumerate(ordered):
        running += weight
        if math.isclose(running, midpoint) and position + 1 < len(ordered):
            return (value + ordered[position + 1][0]) / 2.0
        if running > midpoint:
            return value
    return ordered[-1][0]


def recency_weight(observed_at: datetime, now: datetime) -> float:
    """Exponential decay by age. Today counts 1.0, one half-life ago 0.5."""
    age_days = max(0.0, (now - observed_at).total_seconds() / 86_400.0)
    return math.pow(0.5, age_days / RECENCY_HALF_LIFE_DAYS)


async def record_observations(session: AsyncSession, *, only_ad_id: int | None = None) -> int:
    """Snapshot current single-component listings into `price_observations`.

    Only `SINGLE_COMPONENT` ads contribute. The price of a whole computer says
    nothing about the price of its graphics card, and mixing the two would push
    every component median up towards the price of a PC.

    Uses `INSERT ... ON CONFLICT DO NOTHING` against the dedup constraint, so
    running this nightly is idempotent: an unchanged ad seen again adds nothing
    and does not fake a deeper sample. A genuine price *change* has a different
    `price_sek` and so is correctly recorded as a new data point.
    """
    stmt = (
        select(
            ExtractedComponent.canonical_name,
            ExtractedComponent.component_type,
            Ad.price_sek,
            Ad.condition,
            Ad.source,
            Ad.external_id,
        )
        .join(Ad, Ad.id == ExtractedComponent.ad_id)
        .where(Ad.item_type == ItemTypeEnum.SINGLE_COMPONENT)
        .where(ExtractedComponent.canonical_name.is_not(None))
    )
    if only_ad_id is not None:
        stmt = stmt.where(Ad.id == only_ad_id)

    rows = (await session.execute(stmt)).all()
    if not rows:
        return 0

    payload = [
        {
            "canonical_name": name,
            "component_type": ctype,
            "price_sek": price,
            "condition": condition,
            "source": source,
            "source_ref": external_id,
        }
        for name, ctype, price, condition, source, external_id in rows
    ]

    result = await session.execute(
        sqlite_insert(PriceObservation).on_conflict_do_nothing().values(payload)
    )
    return result.rowcount or 0


async def price_index(
    session: AsyncSession,
    canonical_name: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    now: datetime | None = None,
) -> PriceIndex:
    """The recency-weighted median price for one canonical part."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    stmt = (
        select(PriceObservation.price_sek, PriceObservation.observed_at)
        .where(PriceObservation.canonical_name == canonical_name)
        .where(PriceObservation.observed_at >= cutoff)
    )
    rows = (await session.execute(stmt)).all()

    if not rows:
        return PriceIndex(canonical_name=canonical_name, median_sek=None, sample_count=0)

    weighted = [
        (float(price), recency_weight(_aware(observed_at), now))
        for price, observed_at in rows
    ]
    median = weighted_median(weighted)
    prices = sorted(price for price, _ in weighted)

    return PriceIndex(
        canonical_name=canonical_name,
        median_sek=int(median) if median is not None else None,
        sample_count=len(rows),
        # Interquartile-ish spread. Surfaced so a wide, untrustworthy market
        # ("somewhere between 2 000 and 9 000") is visible rather than hidden
        # behind a confident-looking midpoint.
        spread_sek=int(prices[-1] - prices[0]) if len(prices) > 1 else 0,
        newest_observation=max(_aware(observed) for _, observed in rows),
    )


def _price_or_inf(part: "PartValuation") -> float:
    """Sort helper: an unpriced part is treated as the most expensive option.

    So a priced candidate always beats an unpriced one for the slot. Picking the
    unpriced one would drop the slot out of the estimate entirely and lower
    coverage, which is a worse outcome than using the number we do have.
    """
    return part.index.median_sek if part.index.median_sek is not None else float("inf")


def _aware(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; treat them as the UTC they were stored as."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class PartValuation:
    component_type: ComponentTypeEnum
    canonical_name: str
    index: PriceIndex


@dataclass(frozen=True, slots=True)
class BuildValuation:
    """A listing priced against the sum of its parts."""

    ad_id: int
    listed_price_sek: int
    parts: tuple[PartValuation, ...]
    # Drives how coverage is measured: a build is judged against what a build
    # contains, a single component against itself.
    item_type: ItemTypeEnum = ItemTypeEnum.UNKNOWN

    @property
    def effective_parts(self) -> tuple[PartValuation, ...]:
        """`parts`, with each exclusive slot collapsed to a single entry.

        A computer has one processor and one graphics card. `parts` is the raw
        extraction and may hold several of either — because the ad mentioned the
        card twice, or named an alternative, or is listing more than one machine.
        Summing them all values a computer that does not exist, and the error
        only ever runs one way: **upward**, which reads as a bargain.

        Live example, top of the published leaderboard by a wide margin:
        *"Gaming Dator Ryzen 5 9600x/RTX 5070ti/32GB DDR5"* at 20 000 kr. The
        5070 Ti was extracted twice, counted twice at 11 000 kr, and the machine
        was reported as **13 675 kr below market**. There is one card in it.

        Where several remain, the *cheapest* wins. We cannot tell which card is
        actually in the box, and the two possible errors are not symmetrical: too
        low makes a real bargain look ordinary, too high invents one that is not
        there and puts it at the top of the page. Same pessimism as the
        ambiguous-VRAM rule in `pipeline/hardware.py`, for the same reason.

        `is_inventory_listing` deliberately reads `parts`, not this — it needs to
        see all six graphics cards in order to notice that it is looking at a
        dealer's stock list.
        """
        # Tracked by position, not by object identity. `PartValuation` is a
        # frozen dataclass, so two extractions of the same card may be one
        # object or two indistinguishable ones depending on how the caller built
        # the tuple — and an identity-keyed version of this passed the live
        # example while silently failing on the simpler case of the exact same
        # object listed twice.
        winner: dict[ComponentTypeEnum, int] = {}
        for i, part in enumerate(self.parts):
            if part.component_type not in EXCLUSIVE_SLOTS:
                continue
            best = winner.get(part.component_type)
            if best is None or _price_or_inf(part) < _price_or_inf(self.parts[best]):
                winner[part.component_type] = i

        keep = set(winner.values())
        # Rebuilt in the original order, so the component list in the UI still
        # reads in the order the extractor found things.
        return tuple(
            part
            for i, part in enumerate(self.parts)
            if part.component_type not in EXCLUSIVE_SLOTS or i in keep
        )

    @property
    def priced(self) -> tuple[PartValuation, ...]:
        return tuple(
            p for p in self.effective_parts if p.index.median_sek is not None
        )

    @property
    def estimated_parts_sek(self) -> int | None:
        priced = self.priced
        if not priced:
            return None
        return sum(p.index.median_sek for p in priced)  # type: ignore[misc]

    @property
    def delta_sek(self) -> int | None:
        """Listed price minus parts estimate. **Negative is a good deal.**

        `None` when too few of the build's parts could be priced — see
        `MIN_COVERAGE`. That is not defensiveness; it is the difference between
        a number and a wrong number.

        Found on real data. Every full build came back "overpriced" by
        2 000–4 000 kr, uniformly, which is not what a real market looks like.
        The cause was coverage of 25–50%: only the graphics card had an index
        entry, so "sum of parts" quietly meant "sum of the one part we knew",
        and the CPU, RAM, storage, PSU and case were each counted as worth
        nothing. A systematic one-directional bias, presented as a confident
        figure, on the exact number the leaderboard sorts by.
        """
        if self.coverage < MIN_COVERAGE:
            return None
        estimate = self.estimated_parts_sek
        return None if estimate is None else self.listed_price_sek - estimate

    @property
    def is_inventory_listing(self) -> bool:
        """Whether this ad is a stock list rather than one machine.

        A computer has one processor and, realistically, one graphics card. An
        ad naming six cards and five processors is a dealer's inventory, and
        summing all of it against the single asking price produces a
        spectacular, entirely fictional bargain.

        Live example, top of the leaderboard at 999 kr: *"RTX, GTX, Intel I7,
        Intel i5, RX 7900 GRE, Ryzen, 32gb ram"* — description: "Bud på
        kompletta datorer". Six GPUs, five CPUs, one price.
        """
        counts: dict[ComponentTypeEnum, set[str]] = {}
        for part in self.parts:
            if part.component_type in EXCLUSIVE_SLOTS:
                counts.setdefault(part.component_type, set()).add(part.canonical_name)
        return any(
            len(names) > MAX_DISTINCT_PER_EXCLUSIVE_SLOT for names in counts.values()
        )

    @property
    def is_implausibly_cheap(self) -> bool:
        """Whether the asking price is too low to be a real asking price.

        Requires a confident estimate to compare against — with thin sample
        counts the parts figure is itself uncertain, and calling a listing
        fraudulent on the strength of two observations would be worse than
        saying nothing.

        Deliberately expressed as "implausible" rather than "fraudulent". The
        pipeline cannot know which of financing-per-month, starting bid, typo or
        bait it is looking at. It can know the number is not a sale price.
        """
        estimate = self.estimated_parts_sek
        if estimate is None or not estimate:
            return False
        if self.min_sample_count < MIN_SAMPLES_FOR_IMPLAUSIBILITY:
            return False
        # Note that `estimated_parts_sek` counts unpriced parts as zero, so it
        # is an *under*estimate of what the machine contains. That biases this
        # test toward silence, which is the correct direction for an accusation.
        return self.listed_price_sek < estimate * MIN_PLAUSIBLE_PRICE_RATIO

    @property
    def min_sample_count(self) -> int:
        """The thinnest evidence anywhere in the estimate.

        A build is only as trustworthy as its worst-sampled part, so this — not
        the average — is what gets displayed. Averaging would let one
        well-sampled GPU disguise three parts we know nothing about.
        """
        return min((p.index.sample_count for p in self.priced), default=0)

    @property
    def coverage(self) -> float:
        """Fraction of the parts a machine of this kind *should* have that we priced.

        The denominator is deliberately not `len(self.parts)`. Measuring
        priced-over-identified was the first attempt and it is subtly useless: a
        full PC where only the graphics card was ever recognised scores 100%,
        because the one part we found is the one part we priced.

        Real output made that obvious — a Lenovo Legion asking 7 000 kr was
        reported at "100% coverage, parts worth 2 595 kr", flagging a whole
        computer as wildly overpriced on the strength of its GPU alone.

        So a build is measured against what a build actually contains. A CPU, a
        GPU and RAM are most of the value of a gaming machine; case, PSU and
        motherboard are comparatively minor and rarely itemised in ads. A
        single component is measured against itself.
        """
        if not self.parts:
            return 0.0

        # An inventory is not a machine, whatever it is labelled. Checked before
        # anything else, because the arithmetic below is meaningless once the
        # premise "these parts are one computer" fails.
        if self.is_inventory_listing:
            return 0.0

        if self.item_type in (ItemTypeEnum.FULL_BUILD, ItemTypeEnum.BUNDLE):
            priced_kinds = {p.component_type for p in self.priced}
            return len(priced_kinds & ESSENTIAL_BUILD_PARTS) / len(ESSENTIAL_BUILD_PARTS)

        # A "single component" ad holding several parts is a contradiction, and
        # the arithmetic it produces is spectacular nonsense: the medians of
        # every card get summed against the asking price of one.
        #
        # Live output, top of the leaderboard: an ad titled "RTX 3070 Asus TuF,
        # RX 6700 xt, RTX 2080..." asking 2 690 kr, valued at 38 570 kr of
        # parts — a 35 880 kr "bargain". Multi-item listings are common and the
        # classifier will keep mislabelling some of them, so this is caught here
        # by consistency rather than by trying to classify perfectly.
        if self.item_type is ItemTypeEnum.SINGLE_COMPONENT and len(self.effective_parts) > 1:
            return 0.0

        return len(self.priced) / len(self.effective_parts)


async def value_ad(
    session: AsyncSession, ad_id: int, *, now: datetime | None = None
) -> BuildValuation | None:
    """Price one listing against the index."""
    ad = await session.get(Ad, ad_id)
    if ad is None:
        return None

    stmt = select(
        ExtractedComponent.component_type, ExtractedComponent.canonical_name
    ).where(ExtractedComponent.ad_id == ad_id)
    components = (await session.execute(stmt)).all()

    parts = []
    for component_type, canonical_name in components:
        if not canonical_name:
            continue
        parts.append(
            PartValuation(
                component_type=component_type,
                canonical_name=canonical_name,
                index=await price_index(session, canonical_name, now=now),
            )
        )

    return BuildValuation(
        ad_id=ad.id,
        listed_price_sek=ad.price_sek,
        parts=tuple(parts),
        item_type=ad.item_type,
    )


async def rebuild_index(session: AsyncSession) -> int:
    """Refresh observations from current listings. Safe to run repeatedly."""
    added = await record_observations(session)
    await session.commit()
    logger.info("Recorded %d new price observations.", added)
    return added
