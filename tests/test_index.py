"""Market index tests.

Three properties matter more than the rest, and each has a named test:

1. **An outlier cannot move the median.** The reason this is not a mean.
2. **Recent prices outweigh old ones.** The reason it is not a plain median.
3. **Deleting an ad leaves its price observation intact.** The property the
   entire retention design rests on — if this breaks, aggressive purging starts
   destroying the data that makes valuation work.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from db.models import (
    Ad,
    ComponentTypeEnum,
    ExtractedComponent,
    ItemTypeEnum,
    PriceObservation,
)
from pipeline.index import (
    MIN_CONFIDENT_SAMPLES,
    BuildValuation,
    PartValuation,
    PriceIndex,
    RECENCY_HALF_LIFE_DAYS,
    price_index,
    record_observations,
    recency_weight,
    value_ad,
    weighted_median,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def days_ago(n: int) -> datetime:
    return NOW - timedelta(days=n)


async def observe(session, name, price, *, ref=None, age_days=0, ctype=ComponentTypeEnum.GPU):
    session.add(
        PriceObservation(
            canonical_name=name,
            component_type=ctype,
            price_sek=price,
            source="blocket",
            source_ref=ref or f"{name}-{price}-{age_days}",
            observed_at=days_ago(age_days),
        )
    )


async def _seed_observations(session, name, prices, *, ctype=ComponentTypeEnum.GPU):
    for i, price in enumerate(prices):
        await observe(session, name, price, ref=f"{name}-seed-{i}", ctype=ctype)
    await session.commit()


async def _seed_build(session, *, price_sek, parts):
    """A FULL_BUILD ad with the given canonical parts. Returns its id."""
    from tests.conftest import make_ad

    ad = make_ad("plausibility", price_sek=price_sek, item_type=ItemTypeEnum.FULL_BUILD)
    session.add(ad)
    await session.flush()
    for ctype, canonical in parts:
        session.add(
            ExtractedComponent(
                ad_id=ad.id,
                component_type=ComponentTypeEnum[ctype],
                component_name=canonical.upper(),
                canonical_name=canonical,
            )
        )
    await session.commit()
    return ad.id


# --- Median behaviour --------------------------------------------------------


def test_weighted_median_ignores_an_extreme_outlier():
    """One deluded seller must not move the number.

    Eight honest listings near 3 000 kr and one asking 30 000. The mean is over
    6 000 — double the real price, and biased in the direction that makes every
    build look like a bargain.
    """
    prices = [2800, 2900, 3000, 3000, 3100, 3200, 3300, 30000]
    median = weighted_median([(float(p), 1.0) for p in prices])

    assert median == 3050
    assert sum(prices) / len(prices) > 6000, "the outlier really does wreck the mean"


def test_weighted_median_of_nothing_is_none():
    assert weighted_median([]) is None


def test_zero_total_weight_returns_none():
    """Guards a division by zero when every observation has decayed to nothing."""
    assert weighted_median([(3000.0, 0.0)]) is None


# --- Recency -----------------------------------------------------------------


def test_weight_halves_over_one_half_life():
    assert recency_weight(NOW, NOW) == pytest.approx(1.0)
    assert recency_weight(days_ago(int(RECENCY_HALF_LIFE_DAYS)), NOW) == pytest.approx(0.5)


async def test_recent_prices_pull_the_median_down(session_factory):
    async with session_factory() as session:
        for i in range(5):
            await observe(session, "rtx 3080", 6000, ref=f"old{i}", age_days=170)
        for i in range(5):
            await observe(session, "rtx 3080", 3000, ref=f"new{i}", age_days=1)
        await session.commit()

        weighted = await price_index(session, "rtx 3080", now=NOW)

    # An unweighted median of five 6000s and five 3000s sits at 4500.
    # Recency weighting must land nearer the current market.
    assert weighted.median_sek == 3000
    assert weighted.sample_count == 10


async def test_observations_outside_the_window_are_ignored(session_factory):
    async with session_factory() as session:
        await observe(session, "rtx 3080", 9000, ref="ancient", age_days=400)
        await observe(session, "rtx 3080", 3000, ref="recent", age_days=2)
        await session.commit()

        index = await price_index(session, "rtx 3080", window_days=180, now=NOW)

    assert index.sample_count == 1
    assert index.median_sek == 3000


# --- Honesty about thin data -------------------------------------------------


async def test_an_unseen_part_reports_no_price_rather_than_zero(session_factory):
    """Zero is a price. "We don't know" is not, and conflating them would rank
    an unknown part as free."""
    async with session_factory() as session:
        index = await price_index(session, "rtx 5090", now=NOW)

    assert index.median_sek is None
    assert index.sample_count == 0
    assert not index.is_confident
    assert "no data" in index.describe()


async def test_thin_samples_are_flagged_not_hidden(session_factory):
    async with session_factory() as session:
        for i in range(MIN_CONFIDENT_SAMPLES - 1):
            await observe(session, "rtx 3080", 3000, ref=f"a{i}")
        await session.commit()

        index = await price_index(session, "rtx 3080", now=NOW)

    assert index.median_sek == 3000
    assert not index.is_confident
    assert "low confidence" in index.describe()
    assert f"n={MIN_CONFIDENT_SAMPLES - 1}" in index.describe()


async def test_enough_samples_are_confident(session_factory):
    async with session_factory() as session:
        for i in range(MIN_CONFIDENT_SAMPLES):
            await observe(session, "rtx 3080", 3000, ref=f"a{i}")
        await session.commit()

        index = await price_index(session, "rtx 3080", now=NOW)

    assert index.is_confident
    assert "low confidence" not in index.describe()


async def test_a_wide_spread_is_surfaced(session_factory):
    """"Somewhere between 2 000 and 9 000" should not hide behind a confident
    midpoint."""
    async with session_factory() as session:
        for i, price in enumerate([2000, 3000, 4000, 9000]):
            await observe(session, "rtx 3080", price, ref=f"a{i}")
        await session.commit()

        index = await price_index(session, "rtx 3080", now=NOW)

    assert index.spread_sek == 7000


# --- Recording observations --------------------------------------------------


async def test_only_single_component_ads_become_observations(session_factory, ad_factory):
    """A 9 000 kr PC containing an RTX 3080 says nothing about the card's price.

    Letting builds through would drag every component median up toward the price
    of a whole computer.
    """
    async with session_factory() as session:
        part = ad_factory("1", title="RTX 3080", price_sek=3000,
                          item_type=ItemTypeEnum.SINGLE_COMPONENT)
        part.extracted_components.append(
            ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                               component_name="RTX 3080", canonical_name="rtx 3080")
        )
        build = ad_factory("2", title="Gaming PC", price_sek=9000,
                           item_type=ItemTypeEnum.FULL_BUILD)
        build.extracted_components.append(
            ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                               component_name="RTX 3080", canonical_name="rtx 3080")
        )
        session.add_all([part, build])
        await session.commit()

        assert await record_observations(session) == 1
        await session.commit()

        index = await price_index(session, "rtx 3080", now=NOW)

    assert index.sample_count == 1
    assert index.median_sek == 3000


async def test_rescraping_an_unchanged_ad_adds_no_duplicate(session_factory, ad_factory):
    """Nightly runs must not inflate the sample count and fake deeper evidence
    than actually exists."""
    async with session_factory() as session:
        ad = ad_factory("1", title="RTX 3080", price_sek=3000,
                        item_type=ItemTypeEnum.SINGLE_COMPONENT)
        ad.extracted_components.append(
            ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                               component_name="RTX 3080", canonical_name="rtx 3080")
        )
        session.add(ad)
        await session.commit()

        await record_observations(session)
        await session.commit()
        await record_observations(session)
        await session.commit()

        count = await session.scalar(select(func.count()).select_from(PriceObservation))

    assert count == 1


async def test_a_price_change_is_recorded_as_a_new_observation(session_factory, ad_factory):
    """A seller cutting their price is a genuine new data point, not a duplicate."""
    async with session_factory() as session:
        ad = ad_factory("1", title="RTX 3080", price_sek=3000,
                        item_type=ItemTypeEnum.SINGLE_COMPONENT)
        ad.extracted_components.append(
            ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                               component_name="RTX 3080", canonical_name="rtx 3080")
        )
        session.add(ad)
        await session.commit()
        await record_observations(session)
        await session.commit()

        ad.price_sek = 2500
        await session.commit()
        await record_observations(session)
        await session.commit()

        count = await session.scalar(select(func.count()).select_from(PriceObservation))

    assert count == 2


# --- The property the retention design rests on ------------------------------


async def test_deleting_an_ad_leaves_its_price_observation_intact(session_factory, ad_factory):
    """**The load-bearing test of the whole retention design.**

    Sold ads are purged after 90 days and descriptions dropped after 7. That is
    only affordable because an observation carries no reference to the ad it
    came from — the listing disappears, the data point survives, and the index
    keeps improving while the database keeps forgetting.

    If a foreign key or cascade is ever added to `price_observations`, this test
    is what catches it.
    """
    async with session_factory() as session:
        ad = ad_factory("1", title="RTX 3080", price_sek=3000,
                        item_type=ItemTypeEnum.SINGLE_COMPONENT)
        ad.extracted_components.append(
            ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                               component_name="RTX 3080", canonical_name="rtx 3080")
        )
        session.add(ad)
        await session.commit()

        await record_observations(session)
        await session.commit()

        await session.delete(ad)
        await session.commit()

        assert await session.scalar(select(func.count()).select_from(Ad)) == 0
        index = await price_index(session, "rtx 3080", now=NOW)

    assert index.sample_count == 1, "the ad is gone; its price must not be"
    assert index.median_sek == 3000


async def test_observations_carry_no_personal_data():
    """Checked structurally, so it holds for data never seen."""
    forbidden = {"seller", "owner", "phone", "email", "postal", "zip",
                 "lat", "lng", "url", "description", "title"}
    columns = {c.name.lower() for c in PriceObservation.__table__.columns}

    assert not any(f in col for col in columns for f in forbidden), columns


async def test_observations_have_no_foreign_key_to_ads():
    """The structural guarantee behind the test above it: no FK means no
    cascade, so no purge can ever take an observation with it."""
    assert PriceObservation.__table__.foreign_keys == set()


# --- Valuing a build ---------------------------------------------------------


async def test_a_build_is_priced_against_the_sum_of_its_parts(session_factory, ad_factory):
    async with session_factory() as session:
        for i in range(5):
            await observe(session, "rtx 3080", 3000, ref=f"g{i}")
        for i in range(5):
            await observe(session, "ryzen 5 5600x", 1200, ref=f"c{i}",
                          ctype=ComponentTypeEnum.CPU)

        build = ad_factory("1", title="Gaming PC", price_sek=3500,
                           item_type=ItemTypeEnum.FULL_BUILD)
        for ctype, name in [(ComponentTypeEnum.GPU, "rtx 3080"),
                            (ComponentTypeEnum.CPU, "ryzen 5 5600x")]:
            build.extracted_components.append(
                ExtractedComponent(component_type=ctype, component_name=name,
                                   canonical_name=name)
            )
        session.add(build)
        await session.commit()

        valuation = await value_ad(session, build.id, now=NOW)

    assert valuation.estimated_parts_sek == 4200
    # Negative delta = cheaper than its parts = the thing worth finding.
    assert valuation.delta_sek == -700
    assert valuation.min_sample_count == 5
    # CPU and GPU of the three essential build parts; RAM was not identified.
    assert valuation.coverage == pytest.approx(2 / 3)


async def test_an_unpriceable_build_declines_to_answer(session_factory, ad_factory):
    """Summing zero known parts to 0 kr would rank an unknown machine as the
    greatest bargain ever found."""
    async with session_factory() as session:
        build = ad_factory("1", title="Gaming PC", price_sek=5000,
                           item_type=ItemTypeEnum.FULL_BUILD)
        build.extracted_components.append(
            ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                               component_name="rtx 5090", canonical_name="rtx 5090")
        )
        session.add(build)
        await session.commit()

        valuation = await value_ad(session, build.id, now=NOW)

    assert valuation.estimated_parts_sek is None
    assert valuation.delta_sek is None
    assert valuation.coverage == 0.0


async def test_min_sample_count_reports_the_weakest_link(session_factory, ad_factory):
    """A build is only as trustworthy as its worst-sampled part. Averaging would
    let one well-known GPU disguise three parts nobody has data on."""
    async with session_factory() as session:
        for i in range(20):
            await observe(session, "rtx 3080", 3000, ref=f"g{i}")
        await observe(session, "ryzen 5 5600x", 1200, ref="c0", ctype=ComponentTypeEnum.CPU)

        build = ad_factory("1", title="Gaming PC", price_sek=3500,
                           item_type=ItemTypeEnum.FULL_BUILD)
        for ctype, name in [(ComponentTypeEnum.GPU, "rtx 3080"),
                            (ComponentTypeEnum.CPU, "ryzen 5 5600x")]:
            build.extracted_components.append(
                ExtractedComponent(component_type=ctype, component_name=name,
                                   canonical_name=name)
            )
        session.add(build)
        await session.commit()

        valuation = await value_ad(session, build.id, now=NOW)

    assert valuation.min_sample_count == 1


async def test_partial_coverage_is_reported(session_factory, ad_factory):
    """Half-known builds must be distinguishable from fully-known ones."""
    async with session_factory() as session:
        for i in range(5):
            await observe(session, "rtx 3080", 3000, ref=f"g{i}")

        build = ad_factory("1", title="Gaming PC", price_sek=3500,
                           item_type=ItemTypeEnum.FULL_BUILD)
        for ctype, name in [(ComponentTypeEnum.GPU, "rtx 3080"),
                            (ComponentTypeEnum.CPU, "unknown cpu")]:
            build.extracted_components.append(
                ExtractedComponent(component_type=ctype, component_name=name,
                                   canonical_name=name)
            )
        session.add(build)
        await session.commit()

        valuation = await value_ad(session, build.id, now=NOW)

    # GPU only, of CPU/GPU/RAM. Measured against what a build contains, not
    # against what happened to be recognised in the text.
    assert valuation.coverage == pytest.approx(1 / 3)
    assert valuation.estimated_parts_sek == 3000


# --- Coverage: the difference between a number and a wrong number ------------


async def test_a_delta_from_too_few_parts_is_withheld(session_factory, ad_factory):
    """Found on real data, and it was silently poisoning the whole leaderboard.

    Every build came back "overpriced" by 2 000–4 000 kr, uniformly. Only the
    GPU had an index entry, so the CPU, RAM, storage, PSU and case were each
    counted as worth zero — a one-directional bias presented as a confident
    number, on the exact figure the ranking sorts by.
    """
    async with session_factory() as session:
        for i in range(5):
            await observe(session, "rtx 3080", 3000, ref=f"g{i}")

        build = ad_factory("1", title="Gaming PC", price_sek=7000,
                           item_type=ItemTypeEnum.FULL_BUILD)
        for ctype, name in [
            (ComponentTypeEnum.GPU, "rtx 3080"),
            (ComponentTypeEnum.CPU, "unknown cpu"),
            (ComponentTypeEnum.RAM, "unknown ram"),
            (ComponentTypeEnum.STORAGE, "unknown ssd"),
        ]:
            build.extracted_components.append(
                ExtractedComponent(component_type=ctype, component_name=name,
                                   canonical_name=name)
            )
        session.add(build)
        await session.commit()

        valuation = await value_ad(session, build.id, now=NOW)

    assert valuation.coverage == pytest.approx(1 / 3)
    assert valuation.delta_sek is None, "one part of three cannot support a verdict"
    # The partial estimate stays available for display; only the verdict is withheld.
    assert valuation.estimated_parts_sek == 3000


async def test_good_coverage_still_produces_a_delta(session_factory, ad_factory):
    async with session_factory() as session:
        for i in range(5):
            await observe(session, "rtx 3080", 3000, ref=f"g{i}")
        for i in range(5):
            await observe(session, "ryzen 5 5600x", 1200, ref=f"c{i}",
                          ctype=ComponentTypeEnum.CPU)
        for i in range(5):
            await observe(session, "16gb ddr4", 500, ref=f"r{i}",
                          ctype=ComponentTypeEnum.RAM)

        build = ad_factory("1", title="Gaming PC", price_sek=4000,
                           item_type=ItemTypeEnum.FULL_BUILD)
        for ctype, name in [(ComponentTypeEnum.GPU, "rtx 3080"),
                            (ComponentTypeEnum.CPU, "ryzen 5 5600x"),
                            (ComponentTypeEnum.RAM, "16gb ddr4")]:
            build.extracted_components.append(
                ExtractedComponent(component_type=ctype, component_name=name,
                                   canonical_name=name)
            )
        session.add(build)
        await session.commit()

        valuation = await value_ad(session, build.id, now=NOW)

    assert valuation.coverage == 1.0
    assert valuation.delta_sek == 4000 - 4700


async def test_a_build_known_only_by_its_gpu_is_not_scored(session_factory, ad_factory):
    """The bug the coverage metric was rewritten to catch.

    Measuring priced-over-*identified* gave this case 100%: one part was
    recognised, that part was priced, so it looked fully covered. Real output
    duly reported a 7 000 kr Lenovo Legion as "100% coverage, parts worth
    2 595 kr" — condemning a whole computer on the strength of its GPU.
    """
    async with session_factory() as session:
        for i in range(9):
            await observe(session, "rtx 3060", 2590, ref=f"g{i}")

        build = ad_factory("1", title="Lenovo Legion gamingdator", price_sek=7000,
                           item_type=ItemTypeEnum.FULL_BUILD)
        build.extracted_components.append(
            ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                               component_name="RTX 3060", canonical_name="rtx 3060")
        )
        session.add(build)
        await session.commit()

        valuation = await value_ad(session, build.id, now=NOW)

    # Every identified part is priced — but a PC is not just a graphics card.
    assert len(valuation.priced) == len(valuation.parts)
    assert valuation.coverage == pytest.approx(1 / 3)
    assert valuation.delta_sek is None


async def test_a_single_component_is_measured_against_itself(session_factory, ad_factory):
    """A GPU listing is fully covered by pricing the GPU. Applying the build
    rule here would withhold a verdict on every component ad in the database."""
    async with session_factory() as session:
        for i in range(5):
            await observe(session, "rtx 3060", 2590, ref=f"g{i}")

        ad = ad_factory("1", title="RTX 3060", price_sek=2000,
                        item_type=ItemTypeEnum.SINGLE_COMPONENT)
        ad.extracted_components.append(
            ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                               component_name="RTX 3060", canonical_name="rtx 3060")
        )
        session.add(ad)
        await session.commit()

        valuation = await value_ad(session, ad.id, now=NOW)

    assert valuation.coverage == 1.0
    assert valuation.delta_sek == 2000 - 2590  # a genuine bargain


async def test_a_multi_card_ad_labelled_single_component_is_not_scored(
    session_factory, ad_factory
):
    """Topped the live leaderboard with a 35 880 kr phantom bargain.

    The ad — "RTX 3070 Asus TuF, RX 6700 xt, RTX 2080..." asking 2 690 kr — was
    classified SINGLE_COMPONENT, so three cards' medians were summed against the
    price of one. Multi-item listings are common and no classifier will label
    them all correctly, so this is caught by *consistency* instead: one
    component ad, one component.
    """
    async with session_factory() as session:
        for name, price in [("rtx 3070", 2795), ("rtx 3080", 4250), ("rx 6700", 2500)]:
            for i in range(5):
                await observe(session, name, price, ref=f"{name}{i}")

        ad = ad_factory("1", title="RTX 3070 Asus TuF, RX 6700 xt, RTX 3080",
                        price_sek=2690, item_type=ItemTypeEnum.SINGLE_COMPONENT)
        for name in ("rtx 3070", "rtx 3080", "rx 6700"):
            ad.extracted_components.append(
                ExtractedComponent(component_type=ComponentTypeEnum.GPU,
                                   component_name=name, canonical_name=name)
            )
        session.add(ad)
        await session.commit()

        valuation = await value_ad(session, ad.id, now=NOW)

    assert valuation.coverage == 0.0
    assert valuation.delta_sek is None, "a 35 880 kr phantom bargain must not rank"


class TestImplausiblePricing:
    """The failure mode a deal-finder is most exposed to.

    What the tool searches for and what characterises a fraudulent listing are
    the same observable: an unusually low price. So the top of the leaderboard
    is precisely where bait lands, and "cheapest" being the selection criterion
    is what puts it there.

    Found in live output: "High-End Gaming: i9-13900 | RTX 4070 Ti | 64GB DDR5 |
    4TB NVMe" asking **395 kr**. Every spec parsed cleanly and no risk language
    appeared anywhere, so the deterministic path settled it without a model call
    and it was published as the cheapest qualifying AI workstation on the
    market. A fraudulent advert is written cleanly on purpose, which is exactly
    why a text-only risk check cannot catch this one.
    """

    async def test_a_price_far_below_the_parts_is_flagged(self, session_factory):
        async with session_factory() as session:
            await _seed_observations(session, "rtx 4070 ti", [5000] * 6)
            ad = await _seed_build(session, price_sek=395, parts=[("GPU", "rtx 4070 ti")])
            valuation = await value_ad(session, ad)

        assert valuation.is_implausibly_cheap

    async def test_a_genuine_bargain_is_not_flagged(self, session_factory):
        """The threshold is an absurdity filter, not a deal filter.

        Real listings do go 40% under the index — finding them is the entire
        point of the project — so the floor sits far below any honest sale
        rather than anywhere near the interesting range.
        """
        async with session_factory() as session:
            await _seed_observations(session, "rtx 4070 ti", [5000] * 6)
            ad = await _seed_build(session, price_sek=2900, parts=[("GPU", "rtx 4070 ti")])
            valuation = await value_ad(session, ad)

        # 2 900 against a 5 000 kr part is a 42% discount — squarely in the
        # range this project exists to surface, and far clear of the floor.
        assert not valuation.is_implausibly_cheap

    async def test_two_observations_are_enough_to_reject_a_16x_gap(
        self, session_factory
    ):
        """The evidence bar here is lower than the one for publishing a number.

        Reusing `MIN_CONFIDENT_SAMPLES` was the first attempt, and it failed on
        the exact listing that motivated the check: the real 395 kr ad had n=2
        on its graphics card, so a 16× discrepancy was suppressed on the grounds
        that two observations are not many. Two observations are ample to
        establish that an RTX 4070 Ti does not sell for 395 kr.
        """
        async with session_factory() as session:
            await _seed_observations(session, "rtx 4070 ti", [5000, 5000])
            ad = await _seed_build(session, price_sek=395, parts=[("GPU", "rtx 4070 ti")])
            valuation = await value_ad(session, ad)

        assert valuation.is_implausibly_cheap

    async def test_a_single_observation_does_not_accuse(self, session_factory):
        """One sighting is an anecdote. It could be the mispriced one."""
        async with session_factory() as session:
            await _seed_observations(session, "rtx 4070 ti", [5000])
            ad = await _seed_build(session, price_sek=395, parts=[("GPU", "rtx 4070 ti")])
            valuation = await value_ad(session, ad)

        assert not valuation.is_implausibly_cheap

    async def test_no_estimate_means_no_accusation(self, session_factory):
        async with session_factory() as session:
            ad = await _seed_build(session, price_sek=395, parts=[("GPU", "rtx 4070 ti")])
            valuation = await value_ad(session, ad)

        assert not valuation.is_implausibly_cheap


class TestInventoryListings:
    """An advert can name many machines while claiming to sell one.

    Live example, top of the leaderboard at 999 kr: "RTX, GTX, Intel I7, Intel
    i5, RX 7900 GRE, Ryzen, 32gb ram", described as "Bud på kompletta datorer".
    Six graphics cards and five processors, summed against a single asking
    price. Same error as the multi-card SINGLE_COMPONENT case, one level up:
    there it was several parts in an ad selling one part, here it is several
    machines in an ad selling one machine.
    """

    async def test_a_stock_list_is_not_scored(self, session_factory):
        async with session_factory() as session:
            await _seed_observations(session, "rtx 3060 ti", [2500] * 6)
            ad = await _seed_build(
                session,
                price_sek=999,
                parts=[
                    ("GPU", "rx 7900 xt"),
                    ("GPU", "rx 7900 xtx"),
                    ("GPU", "rtx 3060 ti"),
                    ("GPU", "gtx 1080"),
                    ("CPU", "i7-4790k"),
                    ("CPU", "i5-4690k"),
                    ("CPU", "ryzen 7 3700x"),
                ],
            )
            valuation = await value_ad(session, ad)

        assert valuation.is_inventory_listing
        assert valuation.coverage == 0.0
        assert valuation.delta_sek is None

    async def test_a_dual_gpu_machine_is_still_a_machine(self, session_factory):
        """Two of a kind is unusual but real; three is a catalogue.

        Sellers also legitimately write "passar i5 eller i7", so the threshold
        has to tolerate two without tolerating six.
        """
        async with session_factory() as session:
            await _seed_observations(session, "rtx 3080", [4000] * 6)
            await _seed_observations(
                session, "32gb ddr4", [800] * 6, ctype=ComponentTypeEnum.RAM
            )
            ad = await _seed_build(
                session,
                price_sek=9000,
                parts=[
                    ("GPU", "rtx 3080"),
                    ("GPU", "rtx 3070"),
                    ("CPU", "i7-12700k"),
                    ("RAM", "32gb ddr4"),
                ],
            )
            valuation = await value_ad(session, ad)

        assert not valuation.is_inventory_listing


class TestExclusiveSlotsAreNotDoubleCounted:
    """Regression: the largest fake bargain on the first published leaderboard.

    *"Gaming Dator Ryzen 5 9600x/RTX 5070ti/32GB DDR5"* at 20 000 kr. The card
    was extracted twice, so 11 000 kr of graphics card was counted as 22 000,
    and the machine was published as **13 675 kr below market** — comfortably
    the biggest apparent saving in the country.

    The existing inventory guard did not fire and should not have: it counts
    *distinct* model names, and there was only one. Duplication and enumeration
    are different failures needing different answers.
    """

    def test_a_repeated_card_is_counted_once(self):
        gpu = PartValuation(
            component_type=ComponentTypeEnum.GPU,
            canonical_name="rtx 5070 ti",
            index=PriceIndex("rtx 5070 ti", 11000, 6),
        )
        valuation = BuildValuation(
            ad_id=1, listed_price_sek=20000, parts=(gpu, gpu),
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        assert valuation.estimated_parts_sek == 11000

    def test_two_different_cards_resolve_to_the_cheaper(self):
        """We cannot tell which card is in the box, and the errors are not
        symmetrical: too low understates a real bargain, too high invents one
        and puts it at the top of the page.
        """
        parts = (
            PartValuation(ComponentTypeEnum.GPU, "rtx 4070", PriceIndex("rtx 4070", 6000, 6)),
            PartValuation(ComponentTypeEnum.GPU, "rx 6950 xt", PriceIndex("rx 6950 xt", 4000, 6)),
        )
        valuation = BuildValuation(
            ad_id=1, listed_price_sek=9000, parts=parts,
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        assert valuation.estimated_parts_sek == 4000

    def test_a_priced_candidate_beats_an_unpriced_one(self):
        """Choosing the unpriced part would drop the slot out of the estimate
        entirely and lower coverage — worse than using the number we have."""
        parts = (
            PartValuation(ComponentTypeEnum.GPU, "mystery", PriceIndex("mystery", None, 0)),
            PartValuation(ComponentTypeEnum.GPU, "rtx 4070", PriceIndex("rtx 4070", 6000, 6)),
        )
        valuation = BuildValuation(ad_id=1, listed_price_sek=9000, parts=parts)
        assert valuation.estimated_parts_sek == 6000

    def test_non_exclusive_slots_still_add_up(self):
        """Two sticks of RAM really are more RAM. Only CPU and GPU collapse."""
        stick = PartValuation(
            ComponentTypeEnum.RAM, "16gb ddr4", PriceIndex("16gb ddr4", 500, 8)
        )
        valuation = BuildValuation(ad_id=1, listed_price_sek=1200, parts=(stick, stick))
        assert valuation.estimated_parts_sek == 1000

    def test_the_inventory_guard_still_sees_every_card(self):
        """`is_inventory_listing` must read the raw parts, not the collapsed
        view — it needs all six cards to notice it is looking at a stock list."""
        parts = tuple(
            PartValuation(ComponentTypeEnum.GPU, name, PriceIndex(name, 3000, 6))
            for name in ("rtx 3060", "rtx 3070", "rtx 3080", "rx 580", "gtx 1660")
        )
        valuation = BuildValuation(ad_id=1, listed_price_sek=999, parts=parts)
        assert valuation.is_inventory_listing is True
