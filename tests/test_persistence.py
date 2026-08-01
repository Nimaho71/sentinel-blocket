"""Persistence and retention tests.

The retention design rests on one claim: deleting an ad is safe because the
value it contributed does not live in the ad row. These tests hold the pieces of
that claim that exist today — that descriptions are dropped, that dropping is
not undone by a re-scrape, and that a purge cascades cleanly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from db.crud import (
    content_hash,
    is_wanted_ad,
    drop_stale_descriptions,
    known_external_ids,
    upsert_ad,
)
from db.models import (
    Ad,
    AiEvaluation,
    CategoryEnum,
    ExtractedComponent,
    ItemTypeEnum,
    StatusEnum,
)
from pipeline.persist import persist_records
from sources.base import AdRecord


def record(external_id="1", *, title="RTX 3080", price=3000, description="Fint skick"):
    return AdRecord(
        source="blocket",
        external_id=external_id,
        url=f"https://blocket.se/annons/{external_id}",
        title=title,
        description=description,
        price_sek=price,
        attributes={"condition": "Bra skick"},
        city="Göteborg",
        image_count=3,
    )


async def test_records_are_written_with_no_personal_data_columns():
    """A structural guarantee, checked structurally.

    `Ad` has no column for a phone number, an email, a postal code or a
    coordinate — so no bug downstream can put one there. Asserting on the schema
    rather than on a sample row is what makes this hold for data we never saw.
    """
    forbidden = {"phone", "email", "seller", "owner", "postal", "zip", "lat", "lng",
                 "latitude", "longitude", "position", "coord"}
    columns = {c.name.lower() for c in Ad.__table__.columns}

    assert not any(f in col for col in columns for f in forbidden), columns


async def test_persist_writes_and_is_idempotent(session_factory):
    stats = await persist_records(session_factory, [record("1"), record("2")])
    assert stats.written == 2

    await persist_records(session_factory, [record("1"), record("2")])

    async with session_factory() as session:
        count = await session.scalar(select(func.count()).select_from(Ad))
    assert count == 2, "re-ingesting the same ads must not duplicate them"


async def test_upsert_refreshes_price_and_hash(session_factory):
    await persist_records(session_factory, [record("1", price=3000)])
    await persist_records(session_factory, [record("1", price=2500)])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()

    assert ad.price_sek == 2500
    # The hash must move with the price, or the AI phase will never revisit it.
    assert ad.content_hash == content_hash(ad.title, "Fint skick", 2500)


async def test_a_graphics_card_is_not_filed_as_a_whole_computer(session_factory):
    """Regression, caught on live data.

    The category paths below are verbatim from Blocket's API. An earlier
    heuristic tested `"dator" in path`, which is true of "Datorkomponenter" too,
    so 51 of 52 graphics cards were filed as complete machines — and a 900 kr
    GPU filed as a build would be valued against the price of a whole PC.
    """
    gpu = record("1")
    object.__setattr__(
        gpu, "category_path", ("Elektronik och vitvaror", "Datorer", "Datorkomponenter")
    )
    pc = record("2")
    object.__setattr__(
        pc, "category_path", ("Elektronik och vitvaror", "Datorer", "Stationär dator")
    )

    await persist_records(session_factory, [gpu, pc])

    async with session_factory() as session:
        by_id = {
            ad.external_id: ad.category
            for ad in (await session.execute(select(Ad))).scalars().all()
        }

    assert by_id["1"] is CategoryEnum.COMPONENT
    assert by_id["2"] is CategoryEnum.STATIONARY


async def test_an_unknown_category_defaults_to_component(session_factory):
    """The safe direction: mistaking a build for a part understates its value,
    mistaking a part for a build invents value that isn't there."""
    unknown = record("1")
    object.__setattr__(unknown, "category_path", ("Something", "Unmapped"))

    await persist_records(session_factory, [unknown])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
    assert ad.category is CategoryEnum.COMPONENT


async def test_condition_comes_from_the_source_not_an_llm(session_factory):
    """Blocket publishes this as a structured attribute. It is free."""
    await persist_records(session_factory, [record("1")])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()

    assert ad.condition == "Bra skick"


async def test_seeing_an_ad_again_revives_a_wrongly_sold_row(session_factory):
    """The janitor can mark an ad SOLD on a bad network day. Search proves otherwise."""
    await persist_records(session_factory, [record("1")])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
        ad.status = StatusEnum.SOLD
        await session.commit()

    await persist_records(session_factory, [record("1")])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
    assert ad.status is StatusEnum.ACTIVE


async def test_a_bad_batch_does_not_roll_back_a_good_one(session_factory):
    """Batch isolation, the same property the evaluation stage needs."""
    bad = record("2")
    object.__setattr__(bad, "url", None)  # violates NOT NULL

    stats = await persist_records(session_factory, [record("1"), bad, record("3")], batch_size=1)

    assert stats.written == 2
    assert stats.batches_failed == 1


async def test_known_ids_are_fetched_in_one_query(session_factory):
    await persist_records(session_factory, [record("1"), record("2")])

    async with session_factory() as session:
        assert await known_external_ids(session, "blocket") == {"1", "2"}
        assert await known_external_ids(session, "tradera") == set()


async def test_descriptions_are_dropped_after_extraction(session_factory):
    """GDPR data minimisation: the free text is the field that holds a phone
    number a seller typed into the body, and the specs are already extracted."""
    await persist_records(session_factory, [record("1")])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
        ad.scraped_at = datetime.now(timezone.utc) - timedelta(days=30)
        session.add(
            AiEvaluation(
                ad_id=ad.id,
                item_type=ItemTypeEnum.SINGLE_COMPONENT,
                confidence=0.9,
                ai_reasoning="Fine.",
                content_hash=ad.content_hash,
            )
        )
        await session.commit()

        dropped = await drop_stale_descriptions(session, older_than_days=7)
        await session.commit()

        ad = (await session.execute(select(Ad))).scalars().one()

    assert dropped == 1
    assert ad.raw_description is None
    assert ad.description_dropped_at is not None


async def test_unevaluated_descriptions_are_kept(session_factory):
    """Dropping text the extractor never read would destroy the ad for nothing."""
    await persist_records(session_factory, [record("1")])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
        ad.scraped_at = datetime.now(timezone.utc) - timedelta(days=30)
        await session.commit()

        assert await drop_stale_descriptions(session, older_than_days=7) == 0


async def test_a_rescrape_does_not_resurrect_a_dropped_description(session_factory):
    """Retention that a routine re-scrape silently undoes is not retention."""
    await persist_records(session_factory, [record("1")])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
        ad.raw_description = None
        ad.description_dropped_at = datetime.now(timezone.utc)
        await session.commit()

    await persist_records(session_factory, [record("1")])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
    assert ad.raw_description is None


async def test_deleting_an_ad_cascades_to_its_children(session_factory):
    """Retention purges rows; orphaned evaluations would corrupt every count."""
    await persist_records(session_factory, [record("1")])

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
        session.add_all(
            [
                AiEvaluation(
                    ad_id=ad.id,
                    item_type=ItemTypeEnum.SINGLE_COMPONENT,
                    confidence=1.0,
                    ai_reasoning="x",
                    content_hash=ad.content_hash,
                ),
                ExtractedComponent(
                    ad_id=ad.id,
                    component_type="GPU",
                    component_name="RTX 3080",
                    canonical_name="rtx 3080",
                ),
            ]
        )
        await session.commit()

        await session.delete(ad)
        await session.commit()

        assert await session.scalar(select(func.count()).select_from(AiEvaluation)) == 0
        assert await session.scalar(select(func.count()).select_from(ExtractedComponent)) == 0


class TestWantedAdsAreExcludedAtIngest:
    """Regression: "KÖPER RTX 3090 — 8 000 kr", published as a bargain.

    Blocket files want-to-buy ads in the same categories as items for sale. The
    price on one means the opposite of a sale price — it is what a stranger is
    *offering* — so it is systematically low, which is exactly what puts it near
    the top of a deal leaderboard.

    Marked at ingest rather than filtered downstream. Every later query already
    excludes non-ACTIVE ads, so one assignment covers valuation, the index,
    basket mode and the publisher at once. The alternative was a fifth place to
    remember a guard, which is the shape of bug this project keeps finding.
    """

    @pytest.mark.parametrize(
        "title",
        [
            "KÖPER RTX 3090",
            "köpers stationär dator till roblox",
            "Köpes: gaming dator",
            "Sökes RTX 4070",
            "WANTED: RTX 3080",
            "Letar efter en speldator",
        ],
    )
    def test_wanted_titles_are_recognised(self, title):
        assert is_wanted_ad(title) is True

    @pytest.mark.parametrize(
        "title",
        [
            "Gaming PC RTX 3060 12GB",
            "Säljer min speldator",
            "RTX 3080 till salu",
            # The trap: past tense in a genuine sale. Matching the description
            # instead of the title flags a large share of honest listings, since
            # "jag köpte den 2021" is extremely common phrasing.
            "RTX 3080 som jag köpte 2021",
        ],
    )
    def test_genuine_sales_are_not_flagged(self, title):
        assert is_wanted_ad(title) is False

    async def test_ingest_stores_a_wanted_ad_as_ignored(self, session_factory):
        """Stored rather than discarded, so a re-scrape does not keep
        re-fetching it and the decision stays inspectable."""
        async with session_factory() as session:
            await upsert_ad(session, record("w", title="KÖPER RTX 3090", price=8000))
            await upsert_ad(session, record("s", title="Säljer RTX 3090", price=9000))
            await session.commit()

        async with session_factory() as session:
            by_id = {
                a.external_id: a
                for a in (await session.execute(select(Ad))).scalars().all()
            }
            assert by_id["w"].status is StatusEnum.IGNORED
            assert by_id["s"].status is StatusEnum.ACTIVE

    async def test_a_rescrape_does_not_reactivate_a_wanted_ad(self, session_factory):
        """The refresh path used to set ACTIVE unconditionally, on the reasoning
        that seeing an ad in search results proves it is live. It does — and it
        is still not a thing for sale."""
        async with session_factory() as session:
            await upsert_ad(session, record("w", title="KÖPER RTX 3090", price=8000))
            await session.commit()
        async with session_factory() as session:
            await upsert_ad(session, record("w", title="KÖPER RTX 3090", price=8500))
            await session.commit()

        async with session_factory() as session:
            ad = (await session.execute(select(Ad))).scalars().one()
            assert ad.price_sek == 8500
            assert ad.status is StatusEnum.IGNORED
