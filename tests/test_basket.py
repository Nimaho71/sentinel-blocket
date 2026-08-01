"""Tests for Basket Mode.

Two properties matter more than the arithmetic. First, the basket must be
priced from listings a person could actually go and buy — not from index
medians, and not from parts that are only available as half of a whole
computer. Second, an incomplete basket must never be compared against a
complete machine, because that comparison always favours building and always
for the wrong reason.
"""

from __future__ import annotations

import pytest
from sqlalchemy import update

from db.models import (
    Ad,
    AiEvaluation,
    ComponentTypeEnum as CT,
    ExtractedComponent,
    ItemTypeEnum,
    StatusEnum,
)
from pipeline.basket import CORE_SLOTS, build_or_buy, cheapest_basket
from pipeline.profiles import AI_WORKSTATION, GAMING, HOME_SERVER


async def _add(session, ad_factory, external_id, *, price, item_type, parts, **kwargs):
    ad = ad_factory(external_id, price_sek=price, item_type=item_type, **kwargs)
    session.add(ad)
    await session.flush()
    for component_type, canonical in parts:
        session.add(
            ExtractedComponent(
                ad_id=ad.id,
                component_type=component_type,
                component_name=canonical.upper(),
                canonical_name=canonical,
            )
        )
    await session.commit()
    return ad


@pytest.fixture
async def market(session_factory, ad_factory):
    """A small live market: three GPUs, two CPUs, RAM, a drive, one prebuilt."""
    async with session_factory() as session:
        await _add(
            session, ad_factory, "gpu-cheap-small",
            price=2400, item_type=ItemTypeEnum.SINGLE_COMPONENT,
            parts=[(CT.GPU, "rtx 3070")], title="RTX 3070 8GB",
        )
        await _add(
            session, ad_factory, "gpu-qualifying",
            price=3200, item_type=ItemTypeEnum.SINGLE_COMPONENT,
            parts=[(CT.GPU, "rtx 3060")], title="RTX 3060 12GB",
        )
        await _add(
            session, ad_factory, "gpu-dear",
            price=9000, item_type=ItemTypeEnum.SINGLE_COMPONENT,
            parts=[(CT.GPU, "rtx 3090")], title="RTX 3090",
        )
        await _add(
            session, ad_factory, "cpu",
            price=900, item_type=ItemTypeEnum.SINGLE_COMPONENT,
            parts=[(CT.CPU, "ryzen 5 5600")], title="Ryzen 5 5600",
        )
        await _add(
            session, ad_factory, "ram",
            price=600, item_type=ItemTypeEnum.SINGLE_COMPONENT,
            parts=[(CT.RAM, "32gb ddr4")], title="32GB DDR4",
        )
        await _add(
            session, ad_factory, "ssd",
            price=500, item_type=ItemTypeEnum.SINGLE_COMPONENT,
            parts=[(CT.STORAGE, "1tb nvme")], title="1TB NVMe",
        )
    return session_factory


class TestCheapestBasket:
    async def test_picks_the_cheapest_qualifying_part_per_slot(self, market):
        async with market() as session:
            basket = await cheapest_basket(session, AI_WORKSTATION)

        chosen = {item.slot: item for item in basket.items}
        # Not the 2 400 kr 3070 — it fails the 12 GB floor. Not the 9 000 kr
        # 3090 — it qualifies but costs more.
        assert chosen[CT.GPU].price_sek == 3200
        assert chosen[CT.CPU].price_sek == 900
        assert basket.total_sek == 3200 + 900 + 600 + 500
        assert basket.seller_count == 4

    async def test_a_different_profile_picks_a_different_card(self, market):
        """Same market, same query, different answer — profiles doing their job."""
        async with market() as session:
            basket = await cheapest_basket(session, GAMING)

        chosen = {item.slot: item for item in basket.items}
        assert chosen[CT.GPU].price_sek == 2400  # 8 GB clears the gaming floor

    async def test_reports_the_slots_it_cannot_fill(self, market, ad_factory):
        async with market() as session:
            basket = await cheapest_basket(session, AI_WORKSTATION)
        assert basket.missing_slots == ()
        assert basket.is_complete

        async with market() as session:
            await session.execute(
                update(Ad).where(Ad.external_id == "cpu").values(status=StatusEnum.SOLD)
            )
            await session.commit()
            basket = await cheapest_basket(session, AI_WORKSTATION)
        assert CT.CPU in basket.missing_slots
        assert not basket.is_complete

    async def test_names_the_slots_it_never_tries_to_fill(self, market):
        """Motherboard, PSU and case are excluded — and said to be excluded.

        A basket total that silently omits three real costs tilts every
        build-versus-buy comparison toward building.
        """
        async with market() as session:
            basket = await cheapest_basket(session, AI_WORKSTATION)

        assert CT.MOTHERBOARD in basket.excluded_slots
        assert CT.PSU in basket.excluded_slots
        assert CT.CASE in basket.excluded_slots
        assert "motherboard" in basket.describe()

    async def test_ignores_parts_that_are_only_part_of_a_whole_computer(
        self, market, ad_factory
    ):
        """A GPU inside a prebuilt is not for sale on its own.

        Without this, a 4 000 kr complete PC containing an RTX 3060 would offer
        its graphics card to the basket at 4 000 kr — or worse, a cheap PC would
        undercut every standalone card and the basket would "buy" the same
        machine four times over.
        """
        async with market() as session:
            await _add(
                session, ad_factory, "prebuilt-with-3060",
                price=1000, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[(CT.GPU, "rtx 3060"), (CT.CPU, "i5-12400")],
                title="Komplett dator RTX 3060 12GB",
            )
            basket = await cheapest_basket(session, AI_WORKSTATION)

        chosen = {item.slot: item for item in basket.items}
        assert chosen[CT.GPU].price_sek == 3200
        assert chosen[CT.CPU].price_sek == 900

    async def test_ignores_multi_part_single_component_ads(self, market, ad_factory):
        """The 35 880 kr phantom bargain, in basket form.

        A listing selling three cards for one price cannot contribute any of
        them at that price. The market index refuses these on the same grounds.
        """
        async with market() as session:
            await _add(
                session, ad_factory, "three-cards",
                price=2000, item_type=ItemTypeEnum.SINGLE_COMPONENT,
                parts=[(CT.GPU, "rtx 3060"), (CT.GPU, "rtx 3090"), (CT.GPU, "rx 6800")],
                title="RTX 3060, RTX 3090, RX 6800",
            )
            basket = await cheapest_basket(session, AI_WORKSTATION)

        chosen = {item.slot: item for item in basket.items}
        assert chosen[CT.GPU].price_sek == 3200

    async def test_respects_a_budget_ceiling(self, market):
        async with market() as session:
            basket = await cheapest_basket(session, AI_WORKSTATION, max_price_sek=1000)

        assert all(item.price_sek <= 1000 for item in basket.items)
        assert CT.GPU in basket.missing_slots

    async def test_empty_market_is_empty_not_an_error(self, session_factory):
        async with session_factory() as session:
            basket = await cheapest_basket(session, AI_WORKSTATION)
        assert basket.items == ()
        assert basket.total_sek == 0
        assert "no qualifying parts" in basket.describe()

    async def test_only_core_slots_are_considered(self, market):
        async with market() as session:
            basket = await cheapest_basket(session, AI_WORKSTATION)
        assert all(item.slot in CORE_SLOTS for item in basket.items)


class TestBuildOrBuy:
    async def test_compares_against_the_cheapest_qualifying_prebuilt(
        self, market, ad_factory
    ):
        async with market() as session:
            await _add(
                session, ad_factory, "prebuilt-good",
                price=7200, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[
                    (CT.GPU, "rtx 3060"),
                    (CT.CPU, "i5-12400"),
                    (CT.RAM, "32gb ddr4"),
                ],
                title="Speldator RTX 3060 12GB 32GB RAM",
            )
            result = await build_or_buy(session, AI_WORKSTATION)

        assert result.best_prebuilt_price_sek == 7200
        assert result.saving_sek == 7200 - result.basket.total_sek
        assert "building saves" in result.describe()

    async def test_prebuilt_is_ranked_by_effective_price_not_sticker_price(
        self, market, ad_factory
    ):
        """A cheaper box that needs 800 kr of RAM is not the cheaper box."""
        async with market() as session:
            await _add(
                session, ad_factory, "cheap-but-16gb",
                price=6400, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[(CT.GPU, "rtx 3060"), (CT.RAM, "16gb ddr4")],
                title="Dator RTX 3060 12GB 16GB RAM",
            )
            await _add(
                session, ad_factory, "dearer-but-correct",
                price=6800, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[(CT.GPU, "rtx 3060"), (CT.RAM, "32gb ddr4")],
                title="Dator RTX 3060 12GB 32GB RAM",
            )
            result = await build_or_buy(session, AI_WORKSTATION)

        assert result.best_prebuilt_price_sek == 6800

    async def test_unfit_prebuilts_are_not_offered(self, market, ad_factory):
        async with market() as session:
            await _add(
                session, ad_factory, "prebuilt-8gb-card",
                price=4000, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[(CT.GPU, "rtx 3070"), (CT.RAM, "32gb ddr4")],
                title="Speldator RTX 3070",
            )
            result = await build_or_buy(session, AI_WORKSTATION)

        assert result.best_prebuilt_price_sek is None
        assert "No qualifying prebuilt" in result.describe()

    async def test_incomplete_basket_reports_no_saving(self, market, ad_factory):
        """The comparison that is always favourable and always meaningless.

        With no CPU currently listed the basket is a heap of parts, not a
        machine. It is necessarily cheaper than a whole computer, and quoting
        that difference as a saving would recommend building on the strength of
        the very thing that makes building impossible today.
        """
        async with market() as session:
            await _add(
                session, ad_factory, "prebuilt-good",
                price=7200, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[
                    (CT.GPU, "rtx 3060"),
                    (CT.CPU, "i5-12400"),
                    (CT.RAM, "32gb ddr4"),
                ],
                title="Speldator RTX 3060 12GB 32GB RAM",
            )
            await session.execute(
                update(Ad).where(Ad.external_id == "cpu").values(status=StatusEnum.SOLD)
            )
            await session.commit()
            result = await build_or_buy(session, AI_WORKSTATION)

        assert not result.basket.is_complete
        assert result.best_prebuilt_price_sek == 7200
        assert result.saving_sek is None
        assert "not comparable" in result.describe()


class TestBrokenMachinesAreNotBaselines:
    """Ad 13, from the published snapshot.

    *"DEFEKT Acer Predator Triton 300 SE (PT314-51s) – RTX 3060 / i7 / 144Hz –
    Defekt"*, 2 999 kr, Hisings Backa. Body: "i helt defekt skick som
    reservdelsobjekt", "helt stendöd och får ingen ström alls".

    It was `bestPrebuiltId` for `gaming`, so the site published a 2 300 kr
    penalty for building from working parts — measured against a laptop that
    does not turn on. Nothing here was arithmetically wrong; the comparison was
    simply against a thing that is not a computer.
    """

    _TITLE = (
        "DEFEKT Acer Predator Triton 300 SE (PT314-51s) "
        "– RTX 3060 / i7 / 144Hz – Defekt"
    )
    _BODY = (
        "Säljer en Acer Predator Triton 300 SE (modell PT314-51s-721G) i helt "
        "defekt skick som reservdelsobjekt eller reparationsprojekt. Datorn är "
        "helt stendöd och får ingen ström alls."
    )
    _PARTS = [(CT.GPU, "rtx 3060"), (CT.CPU, "i7-11370h"), (CT.RAM, "16gb ddr4")]

    async def test_the_defekt_acer_is_not_the_best_prebuilt(self, market, ad_factory):
        async with market() as session:
            broken = await _add(
                session, ad_factory, "defekt-acer",
                price=2999, item_type=ItemTypeEnum.FULL_BUILD,
                parts=self._PARTS, title=self._TITLE, description=self._BODY,
            )
            result = await build_or_buy(session, GAMING)

        assert result.best_prebuilt_ad_id != broken.id
        assert result.best_prebuilt_price_sek is None
        assert result.damaged_candidates == 1
        assert "stated as damaged" in result.describe()

    async def test_the_same_machine_working_would_have_won(self, market, ad_factory):
        """The control, without which the test above passes for any reason.

        Identical price and identical parts, with the word removed: it becomes
        the best prebuilt. So it is the stated damage that disqualifies it, not
        the profile, the price or the specs.
        """
        async with market() as session:
            working = await _add(
                session, ad_factory, "working-acer",
                price=2999, item_type=ItemTypeEnum.FULL_BUILD,
                parts=self._PARTS,
                title="Acer Predator Triton 300 SE (PT314-51s) – RTX 3060 / i7 / 144Hz",
                description="Välvårdad, fungerar utmärkt. Laddare medföljer.",
            )
            result = await build_or_buy(session, GAMING)

        assert result.best_prebuilt_ad_id == working.id
        assert result.damaged_candidates == 0

    async def test_damage_stated_only_in_the_body_also_disqualifies(
        self, market, ad_factory
    ):
        """Neither field is privileged: sellers put it in whichever they like."""
        async with market() as session:
            await _add(
                session, ad_factory, "quiet-title",
                price=2999, item_type=ItemTypeEnum.FULL_BUILD,
                parts=self._PARTS,
                title="Acer Predator Triton 300 SE – RTX 3060 / i7 / 144Hz",
                description="Obs: säljes för delar, startar inte längre.",
            )
            result = await build_or_buy(session, GAMING)

        assert result.best_prebuilt_price_sek is None
        assert result.damaged_candidates == 1

    async def test_the_models_own_flag_is_enough(self, market, ad_factory):
        """The description is dropped by the janitor after 30 days; flags persist.

        An ad whose body has been purged must still be disqualified by the
        verdict the model formed while it could still read it.
        """
        async with market() as session:
            broken = await _add(
                session, ad_factory, "purged-body",
                price=2999, item_type=ItemTypeEnum.FULL_BUILD,
                parts=self._PARTS,
                title="Acer Predator Triton 300 SE – RTX 3060 / i7 / 144Hz",
                description="",
            )
            session.add(AiEvaluation(
                ad_id=broken.id, item_type=broken.item_type, confidence=0.1,
                risk_flags="defekt\nsold as-is",
                ai_reasoning="The laptop is explicitly stated as broken.",
                content_hash=broken.content_hash, model="test",
            ))
            await session.commit()
            result = await build_or_buy(session, GAMING)

        assert result.best_prebuilt_price_sek is None
        assert result.damaged_candidates == 1


class TestStrictnessWhenAsserting:
    """Regressions from live output. Both were invisible to every other test.

    They share a root cause: `unknown` is the right answer to "should this be
    shown to a shopper?" and the wrong answer to "does this satisfy the
    profile?". The first question is permissive by design, because most ads
    state the graphics card and nothing else. The second is a claim, and a claim
    needs evidence.
    """

    async def test_a_part_whose_spec_cannot_be_read_is_not_basket_material(
        self, market, ad_factory
    ):
        """Live output picked "server ram ddr4 ecc" for a 32 GB slot.

        No capacity parses out of that name, so the requirement returned
        `unknown`, so it passed — and the basket asserted a 32 GB machine on the
        strength of a string it could not read. It was also ECC server memory,
        which would not have run in the machine anyway.
        """
        async with market() as session:
            await _add(
                session, ad_factory, "unreadable-ram",
                price=100, item_type=ItemTypeEnum.SINGLE_COMPONENT,
                parts=[(CT.RAM, "server ram ddr4 ecc")],
                title="Server RAM DDR4 ECC",
            )
            basket = await cheapest_basket(session, AI_WORKSTATION)

        chosen = {item.slot: item for item in basket.items}
        # Not the 100 kr mystery sticks — the 600 kr 32 GB kit we can verify.
        assert chosen[CT.RAM].price_sek == 600
        assert chosen[CT.RAM].canonical_name == "32gb ddr4"

    async def test_a_vaguely_described_prebuilt_is_not_the_best_prebuilt(
        self, market, ad_factory
    ):
        """Live output announced a 300 kr listing as the cheapest AI workstation.

        Neither its GPU nor its RAM could be read, so both requirements came
        back `unknown`, so it counted as viable and won on price. Viability is
        the leaderboard's test; the headline needs a machine we can vouch for.
        """
        async with market() as session:
            await _add(
                session, ad_factory, "mystery-box",
                price=300, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[(CT.CPU, "i5-4590")],
                title="Dator säljes",
            )
            await _add(
                session, ad_factory, "known-good",
                price=7200, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[
                    (CT.GPU, "rtx 3060"),
                    (CT.CPU, "i5-12400"),
                    (CT.RAM, "32gb ddr4"),
                ],
                title="Speldator RTX 3060 12GB 32GB RAM",
            )
            result = await build_or_buy(session, AI_WORKSTATION)

        assert result.best_prebuilt_price_sek == 7200
        assert result.unverified_candidates >= 1

    async def test_the_vague_ones_are_counted_rather_than_silently_dropped(
        self, market, ad_factory
    ):
        """"Nothing qualifies" and "nothing said enough to check" differ.

        Only the first is a fact about the market. Reporting the second as the
        first would suggest a gap in supply where there is really a gap in the
        listings' descriptions.
        """
        async with market() as session:
            await _add(
                session, ad_factory, "mystery-box",
                price=300, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[(CT.CPU, "i5-4590")],
                title="Dator säljes",
            )
            result = await build_or_buy(session, AI_WORKSTATION)

        assert result.best_prebuilt_price_sek is None
        assert result.unverified_candidates == 1
        assert "too vaguely described" in result.describe()

    async def test_a_fixable_shortfall_still_qualifies_as_best_prebuilt(
        self, market, ad_factory
    ):
        """Strictness applies to *unknown*, not to *upgradeable*.

        A machine with 16 GB of RAM is fully described — we know exactly what it
        has and exactly what the fix costs. That belongs in the comparison, at
        its effective price.
        """
        async with market() as session:
            await _add(
                session, ad_factory, "needs-ram",
                price=6000, item_type=ItemTypeEnum.FULL_BUILD,
                parts=[(CT.GPU, "rtx 3060"), (CT.RAM, "16gb ddr4")],
                title="Dator RTX 3060 12GB 16GB RAM",
            )
            result = await build_or_buy(session, AI_WORKSTATION)

        assert result.best_prebuilt_price_sek == 6000 + 800
        assert result.unverified_candidates == 0


class TestVacuousQualification:
    async def test_a_slot_with_no_requirement_still_needs_a_readable_part(
        self, market, ad_factory
    ):
        """`all()` over an empty set is True, and that was a hole.

        `home-server` has no RAM floor, so once `confirms_part` was tightened it
        *still* accepted "server ram ddr4 ecc" — zero requirements about the RAM
        slot, so zero of them failed. The basket quoted 499 kr for a stick whose
        capacity nobody could read.
        """
        async with market() as session:
            await _add(
                session, ad_factory, "unreadable-ram",
                price=100, item_type=ItemTypeEnum.SINGLE_COMPONENT,
                parts=[(CT.RAM, "server ram ddr4 ecc")],
                title="Server RAM DDR4 ECC",
            )
            basket = await cheapest_basket(session, HOME_SERVER)

        chosen = {item.slot: item for item in basket.items}
        assert chosen[CT.RAM].canonical_name == "32gb ddr4"

    async def test_slots_without_a_defining_quantity_are_still_accepted(self, market):
        """A CPU has no capacity to parse; requiring one would empty the basket."""
        async with market() as session:
            basket = await cheapest_basket(session, HOME_SERVER)

        assert CT.CPU in {item.slot for item in basket.items}
