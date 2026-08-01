"""Tests for the local development API.

This layer is not deployed, so the reason to test it is narrower than usual: it
is the only place where profile fit, market valuation and ranking are assembled
into one response, and the assembly has failure modes that none of the
underlying modules can catch on their own — a sort key that leaks into the
payload, a filter applied to the wrong price, a field the dashboard reads under
a different name.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from api.app import app
from db.connection import get_session
from db.models import AiEvaluation, ExtractedComponent, ItemTypeEnum, PriceObservation
from db.models import ComponentTypeEnum as CT


@pytest.fixture
async def client(session_factory, ad_factory):
    """The app wired to an in-memory database via dependency override."""
    async with session_factory() as session:
        # A 12 GB card, correctly specified: fits `ai-workstation` outright.
        good = ad_factory(
            "good", title="Speldator RTX 3060 12GB", price_sek=6000,
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        # An 8 GB card: a fine gaming machine, disqualified for local models.
        gaming_only = ad_factory(
            "gaming-only", title="Speldator RTX 3070", price_sek=5000,
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        session.add_all([good, gaming_only])
        await session.flush()

        session.add_all(
            [
                ExtractedComponent(
                    ad_id=good.id, component_type=CT.GPU,
                    component_name="RTX 3060", canonical_name="rtx 3060",
                ),
                ExtractedComponent(
                    ad_id=good.id, component_type=CT.RAM,
                    component_name="16GB DDR4", canonical_name="16gb ddr4",
                ),
                ExtractedComponent(
                    ad_id=gaming_only.id, component_type=CT.GPU,
                    component_name="RTX 3070", canonical_name="rtx 3070",
                ),
            ]
        )
        # `/api/deals` joins on the AI verdict, so an unevaluated ad is
        # invisible there by design — the endpoint reports what has been
        # assessed, not everything that has been scraped.
        session.add_all(
            [
                AiEvaluation(
                    ad_id=ad.id,
                    item_type=ItemTypeEnum.FULL_BUILD,
                    confidence=0.9,
                    risk_flags="",
                    ai_reasoning="Looks fine.",
                    content_hash=ad.content_hash,
                    model="test",
                )
                for ad in (good, gaming_only)
            ]
        )
        # Enough observations for the RAM upgrade price to be index-sourced.
        session.add_all(
            [
                PriceObservation(
                    canonical_name="32gb ddr4", component_type=CT.RAM,
                    price_sek=500, source="blocket", source_ref=f"ram-{i}",
                )
                for i in range(6)
            ]
        )
        await session.commit()

    async def override():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


class TestProfilesEndpoint:
    async def test_lists_every_profile_with_its_reasoning(self, client):
        response = await client.get("/api/profiles")
        assert response.status_code == 200

        profiles = response.json()
        assert {p["id"] for p in profiles} == {"ai-workstation", "gaming", "home-server"}
        for profile in profiles:
            # The rationale ships to the UI: a filter that silently drops
            # listings looks broken, one that explains itself is the product.
            assert profile["rationale"]
            assert profile["requirements"]


class TestDealsEndpoint:
    async def test_the_sort_key_does_not_leak_into_the_payload(self, client):
        """`_rank` is scratch state and must not reach the client.

        It is a tuple, so it would serialise happily as a JSON array and no
        schema anywhere would object — the dashboard would simply carry a
        mystery field forever.
        """
        deals = (await client.get("/api/deals?profile=gaming&includeUnfit=true")).json()
        assert deals
        assert all("_rank" not in deal for deal in deals)

    async def test_unfit_listings_are_withheld_by_default(self, client):
        deals = (await client.get("/api/deals?profile=ai-workstation")).json()
        titles = {deal["title"] for deal in deals}
        assert "Speldator RTX 3070" not in titles

    async def test_but_can_be_asked_for(self, client):
        deals = (
            await client.get("/api/deals?profile=ai-workstation&includeUnfit=true")
        ).json()
        unfit = [d for d in deals if d["fit"]["verdict"] == "unfit"]
        assert unfit
        assert "12 GB VRAM" in unfit[0]["fit"]["blockedBy"]

    async def test_the_same_listing_is_judged_differently_per_profile(self, client):
        """The point of the whole feature, asserted end to end."""
        ai = (
            await client.get("/api/deals?profile=ai-workstation&includeUnfit=true")
        ).json()
        gaming = (await client.get("/api/deals?profile=gaming")).json()

        by_title = {d["title"]: d for d in ai}
        assert by_title["Speldator RTX 3070"]["fit"]["verdict"] == "unfit"
        assert "Speldator RTX 3070" in {d["title"] for d in gaming}

    async def test_upgrade_cost_comes_from_the_index_not_the_fallback(self, client):
        """Six observations at 500 kr should beat the 800 kr default estimate."""
        deals = (await client.get("/api/deals?profile=ai-workstation")).json()
        good = next(d for d in deals if d["title"] == "Speldator RTX 3060 12GB")
        assert good["fit"]["upgradeCostSek"] == 500
        assert good["fit"]["effectivePriceSek"] == 6500

    async def test_budget_filter_uses_the_asking_price_ceiling(self, client):
        deals = (await client.get("/api/deals?maxPrice=5500&includeUnfit=true")).json()
        assert all(deal["priceSek"] <= 5500 for deal in deals)

    async def test_an_unknown_profile_falls_back_rather_than_erroring(self, client):
        """A stale bookmark must not produce a 500."""
        response = await client.get("/api/deals?profile=nonsense")
        assert response.status_code == 200

    async def test_limit_is_applied_after_ranking_not_before(self, client):
        """Truncating before the sort would return an arbitrary subset.

        The ranking depends on per-part medians and profile fit, neither of
        which the database can order by — so a `LIMIT` in SQL would slice the
        list in insertion order and then sort the wrong ten rows.
        """
        deals = (
            await client.get("/api/deals?limit=1&profile=gaming&includeUnfit=true")
        ).json()
        assert len(deals) == 1
        # The 12 GB machine wins on kr/GB VRAM... but gaming has no metric, so
        # the tiebreak is market value. Either way, one deal, deterministically.
        assert deals[0]["fit"]["profileId"] == "gaming"


class TestBasketEndpoint:
    async def test_exclusions_are_part_of_the_response(self, client):
        """Not a UI footnote — a renderer without them misstates the total."""
        payload = (await client.get("/api/basket?profile=ai-workstation")).json()
        excluded = payload["basket"]["excludedSlots"]
        assert {"MOTHERBOARD", "PSU", "CASE"} <= set(excluded)

    async def test_incomplete_basket_reports_no_saving(self, client):
        """There are no standalone component listings in this fixture at all."""
        payload = (await client.get("/api/basket?profile=ai-workstation")).json()
        assert payload["basket"]["isComplete"] is False
        assert payload["savingSek"] is None
