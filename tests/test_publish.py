"""Tests for the published artifact.

These are the highest-stakes tests in the suite, and they are not really about
correctness. Everywhere else a bug produces a wrong number on a dashboard that
one person looks at. Here a bug produces a stranger's phone number in a public
git history, which cannot be taken back by fixing the code afterwards.

So the shape is different from the rest of the suite: as well as asserting that
good input passes, every check is fed **deliberately poisoned input** and
required to fail. A safety check that has only ever been shown clean data is not
known to work — it is known to be silent, which is what it would also be if it
were broken.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from db.models import AiEvaluation, ExtractedComponent, ItemTypeEnum, PriceObservation
from db.models import ComponentTypeEnum as CT
from pipeline.publish import (
    BUDGET_BANDS,
    PER_CATEGORY_LIMIT,
    RedactionError,
    assert_capped,
    assert_redacted,
    build_snapshot,
    publish,
    verify_file,
)

TYPES_TS = Path(__file__).resolve().parents[1] / "web" / "src" / "types.ts"


# --------------------------------------------------------------------------
# Fixtures: a small but realistic corpus
# --------------------------------------------------------------------------


@pytest.fixture
async def populated(session_factory, ad_factory):
    """Four listings covering the cases the publisher has to tell apart."""
    async with session_factory() as session:
        # Fits `ai-workstation` outright: 12 GB card.
        good = ad_factory(
            "good",
            title="Speldator RTX 3060 12GB, 32GB RAM",
            price_sek=6000,
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        # An 8 GB card. Great for gaming, blocked for local models — the listing
        # that proves the cap is applied per profile rather than once globally.
        gaming = ad_factory(
            "gaming",
            title="Speldator RTX 3070",
            price_sek=5000,
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        # A single component, so it belongs in the GPU bucket, not FULL_BUILD.
        card = ad_factory(
            "card",
            title="RTX 3060 12GB till salu",
            price_sek=2300,
            item_type=ItemTypeEnum.SINGLE_COMPONENT,
        )
        # A seller who typed their number into the title. Not hypothetical:
        # this is the single most likely way personal data reaches the public
        # file, because the title is the one free-text field that survives.
        leaky = ad_factory(
            "leaky",
            title="RTX 3060 12GB ring 070-123 45 67",
            price_sek=2400,
            item_type=ItemTypeEnum.SINGLE_COMPONENT,
        )
        session.add_all([good, gaming, card, leaky])
        await session.flush()

        session.add_all(
            [
                ExtractedComponent(
                    ad_id=good.id, component_type=CT.GPU,
                    component_name="RTX 3060 12GB", canonical_name="rtx 3060",
                ),
                ExtractedComponent(
                    ad_id=good.id, component_type=CT.RAM,
                    component_name="32GB DDR4", canonical_name="32gb ddr4",
                ),
                ExtractedComponent(
                    ad_id=good.id, component_type=CT.CPU,
                    component_name="i5-11400", canonical_name="i5-11400",
                ),
                ExtractedComponent(
                    ad_id=gaming.id, component_type=CT.GPU,
                    component_name="RTX 3070", canonical_name="rtx 3070",
                ),
                ExtractedComponent(
                    ad_id=gaming.id, component_type=CT.RAM,
                    component_name="16GB DDR4", canonical_name="16gb ddr4",
                ),
                ExtractedComponent(
                    ad_id=gaming.id, component_type=CT.CPU,
                    component_name="i5-12400", canonical_name="i5-12400",
                ),
                ExtractedComponent(
                    ad_id=card.id, component_type=CT.GPU,
                    component_name="RTX 3060", canonical_name="rtx 3060",
                ),
                ExtractedComponent(
                    ad_id=leaky.id, component_type=CT.GPU,
                    component_name="RTX 3060", canonical_name="rtx 3060",
                ),
            ]
        )
        session.add_all(
            [
                AiEvaluation(
                    ad_id=ad.id,
                    item_type=ad.item_type,
                    confidence=0.9,
                    risk_flags="ej testad\nkvitto finns",
                    ai_reasoning="Looks fine, kvitto finns.",
                    content_hash=ad.content_hash,
                    model="test",
                )
                for ad in (good, gaming, card, leaky)
            ]
        )
        # Enough observations that both machines have full coverage and a real
        # delta. Without that every delta is `None`, every rank ties, and the
        # per-profile cap test below passes for the wrong reason.
        market = {
            ("rtx 3060", CT.GPU): 2500,
            ("rtx 3070", CT.GPU): 4000,
            ("32gb ddr4", CT.RAM): 900,
            ("16gb ddr4", CT.RAM): 500,
            ("i5-11400", CT.CPU): 1200,
            ("i5-12400", CT.CPU): 1500,
        }
        session.add_all(
            [
                PriceObservation(
                    canonical_name=name, component_type=ctype,
                    price_sek=price, source="blocket", source_ref=f"{name}-{i}",
                )
                for (name, ctype), price in market.items()
                for i in range(6)
            ]
        )
        await session.commit()

    return session_factory


@pytest.fixture
async def snapshot(populated):
    async with populated() as session:
        return await build_snapshot(session)


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


async def test_snapshot_has_every_field_the_frontend_reads(snapshot):
    """The Python and TypeScript sides of one contract, checked against each other.

    `web/src/types.ts` is the only description of this file's shape, and nothing
    enforces it — the frontend fetches plain JSON and trusts it. A field renamed
    on this side shows up as `undefined` in the browser, which React renders as
    blank rather than as an error.
    """
    top_level = _interface_fields("DealsSnapshot")
    assert top_level <= set(snapshot), f"missing: {top_level - set(snapshot)}"

    deal_fields = _interface_fields("Deal")
    assert deal_fields <= set(snapshot["deals"][0])

    basket_fields = _interface_fields("BasketSnapshot")
    for basket in snapshot["baskets"].values():
        assert basket_fields <= set(basket)


async def test_every_profile_has_a_fit_and_a_basket(snapshot):
    """Switching profile is a client-side re-sort, so all three must ship."""
    ids = {p["id"] for p in snapshot["profiles"]}
    assert ids == set(snapshot["baskets"])
    for deal in snapshot["deals"]:
        assert set(deal["fits"]) == ids


async def test_baskets_name_what_they_exclude(snapshot):
    """A parts total without its exclusions is cheaper by exactly the omission."""
    for basket in snapshot["baskets"].values():
        assert set(basket["excludedSlots"]) == {"MOTHERBOARD", "PSU", "CASE"}


# --------------------------------------------------------------------------
# Redaction: the clean case
# --------------------------------------------------------------------------


async def test_real_snapshot_passes_its_own_check(snapshot):
    assert_redacted(snapshot)
    assert_capped(snapshot)


async def test_no_description_survives_anywhere(snapshot):
    """The listing bodies are in the database and must not be in the file."""
    text = json.dumps(snapshot)
    assert "Fungerar bra" not in text


async def test_a_phone_number_in_a_title_is_scrubbed(snapshot):
    """The live failure mode: a seller types their number into the headline."""
    leaky = next(d for d in snapshot["deals"] if "ring" in d["title"])
    assert "070" not in leaky["title"]
    assert "[phone]" in leaky["title"]


async def test_risk_flags_are_split_by_direction(snapshot):
    """"kvitto finns" is reassurance; rendering it in amber would mislead."""
    flags = {f["label"]: f["severity"] for d in snapshot["deals"] for f in d["ai"]["flags"]}
    assert flags["ej testad"] == "warn"
    assert flags["kvitto finns"] == "info"


# --------------------------------------------------------------------------
# Redaction: the poisoned cases
#
# Each of these would have been a real leak. The check is required to catch it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "poison, because",
    [
        ({"deals": [{"description": "hej"}]}, "a whole free-text body"),
        ({"deals": [{"sellerName": "Anna Svensson"}]}, "seller identity"),
        ({"deals": [{"latitude": 59.33}]}, "coordinates"),
        ({"deals": [{"postal_code": "11122"}]}, "a postal code"),
        ({"deals": [{"images": ["a.jpg"]}]}, "an image array"),
        ({"deals": [{"summary": "maila mig pa a@b.se"}]}, "an email in prose"),
        ({"deals": [{"summary": "ring 070-123 45 67"}]}, "a phone number in prose"),
        ({"deals": [{"summary": "se https://evil.example/x"}]}, "a bare URL in prose"),
        (
            {"deals": [{"summary": "https://images.blocket.se/a.jpg"}]},
            "a hotlinked image",
        ),
        ({"deals": [{"city": "Stockholm 11122"}]}, "a postal code inside a place"),
        ({"deals": [{"url": "https://evil.example/tracker"}]}, "an off-site link"),
        ({"deals": [{"url": "https://blocket.se/img/a.png"}]}, "an image link"),
    ],
)
def test_poisoned_snapshots_are_rejected(poison, because):
    with pytest.raises(RedactionError):
        assert_redacted(poison)


def test_every_violation_is_reported_at_once():
    """One run should list everything wrong, not the first thing wrong.

    Fixing leaks one error message at a time is how the fourth one gets missed.
    """
    with pytest.raises(RedactionError) as exc:
        assert_redacted(
            {"deals": [{"phone": "x"}, {"latitude": 1.0}, {"description": "y"}]}
        )
    assert "3 redaction violation(s)" in str(exc.value)


def test_camel_and_snake_spellings_are_both_caught():
    """The pipeline speaks snake_case and the contract camelCase.

    A blocklist that knows only one of them protects against half the mistakes
    and reads as though it protects against all of them.
    """
    for key in ("postal_code", "postalCode", "PostalCode", "postalcode"):
        with pytest.raises(RedactionError):
            assert_redacted({"x": {key: "11122"}})


def test_nested_leaks_are_found():
    """Depth is not a hiding place — the walk is recursive, not top-level."""
    with pytest.raises(RedactionError):
        assert_redacted({"a": {"b": [{"c": {"email": "x@y.se"}}]}})


def test_a_plain_blocket_link_is_allowed():
    """The site is a pointer to Blocket, so attribution links must survive."""
    assert_redacted({"deals": [{"url": "https://www.blocket.se/annons/12345"}]})


# --------------------------------------------------------------------------
# The cap
# --------------------------------------------------------------------------


def test_over_cap_is_rejected():
    """The cap is the load-bearing part of the database-right argument.

    A "top ten per category" that quietly became "everything" would undo the
    reasoning in ETHICS.md while still passing every redaction test here, so it
    is asserted separately rather than trusted to the builder.
    """
    snapshot = {
        "perCategoryLimit": 1,
        "deals": [{"category": "GPU"} for _ in range(10)],
    }
    with pytest.raises(RedactionError, match="category caps exceeded"):
        assert_capped(snapshot, profiles=3)


async def test_cap_is_the_union_of_each_profile_s_top_n(populated):
    """A listing no profile ranks highly is dropped; one that any profile ranks
    highly is kept.

    Capping on a single ordering would have given the other two profiles a
    leaderboard pre-filtered by criteria they disagree with — a selector that
    looks like it works and does not.
    """
    async with populated() as session:
        snapshot = await build_snapshot(session, per_category_limit=1)

    builds = [d for d in snapshot["deals"] if d["category"] == "FULL_BUILD"]
    titles = {d["title"] for d in builds}
    # The 8 GB machine is unfit for `ai-workstation` and fine for `gaming`. It
    # survives a limit of one *because* gaming ranked it first.
    assert "Speldator RTX 3070" in titles
    assert "Speldator RTX 3060 12GB, 32GB RAM" in titles


# --------------------------------------------------------------------------
# Categorisation
# --------------------------------------------------------------------------


async def test_single_components_are_filed_under_their_part(snapshot):
    card = next(d for d in snapshot["deals"] if d["title"].startswith("RTX 3060 12GB till"))
    assert card["category"] == "GPU"


async def test_an_unclassifiable_listing_is_not_published(session_factory, ad_factory):
    """A "single component" holding two kinds of part has no honest bucket.

    Publishing it under a guess would put a motherboard in the graphics-card
    leaderboard. Omission is the correct answer.
    """
    async with session_factory() as session:
        ad = ad_factory("mixed", title="RTX 3060 + moderkort",
                        item_type=ItemTypeEnum.SINGLE_COMPONENT)
        session.add(ad)
        await session.flush()
        session.add_all([
            ExtractedComponent(ad_id=ad.id, component_type=CT.GPU,
                               component_name="RTX 3060", canonical_name="rtx 3060"),
            ExtractedComponent(ad_id=ad.id, component_type=CT.MOTHERBOARD,
                               component_name="B550", canonical_name="b550"),
        ])
        session.add(AiEvaluation(
            ad_id=ad.id, item_type=ad.item_type, confidence=0.9, risk_flags="",
            ai_reasoning="ok", content_hash=ad.content_hash, model="test",
        ))
        await session.commit()

    async with session_factory() as session:
        snapshot = await build_snapshot(session)
    assert snapshot["deals"] == []


async def test_the_395_kr_listing_never_reaches_the_public_file(
    session_factory, ad_factory
):
    """Regression, found by reading the first real run of this module.

    Ad 289 — *"High-End Gaming: i9-13900 | RTX 4070 Ti | 64GB DDR5 | 4TB NVMe"*
    at **395 kr** — was published as the number-one deal on the public site.

    `pipeline.basket` had refused it since Phase 5. `pipeline.publish` had not,
    because the guard lives on `BuildValuation` and each caller has to remember
    to consult it. Nothing about the failure was visible in 366 passing tests:
    the snapshot was well-formed, fully redacted, correctly capped, and wrong at
    the very top.

    A listing this cheap sorts first *because* the ranking rewards cheapness, so
    the public leaderboard is the single most likely place in the application
    for a bait listing to surface.
    """
    async with session_factory() as session:
        bait = ad_factory(
            "bait",
            title="High-End Gaming: i9-13900 | RTX 4070 Ti | 64GB DDR5",
            price_sek=395,
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        session.add(bait)
        await session.flush()
        session.add(ExtractedComponent(
            ad_id=bait.id, component_type=CT.GPU,
            component_name="RTX 4070 Ti", canonical_name="rtx 4070 ti",
        ))
        session.add(AiEvaluation(
            ad_id=bait.id, item_type=bait.item_type, confidence=0.9, risk_flags="",
            ai_reasoning="Specs read from the listing text.",
            content_hash=bait.content_hash, model="test",
        ))
        session.add_all([
            PriceObservation(
                canonical_name="rtx 4070 ti", component_type=CT.GPU,
                price_sek=6495, source="blocket", source_ref=f"gpu-{i}",
            )
            for i in range(6)
        ])
        await session.commit()

    async with session_factory() as session:
        snapshot = await build_snapshot(session)

    assert snapshot["deals"] == [], "a 395 kr RTX 4070 Ti machine was published"


async def test_affordable_machines_survive_the_cap(session_factory, ad_factory):
    """The cap used to run before the budget filter, and threw away the answer.

    Selection took the ten best machines *at any price*; the browser then
    filtered them to the visitor's budget. So a shopper on 6 000 kr saw whatever
    happened to survive, which was usually almost nothing — the database held
    218 active builds under 6 000 kr and the snapshot published eleven.

    Here twelve dear machines outrank three cheap ones on every profile. Under
    the old selection the cheap three fell off the end of a ten-slot list and
    the budget view was empty. The fix is not a bigger cap — that would
    dismantle the *Innoweb* argument in ETHICS.md — but bands spent inside it.
    """
    async with session_factory() as session:
        for i in range(12):
            dear = ad_factory(
                f"dear-{i}",
                title=f"Speldator RTX 3090 24GB, 32GB RAM #{i}",
                price_sek=20000 + i,
                item_type=ItemTypeEnum.FULL_BUILD,
            )
            session.add(dear)
            await session.flush()
            session.add_all([
                ExtractedComponent(ad_id=dear.id, component_type=CT.GPU,
                                   component_name="RTX 3090", canonical_name="rtx 3090"),
                ExtractedComponent(ad_id=dear.id, component_type=CT.RAM,
                                   component_name="32GB DDR4", canonical_name="32gb ddr4"),
            ])
            session.add(AiEvaluation(
                ad_id=dear.id, item_type=dear.item_type, confidence=0.9,
                risk_flags="", ai_reasoning="ok",
                content_hash=dear.content_hash, model="test",
            ))

        for i in range(3):
            cheap = ad_factory(
                f"cheap-{i}",
                title=f"Speldator RTX 3060 12GB, 32GB RAM #{i}",
                price_sek=5000 + i,
                item_type=ItemTypeEnum.FULL_BUILD,
            )
            session.add(cheap)
            await session.flush()
            session.add_all([
                ExtractedComponent(ad_id=cheap.id, component_type=CT.GPU,
                                   component_name="RTX 3060 12GB", canonical_name="rtx 3060"),
                ExtractedComponent(ad_id=cheap.id, component_type=CT.RAM,
                                   component_name="32GB DDR4", canonical_name="32gb ddr4"),
            ])
            session.add(AiEvaluation(
                ad_id=cheap.id, item_type=cheap.item_type, confidence=0.9,
                risk_flags="", ai_reasoning="ok",
                content_hash=cheap.content_hash, model="test",
            ))
        await session.commit()

    async with session_factory() as session:
        snapshot = await build_snapshot(session)

    builds = [d for d in snapshot["deals"] if d["category"] == "FULL_BUILD"]
    affordable = [d for d in builds if d["priceSek"] <= 6000]
    assert len(affordable) == 3, "in-budget machines were capped out of the file"

    # And the cap itself still holds — this is the part ETHICS.md rests on.
    assert_capped(snapshot)


async def test_budget_bands_match_the_client(snapshot):
    """The bands are duplicated in `web/src/App.tsx`; drift makes the cap lie.

    If the client offers a preset the publisher never reserved slots for, that
    preset silently returns the old near-empty view — the exact bug this was
    written to fix, reintroduced by an edit to a different file.
    """
    source = (TYPES_TS.parent / "App.tsx").read_text(encoding="utf-8")
    block = re.search(r"const BUDGETS = \[(.*?)\];", source, re.DOTALL)
    assert block, "BUDGETS not found in App.tsx — the regex has rotted"

    client = {int(v) for v in re.findall(r"value:\s*(\d+)", block.group(1))}
    assert client == {b for b in BUDGET_BANDS if b is not None}


async def test_the_defekt_acer_is_never_the_basket_baseline(
    session_factory, ad_factory
):
    """Regression, found by reading the published snapshot.

    Ad 13 — *"DEFEKT Acer Predator Triton 300 SE (PT314-51s) – RTX 3060 / i7 /
    144Hz – Defekt"* at **2 999 kr**, Hisings Backa, whose body calls it "helt
    defekt skick", "helt stendöd" and a "reservdelsobjekt" — was published as
    `bestPrebuiltId` for the `gaming` profile. The site therefore told a
    first-time buyer that assembling from working parts costs 2 300 kr *more*
    than buying assembled, where the assembled machine is one that does not turn
    on.

    Every number in that comparison was correct. The specs of a dead laptop
    parse exactly as well as those of a live one, and being dead is what made it
    the cheapest — which is the selection criterion. `build_or_buy` asked
    whether the machine was good enough for the profile and never whether it
    worked.

    The three-way check matters: the machine must still be *published* (a broken
    laptop at 2 999 kr is a legitimate parts listing and someone may want it),
    it must carry a `high` flag rather than the same amber chip as "no receipt",
    and it must not be the thing a parts basket is measured against.
    """
    title = (
        "DEFEKT Acer Predator Triton 300 SE (PT314-51s) "
        "– RTX 3060 / i7 / 144Hz – Defekt"
    )
    description = (
        "Säljer en Acer Predator Triton 300 SE (modell PT314-51s-721G) i helt "
        "defekt skick som reservdelsobjekt eller reparationsprojekt.\n\n"
        "Datorn är helt stendöd och får ingen ström alls."
    )
    async with session_factory() as session:
        broken = ad_factory(
            "defekt-acer",
            title=title,
            description=description,
            price_sek=2999,
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        session.add(broken)
        await session.flush()
        session.add_all([
            ExtractedComponent(ad_id=broken.id, component_type=CT.GPU,
                               component_name="RTX 3060", canonical_name="rtx 3060"),
            ExtractedComponent(ad_id=broken.id, component_type=CT.CPU,
                               component_name="i7-11370H", canonical_name="i7-11370h"),
            ExtractedComponent(ad_id=broken.id, component_type=CT.RAM,
                               component_name="16GB DDR4", canonical_name="16gb ddr4"),
        ])
        # The verdict as it was actually published: the model read the listing
        # correctly and said so. The flags were there; nothing consulted them.
        session.add(AiEvaluation(
            ad_id=broken.id, item_type=broken.item_type, confidence=0.1,
            risk_flags="defekt\nsold as-is\nno returns",
            ai_reasoning="The laptop is explicitly stated as broken.",
            content_hash=broken.content_hash, model="test",
        ))
        await session.commit()

    async with session_factory() as session:
        snapshot = await build_snapshot(session)

    published = next(d for d in snapshot["deals"] if d["id"] == str(broken.id))
    severity = {f["label"]: f["severity"] for f in published["ai"]["flags"]}
    assert severity["defekt"] == "high"
    assert severity["sold as-is"] == "high"
    # "no returns" is a term of sale, not a statement that the thing is broken.
    assert severity["no returns"] == "warn"

    gaming = snapshot["baskets"]["gaming"]
    assert gaming["bestPrebuiltId"] != str(broken.id), (
        "a machine the seller calls stendöd was the build-or-buy baseline"
    )


async def test_damage_stated_only_in_a_title_still_earns_a_high_flag(
    session_factory, ad_factory
):
    """The flag must not depend on the model having been the one to read it.

    A deterministic verdict raises no flags at all by construction, so an ad
    resolved on the fast path would publish "befintligt skick" in its headline
    and "no risk language found" underneath it.
    """
    async with session_factory() as session:
        ad = ad_factory(
            "title-only",
            title="Speldator RTX 3060 12GB, 32GB RAM – säljes i befintligt skick",
            description="Hämtas i Göteborg.",
            price_sek=4000,
            item_type=ItemTypeEnum.FULL_BUILD,
        )
        session.add(ad)
        await session.flush()
        session.add(ExtractedComponent(
            ad_id=ad.id, component_type=CT.GPU,
            component_name="RTX 3060 12GB", canonical_name="rtx 3060",
        ))
        session.add(AiEvaluation(
            ad_id=ad.id, item_type=ad.item_type, confidence=0.75, risk_flags="",
            ai_reasoning="Specs read directly from the listing text by pattern "
                         "matching; no risk language found.",
            content_hash=ad.content_hash, model="deterministic",
        ))
        await session.commit()

    async with session_factory() as session:
        snapshot = await build_snapshot(session)

    flags = next(d for d in snapshot["deals"] if d["id"] == str(ad.id))["ai"]["flags"]
    assert [f["severity"] for f in flags] == ["high"]
    assert flags[0]["label"] == "befintligt skick"


async def test_unevaluated_ads_are_not_published(populated, ad_factory):
    """The file reports what has been assessed, not everything scraped."""
    async with populated() as session:
        session.add(ad_factory("raw", title="RTX 4090", price_sek=100))
        await session.commit()

    async with populated() as session:
        snapshot = await build_snapshot(session)
    assert "RTX 4090" not in {d["title"] for d in snapshot["deals"]}


# --------------------------------------------------------------------------
# Writing to disk
# --------------------------------------------------------------------------


async def test_publish_writes_a_file_that_verifies(populated, tmp_path):
    out = tmp_path / "public" / "deals.json"
    await publish(populated, output=out)

    assert out.exists()
    # Re-read from disk rather than reusing the in-memory object: the artifact
    # that matters is the bytes, and JSON serialisation is a step that can lose
    # or mangle things.
    reloaded = verify_file(out)
    assert reloaded["deals"]


async def test_publish_refuses_to_write_a_leaky_snapshot(populated, tmp_path, monkeypatch):
    """No file at all beats a bad file, because `git add .` is one command.

    Simulates the realistic regression: someone adds a field to the builder
    without thinking about what is in it.
    """
    import pipeline.publish as mod

    original = mod.build_snapshot

    async def leaky(session, **kwargs):
        snapshot = await original(session, **kwargs)
        snapshot["deals"][0]["sellerName"] = "Anna Svensson"
        return snapshot

    monkeypatch.setattr(mod, "build_snapshot", leaky)

    out = tmp_path / "deals.json"
    with pytest.raises(RedactionError):
        await mod.publish(populated, output=out)
    assert not out.exists()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _interface_fields(name: str) -> set[str]:
    """Field names declared on a TypeScript interface in `web/src/types.ts`.

    A regex over a `.ts` file is a blunt instrument, and it is the right one
    here: the alternative is a Node dependency in the Python test suite to guard
    one file that changes rarely. If the parse ever breaks, it breaks loudly —
    an empty set would make the assertion vacuous, so that is checked too.
    """
    source = TYPES_TS.read_text(encoding="utf-8")
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.DOTALL)
    assert match, f"{name} not found in {TYPES_TS}"
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.DOTALL)
    fields = set(re.findall(r"^\s{2}(\w+)\??:", body, re.MULTILINE))
    assert fields, f"parsed no fields from {name} — the regex has rotted"
    return fields
