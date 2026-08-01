"""Basket Mode — what the same machine costs assembled from parts, right now.

### One query, inverted

`pipeline/index.py` values a prebuilt by summing what its parts sell for. Turn
that around and the same data answers a different question: given every
component listing currently live, what is the cheapest set that satisfies a
profile? *"Best prebuilt right now: 7 200 kr — equivalent parts basket: 6 100 kr
across four sellers."*

No new data collection, no new model calls. The component sweep was already
running to feed the price index; this reads it a second way.

### Live listings, not index medians

The basket is priced from **actual current asking prices**, not from the median.
A median says what a part is generally worth. A basket has to be something a
person could go and buy this afternoon, which means real listings with real
prices, and the number is meaningless if it is assembled from parts that sold
last month.

That also makes the basket honestly *volatile*. It can move several hundred
kronor between runs because one cheap card sold. That is the market, not a bug,
and the seller count is shown alongside precisely so the friction is visible: a
1 100 kr saving spread across four strangers in four towns is a different
proposition from the same saving in one transaction.

### What the total deliberately excludes, and why that matters

A parts basket of CPU, GPU, RAM and a drive is **not a computer**. It still
needs a motherboard, a power supply and a case, and those are the three things
Blocket sellers itemise worst — they are usually bundled invisibly into a "hela
datorn" listing and rarely sold alone in a form the extractor recognises.

Quietly omitting them would tilt every comparison toward building, by roughly
the cost of the omission. So `missing_slots` is part of the return value, the
excluded slots are named, and a caller that presents a basket total without
naming them is misrepresenting it. This is the same failure the coverage bug in
`pipeline/index.py` was: an incomplete sum presented as a complete one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    Ad,
    AiEvaluation,
    ComponentTypeEnum,
    ExtractedComponent,
    ItemTypeEnum,
    StatusEnum,
)
from pipeline.extract import find_damage_markers
from pipeline.index import value_ad
from pipeline.profiles import MachineSpec, Profile, evaluate, machine_spec

logger = logging.getLogger(__name__)

# The slots a basket tries to fill. Ordered by how much of the budget they are.
CORE_SLOTS: tuple[ComponentTypeEnum, ...] = (
    ComponentTypeEnum.GPU,
    ComponentTypeEnum.CPU,
    ComponentTypeEnum.RAM,
    ComponentTypeEnum.STORAGE,
)

# Named, not silently dropped. See the module docstring.
EXCLUDED_SLOTS: tuple[ComponentTypeEnum, ...] = (
    ComponentTypeEnum.MOTHERBOARD,
    ComponentTypeEnum.PSU,
    ComponentTypeEnum.CASE,
)


@dataclass(frozen=True, slots=True)
class BasketItem:
    """One listing filling one slot."""

    slot: ComponentTypeEnum
    canonical_name: str
    price_sek: int
    ad_id: int
    url: str
    city: str | None
    title: str


@dataclass(frozen=True, slots=True)
class Basket:
    """The cheapest currently-buyable set of parts meeting a profile."""

    profile_id: str
    items: tuple[BasketItem, ...]
    missing_slots: tuple[ComponentTypeEnum, ...]
    excluded_slots: tuple[ComponentTypeEnum, ...]
    generated_at: datetime

    @property
    def total_sek(self) -> int:
        return sum(item.price_sek for item in self.items)

    @property
    def seller_count(self) -> int:
        """Distinct listings, which is the number of strangers to transact with.

        Counted from ad ids rather than any seller identifier — the schema holds
        no seller identity, by design. Two listings from the same person count
        as two, which slightly overstates the friction. Overstating the cost of
        the do-it-yourself option is the safe direction for a first-time buyer.
        """
        return len({item.ad_id for item in self.items})

    @property
    def is_complete(self) -> bool:
        return not self.missing_slots

    def spec(self) -> MachineSpec:
        return machine_spec(
            ((item.slot, item.canonical_name) for item in self.items),
            text_hint=" ".join(item.title for item in self.items),
        )

    def describe(self) -> str:
        if not self.items:
            return f"{self.profile_id}: no qualifying parts currently listed"
        gaps = (
            f", missing {', '.join(s.value for s in self.missing_slots)}"
            if self.missing_slots
            else ""
        )
        return (
            f"{self.profile_id}: {self.total_sek} kr across {self.seller_count} "
            f"sellers{gaps} (excludes "
            f"{', '.join(s.value.lower() for s in self.excluded_slots)})"
        )


async def cheapest_basket(
    session: AsyncSession,
    profile: Profile,
    *,
    max_price_sek: int | None = None,
    now: datetime | None = None,
) -> Basket:
    """Cheapest live component listings that together satisfy `profile`.

    ### Why greedy is the right algorithm and not a shortcut

    Each slot is filled with the cheapest listing that clears the profile's
    requirements *for that slot* — no search, no backtracking.

    That is optimal here rather than merely convenient, because the profile's
    requirements are per-part floors ("at least 12 GB of VRAM") and not a
    budget to be traded across slots. Nothing about choosing a cheaper CPU
    changes which graphics cards qualify, so the slots are independent and the
    cheapest qualifying choice in each is the cheapest overall.

    It would stop being optimal the moment a requirement spanned two slots —
    total system wattage, or socket compatibility between CPU and motherboard.
    Neither is expressible from what the extractor produces today, and adding
    one means replacing this function rather than patching it.
    """
    now = now or datetime.now(timezone.utc)

    stmt = (
        select(
            ExtractedComponent.component_type,
            ExtractedComponent.canonical_name,
            Ad.price_sek,
            Ad.id,
            Ad.url,
            Ad.city,
            Ad.title,
        )
        .join(Ad, Ad.id == ExtractedComponent.ad_id)
        .where(Ad.status == StatusEnum.ACTIVE)
        # Single-component listings only. A "GPU" pulled out of a whole-PC ad is
        # not for sale on its own, and pricing a basket with it would quote a
        # complete computer's asking price as the cost of its graphics card.
        .where(Ad.item_type == ItemTypeEnum.SINGLE_COMPONENT)
        .where(ExtractedComponent.canonical_name.is_not(None))
        .where(ExtractedComponent.component_type.in_(CORE_SLOTS))
        .order_by(Ad.price_sek.asc())
    )
    if max_price_sek is not None:
        stmt = stmt.where(Ad.price_sek <= max_price_sek)

    rows = (await session.execute(stmt)).all()

    # Ads holding several parts are excluded the same way `BuildValuation`
    # excludes them: one asking price cannot be attributed to one of three
    # cards. Counted here rather than filtered in SQL because the count is a
    # property of the ad, not of the row.
    parts_per_ad: dict[int, int] = {}
    for _, _, _, ad_id, _, _, _ in rows:
        parts_per_ad[ad_id] = parts_per_ad.get(ad_id, 0) + 1

    best: dict[ComponentTypeEnum, BasketItem] = {}
    for slot, canonical_name, price_sek, ad_id, url, city, title in rows:
        if slot in best:
            continue  # rows are price-ascending, so the first is the cheapest
        if parts_per_ad.get(ad_id, 0) > 1:
            continue
        candidate = BasketItem(
            slot=slot,
            canonical_name=canonical_name,
            price_sek=price_sek,
            ad_id=ad_id,
            url=url,
            city=city,
            title=title,
        )
        if _slot_qualifies(profile, candidate):
            best[slot] = candidate

    items = tuple(best[slot] for slot in CORE_SLOTS if slot in best)
    missing = tuple(slot for slot in _required_slots(profile) if slot not in best)

    basket = Basket(
        profile_id=profile.id,
        items=items,
        missing_slots=missing,
        excluded_slots=EXCLUDED_SLOTS,
        generated_at=now,
    )
    logger.info("Basket: %s", basket.describe())
    return basket


def _slot_qualifies(profile: Profile, item: BasketItem) -> bool:
    """Whether one part may be put in a basket that claims to meet the profile.

    Uses `Profile.confirms_part`, which is strict: every requirement *about*
    this slot must be positively true, not merely un-contradicted. Requirements
    about other slots are ignored rather than counted as unknowns.

    An earlier version reused `evaluate` and accepted anything not actively
    failing. Live output showed why that is wrong: the AI-workstation basket
    picked "server ram ddr4 ecc" for its 32 GB slot. No capacity could be read
    from the name, so the requirement returned `unknown`, so it passed — and the
    basket asserted a 32 GB machine on the strength of a string it could not
    parse. Permissiveness is right when deciding what to show a shopper and
    wrong when claiming a set of parts adds up to a working profile.

    The title is passed as `text_hint` for the reason it is everywhere else:
    "RTX 3060 12GB" is the seller stating which variant they hold, and without
    it the pessimistic table default of 8 GB would disqualify the single best
    value card for this profile from its own basket.
    """
    spec = machine_spec([(item.slot, item.canonical_name)], text_hint=item.title)
    return profile.describes_part(item.slot, spec) and profile.confirms_part(
        item.slot, spec
    )


def _required_slots(profile: Profile) -> tuple[ComponentTypeEnum, ...]:
    """Slots a basket must fill for this profile to be a usable machine.

    CPU, RAM and storage are unconditional — every computer needs all three,
    whatever it is for. Only the graphics card is profile-dependent: an AI
    workstation is pointless without one, a home server is better off without.

    An earlier version derived storage from the profile's requirements, which
    produced a quiet inconsistency: the `ai-workstation` profile has no storage
    *requirement*, so a basket with no drive in it reported itself complete
    while `CORE_SLOTS` had gone looking for one anyway. Complete meant
    "everything I asked for" rather than "everything you need", which is the
    same shape of error as the coverage bug in `pipeline/index.py`.
    """
    labels = " ".join(r.label.lower() for r in profile.requirements)
    slots = [ComponentTypeEnum.CPU, ComponentTypeEnum.RAM, ComponentTypeEnum.STORAGE]
    if "vram" in labels:
        slots.insert(0, ComponentTypeEnum.GPU)
    return tuple(slots)


@dataclass(frozen=True, slots=True)
class BuildOrBuy:
    """The two options side by side. Neither is recommended over the other here."""

    basket: Basket
    best_prebuilt_ad_id: int | None
    best_prebuilt_price_sek: int | None
    best_prebuilt_title: str | None
    # Listings that might qualify but did not state enough to say. Surfaced
    # rather than dropped: "no qualifying prebuilt" and "no prebuilt described
    # its specs fully enough to check" are different facts about the market, and
    # only one of them is about the market.
    unverified_candidates: int = 0
    # Listings excluded because their seller says the machine is broken. Counted
    # for the same reason as `unverified_candidates`: "nothing qualified" and
    # "the cheapest thing that qualified was defective" are different facts.
    damaged_candidates: int = 0

    @property
    def saving_sek(self) -> int | None:
        """Prebuilt price minus basket total. Positive means building is cheaper.

        `None` when the basket is incomplete — a partial basket is cheaper than
        a whole computer for the trivial reason that it is not one, and
        reporting that difference as a saving would be a lie by arithmetic.
        """
        if not self.basket.is_complete or self.best_prebuilt_price_sek is None:
            return None
        return self.best_prebuilt_price_sek - self.basket.total_sek

    def describe(self) -> str:
        if self.best_prebuilt_price_sek is None:
            reasons = []
            if self.unverified_candidates:
                reasons.append(f"{self.unverified_candidates} too vaguely described to check")
            if self.damaged_candidates:
                reasons.append(f"{self.damaged_candidates} stated as damaged")
            note = f" ({'; '.join(reasons)})" if reasons else ""
            return (
                f"No qualifying prebuilt listed{note}. "
                f"Parts basket: {self.basket.describe()}"
            )
        saving = self.saving_sek
        verdict = (
            "incomplete basket — not comparable"
            if saving is None
            else f"building saves {saving} kr"
            if saving > 0
            else f"buying assembled saves {-saving} kr"
        )
        return (
            f"Best prebuilt: {self.best_prebuilt_price_sek} kr — "
            f"parts basket: {self.basket.total_sek} kr across "
            f"{self.basket.seller_count} sellers — {verdict}"
        )


async def build_or_buy(
    session: AsyncSession,
    profile: Profile,
    *,
    max_price_sek: int | None = None,
    now: datetime | None = None,
) -> BuildOrBuy:
    """Cheapest qualifying assembled machine versus the cheapest parts basket.

    "Cheapest qualifying" is by **effective** price — asking price plus whatever
    the profile says it costs to bring the machine up to standard. A 6 400 kr
    box with 16 GB of RAM is not cheaper than a 6 800 kr box with 32 GB, and
    ranking on the sticker price would tell a first-time buyer that it is.
    """
    basket = await cheapest_basket(
        session, profile, max_price_sek=max_price_sek, now=now
    )

    stmt = (
        select(Ad.id, Ad.price_sek, Ad.title, Ad.raw_description, AiEvaluation.risk_flags)
        # Outer join: an ad with no verdict yet still has a title and a body to
        # read, and this check must not depend on the model having run.
        .outerjoin(AiEvaluation, AiEvaluation.ad_id == Ad.id)
        .where(Ad.status == StatusEnum.ACTIVE)
        .where(Ad.item_type == ItemTypeEnum.FULL_BUILD)
        .order_by(Ad.price_sek.asc())
    )
    if max_price_sek is not None:
        stmt = stmt.where(Ad.price_sek <= max_price_sek)

    candidates = (await session.execute(stmt)).all()

    best_id = best_price = best_title = None
    unverified = 0
    damaged = 0
    for ad_id, price_sek, title, description, risk_flags in candidates:
        # A broken machine is not a comparison baseline.
        #
        # Everything else in this loop asks whether a listing is *good enough*
        # for the profile — enough VRAM, enough RAM, a believable price. None of
        # it asks whether the thing works, because until ad 13 nothing had to:
        # the specs of a dead laptop parse exactly as well as those of a live
        # one, and being dead makes it cheap, which is the selection criterion.
        #
        # So "DEFEKT Acer Predator Triton 300 SE ... – Defekt" at 2 999 kr won
        # the gaming headline, and the site published a 2 300 kr penalty for
        # building from working parts — measured against a computer its own
        # seller describes as stendöd and sells as a reservdelsobjekt. The
        # arithmetic was right and the comparison was meaningless.
        markers = find_damage_markers(title, description, risk_flags)
        if markers:
            logger.info(
                "%s: skipping ad %s at %d kr — seller states %s.",
                profile.id,
                ad_id,
                price_sek,
                ", ".join(f'"{m}"' for m in markers[:3]),
            )
            damaged += 1
            continue

        parts = (
            await session.execute(
                select(
                    ExtractedComponent.component_type,
                    ExtractedComponent.canonical_name,
                ).where(ExtractedComponent.ad_id == ad_id)
            )
        ).all()
        spec = machine_spec(
            ((ctype, name) for ctype, name in parts if name), text_hint=title
        )
        # A price far below the parts it claims to contain is not a bargain.
        # This headline is the most attractive place in the whole application
        # for a bait listing to land, since "cheapest" is literally the
        # selection criterion.
        valuation = await value_ad(session, ad_id, now=now)
        if valuation is not None and (
            valuation.is_implausibly_cheap or valuation.is_inventory_listing
        ):
            logger.info(
                "%s: skipping ad %s at %d kr — %s.",
                profile.id,
                ad_id,
                price_sek,
                "an inventory listing, not one machine"
                if valuation.is_inventory_listing
                else f"implausible against ~{valuation.estimated_parts_sek} kr of parts",
            )
            continue

        fit = evaluate(profile, spec, price_sek)

        # `verdict != "fits"` and not merely `is_viable`.
        #
        # This headline claims a specific machine meets the profile, so it has
        # to be one we can vouch for. `is_viable` is the leaderboard's test —
        # permissive, because an ad that simply never mentioned its RAM should
        # still be shown to someone shopping. Reused here it produced a real
        # absurdity: a 300 kr listing whose GPU and RAM were both unreadable
        # came back "unverified", passed, and was announced as the cheapest
        # qualifying AI workstation on the market.
        #
        # A fixable shortfall is still fine — the effective price already
        # carries the upgrade cost. It is *unverified* that must not win.
        if fit.blocked_by or fit.unknown:
            if fit.is_viable:
                unverified += 1
            continue
        if best_price is None or fit.effective_price_sek < best_price:
            best_id, best_price, best_title = ad_id, fit.effective_price_sek, title

    if unverified:
        logger.info(
            "%s: %d listings not offered as the best prebuilt — specs incomplete.",
            profile.id,
            unverified,
        )
    if damaged:
        logger.info(
            "%s: %d listings not offered as the best prebuilt — stated as damaged.",
            profile.id,
            damaged,
        )

    return BuildOrBuy(
        basket=basket,
        best_prebuilt_ad_id=best_id,
        best_prebuilt_price_sek=best_price,
        best_prebuilt_title=best_title,
        unverified_candidates=unverified,
        damaged_candidates=damaged,
    )
