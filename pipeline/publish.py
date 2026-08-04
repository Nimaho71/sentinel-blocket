"""Turn the local database into a file that is safe to put on the internet.

### The only stage whose output leaves this machine

Everything before this runs on Nils's laptop against `sentinel_blocket.db`, a
file that holds real listings written by real people. This stage produces
`web/public/deals.json`, which gets committed to a public repository and served
from a CDN. The asymmetry is the whole point of the module: a mistake anywhere
else costs a bad number on a dashboard, and a mistake here is a stranger's
phone number in a git history that cannot be rewritten.

So redaction is not a filter applied at the end. It is the construction method.

### Allowlist, not blocklist

Every field in the published snapshot is named explicitly in `_deal()`,
`_basket()` and `_snapshot()`. Nothing is copied wholesale from an ORM object,
nothing is `dict(**row)`, nothing is `model_dump()`. That inversion is what
makes the guarantee survive schema change: adding a column to `Ad` cannot leak
it, because nothing here would ever read it.

A blocklist has the opposite property. It protects against the fields you
thought of on the day you wrote it, and silently stops protecting the moment
someone adds a field you did not — which is precisely when you need it.

### And then an independent check, because the allowlist can be edited

`pipeline/redact.py` re-reads the finished JSON and refuses to let it through if
it finds contact details, image URLs, coordinates, postal codes, or a key whose
*name* suggests any of those. It is a separate module for two reasons. It shares
no code with the builder, so it is a genuine second opinion rather than the same
assumption asserted twice. And it depends on nothing outside the standard
library, so CI can verify the bytes actually committed without being able to
build a snapshot at all — a check that always runs beats a better one that
sometimes does not.

`tests/test_publish.py` feeds it deliberately poisoned input. A safety check
that has only ever seen clean data is not known to work; it is known to be
silent, which is also what it would be if it were broken. The pre-commit hook in
`.githooks/` applies a coarser version of the same idea at commit time. Three
layers, because this is the one failure that cannot be undone afterwards by
fixing the code.

### What is published, and what is not

| Published | Withheld |
|---|---|
| Title (scrubbed) | The description, in full |
| Price, city, condition | Postal code, coordinates — never ingested at all |
| Canonical part names | Seller name, phone, email |
| Index medians and sample counts | Image URLs |
| The model's risk flags and prose | The raw listing record |
| A link back to the original ad | Anything not in this table |

The link back is deliberate: the site is a pointer to Blocket, not a substitute
for it. Every listing is attributed and every price is one click from its
source.

### Why the cap is per profile *and* per category

Capping the file to a top-N per category needs a sort order, and the sort order
is profile-dependent — that is the entire premise of `pipeline/profiles.py`. Cap
by one profile's ranking and the other two get a leaderboard filtered by
criteria they disagree with, which looks like a working selector while quietly
being a broken one.

So the snapshot carries the union of each profile's own top-N per category. At
most three times the rows, a few hundred kilobytes, and every profile is
complete to its stated depth. The alternative — publishing three files that can
disagree about the same listing — is worse in every dimension except size.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from ai.scrub import scrub
from db.models import Ad, AiEvaluation, ComponentTypeEnum, ItemTypeEnum, StatusEnum
from pipeline.basket import build_or_buy
from pipeline.extract import find_damage_markers
from pipeline.index import price_index, value_ad
from pipeline.redact import (
    RedactionError,
    assert_capped,
    assert_redacted,
    verify_file,
)
from pipeline.profiles import (
    PROFILES,
    Profile,
    describe_all,
    evaluate,
    machine_spec,
    rank_key,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("web/public/deals.json")

# Per category, per profile. Ten is enough to browse and few enough that the
# public artifact is a leaderboard rather than a copy of the collection — the
# distinction the EU database-right case law turns on. See ETHICS.md.
PER_CATEGORY_LIMIT = 10

SOURCE = "blocket"

# Categories the leaderboard is bucketed into. Anything that does not land in
# one of these is not published at all, rather than swept into a junk drawer.
PUBLISHED_CATEGORIES: tuple[str, ...] = (
    "FULL_BUILD",
    ComponentTypeEnum.GPU.value,
    ComponentTypeEnum.CPU.value,
    ComponentTypeEnum.RAM.value,
    ComponentTypeEnum.STORAGE.value,
    ComponentTypeEnum.MOTHERBOARD.value,
)

# Risk flags are, by name, risks. These are the exceptions — phrases that are
# reassurance rather than warning, and would be misleading rendered in amber.
_POSITIVE_MARKERS: tuple[str, ...] = (
    "kvitto finns",
    "garanti",
    "original",
    "oanvänd",
    "nyskick",
    "receipt included",
    "warranty",
    "unused",
)


# --------------------------------------------------------------------------
# Building the snapshot
# --------------------------------------------------------------------------


def _slug(text: str) -> str:
    """A stable identifier for a free-text risk flag."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "flag"


def _severity(label: str) -> str:
    """Three levels, because "defekt" and "ej testad" are not the same warning.

    Rendered identically in amber, "no receipt" and "the machine is dead" read
    as the same mild caveat. They are not: one is a hedge about provenance and
    the other is the seller stating the item does not work.
    """
    lowered = label.lower()
    if any(marker in lowered for marker in _POSITIVE_MARKERS):
        return "info"
    return "high" if find_damage_markers(lowered) else "warn"


def _flags(ad: Ad, evaluation: AiEvaluation) -> list[dict[str, str]]:
    """The model's flags, plus one the listing text earns on its own.

    A synthesised flag rather than a trusted one: the model raised "defekt" for
    ad 13, but a deterministic verdict raises nothing at all, and "DEFEKT ... –
    Defekt" in a title should not be a warning only when a model happened to
    read it.
    """
    flags = [
        {"id": _slug(label), "label": scrub(label), "severity": _severity(label)}
        for label in evaluation.flags
    ]
    if not any(flag["severity"] == "high" for flag in flags):
        markers = find_damage_markers(ad.title, ad.raw_description)
        if markers:
            flags.insert(
                0,
                {"id": _slug(markers[0]), "label": scrub(markers[0]), "severity": "high"},
            )
    return flags


def _category(ad: Ad) -> str | None:
    """Which leaderboard bucket a listing belongs in, or None to skip it.

    A single-component ad is filed under the part it is, which requires knowing
    what that part is. An ad the extractor could not resolve to exactly one
    component type has no bucket, and gets no entry — publishing it under a
    guess would put a motherboard in the graphics-card leaderboard.
    """
    if ad.item_type is ItemTypeEnum.FULL_BUILD:
        return "FULL_BUILD"
    if ad.item_type is not ItemTypeEnum.SINGLE_COMPONENT:
        return None
    types = {c.component_type for c in ad.extracted_components}
    if len(types) != 1:
        return None
    value = next(iter(types)).value
    return value if value in PUBLISHED_CATEGORIES else None


async def _upgrade_prices(session: AsyncSession, profile: Profile) -> dict[str, int]:
    """Observed medians for the parts a profile knows how to upgrade."""
    prices: dict[str, int] = {}
    for key in profile.upgrade_price_keys():
        index = await price_index(session, key)
        if index.median_sek is not None and index.is_confident:
            prices[key] = index.median_sek
    return prices


def _fit_payload(fit) -> dict[str, Any]:
    return {
        "profileId": fit.profile_id,
        "verdict": fit.verdict,
        "met": list(fit.met),
        "unmet": list(fit.unmet),
        "unknown": list(fit.unknown),
        "blockedBy": list(fit.blocked_by),
        "upgradeCostSek": fit.upgrade_cost_sek,
        "effectivePriceSek": fit.effective_price_sek,
        "metric": fit.metric_label,
        "metricUnit": fit.metric_unit,
        "metricValue": fit.metric_value,
        "vramCaveat": fit.vram_caveat,
    }


def _deal(ad: Ad, valuation, fits: dict[str, Any], category: str) -> dict[str, Any]:
    """One published listing. Every field is named here; nothing is copied.

    `title` and the model's prose are the only free text that survives, and both
    go through `scrub` on the way out even though the ingest layer already
    scrubbed what it stored. Cheap, and the one place where paying twice for the
    same guarantee is obviously worth it.
    """
    evaluation = ad.ai_evaluation
    return {
        "id": str(ad.id),
        "title": scrub(ad.title),
        "priceSek": ad.price_sek,
        "url": ad.url,
        "category": category,
        "city": ad.city,
        "listedAt": ad.listed_at.isoformat() if ad.listed_at else None,
        "condition": ad.condition,
        "math": {
            "estimatedPartsSek": valuation.estimated_parts_sek if valuation else None,
            "deltaSek": valuation.delta_sek if valuation else None,
            "minSampleCount": valuation.min_sample_count if valuation else 0,
            "components": [
                {
                    "type": part.component_type.value,
                    "name": part.canonical_name,
                    "medianSek": part.index.median_sek,
                    "sampleCount": part.index.sample_count,
                }
                for part in (valuation.parts if valuation else ())
            ],
        },
        "ai": {
            "summary": scrub(evaluation.ai_reasoning),
            "flags": _flags(ad, evaluation),
            "confidence": evaluation.confidence,
        }
        if evaluation
        else None,
        "fits": fits,
    }


def _basket_payload(profile_id: str, result) -> dict[str, Any]:
    basket = result.basket
    return {
        "profileId": profile_id,
        "totalSek": basket.total_sek,
        "sellerCount": basket.seller_count,
        "isComplete": basket.is_complete,
        "missingSlots": [slot.value for slot in basket.missing_slots],
        # Travels with the data, not as a UI footnote. A parts total that omits
        # motherboard, PSU and case is cheaper than a whole computer by exactly
        # the amount it left out.
        "excludedSlots": [slot.value for slot in basket.excluded_slots],
        "items": [
            {
                "slot": item.slot.value,
                "name": item.canonical_name,
                "priceSek": item.price_sek,
                "title": scrub(item.title),
                "city": item.city,
                "url": item.url,
            }
            for item in basket.items
        ],
        "bestPrebuiltId": str(result.best_prebuilt_ad_id)
        if result.best_prebuilt_ad_id is not None
        else None,
        "bestPrebuiltPriceSek": result.best_prebuilt_price_sek,
        "savingSek": result.saving_sek,
    }


async def build_snapshot(
    session: AsyncSession,
    *,
    per_category_limit: int = PER_CATEGORY_LIMIT,
    budget_sek: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the public snapshot from the local database.

    Read-only. Produces plain JSON-serialisable data and touches no files, so it
    is testable without a filesystem and callable from the API if that is ever
    wanted.
    """
    now = now or datetime.now(timezone.utc)

    stmt = (
        select(Ad)
        .join(AiEvaluation)
        .where(Ad.status == StatusEnum.ACTIVE)
        .options(selectinload(Ad.ai_evaluation), selectinload(Ad.extracted_components))
    )
    ads = list((await session.execute(stmt)).scalars().all())

    upgrade_prices = {p.id: await _upgrade_prices(session, p) for p in PROFILES}

    # One pass over the ads, building every profile's view at once. Valuation
    # hits the index per ad and is the expensive part, so it happens once here
    # rather than once per profile.
    scored: list[tuple[dict[str, Any], dict[str, tuple]]] = []
    for ad in ads:
        category = _category(ad)
        if category is None:
            continue

        valuation = await value_ad(session, ad.id, now=now)
        # Both exclusions, and both were learned the same way.
        #
        # A stock list summed against one asking price is a spectacular
        # fictional bargain, and an asking price far below the parts it claims
        # to contain is not an asking price. Either lands at the *top* of a
        # leaderboard, because cheapness is what the ranking rewards.
        #
        # `pipeline.basket` already refused both. This call site did not, and
        # the first real run of this module published ad 289 — "i9-13900 | RTX
        # 4070 Ti | 64GB DDR5 | 4TB NVMe" at 395 kr, the very listing the
        # plausibility floor was written for — as the number-one deal on the
        # public site. A guard that exists is not a guard that is applied, and
        # the two places that need it are the two places least able to share
        # code. Hence `test_the_395_kr_listing_never_reaches_the_public_file`.
        if valuation is not None and (
            valuation.is_inventory_listing or valuation.is_implausibly_cheap
        ):
            logger.info(
                "Withholding ad %s at %d kr: %s.",
                ad.id,
                ad.price_sek,
                "an inventory listing, not one machine"
                if valuation.is_inventory_listing
                else f"implausible against ~{valuation.estimated_parts_sek} kr of parts",
            )
            continue

        spec = machine_spec(
            (
                (component.component_type, component.canonical_name)
                for component in ad.extracted_components
                if component.canonical_name
            ),
            text_hint=ad.title,
        )
        confidence = ad.ai_evaluation.confidence if ad.ai_evaluation else 1.0
        delta = valuation.delta_sek if valuation else None

        fits: dict[str, Any] = {}
        ranks: dict[str, tuple] = {}
        for profile in PROFILES:
            fit = evaluate(
                profile, spec, ad.price_sek, upgrade_prices=upgrade_prices[profile.id]
            )
            fits[profile.id] = _fit_payload(fit)
            ranks[profile.id] = rank_key(fit, delta, confidence)

        scored.append((_deal(ad, valuation, fits, category), ranks))

    deals = _select_top(scored, per_category_limit)

    baskets = {}
    for profile in PROFILES:
        result = await build_or_buy(
            session, profile, max_price_sek=budget_sek, now=now
        )
        baskets[profile.id] = _basket_payload(profile.id, result)

    return {
        "generatedAt": now.isoformat(),
        "source": SOURCE,
        "perCategoryLimit": per_category_limit,
        "deals": deals,
        "profiles": list(describe_all()),
        "baskets": baskets,
    }


# The client's budget presets, tightest first, with "any price" last.
#
# Duplicated from `web/src/App.tsx` rather than derived, and that duplication is
# checked by `test_publish.py`. The alternative — publishing the bands and
# having the client read them — would let a snapshot silently redefine the
# filter the user thinks they are applying.
BUDGET_BANDS: tuple[int | None, ...] = (6000, 8000, 10000, None)


def _effective_price(deal: dict[str, Any], profile_id: str) -> int | None:
    """What this machine costs a buyer with this profile, upgrades included."""
    return deal["fits"][profile_id]["effectivePriceSek"]


def _select_for_profile(
    bucket: Sequence[tuple[dict[str, Any], dict[str, tuple]]],
    profile_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """One profile's slice of one category, stratified across budget bands.

    ### The bug this replaces

    The cap ran first and the budget filter ran second, in the browser. So the
    file carried the ten best machines *at any price*, and a visitor on a 6 000
    kr budget saw whatever survived — which was usually almost nothing. The
    database held 218 active builds under 6 000 kr and the snapshot published
    eleven of them. The site looked like a market with no affordable machines
    in it, and the market was not the thing that was empty.

    ### Why the cap is not simply raised

    Because the cap is not a UI preference. It is the load-bearing part of the
    *Innoweb* argument in ETHICS.md: this project publishes a leaderboard, not
    a copy of Blocket's collection, and `assert_capped` enforces that as an
    invariant. Giving each of four bands its own top-ten would multiply the
    published rows by four and take the file from ~9% of the corpus to most of
    it, dismantling that reasoning to fix a presentation bug.

    So the budget bands are spent *inside* the existing cap. Each bounded band
    reserves a couple of the ten slots for machines that actually fit it, and
    the unbounded pass fills whatever remains. Same number of published rows,
    composition that answers the question a shopper is really asking.
    """
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    reserved = max(1, limit // len(BUDGET_BANDS))

    for band in BUDGET_BANDS:
        if band is None:
            pool = list(bucket)
            quota = limit  # the last pass fills every slot still open
        else:
            pool = [
                entry
                for entry in bucket
                if (price := _effective_price(entry[0], profile_id)) is not None
                and price <= band
            ]
            quota = reserved

        taken = 0
        for deal, _ in sorted(pool, key=lambda entry: entry[1][profile_id]):
            if len(chosen) >= limit or taken >= quota:
                break
            if deal["id"] in seen:
                continue
            chosen.append(deal)
            seen.add(deal["id"])
            taken += 1

    return chosen


def _select_top(
    scored: Sequence[tuple[dict[str, Any], dict[str, tuple]]], limit: int
) -> list[dict[str, Any]]:
    """Union of each profile's top-N per category. See the module docstring.

    Deduplicated by ad id, because a listing that is good for two profiles is
    still one listing — and it carries all three verdicts anyway, so there is
    nothing profile-specific about the row to duplicate.
    """
    keep: dict[str, dict[str, Any]] = {}
    categories = {deal["category"] for deal, _ in scored}

    for category in sorted(categories):
        bucket = [entry for entry in scored if entry[0]["category"] == category]
        for profile in PROFILES:
            for deal in _select_for_profile(bucket, profile.id, limit):
                keep[deal["id"]] = deal

    # Sorted by the default profile so the file has a stable, meaningful order
    # even before the client applies its own. Ties broken by id to keep diffs
    # between nightly runs readable.
    default = PROFILES[0].id
    order = {deal["id"]: ranks[default] for deal, ranks in scored}
    return sorted(keep.values(), key=lambda d: (order[d["id"]], d["id"]))


# --------------------------------------------------------------------------
# Writing it out
# --------------------------------------------------------------------------


async def publish(
    session_factory: async_sessionmaker,
    *,
    output: Path = DEFAULT_OUTPUT,
    per_category_limit: int = PER_CATEGORY_LIMIT,
    budget_sek: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build, check, then write. In that order, and the check cannot be skipped.

    There is no `force` flag and no way to write an unchecked snapshot. A
    parameter that disables the safety check is a parameter that will eventually
    be passed.
    """
    async with session_factory() as session:
        snapshot = await build_snapshot(
            session,
            per_category_limit=per_category_limit,
            budget_sek=budget_sek,
            now=now,
        )

    assert_redacted(snapshot)
    assert_capped(snapshot)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Indented and newline-terminated: this file is committed, so a readable
    # diff between nightly runs is worth the extra bytes. `ensure_ascii=False`
    # keeps Swedish place names as themselves rather than escape sequences.
    output.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "Published %d deals across %d categories to %s",
        len(snapshot["deals"]),
        len({d["category"] for d in snapshot["deals"]}),
        output,
    )
    return snapshot


# `verify_file` is re-exported from `pipeline.redact` above, so callers that
# think of verification as part of publishing can find it here, while CI can
# import it without the database stack.


def published_ad_ids(path: Path = DEFAULT_OUTPUT) -> tuple[int, ...]:
    """Ad ids in the published snapshot, or empty if there is not one yet.

    Reads the finished artifact rather than re-deriving the ranking. Two copies
    of "which ads are on the board" would drift, and the whole point of the
    caller — `pipeline.run.stage_janitor` — is to verify precisely what a
    visitor sees.

    Never raises. A missing or malformed file means "no priorities known", and
    the janitor falls back to its ordinary oldest-first rotation; a first run,
    before anything has been published, must not be a failure.
    """
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            int(deal["id"])
            for deal in snapshot.get("deals", ())
            if str(deal.get("id", "")).isdigit()
        )
    except (OSError, ValueError, TypeError):
        logger.warning("No usable snapshot at %s; janitor will not prioritise.", path)
        return ()
