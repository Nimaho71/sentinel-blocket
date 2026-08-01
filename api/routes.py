"""Local development API.

**Not deployed.** The public site is a static bundle reading a redacted
`deals.json`; this exists so the pipeline's output can be browsed unredacted on
the machine that produced it. `api/app.py` binds it to localhost only.

The response shape deliberately mirrors `web/src/types.ts`, so the dashboard can
be pointed at either this or the published snapshot without a code change.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db.connection import get_session
from db.models import Ad, AiEvaluation, StatusEnum
from pipeline.basket import build_or_buy
from pipeline.index import price_index, value_ad
from pipeline.profiles import describe_all, evaluate, machine_spec, rank_key, resolve

router = APIRouter()


async def _upgrade_prices(session: AsyncSession, profile) -> dict[str, int]:
    """Current medians for the parts a profile knows how to upgrade.

    Sourcing these from the index rather than the hardcoded fallbacks keeps the
    "count, don't guess" rule intact for the one number profiles add to a
    listing's price. A part the index has not seen falls back to the estimate in
    `DEFAULT_UPGRADE_SEK`, which is deliberately pitched high.
    """
    prices: dict[str, int] = {}
    for key in profile.upgrade_price_keys():
        index = await price_index(session, key)
        if index.median_sek is not None and index.is_confident:
            prices[key] = index.median_sek
    return prices


@router.get("/api/profiles")
async def get_profiles():
    """Profile metadata for the selector, including each profile's rationale.

    The rationale ships to the UI on purpose. A filter that silently drops a
    listing the user can see elsewhere is infuriating; one that says "8 GB VRAM,
    below this profile's floor" is a feature.
    """
    return describe_all()


@router.get("/api/deals")
async def get_deals(
    limit: int = 50,
    profile: str | None = None,
    max_price_sek: int | None = Query(None, alias="maxPrice"),
    include_unfit: bool = Query(False, alias="includeUnfit"),
    session: AsyncSession = Depends(get_session),
):
    """Evaluated live ads, best fit-then-value first.

    Ranking is `pipeline.profiles.rank_key`: qualifying machines, then the
    profile's own metric, then the deterministic delta scaled by the AI's
    confidence — the "math owns value, AI owns risk" split. There is no
    `deal_score` any more: a single opaque 0-100 from a language model was
    neither auditable nor reproducible, and it was the field that silently
    defaulted to zero.

    **`maxPrice` is applied here and not at ingest.** Filtering the scrape to a
    budget would bias the index, because a median computed only over cheap
    listings sits low and makes everything look like a bad deal. The filter
    would corrupt the very number used to judge. So ingest stays broad and the
    budget is a presentation concern.
    """
    selected = resolve(profile)
    upgrade_prices = await _upgrade_prices(session, selected)

    stmt = (
        select(Ad)
        .join(AiEvaluation)
        .where(Ad.status == StatusEnum.ACTIVE)
        .options(selectinload(Ad.ai_evaluation), selectinload(Ad.extracted_components))
    )
    if max_price_sek is not None:
        stmt = stmt.where(Ad.price_sek <= max_price_sek)
    ads = list((await session.execute(stmt)).scalars().all())

    payload = []
    for ad in ads:
        valuation = await value_ad(session, ad.id)
        evaluation = ad.ai_evaluation

        spec = machine_spec(
            (
                (component.component_type, component.canonical_name)
                for component in ad.extracted_components
                if component.canonical_name
            ),
            text_hint=ad.title,
        )
        fit = evaluate(
            selected, spec, ad.price_sek, upgrade_prices=upgrade_prices
        )
        if not include_unfit and not fit.is_viable:
            continue

        payload.append(
            {
                "id": ad.id,
                "title": ad.title,
                "priceSek": ad.price_sek,
                "city": ad.city,
                "condition": ad.condition,
                "itemType": ad.item_type.value,
                "url": ad.url,
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
                    "coverage": valuation.coverage if valuation else 0.0,
                },
                "ai": {
                    "summary": evaluation.ai_reasoning,
                    "flags": evaluation.flags,
                    "confidence": evaluation.confidence,
                }
                if evaluation
                else None,
                "fit": {
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
                },
                # Held aside for sorting, then dropped — see below.
                "_rank": rank_key(
                    fit,
                    valuation.delta_sek if valuation else None,
                    evaluation.confidence if evaluation else 1.0,
                ),
            }
        )

    # Sorted here rather than in SQL: the delta comes from per-part medians
    # computed in Python, and the profile fit from a spec table, so the database
    # can order by neither.
    payload.sort(key=lambda entry: entry["_rank"])
    for entry in payload:
        del entry["_rank"]
    return payload[:limit]


@router.get("/api/basket")
async def get_basket(
    profile: str | None = None,
    max_price_sek: int | None = Query(None, alias="maxPrice"),
    session: AsyncSession = Depends(get_session),
):
    """Build-versus-buy: the cheapest qualifying prebuilt against a parts basket.

    The excluded slots travel with the response rather than being footnoted in
    the UI. A basket total that omits motherboard, PSU and case is not
    comparable to a whole computer, and a client that renders the number without
    the caveat is misrepresenting it.
    """
    selected = resolve(profile)
    result = await build_or_buy(session, selected, max_price_sek=max_price_sek)
    basket = result.basket

    return {
        "profileId": selected.id,
        "basket": {
            "totalSek": basket.total_sek,
            "sellerCount": basket.seller_count,
            "isComplete": basket.is_complete,
            "missingSlots": [slot.value for slot in basket.missing_slots],
            "excludedSlots": [slot.value for slot in basket.excluded_slots],
            "items": [
                {
                    "slot": item.slot.value,
                    "name": item.canonical_name,
                    "priceSek": item.price_sek,
                    "title": item.title,
                    "city": item.city,
                    "url": item.url,
                }
                for item in basket.items
            ],
        },
        "bestPrebuilt": {
            "id": result.best_prebuilt_ad_id,
            "title": result.best_prebuilt_title,
            "effectivePriceSek": result.best_prebuilt_price_sek,
        }
        if result.best_prebuilt_ad_id is not None
        else None,
        "savingSek": result.saving_sek,
        "summary": result.describe(),
    }
