from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from database.connection import get_session
from database.models import Ad, AiEvaluation

router = APIRouter()

@router.get("/api/deals")
async def get_deals(session: AsyncSession = Depends(get_session)):
    # Fetch ads that have an AI evaluation, ordered by deal_score
    stmt = (
        select(Ad)
        .join(AiEvaluation)
        .options(selectinload(Ad.ai_evaluation))
        .order_by(AiEvaluation.deal_score.desc())
        .limit(50)
    )
    result = await session.execute(stmt)
    ads = result.scalars().all()

    return [
        {
            "id": ad.id,
            "title": ad.title,
            "price": ad.price_sek,
            "image": ad.main_image_url, # Send the image URL
            "item_type": ad.item_type,
            "score": ad.ai_evaluation.deal_score if ad.ai_evaluation else 0,
            "reasoning": ad.ai_evaluation.ai_reasoning if ad.ai_evaluation else "",
            "url": ad.url
        }
        for ad in ads
    ]
