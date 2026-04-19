from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database.models import Ad, StatusEnum

async def upsert_ad(session: AsyncSession, ad_data: dict) -> Ad:
    """
    Upserts an ad into the database.
    If the blocket_id already exists, it updates the price and last_checked_at.
    If it doesn't exist, it creates a new Ad record.
    """
    blocket_id = ad_data.get("blocket_id")
    if not blocket_id:
        raise ValueError("ad_data must contain 'blocket_id'")

    # Try to find the existing ad
    stmt = select(Ad).where(Ad.blocket_id == blocket_id)
    result = await session.execute(stmt)
    existing_ad = result.scalars().first()

    now = datetime.now(timezone.utc)

    if existing_ad:
        # Update fields if ad already exists
        existing_ad.price_sek = ad_data.get("price_sek", existing_ad.price_sek)
        existing_ad.main_image_url = ad_data.get("main_image_url", existing_ad.main_image_url) # Update
        existing_ad.last_checked_at = now
        
        # If we see it again, it means it's active. Let's make sure status reflects that.
        existing_ad.status = StatusEnum.ACTIVE
        
        ad = existing_ad
    else:
        # Insert new ad
        ad = Ad(
            blocket_id=blocket_id,
            url=ad_data.get("url"),
            title=ad_data.get("title"),
            raw_description=ad_data.get("raw_description"),
            price_sek=ad_data.get("price_sek"),
            main_image_url=ad_data.get("main_image_url"), # Save image
            category=ad_data.get("category"),
            status=StatusEnum.ACTIVE,
            scraped_at=now,
            last_checked_at=now
        )
        session.add(ad)

    # Commit changes and refresh object to get the assigned DB ID
    await session.commit()
    await session.refresh(ad)
    
    return ad
