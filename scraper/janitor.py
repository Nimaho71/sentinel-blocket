import asyncio
import logging
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from database.models import Ad, StatusEnum

logger = logging.getLogger(__name__)

# Modern browsers often have this User-Agent structure
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
}

async def check_ad_status(client: httpx.AsyncClient, ad: Ad) -> StatusEnum:
    """
    Checks a single ad URL. Returns ACTIVE if it's live, SOLD if it's not.
    Blocket redirects sold/deleted ads to the homepage, which we can detect.
    """
    try:
        # We use a HEAD request because we only need the status code, not the full HTML body. It's much faster.
        # allow_redirects=True is crucial because Blocket's response is a redirect.
        response = await client.head(ad.url, headers=HEADERS, timeout=10, follow_redirects=True)
        
        # If the final URL after redirects is the Blocket homepage, the ad is gone.
        if "blocket.se" == response.url.host and response.url.path == '/':
            logger.info(f"Ad {ad.blocket_id} has been removed (redirected to homepage).")
            return StatusEnum.SOLD
        
        # If we get a 404, it's also gone.
        if response.status_code == 404:
            logger.info(f"Ad {ad.blocket_id} returned 404 Not Found.")
            return StatusEnum.SOLD
            
        return StatusEnum.ACTIVE

    except httpx.RequestError as e:
        logger.warning(f"Network error checking ad {ad.blocket_id}: {e}")
        # We assume it's still active if we can't connect, to be safe.
        return StatusEnum.ACTIVE

async def run_janitor(session: AsyncSession):
    """
    Finds all active ads and runs a check on them, updating their status if they've been sold.
    """
    logger.info("🧹 The Janitor is starting its daily cleaning run...")
    
    # 1. Get all ads that are currently marked as ACTIVE
    stmt = select(Ad).where(Ad.status == StatusEnum.ACTIVE)
    result = await session.execute(stmt)
    active_ads = result.scalars().all()

    if not active_ads:
        logger.info("No active ads to check. Janitor is finished.")
        return

    logger.info(f"Found {len(active_ads)} active ads to verify.")
    
    ads_to_mark_sold = []
    
    # Use httpx.AsyncClient for connection pooling and efficiency
    async with httpx.AsyncClient() as client:
        # Create a list of tasks to run concurrently
        tasks = [check_ad_status(client, ad) for ad in active_ads]
        results = await asyncio.gather(*tasks)

        for ad, status in zip(active_ads, results):
            if status == StatusEnum.SOLD:
                ads_to_mark_sold.append(ad.id)

    if not ads_to_mark_sold:
        logger.info("All active ads are still live. No changes made.")
        return

    # 3. Update the database in a single bulk operation
    try:
        update_stmt = (
            update(Ad)
            .where(Ad.id.in_(ads_to_mark_sold))
            .values(status=StatusEnum.SOLD)
        )
        await session.execute(update_stmt)
        await session.commit()
        logger.info(f"✅ Successfully marked {len(ads_to_mark_sold)} ads as SOLD.")
    except Exception as e:
        await session.rollback()
        logger.error(f"❌ Janitor failed to update database: {e}")
