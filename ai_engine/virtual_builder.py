import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from database.models import Ad, ExtractedComponent, CategoryEnum, ItemTypeEnum

logger = logging.getLogger(__name__)

async def get_average_component_price(session: AsyncSession, component_name: str) -> float:
    """
    Calculates the average price of an individual component by querying all 'COMPONENT' ads.
    """
    stmt = (
        select(func.avg(Ad.price_sek))
        .join(ExtractedComponent)
        # ONLY look at ads the AI confirmed are SINGLE components
        .where(Ad.item_type == ItemTypeEnum.SINGLE_COMPONENT) 
        .where(ExtractedComponent.component_name.ilike(f"%{component_name}%"))
    )
    result = await session.execute(stmt)
    average_price = result.scalar_one_or_none()
    
    return float(average_price) if average_price else 0.0

async def analyze_full_build_value(session: AsyncSession, ad_id: int):
    """
    Analyzes a 'STATIONARY' (Full Build) ad by comparing its price to the
    sum of its virtually-built parts.
    """
    # 1. Fetch the full build ad and its already-extracted components
    stmt = (
        select(Ad)
        .where(Ad.id == ad_id)
        .where(Ad.category == CategoryEnum.STATIONARY)
        .options(selectinload(Ad.extracted_components)) # Eagerly load components
    )
    result = await session.execute(stmt)
    full_build_ad = result.scalars().first()

    if not full_build_ad:
        logger.warning(f"Could not find a 'STATIONARY' build with ad_id {ad_id} to analyze.")
        return

    logger.info(f"--- Starting Virtual Build Analysis for Ad ID: {full_build_ad.id} ---")
    logger.info(f"Listing Price: {full_build_ad.price_sek} kr for '{full_build_ad.title}'")

    virtual_build_price = 0.0
    
    # 2. Iterate through each component Gemini found
    for component in full_build_ad.extracted_components:
        # Query the database for the average price of this component when sold alone
        avg_price = await get_average_component_price(session, component.component_name)
        
        if avg_price > 0:
            logger.info(f"  - Found component '{component.component_name}'. Avg market price: {avg_price:.0f} kr")
            virtual_build_price += avg_price
        else:
            logger.info(f"  - Found component '{component.component_name}'. No market price data available.")

    logger.info(f"Total Virtual Build Price (sum of parts): {virtual_build_price:.0f} kr")

    # 3. Compare and give a final verdict
    price_difference = virtual_build_price - full_build_ad.price_sek
    
    if virtual_build_price == 0:
        logger.warning("Could not calculate a virtual price. Not enough component data in DB.")
    elif price_difference > 0:
        logger.info(f"✅ VERDICT: This build is a GOOD DEAL. It is approx. {price_difference:.0f} kr CHEAPER than buying parts separately.")
    else:
        logger.info(f"❌ VERDICT: This build is NOT a good deal. It is approx. {abs(price_difference):.0f} kr MORE EXPENSIVE than buying parts separately.")
        
    logger.info(f"--- End of Virtual Build Analysis for Ad ID: {full_build_ad.id} ---")
