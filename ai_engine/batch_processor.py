import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
# 1. Added ItemTypeEnum to imports
from database.models import Ad, AiEvaluation, ExtractedComponent, ComponentTypeEnum, ItemTypeEnum
from ai_engine.gemini_client import analyze_ads_batch

logger = logging.getLogger(__name__)

def chunk_list(data: list, chunk_size: int):
    """Yield successive chunks from a list."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

async def process_unscored_ads(session: AsyncSession, batch_size: int = 10):
    """
    Queries the database for Ads without AI evaluations, batches them, 
    gets Gemini to score and CATEGORIZE them, and saves the results.
    """
    # Find ads that don't have an AI evaluation yet
    stmt = select(Ad).where(~Ad.ai_evaluation.has()).limit(50)
    result = await session.execute(stmt)
    unscored_ads = result.scalars().all()

    if not unscored_ads:
        logger.info("No unscored ads found for AI analysis.")
        return

    logger.info(f"Found {len(unscored_ads)} unscored ads. Processing in batches of {batch_size}...")

    for batch in chunk_list(unscored_ads, batch_size):
        # Prepare payload
        payload = [
            {
                "id": ad.id,
                "title": ad.title,
                "price_sek": ad.price_sek,
                "raw_description": ad.raw_description
            }
            for ad in batch
        ]

        # Send to Gemini
        ai_results = await analyze_ads_batch(payload)

        if not ai_results:
            continue

        for result_data in ai_results:
            try:
                ad_id = result_data.get("ad_id")
                if not ad_id:
                    continue
                
                # --- NEW: UPDATE THE AD CATEGORY ---
                # We find the specific ad object from our 'batch' list
                current_ad = next((a for a in batch if a.id == ad_id), None)
                if current_ad:
                    # Get the classification Gemini decided on
                    ai_type_str = result_data.get("item_type", "UNKNOWN")
                    try:
                        # Convert string to Enum (e.g., "SINGLE_COMPONENT" -> ItemTypeEnum.SINGLE_COMPONENT)
                        current_ad.item_type = ItemTypeEnum(ai_type_str.upper())
                        logger.info(f"AI classified Ad {ad_id} as {current_ad.item_type}")
                    except ValueError:
                        current_ad.item_type = ItemTypeEnum.UNKNOWN

                # --- SAVE THE EVALUATION ---
                evaluation = AiEvaluation(
                    ad_id=ad_id,
                    deal_score=result_data.get("deal_score", 0),
                    estimated_market_value_sek=result_data.get("estimated_market_value_sek", 0),
                    ai_reasoning=result_data.get("ai_reasoning", "No reasoning provided.")
                )
                session.add(evaluation)

                # --- SAVE THE COMPONENTS ---
                components_data = result_data.get("extracted_components", [])
                for comp in components_data:
                    comp_type_str = comp.get("component_type", "OTHER")
                    try:
                        comp_type = ComponentTypeEnum(comp_type_str.upper())
                    except ValueError:
                        comp_type = ComponentTypeEnum.OTHER

                    extracted_comp = ExtractedComponent(
                        ad_id=ad_id,
                        component_type=comp_type,
                        component_name=comp.get("component_name", "Unknown Part"),
                        estimated_individual_value=comp.get("estimated_individual_value")
                    )
                    session.add(extracted_comp)
                    
            except Exception as e:
                logger.error(f"Error processing AI result for ad_id {ad_id}: {str(e)}")

        # Commit everything for this batch
        try:
            await session.commit()
            logger.info(f"✅ Successfully updated categories and saved evaluations for batch.")
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Database commit failed: {str(e)}")