import asyncio
import logging
from playwright.async_api import async_playwright
from database.connection import init_db, AsyncSessionLocal
from database.crud import upsert_ad
from sqlalchemy import select
from database.models import Ad, CategoryEnum

from scraper.stealth_config import create_stealth_context, setup_stealth_page
from scraper.crawler import search_and_fetch_urls, scrape_ad_details
from ai_engine.batch_processor import process_unscored_ads
from scraper.janitor import run_janitor
from ai_engine.virtual_builder import analyze_full_build_value

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_pipeline():
    logger.info("Initializing Database...")
    await init_db()

    logger.info("Starting Scraper Phase...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await create_stealth_context(browser)
        page = await setup_stealth_page(context)
        
        # CATEGORY 5023 = Components | CATEGORY 5022 = Stationary PCs
        tasks = [
            {"url": "https://www.blocket.se/recommerce/forsale/search?cg=5023&q=gtx+1080", "cat": CategoryEnum.COMPONENT},
            {"url": "https://www.blocket.se/recommerce/forsale/search?cg=5022&q=gaming+dator", "cat": CategoryEnum.STATIONARY}
        ]
        
        for task in tasks:
            logger.info(f"--- Scraping URL: {task['url']} ---")
            await page.goto(task['url'], wait_until="domcontentloaded")
            from scraper.crawler import accept_cookies, simulate_human_behavior
            await accept_cookies(page)
            await simulate_human_behavior(page)

            # Manual URL extraction for this run
            links = await page.locator("a").all()
            ad_urls = []
            for link in links:
                href = await link.get_attribute("href")
                if href and ("/item/" in href or "/annons/" in href):
                    full_url = f"https://www.blocket.se{href}" if href.startswith("/") else href
                    if full_url not in ad_urls: ad_urls.append(full_url)

            async with AsyncSessionLocal() as session:
                for url in ad_urls[:5]: # Get 5 of each
                    ad_data = await scrape_ad_details(page, url, task['cat'])
                    if ad_data:
                        await upsert_ad(session, ad_data)
        
        await browser.close()


    # --- 2. AI INTELLIGENCE PHASE ---
    logger.info("Starting AI Analysis Phase...")
    async with AsyncSessionLocal() as session:
        await process_unscored_ads(session, batch_size=10)
    
    # --- 3. VIRTUAL BUILDER PHASE ---
    logger.info("Starting Virtual Builder Analysis...")
    async with AsyncSessionLocal() as session:
        # Find the latest stationary PC that we just analyzed
        stmt = (
            select(Ad)
            .where(Ad.category == CategoryEnum.STATIONARY)
            .order_by(Ad.id.desc())
        )
        result = await session.execute(stmt)
        ad_to_analyze = result.scalars().first()
        
        if ad_to_analyze:
            # This function will now find the STREATIONARY ad and look up prices
            # for the components Gemini extracted from it!
            await analyze_full_build_value(session, ad_to_analyze.id)
        else:
            logger.warning("No stationary PCs found in DB to analyze.")

    # --- 4. JANITOR PHASE ---
    logger.info("Starting Janitor Phase...")
    async with AsyncSessionLocal() as session:
        await run_janitor(session)
    
    logger.info("🎉 Sentinel-Blocket Pipeline Run Complete!")

if __name__ == "__main__":
    asyncio.run(run_pipeline())