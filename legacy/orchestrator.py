import asyncio
import logging
from playwright.async_api import async_playwright
from db.connection import init_db, AsyncSessionLocal
from db.crud import upsert_ad, ad_exists
from sqlalchemy import select
from db.models import Ad, CategoryEnum

from scraper.stealth_config import create_stealth_context, setup_stealth_page
from scraper.crawler import search_and_fetch_urls, scrape_ad_details
from ai.batch_processor import process_unscored_ads
from pipeline.janitor import run_janitor
from ai.virtual_builder import analyze_full_build_value

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def process_single_ad(context, ad_url, category, session, semaphore):
    """Worker function to process a single ad concurrently in its own tab."""
    async with semaphore:  # Wait for a "green light" to proceed
        # Open a new tab specifically for this ad
        from scraper.stealth_config import setup_stealth_page
        page = await setup_stealth_page(context)
        
        try:
            ad_data = await scrape_ad_details(page, ad_url, category)
            if ad_data:
                await upsert_ad(session, ad_data)
        except Exception as e:
            logger.error(f"Error concurrently scraping {ad_url}: {e}")
        finally:
            await page.close() # Always close the tab to free up RAM!

async def run_pipeline():
    logger.info("Initializing Database...")
    await init_db()

    search_targets = [
        # Components to establish market averages
        {"q": "RTX 3060", "cat": CategoryEnum.COMPONENT},
        {"q": "RTX 3070", "cat": CategoryEnum.COMPONENT},
        {"q": "RTX 3080", "cat": CategoryEnum.COMPONENT},
        {"q": "RTX 4070", "cat": CategoryEnum.COMPONENT},
        {"q": "RTX 4080", "cat": CategoryEnum.COMPONENT},
        {"q": "Ryzen 5600X", "cat": CategoryEnum.COMPONENT},
        {"q": "Ryzen 7800X3D", "cat": CategoryEnum.COMPONENT},
        # Full builds to find deals
        {"q": "Gaming dator", "cat": CategoryEnum.STATIONARY},
        {"q": "Speldator", "cat": CategoryEnum.STATIONARY},
    ]


    logger.info("Starting Scraper Phase...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await create_stealth_context(browser)
        page = await setup_stealth_page(context)

        for target in search_targets:
            logger.info(f"🚀 TARGETED SEARCH: {target['q']} in {target['cat']}")
            # Use cg=5023 for components, cg=5022 for stationary
            cg = "5023" if target['cat'] == CategoryEnum.COMPONENT else "5022"
            base_url = f"https://www.blocket.se/recommerce/forsale/search?cg={cg}&q={target['q']}"
            
            # Scrape 2 pages of each to get a wide sample size
            for page_num in range(1, 3): 
                url_with_page = f"{base_url}&page={page_num}"
                await page.goto(url_with_page, wait_until="domcontentloaded")
                from scraper.crawler import accept_cookies, simulate_human_behavior
                await accept_cookies(page)
                
                # Extract URLs
                links = await page.locator("a").all()
                ad_urls = []
                for link in links:
                    href = await link.get_attribute("href")
                    if href and "/recommerce/forsale/item/" in href:
                        ad_urls.append(f"https://www.blocket.se{href}" if href.startswith("/") else href)

                # -- HIGH PERFORMANCE CONCURRENT SCRAPING --
                # Change the number '5' below based on your computer's RAM. 
                # 5 = 5 tabs open at once.
                semaphore = asyncio.Semaphore(5) 
                
                # Use one session for the check AND the tasks
                async with AsyncSessionLocal() as session:
                    unique_urls = list(set(ad_urls))
                    new_urls = []
                    
                    for url in unique_urls:
                        # Now this will work because of the import!
                        if not await ad_exists(session, url):
                            new_urls.append(url)
                    
                    if not new_urls:
                        logger.info(f"All ads on page {page_num} already exist in DB. Skipping...")
                        continue

                    logger.info(f"Found {len(new_urls)} NEW ads. Scraping concurrently...")
                    tasks = [
                        process_single_ad(context, url, target['cat'], session, semaphore) 
                        for url in new_urls
                    ]
                    await asyncio.gather(*tasks)
                
                logger.info(f"Finished page {page_num} for {target['q']}")

        await browser.close()


    # --- 2. AI INTELLIGENCE PHASE ---
    logger.info("Starting AI Analysis Phase...")
    async with AsyncSessionLocal() as session:
        await process_unscored_ads(session, batch_size=15)
    
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