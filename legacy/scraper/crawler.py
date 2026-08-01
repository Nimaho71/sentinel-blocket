import asyncio
import random
import re
import json
from typing import List, Dict, Optional
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def simulate_human_behavior(page: Page):
    """The Stealth Protocol: Random sleep and mouse scrolling."""
    delay = random.uniform(3.0, 6.0)
    logger.info(f"Stealth Protocol: Sleeping for {delay:.2f} seconds...")
    await asyncio.sleep(delay)
    try:
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
        await asyncio.sleep(random.uniform(0.5, 1.5))
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.5)")
        await asyncio.sleep(random.uniform(0.5, 1.0))
        await page.evaluate("window.scrollBy(0, -300)")
    except Exception:
        pass 

async def accept_cookies(page: Page):
    """Looks for the cookie banner inside the Sourcepoint iframe and clicks Accept."""
    try:
        # 1. The banner is injected into an iframe that starts with 'sp_message_iframe'
        iframe_locator = page.frame_locator('iframe[id^="sp_message_iframe"]')
        btn = iframe_locator.get_by_role("button", name=re.compile(r"godkänn|acceptera", re.IGNORECASE)).first
        
        await btn.click(timeout=4000)
        logger.info("🍪 Clicked 'Godkänn alla' inside the cookie iframe.")
        await asyncio.sleep(1.5)
        return
    except Exception:
        pass

    try:
        # 2. Fallback to main page just in case
        btn = page.get_by_role("button", name=re.compile(r"godkänn|acceptera", re.IGNORECASE)).first
        await btn.click(timeout=2000)
        logger.info("🍪 Clicked 'Godkänn alla' on the main page.")
        await asyncio.sleep(1.5)
    except Exception:
        pass

async def check_for_captcha(page: Page) -> bool:
    content = await page.content()
    if "Access Denied" in content or "Please verify you are a human" in content or "cf-browser-verification" in content:
        return True
    return False

async def search_and_fetch_urls(page: Page, query: str, max_pages: int = 1) -> List[str]:
    """
    Simulates a human navigating to the homepage, typing a query, and extracting URLs.
    """
    ad_urls =[]
    logger.info("Navigating to Blocket Homepage...")
    await page.goto("https://www.blocket.se/", wait_until="domcontentloaded")
    
    await accept_cookies(page)
    
    logger.info(f"Typing search query: '{query}'")
    search_input = page.get_by_placeholder(re.compile(r"sök", re.IGNORECASE)).first
    await search_input.fill(query)
    await asyncio.sleep(random.uniform(0.5, 1.5))
    await search_input.press("Enter")
    
    await page.wait_for_load_state("domcontentloaded")
    await simulate_human_behavior(page)

    if await check_for_captcha(page):
        logger.error("CAPTCHA DETECTED! Pausing...")
        return[]

    logger.info("Scanning for ad links...")
    links = await page.locator("a").all()
    for link in links:
        href = await link.get_attribute("href")
        # Blocket uses /item/ or /annons/ for ad links now
        if href and ("/item/" in href or "/annons/" in href):
            full_url = f"https://www.blocket.se{href}" if href.startswith("/") else href
            if full_url not in ad_urls:
                ad_urls.append(full_url)
                
    logger.info(f"Found {len(ad_urls)} unique ad URLs.")
    return ad_urls

async def scrape_ad_details(page: Page, url: str, category_enum: str) -> Optional[Dict]:
    """Deep-Scrape Logic using JSON-LD metadata for bulletproof extraction."""
    logger.info(f"Deep scraping ad: {url}")
    try:
        await page.goto(url, wait_until="domcontentloaded")
    except PlaywrightTimeoutError:
        logger.error(f"Timeout while loading ad: {url}")
        return None

    await accept_cookies(page)
    await simulate_human_behavior(page)

    if await check_for_captcha(page):
        logger.error("CAPTCHA DETECTED on ad page!")
        return None

    # Default Data Structure
    match = re.search(r'/(?:item|annons)/(\d+)', url)
    blocket_id = match.group(1) if match else "UNKNOWN"
    
    ad_data = {
        "blocket_id": blocket_id,
        "url": url,
        "title": "",
        "raw_description": "",
        "price_sek": 0,
        "category": category_enum,
        "_image_urls":[]
    }

    try:
        # STRATEGY 1: Parse the hidden JSON-LD script (Highly Reliable)
        # 1. Parse JSON-LD for Title, Price, Description, AND Image
        json_scripts = await page.locator('script[type="application/ld+json"]').all_inner_texts()
        for script_text in json_scripts:
            try:
                data = json.loads(script_text)
                if data.get("@type") == "Product":
                    ad_data["title"] = data.get("name", "")
                    ad_data["raw_description"] = data.get("description", "")
                    # FIX: Get image from metadata
                    if "image" in data:
                        ad_data["main_image_url"] = data["image"]
                    
                    offers = data.get("offers", {})
                    if isinstance(offers, dict) and "price" in offers:
                        ad_data["price_sek"] = int(offers["price"])
            except: continue

        # STRATEGY 2: DOM Fallback (In case JSON-LD is missing)
        if not ad_data["title"]:
            title_loc = page.locator("h1[data-testid='object-title']")
            if await title_loc.count() > 0:
                ad_data["title"] = await title_loc.first.inner_text()

        if ad_data["price_sek"] == 0:
            price_loc = page.locator(".h2:has-text('kr')").first
            if await price_loc.count() > 0:
                price_str = await price_loc.inner_text()
                clean_str = re.sub(r'[^\d]', '', price_str)
                if clean_str:
                    ad_data["price_sek"] = int(clean_str)

        if not ad_data["raw_description"]:
            desc_loc = page.locator("[data-testid='description']")
            if await desc_loc.count() > 0:
                ad_data["raw_description"] = await desc_loc.first.inner_text()

        # Extract Images (we still need the DOM for this)
        img_tags = await page.locator("img").all()
        for img in img_tags:
            src = await img.get_attribute("src")
            if src and "images.blocket.se" in src and src not in ad_data["_image_urls"]:
                ad_data["_image_urls"].append(src)

        logger.info(f"Successfully scraped: {ad_data['title']} | {ad_data['price_sek']} kr")
        return ad_data

    except Exception as e:
        logger.error(f"Failed to extract data from {url}: {str(e)}")
        return None