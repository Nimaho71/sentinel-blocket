import random
from playwright.async_api import async_playwright, BrowserContext, Page, Route
# Updated import for Playwright-Stealth v2.0+
from playwright_stealth import Stealth 

# A small pool of realistic, modern User-Agents
USER_AGENTS =[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

async def block_unnecessary_resources(route: Route):
    """
    Blocks images, fonts, and media from being downloaded by the browser.
    This saves massive amounts of bandwidth and speeds up the crawl.
    (The DOM <img> tags will still exist for us to scrape their 'src' URLs!)
    """
    excluded_resource_types = ["image", "media", "font", "stylesheet"]
    if route.request.resource_type in excluded_resource_types:
        await route.abort()
    else:
        await route.continue_()

async def create_stealth_context(browser) -> BrowserContext:
    """
    Creates a new BrowserContext equipped with stealth patches.
    """
    context = await browser.new_context(
        user_agent=random.choice(USER_AGENTS),
        viewport={"width": random.randint(1366, 1920), "height": random.randint(768, 1080)},
        locale="sv-SE",
        timezone_id="Europe/Stockholm",
        has_touch=False,
    )
    
    # --- NEW V2 API IMPLEMENTATION ---
    # We now apply stealth to the entire context, so every page spawned is protected
    stealth = Stealth()
    await stealth.apply_stealth_async(context)
    
    return context

async def setup_stealth_page(context: BrowserContext) -> Page:
    """
    Opens a new page and sets up network interception to block images.
    """
    page = await context.new_page()
    
    # Route interception to block heavy resources
    await page.route("**/*", block_unnecessary_resources)
    
    return page
