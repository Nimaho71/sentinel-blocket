import asyncio
from playwright.async_api import async_playwright
from scraper.stealth_config import create_stealth_context, setup_stealth_page
from scraper.crawler import search_and_fetch_urls, scrape_ad_details

async def run_test():
    print("🚀 Starting Stealth Scraper Test (Homepage Strategy)...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False) 
        context = await create_stealth_context(browser)
        page = await setup_stealth_page(context)
        
        # Step 1: Start at homepage and search!
        query = "gtx 970"
        urls = await search_and_fetch_urls(page, query, max_pages=1) 
        
        if not urls:
            print("❌ No URLs found.")
            await browser.close()
            return

        print(f"✅ Found {len(urls)} URLs. Moving to Deep Scrape...\n")
        
        # Step 2: Deep Scrape the first 2 ads
        ads_to_scrape = urls[:2] 
        for url in ads_to_scrape:
            print(f"[Test 2] Scraping Ad: {url}")
            ad_data = await scrape_ad_details(page, url, "COMPONENT")
            
            if ad_data:
                print("\n✅ Successfully Extracted Data:")
                print(f"   ID: {ad_data['blocket_id']}")
                print(f"   Title: {ad_data['title']}")
                print(f"   Price: {ad_data['price_sek']} kr")
                print(f"   Description Snippet: {ad_data['raw_description'][:100]}...\n")
            else:
                print("❌ Failed to extract ad data.\n")
                
        print("🎉 Test Complete. Closing browser.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_test())