import os
import asyncio
import time
import json
import logging
from dotenv import load_dotenv

# NEW 2026 SDK Imports
from google import genai
from google.genai import types

from ai_engine.prompt_templates import get_appraiser_system_prompt

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

# Initialize the NEW SDK Client
client = genai.Client(api_key=api_key)

# Global tracking for our Rate Limiter (15 RPM = Max 1 request every 4.5 seconds)
LAST_REQUEST_TIME = 0.0
RATE_LIMIT_DELAY = 4.5  

async def _enforce_rate_limit():
    global LAST_REQUEST_TIME
    now = time.time()
    elapsed = now - LAST_REQUEST_TIME
    if elapsed < RATE_LIMIT_DELAY:
        sleep_time = RATE_LIMIT_DELAY - elapsed
        logger.info(f"Rate Limiter: Sleeping for {sleep_time:.2f}s to respect 15 RPM limit.")
        await asyncio.sleep(sleep_time)
    LAST_REQUEST_TIME = time.time()

async def analyze_ads_batch(ads_data: list[dict]) -> list[dict]:
    if not ads_data: return []

    max_retries = 3
    retry_delay = 10 

    for attempt in range(max_retries):
        try:
            await _enforce_rate_limit()
            logger.info(f"Sending batch to Gemini (Attempt {attempt + 1})...")
            
            payload = json.dumps(ads_data, ensure_ascii=False)
            response = await client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=payload,
                config=types.GenerateContentConfig(
                    system_instruction=get_appraiser_system_prompt(),
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )
            return json.loads(response.text)

        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                logger.warning(f"Gemini busy. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2 
            else:
                logger.error(f"Gemini API Error: {str(e)}")
                return []
    return []

