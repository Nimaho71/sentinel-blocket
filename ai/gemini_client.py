"""Gemini client.

Everything that governs *whether and how* a request may be made lives here, so
that no caller can skip it:

- **PII scrub** (`ai.scrub`) is applied in `build_payload`, the only function
  that shapes a request body — and asserted again immediately before the
  network call, so a future caller that builds a payload by hand still cannot
  leak.
- **Quota** (`ai.quota`) is checked and recorded per request against a
  persistent ledger, covering RPM and the RPD cap that the old in-memory
  limiter forgot on every restart.
- **Model rotation** drains the two independent Flash Lite pools in order and
  reserves the 20-RPD Flash models for escalation.

Errors raise. Every failure path in the previous version returned `[]`, which is
also what an empty input returns, so the caller could not tell a broken API from
a quiet day — and dropped the batch. See `ai/batch_processor.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

from ai.prompt_templates import get_appraiser_system_prompt
from ai.quota import MODELS_BY_NAME, ModelQuota, QuotaLedger
from ai.schemas import RESPONSE_SCHEMA, AdVerdict, AiResponseError, parse_ad_verdicts
from ai.scrub import contains_pii, scrub

logger = logging.getLogger(__name__)

# 250K TPM over ~15 RPM leaves roughly 16K tokens per request for a whole batch.
# PC specs live in the first paragraph of an ad; the rest is shipping terms and
# pleasantries. Truncating is both a quota saving and a quality improvement.
MAX_DESCRIPTION_CHARS = 1_500

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 10.0


class GeminiError(RuntimeError):
    """A call to Gemini failed after exhausting retries."""


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Verdicts plus which model produced them.

    The model is part of the result rather than something the caller passed in,
    because the caller no longer chooses it — rotation does. Recording it on the
    evaluation row is what makes "the 3.5 Lite pool gives worse answers"
    answerable later instead of a hunch.
    """

    verdicts: list[AdVerdict]
    model: str


class PiiLeakError(RuntimeError):
    """Personal data reached the network boundary. Refuse to send.

    Deliberately fatal rather than self-healing. Reaching here means something
    built a payload without going through `build_payload`, and quietly cleaning
    it up would hide that the guarantee had been broken.
    """


class _Throttle:
    """Serialises requests to a per-model minimum spacing.

    Holds the lock across the sleep on purpose. Releasing it first would let
    every waiting coroutine wake, observe the same satisfied interval, and fire
    at once — the exact race the old unlocked module global had.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._last: dict[str, float] = {}

    async def acquire(self, model: str, min_interval: float) -> None:
        async with self._lock:
            last = self._last.get(model, 0.0)
            elapsed = time.monotonic() - last
            if last and elapsed < min_interval:
                wait = min_interval - elapsed
                logger.debug("Throttling %s for %.1fs", model, wait)
                await asyncio.sleep(wait)
            self._last[model] = time.monotonic()


_throttle = _Throttle()
_client = None
_ledger: QuotaLedger | None = None


def get_ledger() -> QuotaLedger:
    """The process-wide quota ledger, opened on first use."""
    global _ledger
    if _ledger is None:
        _ledger = QuotaLedger(os.getenv("QUOTA_LEDGER_PATH", "quota_ledger.db"))
    return _ledger


def set_ledger(ledger: QuotaLedger | None) -> None:
    """Override the ledger. For tests, and for pointing at a scratch file."""
    global _ledger
    _ledger = ledger


def _get_types():
    """The SDK's config types, imported on first use."""
    from google.genai import types

    return types


def _get_client():
    """Build the SDK client on first use.

    Imported lazily so that importing this module — which `ai.batch_processor`
    and the whole test suite do — needs neither the SDK nor a key.
    """
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiError("GEMINI_API_KEY is not set; add it to .env")
        _client = genai.Client(api_key=api_key)
    return _client


def build_payload(ads: Sequence[Any]) -> list[dict[str, Any]]:
    """Shape `Ad` rows into the minimal batch the model needs, scrubbed.

    Only four fields go out. Everything else the row holds — url, city, source
    ids — is irrelevant to appraisal and would only spend tokens.

    The scrub is here, in the one function that builds a request body, rather
    than at each call site. A guarantee that depends on every caller
    remembering is not a guarantee.
    """
    payload = []
    for ad in ads:
        payload.append(
            {
                "ad_id": ad.id,
                # Titles are user-written too, and people do put numbers in them.
                "title": scrub(ad.title),
                "price_sek": ad.price_sek,
                "description": scrub(ad.raw_description)[:MAX_DESCRIPTION_CHARS],
            }
        )
    return payload


def _assert_clean(contents: str) -> None:
    """Last line of defence before the request leaves the machine."""
    if contains_pii(contents):
        raise PiiLeakError(
            "Refusing to send: payload still contains contact details after "
            "scrubbing. Build payloads with build_payload()."
        )


async def analyze_ads_batch(
    ads_data: Sequence[dict[str, Any]],
    *,
    model: str | None = None,
    escalate: bool = False,
) -> BatchResult:
    """Appraise a batch of ads.

    With no `model`, picks the best one with quota remaining. Returns validated
    verdicts, or raises. An empty return means exactly one thing: an empty input.
    """
    if not ads_data:
        return BatchResult(verdicts=[], model="")

    ledger = get_ledger()
    quota: ModelQuota = (
        MODELS_BY_NAME[model] if model else ledger.choose_model(escalate=escalate)
    )

    expected_ids = [entry["ad_id"] for entry in ads_data]
    contents = json.dumps(list(ads_data), ensure_ascii=False)
    _assert_clean(contents)

    backoff = INITIAL_BACKOFF_S
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        if ledger.remaining_today(quota.name) <= 0:
            # Mid-retry exhaustion: fall over to whatever pool still has room
            # rather than spending the attempt on a guaranteed 429.
            quota = ledger.choose_model(escalate=escalate)

        await _throttle.acquire(quota.name, quota.min_interval_s)
        # Recorded before the call: Google counts an attempt whether or not we
        # like the answer, so recording on success would drift the ledger low.
        ledger.record(quota.name)

        # Imported outside the try: an SDK import problem is a broken install,
        # not a failed API call, and must not be retried as though it were.
        types = _get_types()

        try:
            response = await _get_client().aio.models.generate_content(
                model=quota.name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=get_appraiser_system_prompt(),
                    response_mime_type="application/json",
                    # The fix for silent zero-scores: the reply is constrained to
                    # this shape at decode time, then validated again on arrival.
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.2,
                ),
            )
            return BatchResult(
                verdicts=parse_ad_verdicts(
                    json.loads(response.text), expected_ids=expected_ids
                ),
                model=quota.name,
            )

        except (AiResponseError, json.JSONDecodeError) as exc:
            # A malformed reply is worth one more sample — decoding is
            # stochastic — but not an escalating backoff. Nothing is overloaded.
            last_error = exc
            logger.warning("%s returned an unusable reply (attempt %d): %s",
                           quota.name, attempt, exc)

        except Exception as exc:  # noqa: BLE001 — the SDK raises a wide range
            last_error = exc
            if _is_transient(exc):
                logger.warning(
                    "%s unavailable (attempt %d), retrying in %.0fs: %s",
                    quota.name, attempt, backoff, exc,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                # Auth failures, bad model names, malformed requests: retrying
                # cannot help and only spends quota against the daily cap.
                raise GeminiError(f"Gemini call failed: {exc}") from exc

    raise GeminiError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")


def _is_transient(exc: Exception) -> bool:
    """Whether an SDK error is worth retrying.

    String matching is unlovely, but the SDK surfaces HTTP status through
    several exception types and this stays correct across their variations.
    """
    text = str(exc)
    return any(code in text for code in ("429", "500", "502", "503", "504"))
