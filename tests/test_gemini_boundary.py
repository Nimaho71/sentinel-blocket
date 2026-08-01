"""Tests at the network boundary.

The scrub is easy to test as a function (`tests/test_scrub.py`) and that proves
the regexes work. It does not prove the *system* is safe — that depends on the
scrub being unskippable, which is a property of where it sits, not of what it
does.

So these tests drive `analyze_ads_batch` with a fake transport and assert on the
bytes that would have crossed the wire. A future caller who assembles a payload
by hand is exactly the failure mode being guarded against, and it is tested
here explicitly.
"""

from __future__ import annotations

import json

import pytest

import ai.gemini_client as gc
from ai.gemini_client import PiiLeakError, analyze_ads_batch, build_payload
from ai.quota import MODELS, QuotaExhausted, QuotaLedger

PRIMARY = MODELS[0].name


class FakeAd:
    """Minimal stand-in for an `Ad` row."""

    def __init__(self, ad_id=1, title="RTX 3080 säljes", description=""):
        self.id = ad_id
        self.title = title
        self.raw_description = description
        self.price_sek = 3000


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Capture what would be sent, and answer with a canned valid reply."""
    sent: dict = {}

    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeModels:
        async def generate_content(self, *, model, contents, config):
            sent["model"] = model
            sent["contents"] = contents
            ids = [entry["ad_id"] for entry in json.loads(contents)]
            return FakeResponse(
                json.dumps(
                    [
                        {
                            "ad_id": i,
                            "item_type": "SINGLE_COMPONENT",
                            "confidence": 0.8,
                            "risk_flags": [],
                            "reasoning": "ok",
                            "components": [],
                        }
                        for i in ids
                    ]
                )
            )

    class FakeClient:
        aio = type("aio", (), {"models": FakeModels()})()

    monkeypatch.setattr(gc, "_get_client", lambda: FakeClient())
    # No real spacing: the throttle is tested separately and 4s per call here
    # would make the suite unusable.
    monkeypatch.setattr(gc._throttle, "acquire", lambda *a, **k: _noop())
    gc.set_ledger(QuotaLedger(tmp_path / "quota.db"))
    yield sent
    gc.set_ledger(None)


async def _noop():
    return None


# --- The scrub is unskippable ------------------------------------------------


async def test_a_phone_number_never_reaches_the_wire(captured):
    ads = [FakeAd(description="Fint skick. Ring 070-123 45 67 för visning.")]

    await analyze_ads_batch(build_payload(ads))

    assert "070" not in captured["contents"]
    assert "[phone]" in captured["contents"]


async def test_an_email_never_reaches_the_wire(captured):
    ads = [FakeAd(description="Maila info@teknikfronten.se")]

    await analyze_ads_batch(build_payload(ads))

    assert "teknikfronten" not in captured["contents"]


async def test_a_phone_number_in_the_title_is_also_scrubbed(captured):
    """Titles are user-written too, and it would be easy to scrub only bodies."""
    ads = [FakeAd(title="RTX 3080 ring 0701234567")]

    await analyze_ads_batch(build_payload(ads))

    assert "0701234567" not in captured["contents"]


async def test_a_hand_built_payload_that_skips_the_scrub_is_refused(captured):
    """The guarantee must not depend on callers remembering.

    This is the whole reason for a boundary assertion rather than trusting
    `build_payload`. Someone adding a new call site in six months will not read
    this file first.
    """
    dirty = [
        {
            "ad_id": 1,
            "title": "RTX 3080",
            "price_sek": 3000,
            "description": "Ring 070-123 45 67",
        }
    ]

    with pytest.raises(PiiLeakError, match="Refusing to send"):
        await analyze_ads_batch(dirty)

    assert "contents" not in captured, "nothing may reach the transport"


async def test_hardware_specs_still_reach_the_model(captured):
    """The scrub must not cost the pipeline the data it exists to read."""
    ads = [FakeAd(description="Ryzen 7 5800X, 32GB DDR4 3200MHz, 850W. Ring 070-1234567")]

    await analyze_ads_batch(build_payload(ads))

    for spec in ("Ryzen 7 5800X", "32GB DDR4 3200MHz", "850W"):
        assert spec in captured["contents"]


async def test_descriptions_are_truncated_before_sending(captured):
    ads = [FakeAd(description="x" * 9000)]

    await analyze_ads_batch(build_payload(ads))

    assert len(json.loads(captured["contents"])[0]["description"]) == gc.MAX_DESCRIPTION_CHARS


# --- Quota is enforced at the same boundary ----------------------------------


async def test_each_call_is_recorded_against_the_ledger(captured):
    await analyze_ads_batch(build_payload([FakeAd()]))
    await analyze_ads_batch(build_payload([FakeAd(2)]))

    assert gc.get_ledger().used_today(PRIMARY) == 2


async def test_the_model_used_is_reported_back(captured):
    result = await analyze_ads_batch(build_payload([FakeAd()]))

    assert result.model == PRIMARY
    assert captured["model"] == PRIMARY


async def test_calls_rotate_to_the_second_pool_when_the_first_is_spent(captured):
    ledger = gc.get_ledger()
    for _ in range(MODELS[0].rpd):
        ledger.record(PRIMARY)

    result = await analyze_ads_batch(build_payload([FakeAd()]))

    assert result.model == MODELS[1].name


async def test_an_exhausted_day_raises_rather_than_calling(captured):
    ledger = gc.get_ledger()
    for model in MODELS[:2]:
        for _ in range(model.rpd):
            ledger.record(model.name)

    with pytest.raises(QuotaExhausted):
        await analyze_ads_batch(build_payload([FakeAd()]))

    assert "contents" not in captured


async def test_an_empty_batch_costs_no_quota(captured):
    result = await analyze_ads_batch([])

    assert result.verdicts == []
    assert gc.get_ledger().used_today(PRIMARY) == 0
