"""Blocket adapter tests, run against payloads recorded from the live API.

No network. Fixtures in `tests/fixtures/` were captured 2026-07; when the
unofficial API drifts, these are what will catch it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from sources.base import AdSummary, CircuitOpen, SourceError
from sources.blocket import BlocketSource, _category_path, _extras_to_dict
from sources.throttle import CircuitBreaker, RateLimiter

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeAPI:
    """Stands in for BlocketAPI. Records calls; can be told to fail."""

    def __init__(self, search_payload=None, detail_payload=None, error=None):
        self.search_payload = search_payload or {"docs": []}
        self.detail_payload = detail_payload or {}
        self.error = error
        self.calls: list[str] = []

    def search(self, query, **kwargs):
        self.calls.append(f"search:{query}")
        if self.error:
            raise self.error
        return self.search_payload

    def get_ad(self, ad):
        self.calls.append(f"get_ad:{ad}")
        if self.error:
            raise self.error
        return self.detail_payload


def make_source(**kwargs) -> BlocketSource:
    """A source with throttling neutralised, so tests don't sleep."""
    kwargs.setdefault("rate_limiter", RateLimiter(min_interval=0.0, jitter=0.0))
    return BlocketSource(**kwargs)


# ------------------------------------------------------------------------ search


def test_search_parses_live_payload():
    source = make_source(api=FakeAPI(search_payload=load("search_rtx3060.json")))

    results = source.search("RTX 3060")

    assert len(results) == 3
    first = results[0]
    assert first.source == "blocket"
    assert first.external_id
    assert first.title
    assert first.price_sek > 0
    assert first.url.startswith("https://www.blocket.se/")


def test_search_keeps_only_coarse_city_never_coordinates():
    """Search results carry GPS. It must not survive into our model."""
    raw = load("search_rtx3060.json")
    assert "coordinates" in raw["docs"][0], "fixture should still contain GPS"

    results = make_source(api=FakeAPI(search_payload=raw)).search("RTX 3060")

    summary = results[0]
    assert summary.city  # coarse place name is kept
    assert not hasattr(summary, "coordinates")
    blob = json.dumps(summary.__getstate__() if hasattr(summary, "__getstate__") else
                      {f: getattr(summary, f) for f in summary.__slots__}, default=str)
    assert "lat" not in blob and "lon" not in blob


def test_search_skips_unparseable_results_without_dropping_good_ones():
    payload = {"docs": [
        {"ad_id": "1", "heading": "ok", "price": {"amount": 100},
         "canonical_url": "https://www.blocket.se/x/1"},
        {"heading": "no id"},
        {"ad_id": "3", "heading": "no price"},
        {"ad_id": "4", "heading": "ok2", "price": {"amount": 200},
         "canonical_url": "https://www.blocket.se/x/4"},
    ]}

    results = make_source(api=FakeAPI(search_payload=payload)).search("q")

    assert [r.external_id for r in results] == ["1", "4"]


def test_search_failure_raises_rather_than_returning_empty():
    """'Blocked' and 'no results' must never look the same to the caller."""
    source = make_source(api=FakeAPI(error=RuntimeError("boom")))

    with pytest.raises(SourceError):
        source.search("RTX 3060")


# ------------------------------------------------------------------------ detail


def summary_for(ad_id: str = "25234457") -> AdSummary:
    return AdSummary(
        source="blocket",
        external_id=ad_id,
        url=f"https://www.blocket.se/recommerce/forsale/item/{ad_id}",
        title="placeholder",
        price_sek=3200,
        city="Älvsjö",
    )


def test_fetch_detail_returns_full_description():
    """The whole architecture rests on detail carrying the description."""
    source = make_source(api=FakeAPI(detail_payload=load("detail_item.json")))

    record = source.fetch_detail(summary_for())

    assert record is not None
    assert len(record.description) > 50
    assert record.title
    assert record.price_sek > 0


def test_fetch_detail_extracts_structured_attributes():
    """`extras` give condition and component type without an LLM call."""
    source = make_source(api=FakeAPI(detail_payload=load("detail_item.json")))

    record = source.fetch_detail(summary_for())

    assert record is not None
    assert record.condition, "condition should come straight from extras"
    assert "computer_components_type" in record.attributes


def test_fetch_detail_never_ingests_gps_or_seller_id():
    """PII is dropped at the boundary, so downstream cannot leak it."""
    raw = load("detail_item.json")
    item = raw["loaderData"]["item-recommerce"]["itemData"]
    assert "location" in item, "fixture should still contain the GPS payload"

    record = make_source(api=FakeAPI(detail_payload=raw)).fetch_detail(summary_for())

    assert record is not None
    blob = json.dumps(
        {
            "attributes": dict(record.attributes),
            "category_path": list(record.category_path),
            "city": record.city,
            "title": record.title,
            "url": record.url,
        },
        ensure_ascii=False,
    )
    for forbidden in ("lat", "lng", "ownerId", "position"):
        assert forbidden not in blob, f"{forbidden} leaked into AdRecord"
    assert record.city == "Älvsjö", "coarse city is fine and should survive"


def test_fetch_detail_returns_none_for_removed_listing():
    payload = {"loaderData": {"item-recommerce": {"itemData": {"disposed": True}}}}

    assert make_source(api=FakeAPI(detail_payload=payload)).fetch_detail(summary_for()) is None


def test_fetch_detail_returns_none_when_payload_shape_unexpected():
    source = make_source(api=FakeAPI(detail_payload={"loaderData": {}}))

    assert source.fetch_detail(summary_for()) is None


# ----------------------------------------------------- breaker wired into source


def test_repeated_failures_open_the_circuit():
    source = make_source(
        api=FakeAPI(error=RuntimeError("503")),
        breaker=CircuitBreaker(threshold=3, cooldown=900.0),
    )

    for _ in range(3):
        with pytest.raises(SourceError):
            source.search("q")

    # Fourth call is refused before any request is issued.
    with pytest.raises(CircuitOpen):
        source.search("q")


def test_open_circuit_stops_calling_the_api():
    api = FakeAPI(error=RuntimeError("503"))
    source = make_source(api=api, breaker=CircuitBreaker(threshold=2, cooldown=900.0))

    for _ in range(2):
        with pytest.raises(SourceError):
            source.search("q")
    calls_when_tripped = len(api.calls)

    for _ in range(5):
        with pytest.raises(CircuitOpen):
            source.search("q")

    assert len(api.calls) == calls_when_tripped, "no requests may go out while open"


# ------------------------------------------------------------------- pure helpers


def test_extras_to_dict_flattens_blocket_shape():
    extras = [
        {"id": "condition", "label": "Skick", "value": "Mycket bra skick", "valueId": 2},
        {"id": "computer_components_type", "label": "Typ", "value": "Grafikkort"},
    ]

    assert _extras_to_dict(extras) == {
        "condition": "Mycket bra skick",
        "computer_components_type": "Grafikkort",
    }


@pytest.mark.parametrize("junk", [None, "string", 42, [], [None, "x", {"no": "id"}]])
def test_extras_to_dict_survives_junk(junk):
    assert isinstance(_extras_to_dict(junk), dict)


def test_category_path_is_root_first():
    category = {"value": "Datorkomponenter", "parent": {"value": "Datorer", "parent": None}}

    assert _category_path(category) == ("Datorer", "Datorkomponenter")


def test_category_path_terminates_on_cycle():
    """A self-referential payload must not hang the pipeline."""
    node: dict = {"value": "Loop"}
    node["parent"] = node

    assert len(_category_path(node)) <= 10
