"""Response-validation tests.

Every test here is a shape the old code accepted and turned into a stored
number. The point is not that the parser is strict for its own sake — it is that
each of these previously produced a plausible-looking row in the database.
"""

from __future__ import annotations

import pytest

from ai.schemas import MAX_RISK_FLAGS, AiResponseError, parse_ad_verdicts
from db.models import ComponentTypeEnum, ItemTypeEnum


def valid_entry(ad_id: int = 1, **overrides) -> dict:
    entry = {
        "ad_id": ad_id,
        "item_type": "FULL_BUILD",
        "confidence": 0.8,
        "risk_flags": ["ej testad"],
        "reasoning": "Complete build, seller has receipt.",
        "components": [{"component_type": "GPU", "component_name": "RTX 3080"}],
    }
    entry.update(overrides)
    return entry


def test_parses_a_well_formed_reply():
    [verdict] = parse_ad_verdicts([valid_entry()], expected_ids=[1])

    assert verdict.ad_id == 1
    assert verdict.item_type is ItemTypeEnum.FULL_BUILD
    assert verdict.confidence == 0.8
    assert verdict.risk_flags == ("ej testad",)
    assert verdict.components[0].component_type is ComponentTypeEnum.GPU


def test_missing_confidence_raises_instead_of_defaulting():
    """The exact bug: `.get(field, 0)` turned an absent field into a real value.

    A model that omits confidence has told us nothing about the ad. The old code
    recorded that as a score of zero — a definite claim, and a wrong one.
    """
    entry = valid_entry()
    del entry["confidence"]

    with pytest.raises(AiResponseError, match="confidence"):
        parse_ad_verdicts([entry], expected_ids=[1])


def test_confidence_outside_range_raises():
    with pytest.raises(AiResponseError, match="outside"):
        parse_ad_verdicts([valid_entry(confidence=7.3)], expected_ids=[1])


def test_unknown_ad_id_raises():
    """A hallucinated id would write one seller's verdict onto another's ad.

    `response_schema` cannot catch this — 999 is a perfectly valid integer.
    This is why the reply is validated as well as constrained.
    """
    with pytest.raises(AiResponseError, match="not in the batch"):
        parse_ad_verdicts([valid_entry(ad_id=999)], expected_ids=[1, 2])


def test_duplicate_ad_id_raises():
    with pytest.raises(AiResponseError, match="twice"):
        parse_ad_verdicts([valid_entry(1), valid_entry(1)], expected_ids=[1])


def test_unknown_item_type_raises():
    with pytest.raises(AiResponseError, match="not a valid"):
        parse_ad_verdicts([valid_entry(item_type="GAMING_RIG")], expected_ids=[1])


def test_bare_object_instead_of_array_raises():
    """A single-ad batch is a common case for a model to answer unwrapped."""
    with pytest.raises(AiResponseError, match="expected a JSON array"):
        parse_ad_verdicts(valid_entry(), expected_ids=[1])


def test_booleans_are_not_accepted_as_numbers():
    """`isinstance(True, int)` is True in Python, so this needs an explicit guard."""
    with pytest.raises(AiResponseError):
        parse_ad_verdicts([valid_entry(ad_id=True)], expected_ids=[1])


def test_risk_flags_are_capped():
    entry = valid_entry(risk_flags=[f"flag {i}" for i in range(20)])
    [verdict] = parse_ad_verdicts([entry], expected_ids=[1])

    assert len(verdict.risk_flags) == MAX_RISK_FLAGS


def test_unnamed_components_are_dropped_not_fatal():
    """An unnamed part cannot be priced, but it should not lose the whole ad."""
    entry = valid_entry(
        components=[
            {"component_type": "GPU", "component_name": "  "},
            {"component_type": "CPU", "component_name": "Ryzen 5 5600X"},
        ]
    )
    [verdict] = parse_ad_verdicts([entry], expected_ids=[1])

    assert len(verdict.components) == 1
    assert verdict.components[0].component_name == "Ryzen 5 5600X"


def test_missing_ads_are_simply_absent_from_the_result():
    """Detecting the omission is the caller's job, and it must be possible.

    The parser does not fail a reply that covers only part of the batch — but it
    also must not invent placeholder verdicts for the rest, or the caller loses
    the ability to tell which ads went unanswered.
    """
    verdicts = parse_ad_verdicts([valid_entry(1)], expected_ids=[1, 2, 3])

    assert {v.ad_id for v in verdicts} == {1}
