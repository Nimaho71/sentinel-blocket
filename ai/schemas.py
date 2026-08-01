"""The contract for what Gemini returns, and the code that enforces it.

The bug this module exists to kill: the old client asked only for
`response_mime_type="application/json"` — valid JSON, of any shape — and the
consumer then read it with `result.get("deal_score", 0)`. A reply that omitted
the field, or named it differently, or came back as a bare object instead of an
array, produced **a confidently stored score of zero**. A wrong answer that
looks like a real answer is worse than an exception, because nothing upstream
can tell the two apart.

So the shape is declared twice, deliberately:

1. `RESPONSE_SCHEMA` is sent to Gemini as `response_schema`, which constrains
   decoding so the model cannot emit a different shape in the first place.
2. `parse_ad_verdicts` validates the reply anyway and *raises* on anything that
   does not fit.

The second is not redundant. Schema-constrained decoding governs structure, not
meaning: it cannot stop the model returning an `ad_id` that was not in the
batch, or a confidence of 7.3. Those are the checks here.

Note what is absent: **no price, no market value, no deal score.** Valuation is
arithmetic over observed listings (`pipeline/index.py`), not something a language
model recalls. Asking for a number it cannot know is how the old pipeline ended
up with unfalsifiable Swedish price estimates recalled from training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from db.models import ComponentTypeEnum, ItemTypeEnum

# Ceiling on flags per ad. The UI renders these as chips and a model given an
# unbounded list will happily produce twenty.
MAX_RISK_FLAGS = 6


class AiResponseError(ValueError):
    """The model's reply did not satisfy the contract.

    Raised, never swallowed. The caller records it against the ad as a
    dead-letter row so the failure is visible and bounded.
    """


# Passed to the SDK as `response_schema`. Kept as a plain dict rather than SDK
# types so this module imports without google-genai installed — which is what
# lets the tests exercise validation without network or credentials.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "required": ["ad_id", "item_type", "confidence", "risk_flags", "reasoning"],
        "properties": {
            "ad_id": {"type": "INTEGER"},
            "item_type": {
                "type": "STRING",
                "enum": [t.value for t in ItemTypeEnum],
            },
            "confidence": {
                "type": "NUMBER",
                "description": "0.0-1.0. How much to trust that this listing is what it claims.",
            },
            "risk_flags": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "Short buyer-facing warnings, e.g. 'ej testad', 'no receipt'.",
            },
            "reasoning": {"type": "STRING"},
            "components": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "required": ["component_type", "component_name"],
                    "properties": {
                        "component_type": {
                            "type": "STRING",
                            "enum": [t.value for t in ComponentTypeEnum],
                        },
                        "component_name": {"type": "STRING"},
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True, slots=True)
class ComponentVerdict:
    component_type: ComponentTypeEnum
    component_name: str


@dataclass(frozen=True, slots=True)
class AdVerdict:
    ad_id: int
    item_type: ItemTypeEnum
    confidence: float
    risk_flags: tuple[str, ...] = ()
    reasoning: str = ""
    components: tuple[ComponentVerdict, ...] = field(default_factory=tuple)


def _enum_or_raise(enum_cls, raw: Any, ad_id: Any):
    if not isinstance(raw, str):
        raise AiResponseError(f"ad {ad_id}: {enum_cls.__name__} must be a string, got {raw!r}")
    try:
        return enum_cls(raw.strip().upper())
    except ValueError:
        raise AiResponseError(f"ad {ad_id}: {raw!r} is not a valid {enum_cls.__name__}") from None


def _components(raw: Any, ad_id: int) -> tuple[ComponentVerdict, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise AiResponseError(f"ad {ad_id}: 'components' must be a list, got {type(raw).__name__}")

    out: list[ComponentVerdict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise AiResponseError(f"ad {ad_id}: component entries must be objects")
        name = str(entry.get("component_name", "")).strip()
        if not name:
            # An unnamed component cannot be priced or matched, so it is noise
            # rather than an error — drop it and keep the rest of the ad.
            continue
        out.append(
            ComponentVerdict(
                component_type=_enum_or_raise(
                    ComponentTypeEnum, entry.get("component_type"), ad_id
                ),
                component_name=name,
            )
        )
    return tuple(out)


def parse_ad_verdicts(payload: Any, *, expected_ids: Iterable[int]) -> list[AdVerdict]:
    """Validate a decoded reply, or raise.

    `expected_ids` is the batch that was sent. Checking returned ids against it
    catches the failure mode schema validation cannot: a model that invents an
    id, or repeats one, would otherwise have its verdict written onto a
    different seller's ad.
    """
    if not isinstance(payload, list):
        raise AiResponseError(f"expected a JSON array, got {type(payload).__name__}")

    allowed = set(expected_ids)
    seen: set[int] = set()
    verdicts: list[AdVerdict] = []

    for entry in payload:
        if not isinstance(entry, dict):
            raise AiResponseError(f"expected objects in array, got {type(entry).__name__}")

        raw_id = entry.get("ad_id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            raise AiResponseError(f"'ad_id' must be an integer, got {raw_id!r}")
        if raw_id not in allowed:
            raise AiResponseError(f"ad_id {raw_id} was not in the batch that was sent")
        if raw_id in seen:
            raise AiResponseError(f"ad_id {raw_id} appeared twice in one reply")
        seen.add(raw_id)

        confidence = entry.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise AiResponseError(f"ad {raw_id}: 'confidence' must be a number, got {confidence!r}")
        if not 0.0 <= float(confidence) <= 1.0:
            raise AiResponseError(f"ad {raw_id}: confidence {confidence} is outside 0.0-1.0")

        flags = entry.get("risk_flags") or []
        if not isinstance(flags, list):
            raise AiResponseError(f"ad {raw_id}: 'risk_flags' must be a list")

        verdicts.append(
            AdVerdict(
                ad_id=raw_id,
                item_type=_enum_or_raise(ItemTypeEnum, entry.get("item_type"), raw_id),
                confidence=float(confidence),
                risk_flags=tuple(
                    str(f).strip() for f in flags[:MAX_RISK_FLAGS] if str(f).strip()
                ),
                reasoning=str(entry.get("reasoning", "")).strip(),
                components=_components(entry.get("components"), raw_id),
            )
        )

    return verdicts
