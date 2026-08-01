"""The independent redaction check for the published artifact.

### Why this is its own module, importing nothing

`pipeline/publish.py` builds the public snapshot from an explicit allowlist of
fields. This module re-reads the finished JSON and refuses it if it finds
anything it should not. Two properties fall out of keeping them apart, and both
are the point:

**It is a genuine second opinion.** The builder and the checker share no code,
so a mistaken assumption in one is not silently reasserted by the other. The
checker does not know what fields the builder intended to emit; it knows what a
public file is not allowed to contain, which is a different question.

**It runs anywhere.** The only dependency is `ai.scrub`, which is pure `re`. No
SQLAlchemy, no database, no ORM models. That is what lets CI verify the bytes
actually committed to the repository, on a bare Python image, without being able
to build a snapshot at all — and a check that cannot fail to *run* is worth more
than one with better coverage that gets skipped when an import breaks.

Everything here is deliberately paranoid about names as well as values. A field
called `sellerName` is rejected on sight even if it happens to be empty this
run, because the next run it will not be.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ai.scrub import contains_pii


class RedactionError(AssertionError):
    """The snapshot contains something that must not be published.

    An `AssertionError` subclass on purpose: this is a broken invariant, not a
    recoverable condition. There is no correct way to handle it other than to
    stop and fix the builder.
    """



# Key names that must never appear at any depth. Matched case-insensitively
# against both snake_case and camelCase, since the pipeline speaks one and the
# published contract the other.
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "description",
        "rawdescription",
        "raw_description",
        "body",
        "text",
        "seller",
        "sellername",
        "seller_name",
        "sellerid",
        "contact",
        "phone",
        "mobile",
        "email",
        "postalcode",
        "postal_code",
        "zip",
        "zipcode",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "lng",
        "coordinates",
        "gps",
        "images",
        "imageurl",
        "image_url",
        "thumbnail",
    }
)

# The only URLs allowed in the artifact: attribution links to a Blocket ad page.
_ALLOWED_URL = re.compile(r"^https://(?:www\.)?blocket\.se/[^\s\"'<>]*$")
_URL_KEYS = frozenset({"url"})

# Any image host, not just Blocket's. Hotlinking someone else's bandwidth is
# the problem; whose CDN it is does not change that.
_IMAGE_URL = re.compile(r"https?://\S+\.(?:jpe?g|png|webp|gif|avif)\b", re.IGNORECASE)

# A Swedish postal code is five digits, conventionally written "123 45". City
# names do not contain digit runs, so any digits in a place field are suspect.
_POSTAL_CODE = re.compile(r"\b\d{3}\s?\d{2}\b")
_PLACE_KEYS = frozenset({"city"})


def assert_redacted(snapshot: Any, *, path: str = "$") -> None:
    """Refuse a snapshot that carries anything it must not.

    Walks the finished structure rather than inspecting the code that built it.
    Every violation is collected before raising, so a single run tells you
    everything that is wrong instead of one thing at a time.
    """
    problems: list[str] = []
    _walk(snapshot, path, problems)
    if problems:
        raise RedactionError(
            f"{len(problems)} redaction violation(s):\n  "
            + "\n  ".join(sorted(problems))
        )


def _walk(node: Any, path: str, problems: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            normalized = str(key).lower().replace("-", "").replace("_", "")
            here = f"{path}.{key}"
            if normalized in FORBIDDEN_KEYS or key.lower() in FORBIDDEN_KEYS:
                problems.append(f"{here}: forbidden field name")
                continue
            if str(key).lower() in _URL_KEYS:
                _check_url(value, here, problems)
                continue
            if str(key).lower() in _PLACE_KEYS:
                _check_place(value, here, problems)
                continue
            _walk(value, here, problems)
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _walk(value, f"{path}[{i}]", problems)
    elif isinstance(node, str):
        _check_text(node, path, problems)


def _check_text(value: str, path: str, problems: list[str]) -> None:
    if _IMAGE_URL.search(value):
        problems.append(f"{path}: image URL in text")
    if contains_pii(value):
        # `contains_pii` covers emails, phone numbers, bare long digit runs and
        # URLs. In a text field none of those are legitimate — attribution links
        # live in `url`, which is checked separately and more strictly.
        problems.append(f"{path}: contact details or a bare URL in free text")


def _check_url(value: Any, path: str, problems: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _ALLOWED_URL.match(value):
        problems.append(f"{path}: not a plain Blocket attribution link ({value!r})")
    elif _IMAGE_URL.search(value):
        problems.append(f"{path}: image URL")


def _check_place(value: Any, path: str, problems: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        problems.append(f"{path}: place must be a string")
    elif _POSTAL_CODE.search(value) or any(ch.isdigit() for ch in value):
        problems.append(f"{path}: digits in a place name suggest a postal code")


def assert_capped(snapshot: dict[str, Any], *, profiles: int | None = None) -> None:
    """The public artifact is a leaderboard, not a copy of the collection.

    Enforced as an invariant rather than trusted to the builder, because the cap
    is the load-bearing part of the *Innoweb* argument in ETHICS.md — a "top ten
    per category" that quietly became "everything" would undo the reasoning
    while still passing every other test here.

    The multiplier is the profile count, because the snapshot carries the union
    of each profile's own top-N (see `pipeline/publish.py`). It is read from the
    snapshot rather than imported, which is what keeps this module free of the
    database stack and therefore runnable in CI.
    """
    if profiles is None:
        profiles = len(snapshot.get("profiles") or ()) or 1
    limit = snapshot["perCategoryLimit"] * profiles
    counts: dict[str, int] = {}
    for deal in snapshot["deals"]:
        counts[deal["category"]] = counts.get(deal["category"], 0) + 1
    over = {cat: n for cat, n in counts.items() if n > limit}
    if over:
        raise RedactionError(f"category caps exceeded (max {limit}): {over}")



def verify_file(path: Path) -> dict[str, Any]:
    """Re-check a snapshot already on disk. This is what CI runs.

    Separate from `publish` on purpose: CI has no database, and the artifact
    that matters is the one actually committed, not the one the pipeline
    believes it wrote.
    """
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert_redacted(snapshot)
    assert_capped(snapshot)
    return snapshot


def main() -> int:
    """`python -m pipeline.redact web/public/deals.json` — verify only."""
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("web/public/deals.json")
    if not target.exists():
        print(f"{target} does not exist; nothing to verify.")
        return 1
    try:
        snapshot = verify_file(target)
    except RedactionError as exc:
        print(f"{target} FAILED redaction check\n{exc}")
        return 1
    print(
        f"{target} is clean: {len(snapshot['deals'])} deals, "
        f"generated {snapshot['generatedAt']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
