"""Canonical part names — the join key the market index depends on.

Its own module because both the writer (`ai/batch_processor.py`, storing an
extracted component) and the reader (`pipeline/index.py`, looking a part up)
must agree exactly. Two copies of this logic drifting apart would not raise
anything; the index would quietly stop matching and every median would read
`n=0`.

### The fragmentation this exists to fix

Found by reading real index output. The two extraction paths disagreed about
what a part is called:

    deterministic (regex) ->  "rtx 3070"
    Gemini                ->  "asus dual geforce rtx 3070 oc"
                              "asus geforce rtx 3070 dual v2"
                              "gigabyte geforce rtx 3070"

Same graphics card, five index entries. What should have been `n=9` was `n=5`
plus four singletons — and singletons are reported as low confidence, so the
system was correctly telling us it was uncertain about a part it had actually
seen nine times.

The fix is to run *every* name, whatever produced it, through the same model
pattern. "Asus Dual GeForce RTX 3070 OC" contains "RTX 3070", so that is what it
canonicalises to. Board partner, cooler name and OC suffix are marketing; the
silicon is what sets the price.

This is cheaper and sharper than the embedding-similarity approach originally
planned for this problem. Embeddings would cost quota, need a curated parts
list, and introduce a similarity threshold to tune — all to solve a case that a
regex already solves exactly. Embeddings remain the right tool for genuinely
fuzzy names ("nvidia 3080ti founders"), which is a smaller residue than it
looked before this was measured.

### What it deliberately will not do

Merge "RTX 3070" with "RTX 3070 Ti". Model names nest, the longer one is the
more expensive part, and over-merging silently corrupts the index — whereas
under-merging merely thins it. Wrong in the cheap direction, on purpose.
"""

from __future__ import annotations

import re

# Model designations that determine price, each paired with a formatter.
#
# Every component of the designation is captured *separately* and reassembled to
# a fixed shape. Reusing the seller's own spacing would not be canonicalisation
# at all: "RX 6800 XT" and "RX6800XT" are the same card written two ways, and
# echoing the input back would keep them as two index entries — the exact bug
# this module exists to fix, just moved one level down.
_MODEL_RULES: tuple[tuple[re.Pattern[str], "object"], ...] = (
    # NVIDIA. The Ti/Super suffix is part of the model and must survive.
    #
    # Three *or* four digits: the GTX 970 and 980 are still traded here, and a
    # four-digit-only pattern silently dropped them — they matched nothing, fell
    # through to the raw-name fallback, and every board partner's wording became
    # its own index entry. Greedy `{3,4}` still prefers the four-digit reading
    # where one exists, so "RTX 3080" is unaffected.
    (
        re.compile(r"\b(RTX|GTX)\s*(\d{3,4})\s*(Ti\s*Super|Super|Ti)?\b", re.IGNORECASE),
        lambda m: " ".join(
            p for p in (m.group(1), m.group(2), _squash(m.group(3) or "")) if p
        ),
    ),
    # AMD Radeon. Three digits matters more here than on the NVIDIA side — the
    # RX 570 and RX 580 are among the most common used cards on the market.
    (
        re.compile(r"\b(RX)\s*(\d{3,4})\s*(XTX|XT)?\b", re.IGNORECASE),
        lambda m: " ".join(p for p in (m.group(1), m.group(2), m.group(3) or "") if p),
    ),
    # AMD Ryzen: "Ryzen5 5600" and "Ryzen 5 5600" must collapse.
    (
        re.compile(r"\b(Ryzen)\s*([3579])\s*(\d{4}[A-Z]{0,3})\b", re.IGNORECASE),
        lambda m: f"ryzen {m.group(2)} {m.group(3)}",
    ),
    # Intel Core: hyphenated regardless of how it was written.
    (
        re.compile(r"\b(i[3579])[\s-]*(\d{4,5}[A-Z]{0,2})\b", re.IGNORECASE),
        lambda m: f"{m.group(1)}-{m.group(2)}",
    ),
    # Memory: capacity plus generation. Speed is deliberately excluded — it
    # varies by kit and barely moves the second-hand price.
    (
        re.compile(r"\b(\d{1,3})\s*GB\s+(DDR[345])\b", re.IGNORECASE),
        lambda m: f"{m.group(1)}gb {m.group(2)}",
    ),
    # Storage, written either way round: "1TB NVMe" and "SSD 500GB" are the same
    # drive. Brand and model are dropped deliberately — a 1 TB SATA SSD trades
    # within a narrow band regardless of whose logo is on it, and keeping the
    # model would fragment the index for no pricing gain.
    (
        re.compile(r"\b(\d{1,4})\s*(TB|GB)\s*(NVMe|SSD|HDD|M\.?2)\b", re.IGNORECASE),
        lambda m: f"{m.group(1)}{m.group(2).lower()} {_medium(m.group(3))}",
    ),
    (
        re.compile(r"\b(NVMe|SSD|HDD|M\.?2)\s*(\d{1,4})\s*(TB|GB)\b", re.IGNORECASE),
        lambda m: f"{m.group(2)}{m.group(3).lower()} {_medium(m.group(1))}",
    ),
    # Power supplies. Wattage is what sets the price; the 80 PLUS rating and
    # modularity move it far less than the spread between individual listings,
    # so neither is kept.
    #
    # Last on purpose. A bare wattage is the least specific thing in this list
    # and would happily claim "RTX 3080 320W" if it were given first refusal.
    (
        re.compile(r"\b([2-9]\d{2}|1\d{3})\s*W\b", re.IGNORECASE),
        lambda m: f"{m.group(1)}w psu",
    ),
)


def _medium(raw: str) -> str:
    """M.2 and NVMe name the same kind of drive; SATA SSD and HDD do not."""
    return "nvme" if raw.lower().replace(".", "") == "m2" else raw.lower()


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def canonicalize(name: str) -> str:
    """Reduce a component name to the model designation that sets its price.

    Falls back to the whitespace-and-case-normalised name when no known model
    pattern matches — an unrecognised part still deserves a stable key, it just
    will not merge with differently-worded sightings of itself.
    """
    if not name:
        return ""

    for pattern, format_match in _MODEL_RULES:
        match = pattern.search(name)
        if match:
            return _squash(format_match(match))

    return _squash(name)
