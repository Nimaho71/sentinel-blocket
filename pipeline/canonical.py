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

Nor will it merge a laptop GPU with the desktop card of the same name. Nvidia
gives both the same number and they are not the same part: the mobile version
is cut down and trades for substantially less. Found live, and it had reached
the top of the published board — an Acer Predator Helios laptop was credited
with a 38 000 kr desktop RTX 5090 and advertised as 22 700 kr below parts, on
the strength of a single desktop observation. The card in it is worth a
fraction of that.

Detecting this needs the *ad*, not just the part name: that listing said only
"5090". See `is_laptop_listing` for what stands in for the missing word.
"""

from __future__ import annotations

import re

# Named separately from the rule table below so `canonicalize` can ask whether
# the match it made was a graphics card. Only GPUs take the laptop suffix: a
# mobile CPU already announces itself in its own model number (i7-8850H), and
# laptop RAM and storage are the same parts at the same prices.
_NVIDIA_GPU = re.compile(
    r"\b(RTX|GTX)\s*(\d{3,4})\s*(Ti\s*Super|Super|Ti)?\b", re.IGNORECASE
)
_AMD_GPU = re.compile(r"\b(RX)\s*(\d{3,4})\s*(XTX|XT)?\b", re.IGNORECASE)
_GPU_PATTERNS = (_NVIDIA_GPU, _AMD_GPU)

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
        _NVIDIA_GPU,
        lambda m: " ".join(
            p for p in (m.group(1), m.group(2), _squash(m.group(3) or "")) if p
        ),
    ),
    # AMD Radeon. Three digits matters more here than on the NVIDIA side — the
    # RX 570 and RX 580 are among the most common used cards on the market.
    (
        _AMD_GPU,
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


# Words that name the form factor outright. Matched as substrings on purpose:
# "gaminglaptop" is written as one word about as often as two.
_LAPTOP_WORDS = (
    "laptop", "bärbar", "notebook", "ultrabook", "max-q", "max q", "mobile",
)

# A mobile CPU is the giveaway when no word says "laptop" — you cannot put one
# in a desktop, so it settles the question on its own. Intel's mobile parts end
# H/HX/HQ/HK/U and AMD's H/HS/HX, against desktop K/KF/F/X/X3D.
#
# The trailing `3D` is not decorative: "Ryzen 9 9955HX3D" is a laptop chip, and
# a plain word boundary after the suffix rejects it — measured against the
# corpus, which is the only reason it is here.
# Three digits, not four: Intel's Core Ultra line numbers its parts "275HX",
# and a four-digit floor rejected the exact listing that prompted this. The
# suffix is what carries the meaning, so a shorter number costs no precision —
# desktop Core Ultra parts end K, and no desktop part ends H, HX or U.
_MOBILE_CPU = re.compile(
    r"(?:i[3579]|ultra\s*[3579]|ryzen\s*[3579])[\s-]*\d{3,5}\s*"
    r"(?:HX|HQ|HS|HK|H|U)(?:3D)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def is_laptop_listing(*texts: str | None) -> bool:
    """Whether an ad is selling a laptop, from its title and part names.

    Deliberately **not** given the description. Measured: "laptop" appears in
    292 descriptions, and the first one read was an "Uppgraderad stationär" —
    a desktop whose seller merely mentioned laptops. The title and the parts
    list are where the machine describes itself.

    Over-detection is the safer error. Calling a desktop a laptop costs its GPU
    an index match and leaves the ad unpriced; the reverse publishes a laptop
    at desktop prices as the best deal on the site.
    """
    haystack = " ".join(text for text in texts if text).lower()
    if any(word in haystack for word in _LAPTOP_WORDS):
        return True
    return _MOBILE_CPU.search(haystack) is not None


# Model families that name a complete computer. Deliberately excludes ROG, TUF,
# Nitro, Predator, Omen and Victus: those are graphics-card and motherboard
# brands as often as they are machines, and excluding them would throw away
# genuine component listings.
_MACHINE_FAMILIES = (
    "macbook", "imac", "mac mini", "mac studio", "mac pro",
    "thinkpad", "thinkcentre", "ideapad", "legion go",
    "elitebook", "probook", "elitedesk", "prodesk",
    "latitude", "inspiron", "optiplex", "precision 5", "precision 7",
    "pavilion", "zenbook", "vivobook", "chromebook", "zbook",
    "surface pro", "surface laptop",
)

# Phrases that mark a listing as a part *for* a laptop rather than a laptop.
# Without these, "DDR4 16GB (2x8GB) laptopminne" at 610 kr is discarded as a
# whole computer — and it is exactly the cheap, correct observation the RAM
# index most wants.
_PART_FOR_LAPTOP = (
    "laptopminne", "laptop minne", "laptop ram", "till laptop", "för laptop",
    "so-dimm", "sodimm", "bärbar ram", "minne till", "minnen",
)


def names_a_whole_machine(title: str | None) -> bool:
    """Whether a listing is selling a complete computer rather than a part.

    The market index takes the *ad's price* as the price of the component it
    names, which is only sound when the ad is selling that component alone.

    Found live, and it was corrupting every valuation on the site. The "1TB SSD"
    index read 5 500 kr across 34 observations, built from listings like
    "Apple MacBook Pro 16 M3 36GB Ram 1TB ssd" at 27 000 kr. Those were
    classified as single components because "36GB Ram" does not match the RAM
    pattern — it carries no DDR generation — so a whole laptop extracted exactly
    one part and looked like a legitimate SSD listing. Excluding them moves that
    median to about 2 000 kr.

    Not the same question as `is_laptop_listing`, which asks what *kind* of
    machine an ad is selling. A laptop memory stick is a laptop part and a
    perfectly good price observation; a MacBook is not.
    """
    if not title:
        return False
    lowered = title.lower()

    if any(family in lowered for family in _MACHINE_FAMILIES):
        return True
    # A mobile CPU cannot be bought loose in any quantity worth indexing, so its
    # presence means the ad is selling the machine around it.
    if _MOBILE_CPU.search(lowered):
        return True
    if any(word in lowered for word in ("laptop", "bärbar", "notebook", "ultrabook")):
        return not any(phrase in lowered for phrase in _PART_FOR_LAPTOP)
    return False


def _medium(raw: str) -> str:
    """M.2 and NVMe name the same kind of drive; SATA SSD and HDD do not."""
    return "nvme" if raw.lower().replace(".", "") == "m2" else raw.lower()


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def canonicalize(name: str, *, laptop: bool = False) -> str:
    """Reduce a component name to the model designation that sets its price.

    Falls back to the whitespace-and-case-normalised name when no known model
    pattern matches — an unrecognised part still deserves a stable key, it just
    will not merge with differently-worded sightings of itself.

    `laptop=True` marks a graphics card as the mobile variant, giving it its own
    index key. Callers get it from `is_laptop_listing`, because the ad usually
    knows and the part name usually does not — "RTX 5090" is written the same
    either way. A name that says so itself ("RTX 4080 Laptop GPU") is caught
    here regardless of the flag.
    """
    if not name:
        return ""

    mobile = laptop or any(word in name.lower() for word in _LAPTOP_WORDS)

    for pattern, format_match in _MODEL_RULES:
        match = pattern.search(name)
        if match:
            canonical = _squash(format_match(match))
            if mobile and pattern in _GPU_PATTERNS:
                # A suffix rather than a separate field: the index joins on this
                # one string, and every reader already compares it exactly.
                return f"{canonical} laptop"
            return canonical

    return _squash(name)
