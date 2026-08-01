"""Static hardware reference data — the specs that decide whether a part *fits*.

### Why a hand-maintained table, when the rest of the project fought so hard to avoid one

Everywhere else in this codebase, asking a model for a fact was the wrong move:
`estimated_market_value_sek` was deleted precisely because an LLM's recollection
of Swedish used prices is fluent, undated and unfalsifiable.

VRAM is the opposite kind of fact. An RTX 3070 has 8 GB. It had 8 GB in 2020, it
has 8 GB today, and it will still have 8 GB when this repository is archived. It
is published by the manufacturer, it never moves, and a wrong entry here is a
bounded bug that one line fixes — not a silent drift that corrupts a median.

So: prices are counted, specs are looked up, and nothing is recalled.

### The variant problem, and why the table leans pessimistic

Several model names cover more than one card. The RTX 3060 shipped with 12 GB
*and* 8 GB. The RTX 3080 shipped with 10 GB *and* 12 GB. `canonicalize()`
deliberately collapses board-partner naming, so both variants arrive here as the
same string, and there is genuinely no way to tell them apart from the model
name alone.

This matters more than it sounds. The whole point of the `ai-workstation`
profile is a 12 GB VRAM floor, and the RTX 3060 — the single best value card for
local LLMs — is exactly one of the ambiguous ones.

Two rules resolve it, in order:

1. **If the ad states a capacity, believe the ad.** "RTX 3060 12GB" in a title
   is the seller telling us which variant they have, and it is better evidence
   than any table. `vram_from_text` reads it.
2. **Otherwise assume the smaller variant.** These figures feed a *hard floor*.
   Guessing high admits a card that does not qualify; guessing low withholds one
   that might. Being wrong in the second direction costs a listing the user could
   have looked at anyway — being wrong in the first direction is how somebody
   buys an 8 GB card to run a model that needs 12.

`AMBIGUOUS_VRAM` names the affected models explicitly so the UI can say "8 GB
assumed — this model also ships with 12 GB, check the listing" rather than
quietly presenting a guess as a fact.

TDP figures are reference-board numbers and are used only as a coarse
power-draw signal for the `home-server` profile, where the question is "is this
a 300 W gaming card" and not "what is the exact wall draw".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- GPUs --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GpuSpec:
    """Manufacturer specs for one canonical GPU name.

    `vram_gb` is the *smallest* shipping variant — see the module docstring.
    """

    vram_gb: int
    tdp_w: int


# Keyed by the output of `pipeline.canonical.canonicalize`. Scope is roughly the
# last eight years, which is what actually turns up on a Swedish used market;
# anything older is not worth the row.
GPU_SPECS: dict[str, GpuSpec] = {
    # --- NVIDIA GeForce 10 series ---
    "gtx 1050": GpuSpec(2, 75),
    "gtx 1050 ti": GpuSpec(4, 75),
    "gtx 1060": GpuSpec(3, 120),
    "gtx 1070": GpuSpec(8, 150),
    "gtx 1070 ti": GpuSpec(8, 180),
    "gtx 1080": GpuSpec(8, 180),
    "gtx 1080 ti": GpuSpec(11, 250),
    # --- NVIDIA GeForce 16 series ---
    "gtx 1630": GpuSpec(4, 75),
    "gtx 1650": GpuSpec(4, 75),
    "gtx 1650 super": GpuSpec(4, 100),
    "gtx 1660": GpuSpec(6, 120),
    "gtx 1660 super": GpuSpec(6, 125),
    "gtx 1660 ti": GpuSpec(6, 120),
    # --- NVIDIA GeForce 20 series ---
    "rtx 2060": GpuSpec(6, 160),
    "rtx 2060 super": GpuSpec(8, 175),
    "rtx 2070": GpuSpec(8, 175),
    "rtx 2070 super": GpuSpec(8, 215),
    "rtx 2080": GpuSpec(8, 215),
    "rtx 2080 super": GpuSpec(8, 250),
    "rtx 2080 ti": GpuSpec(11, 250),
    # --- NVIDIA GeForce 30 series ---
    "rtx 3050": GpuSpec(6, 130),
    "rtx 3060": GpuSpec(8, 170),
    "rtx 3060 ti": GpuSpec(8, 200),
    "rtx 3070": GpuSpec(8, 220),
    "rtx 3070 ti": GpuSpec(8, 290),
    "rtx 3080": GpuSpec(10, 320),
    "rtx 3080 ti": GpuSpec(12, 350),
    "rtx 3090": GpuSpec(24, 350),
    "rtx 3090 ti": GpuSpec(24, 450),
    # --- NVIDIA GeForce 40 series ---
    "rtx 4060": GpuSpec(8, 115),
    "rtx 4060 ti": GpuSpec(8, 160),
    "rtx 4070": GpuSpec(12, 200),
    "rtx 4070 super": GpuSpec(12, 220),
    "rtx 4070 ti": GpuSpec(12, 285),
    "rtx 4070 ti super": GpuSpec(16, 285),
    "rtx 4080": GpuSpec(16, 320),
    "rtx 4080 super": GpuSpec(16, 320),
    "rtx 4090": GpuSpec(24, 450),
    # --- NVIDIA GeForce 50 series ---
    "rtx 5060": GpuSpec(8, 145),
    "rtx 5060 ti": GpuSpec(8, 180),
    "rtx 5070": GpuSpec(12, 250),
    "rtx 5070 ti": GpuSpec(16, 300),
    "rtx 5080": GpuSpec(16, 360),
    "rtx 5090": GpuSpec(32, 575),
    # --- AMD Radeon RX 500 series ---
    "rx 570": GpuSpec(4, 150),
    "rx 580": GpuSpec(4, 185),
    "rx 590": GpuSpec(8, 175),
    # --- AMD Radeon RX 5000 series ---
    "rx 5500 xt": GpuSpec(4, 130),
    "rx 5600 xt": GpuSpec(6, 150),
    "rx 5700": GpuSpec(8, 180),
    "rx 5700 xt": GpuSpec(8, 225),
    # --- AMD Radeon RX 6000 series ---
    "rx 6400": GpuSpec(4, 53),
    "rx 6500 xt": GpuSpec(4, 107),
    "rx 6600": GpuSpec(8, 132),
    "rx 6600 xt": GpuSpec(8, 160),
    "rx 6650 xt": GpuSpec(8, 180),
    "rx 6700 xt": GpuSpec(12, 230),
    "rx 6750 xt": GpuSpec(12, 250),
    "rx 6800": GpuSpec(16, 250),
    "rx 6800 xt": GpuSpec(16, 300),
    "rx 6900 xt": GpuSpec(16, 300),
    "rx 6950 xt": GpuSpec(16, 335),
    # --- AMD Radeon RX 7000 series ---
    "rx 7600": GpuSpec(8, 165),
    "rx 7600 xt": GpuSpec(16, 190),
    "rx 7700 xt": GpuSpec(12, 245),
    "rx 7800 xt": GpuSpec(16, 263),
    "rx 7900 xt": GpuSpec(20, 315),
    "rx 7900 xtx": GpuSpec(24, 355),
    # --- AMD Radeon RX 9000 series ---
    "rx 9060 xt": GpuSpec(8, 150),
    "rx 9070": GpuSpec(16, 220),
    "rx 9070 xt": GpuSpec(16, 304),
}

# Models sold under one name in more than one memory configuration. The table
# above records the smaller; these are the ones where that assumption is doing
# real work and the user deserves to be told.
AMBIGUOUS_VRAM: dict[str, tuple[int, ...]] = {
    "rtx 2060": (6, 12),
    "rtx 3050": (6, 8),
    "rtx 3060": (8, 12),
    "rtx 3080": (10, 12),
    "rtx 4060 ti": (8, 16),
    "rtx 5060 ti": (8, 16),
    "rx 570": (4, 8),
    "rx 580": (4, 8),
    "rx 5500 xt": (4, 8),
}


# A capacity written *immediately* after a GPU model: "RTX 3060 12GB",
# "RTX 3080-10GB", "RX 6700 XT 12 GB".
#
# Two guards, both added because the first version got this wrong. It allowed a
# comma as a separator and read "RTX 3070, 32GB RAM, 1TB SSD" as a 32 GB
# graphics card — a card that does not exist, clearing a 12 GB floor it should
# have failed. A comma is a list separator, not a spacer, so only whitespace and
# hyphens join a model to its capacity now.
#
# The trailing lookahead catches the same mistake without the comma: in
# "RTX 3070 32GB RAM" the capacity is adjacent, and only the noun after it says
# what it belongs to.
_VRAM_NEAR_MODEL = re.compile(
    r"\b(?:RTX|GTX|RX)\s*\d{3,4}\s*(?:Ti\s*Super|Super|Ti|XTX|XT)?[\s-]{0,3}"
    r"(\d{1,2})\s*GB\b"
    r"(?!\s*(?:RAM|DDR|minne|SSD|HDD|NVMe|M\.?2))",
    re.IGNORECASE,
)

# Plausible VRAM capacities. A "2GB" or "48GB" match next to a consumer model
# name is far more likely to be a typo or a mis-parse than a real card.
_VRAM_VALUES = frozenset({2, 3, 4, 6, 8, 10, 11, 12, 16, 20, 24, 32})


def vram_from_text(text: str) -> int | None:
    """The VRAM the *seller* stated, if they stated one.

    Preferred over `GPU_SPECS` for the ambiguous models: the person holding the
    card knows which variant it is, and a table cannot.
    """
    if not text:
        return None
    for match in _VRAM_NEAR_MODEL.finditer(text):
        value = int(match.group(1))
        if value in _VRAM_VALUES:
            return value
    return None


def possible_vram(canonical_name: str) -> tuple[int, ...]:
    """Every capacity this model was ever sold with, or `()` if unrecognised.

    `()` means "no opinion", not "no valid capacity" — an unknown card must not
    have the seller's stated figure thrown away on the authority of a table that
    has never heard of it.
    """
    key = canonical_name.lower().strip()
    if key in AMBIGUOUS_VRAM:
        return AMBIGUOUS_VRAM[key]
    spec = GPU_SPECS.get(key)
    return (spec.vram_gb,) if spec else ()


def gpu_vram_gb(canonical_name: str, *, text_hint: str = "") -> int | None:
    """VRAM in GB for a canonical GPU name, or None if the model is unknown.

    `None` means "we do not know", and callers must not read it as zero. An
    unrecognised card failing a VRAM floor it might well clear is a different
    outcome from a card that genuinely does not clear it — see
    `pipeline/profiles.py`, where the two are reported separately.

    ### A stated capacity is believed only if the model could have it

    The seller is holding the card and a table is not, so a stated figure wins —
    but only among capacities that model was actually built with.

    Found by reading the first published snapshot. Top of the AI-workstation
    leaderboard, at 453 kr per gigabyte of VRAM, beating every real candidate:
    *"Nybyggd gaming dator ryzen 7 5800x rtx 3080 32gb vit."* The 32 GB is
    system memory. Written without a comma and followed by "vit" rather than
    "RAM", it cleared both guards on `_VRAM_NEAR_MODEL` — one looks at the
    punctuation before the number, the other at the noun after it.

    Tightening that regex again would have been the third attempt to infer the
    answer from surrounding words, and an ad phrased a fourth way would beat it.
    But the capacity of an RTX 3080 is not a question about Swedish word order.
    It is 10 GB or 12 GB, that is already written down two hundred lines above,
    and 32 is not on the list — so the parse is rejected and the table answers.

    This is the module's own thesis turned on itself: specs are looked up,
    prices are counted, and nothing is inferred from prose that could have been
    read from a table.
    """
    variants = possible_vram(canonical_name)
    stated = vram_from_text(text_hint)
    if stated is not None and (not variants or stated in variants):
        return stated
    return variants[0] if variants else None


def gpu_tdp_w(canonical_name: str) -> int | None:
    spec = GPU_SPECS.get(canonical_name.lower().strip())
    return spec.tdp_w if spec else None


# Read off the model prefix rather than stored per row. The naming schemes are
# vendor-exclusive and have been for a decade — no AMD card is called "RTX", no
# Nvidia card is called "RX" — so a prefix is as reliable as a column here, and
# cannot fall out of step with `GPU_SPECS` when a row is added.
_VENDOR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("rtx ", "nvidia"),
    ("gtx ", "nvidia"),
    ("rx ", "amd"),
    ("arc ", "intel"),
)


def gpu_vendor(canonical_name: str) -> str | None:
    """Who made this card: "nvidia", "amd", "intel", or None if unrecognised.

    None is a real answer rather than a failure. It means the name matched no
    known scheme, and a caller asserting "this machine can run CUDA" has to
    treat that as unproven instead of assuming the common case.
    """
    lowered = canonical_name.lower().strip()
    for prefix, vendor in _VENDOR_PREFIXES:
        if lowered.startswith(prefix):
            return vendor
    return None


def is_vram_ambiguous(canonical_name: str) -> bool:
    return canonical_name.lower().strip() in AMBIGUOUS_VRAM


def vram_variants(canonical_name: str) -> tuple[int, ...]:
    return AMBIGUOUS_VRAM.get(canonical_name.lower().strip(), ())


# --- Capacities parsed back out of canonical names ---------------------------
#
# `canonicalize` already normalised these to a fixed shape ("32gb ddr4",
# "1tb ssd", "750w psu"), so reading a number back out is a parse of our own
# output rather than a second attempt at the seller's free text.

_RAM_CAPACITY = re.compile(r"^(\d{1,3})gb\s+ddr[345]$")
_STORAGE_CAPACITY = re.compile(r"^(\d{1,4})(tb|gb)\s+(ssd|hdd|nvme)$")
_PSU_WATTS = re.compile(r"^(\d{3,4})w\s+psu$")


def ram_capacity_gb(canonical_name: str) -> int | None:
    match = _RAM_CAPACITY.match(canonical_name.lower().strip())
    return int(match.group(1)) if match else None


def storage_capacity_gb(canonical_name: str) -> int | None:
    """Capacity in GB. TB is treated as 1000 GB, matching how drives are sold."""
    match = _STORAGE_CAPACITY.match(canonical_name.lower().strip())
    if not match:
        return None
    value = int(match.group(1))
    return value * 1000 if match.group(2) == "tb" else value


def psu_watts(canonical_name: str) -> int | None:
    match = _PSU_WATTS.match(canonical_name.lower().strip())
    return int(match.group(1)) if match else None
