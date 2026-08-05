"""Deterministic spec extraction: resolve what regex can, before spending an LLM call.

### An honest accounting of what this is for

The obvious justification is quota, and that one is weaker than it looks. Two
Flash Lite pools give 1 000 requests/day; at batch size 10 that is ~10 000 ads
daily, against a working corpus in the hundreds. **Quota is not currently the
binding constraint,** and claiming this module exists to save it would be
retrofitting a reason.

The real reasons are better ones:

- **Determinism.** `"RTX 3080"` in a title parses the same way every time. An
  LLM is stochastic, and a component name that varies between runs produces a
  market index that quietly disagrees with itself.
- **Testability.** These rules can be checked against 474 real ads in
  milliseconds with no network and no key.
- **Fewer tokens.** TPM is a tighter ceiling than RPM, and every ad resolved
  here is one whose text never needs sending.
- **A floor under failures.** When the AI stage dead-letters an ad, the
  deterministic parse still gives the index something to work with.

Quota headroom is a real benefit, just a secondary one — and it is the reason
this can afford to be conservative. It never guesses. An ad it is unsure about
goes to the model, which is the expensive-but-better path, and being wrong is
worse than being slow.

### What is deliberately *not* done here

Risk assessment. "Ej testad", "inget kvitto", "säljes i befintligt skick" and
the dozen other ways a Swedish seller hedges are exactly what language models
read well and regexes read badly. Any ad carrying a risk signal is sent onward
regardless of how cleanly its specs parsed — see `RISK_SIGNALS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai.schemas import ComponentVerdict
from db.models import ComponentTypeEnum, ItemTypeEnum

# --- Part patterns -----------------------------------------------------------
#
# Each pattern captures a canonical name in group 1. Ordered most-specific
# first: "RTX 3080 Ti" must be tried before "RTX 3080", or the base pattern
# claims the prefix and the Ti suffix is lost — the same nesting problem that
# made `ilike '%name%'` wrong in the market index.

_GPU = re.compile(
    r"\b((?:RTX|GTX)\s*\d{3,4}(?:\s*(?:Ti\s*Super|Super|Ti))?|RX\s*\d{3,4}(?:\s*(?:XTX|XT))?)\b",
    re.IGNORECASE,
)
_CPU_INTEL = re.compile(r"\b(i[3579])[\s-]*(\d{4,5}[A-Z]{0,2})\b", re.IGNORECASE)
# The tier digit is captured separately so that "Ryzen5 5600" and "Ryzen 5 5600"
# canonicalise identically. Emitting the seller's spacing verbatim would split
# one CPU into two entries in the market index — halving both sample counts and
# making each median less trustworthy than the data actually warrants.
_CPU_AMD = re.compile(r"\bRyzen\s*([3579])\s*(\d{4}[A-Z]{0,3})\b", re.IGNORECASE)
# Memory. Generation and capacity are matched separately and paired by
# `_ram_capacity_near`, because the relationship between them is not positional
# — see that function for the three shapes sellers use and why the largest
# capacity in the window is the right one to take.
_RAM_GEN = re.compile(r"\bDDR([345])\b", re.IGNORECASE)
_RAM_CAP = re.compile(r"\b(\d{1,3})\s*GB\b", re.IGNORECASE)

# How far either side of "DDR4" a capacity may sit and still belong to it.
# Wide enough for "16GB (2x8GB DDR4 3200MHz)", narrow enough that a capacity
# from a different sentence cannot be captured.
_RAM_WINDOW = 40


def _ram_capacity_near(text: str, start: int, end: int) -> str | None:
    """The memory capacity a `DDR…` mention belongs to, if any.

    Not a regex, because the rule is not positional. Sellers write memory three
    ways and the old pattern read only the first:

        "16GB DDR4"                  capacity first, adjacent
        "DDR4 16GB"                  generation first
        "16GB (2x8GB DDR4 3200MHz)"  kit notation, generation inside the pair

    The kit form is the usual way to advertise a matched pair, so this was not
    a rare edge — one published build lost its memory entirely and was valued
    1 100 kr low on an ad that stated it plainly.

    **The largest capacity in the window wins**, and that is the load-bearing
    choice. In "16GB (2x8GB DDR4)" the *nearest* figure is the per-stick 8 GB
    and the right answer is the 16 GB total; indexing a kit as half its size
    would be worse than missing it. The same rule reads "8GB VRAM och 32GB
    DDR5" correctly, where the nearest-figure rule would have taken the
    graphics card's memory.
    """
    before = text[max(0, start - _RAM_WINDOW) : start]
    after = text[end : end + _RAM_WINDOW]
    sizes = [int(m.group(1)) for m in _RAM_CAP.finditer(before)]
    if not sizes:
        sizes = [int(m.group(1)) for m in _RAM_CAP.finditer(after)]
    if not sizes:
        return None
    return str(max(sizes))

# Storage. The medium word is mandatory in both orderings, and that is the whole
# safety mechanism: a bare "500GB" in a PC listing is as likely to be the second
# drive's capacity mentioned in passing, or a phone's storage in an unrelated
# sentence, as it is a drive being sold. Requiring "SSD"/"HDD"/"NVMe"/"M.2"
# next to the number costs a few real matches and prevents a class of nonsense.
_STORAGE = re.compile(
    r"\b(?:(\d{1,4})\s*(TB|GB)\s*(NVMe|SSD|HDD|M\.?2)"
    r"|(NVMe|SSD|HDD|M\.?2)\s*(\d{1,4})\s*(TB|GB))\b",
    re.IGNORECASE,
)

# Power supplies. A wattage on its own is the single most treacherous number in
# a hardware ad — it is the monitor's rating, the kettle in the photo, the
# card's recommended PSU size, or the actual power supply, and the digits look
# identical in all four cases. So a wattage only counts when a power-supply word
# sits within `_PSU_WINDOW` characters of it.
#
# This is why the module docstring left PSU out until there was a profile rule
# to test it against: an unvalidated pattern for an ambiguous number is exactly
# the kind of thing that quietly poisons an index.
_PSU_WATTS = re.compile(r"\b([2-9]\d{2}|1\d{3})\s*W\b", re.IGNORECASE)
_PSU_WORD = re.compile(
    r"\b(?:nätagg\w*|natagg\w*|nätdel\w*|aggregat|psu|power\s*supply|nätaggregat"
    r"|strömförsörjning|80\s*\+?\s*plus|seasonic|corsair\s*rm|be\s*quiet)\b",
    re.IGNORECASE,
)
_PSU_WINDOW = 40

# Words that mean "a human should read this listing". Their presence sends the
# ad to the model no matter how cleanly the specs parsed.
RISK_SIGNALS = (
    "ej testad", "ej testat", "otestad", "obeprövad",
    "befintligt skick", "i befintligt",
    "defekt", "trasig", "trasigt", "går ej", "fungerar ej", "fungerar inte",
    "spricka", "sprucken", "skadad", "skada",
    "utan kvitto", "inget kvitto", "saknar kvitto",
    "mining", "minat", "minad",
    "reservdel", "för delar", "som delar",
    "osäker", "vet ej", "vet inte",
    "as-is", "untested", "faulty", "broken", "for parts",
    "startar inte", "startar ej", "stendöd", "obs läs beskrivning",
    # Cosmetic damage. It costs real money without stopping the machine
    # working, which is exactly why these belong here and *not* in
    # `DAMAGE_MARKERS`: a shattered side panel must send the ad to a model, but
    # it must not disqualify the listing from being a valid parts-comparison
    # baseline the way "defekt" does. The two lists answer different questions.
    #
    # From a live listing: "Glaset sprängdes precis innan jag skulle lägga upp
    # annonsen så behövs ett nytt glas" — a shattered tempered-glass panel on a
    # Phanteks P400A, several hundred kronor to replace and awkward to source,
    # inside an ad that otherwise reads as a clean, well-specified build.
    # Nothing caught it: "spricka" and "sprucken" were already here, but
    # "sprängdes" was not, and the machine works, so no damage marker applied
    # either. It reached the board as an unqualified bargain.
    "sprängdes", "sprängde", "sprängt", "spräckt", "spräcktes",
    "krossad", "krossat", "krossade",
    "nytt glas", "trasigt glas", "sprucket glas", "saknar glas",
    "repor", "repad", "bucklor", "bucklig",
)

# Phrases meaning a part named nearby is what the seller **wants**, not what
# they are selling. A used-marketplace listing routinely names hardware it does
# not contain: trade offers, upgrade wishlists, compatibility notes.
#
# Found live. A HP ZBook 17 G5 — a laptop with a Quadro P4200 — carried the
# line "Främst intresserad av RTX 3080 Ti eller bättre" in its description.
# Pattern matching read the model name and recorded an RTX 3080 Ti as a
# component, valuing the machine 5 500 kr above what it contained and
# publishing it as the best deal on the board. The text was parsed correctly;
# the wrong meaning was taken from it.
#
# Separate from `RISK_SIGNALS` because the two say different things: risk means
# "the condition needs a human", trade means "the parts list may be fiction".
# Both route to the model, where the surrounding sentence is available and
# "interested in" is not hard to understand.
#
# Every phrase below was checked against the live corpus before being kept, and
# most candidates did not survive. The instinct to be generous — a false
# positive only costs one model call — is wrong here for a different reason:
# a phrase that matches ordinary sales copy is not a weak signal, it is not a
# signal, and a list of those is indistinguishable from no list at all.
# Measured and rejected:
#
#   "vill ha"           262 hits, almost all "om du vill ha fler bilder"
#   "söker"              24 hits, "Söker du en välbalanserad dator?"
#   "önskar"             16 hits, "om köparen önskar"
#   "uppgradering till"  25 hits, "säljer inför uppgradering till nytt kort"
#   "looking for"         2 hits, "unless you are looking for them closely"
#   "byten"              36 hits, overwhelmingly "Inga byten!" — an ad
#                        *refusing* trades, whose parts list is the most
#                        trustworthy kind there is. Precisely inverted.
TRADE_SIGNALS = (
    "byta mot", "byte mot", "bytes mot", "byter mot",
    "i byte", "i utbyte",
    "intresserad av", "intresserad utav",
    "eller bättre", "eller liknande",
    "want to trade", "trade for", "willing to trade", "in trade", "or better",
)

# The subset of `RISK_SIGNALS` that does not hedge — the seller is saying the
# machine is broken, is being sold for parts, or is not being vouched for at
# all. Everything in `RISK_SIGNALS` means "a human should read this"; these
# additionally mean "this is not a working computer".
#
# Kept separate because the two lists are consumed by different decisions.
# `RISK_SIGNALS` decides whether to spend a model call, where over-matching
# costs nothing but quota. `DAMAGE_MARKERS` disqualifies a listing from being
# the baseline the whole site compares parts prices against, where a false
# negative publishes "a working parts build costs 2 300 kr more than this
# machine" about a laptop its own seller calls stendöd.
#
# "befintligt skick" and "obs läs beskrivning" are hedges rather than
# statements of damage, and are here deliberately. For a headline comparison
# the safe direction is to decline a machine nobody will vouch for.
DAMAGE_MARKERS = (
    "defekt", "trasig", "trasigt",
    "går ej", "startar inte", "startar ej", "fungerar ej", "fungerar inte",
    "stendöd",
    "för delar", "som delar", "reservdel",
    "befintligt skick",
    "obs läs beskrivning",
    "broken", "faulty", "for parts", "as-is", "not working",
)

# Words implying a complete machine.
#
# Matched as whole words rather than fixed phrases, after a real
# misclassification: "Komplett Gaming Dator Setup" does not contain the literal
# "komplett dator" — "Gaming" sits between them — so a full PC was filed as a
# bare graphics card. That is the worst direction to be wrong in, since a whole
# computer priced against a GPU median looks like the deal of the century.
#
# Over-matching here is safe by construction. A build word without both a CPU
# and a GPU yields None ("ask the model"), never a confident answer — so a false
# positive costs one LLM call while a false negative corrupts a valuation.
_BUILD_WORDS = re.compile(
    r"\b(?:dator|datorn|datorer|pc|speldator|gamingdator|stationär|stationära"
    r"|desktop|setup|komplett|rig|system|burk|tower|chassi)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Extraction:
    """What could be read from an ad without asking a model."""

    components: tuple[ComponentVerdict, ...]
    item_type: ItemTypeEnum | None
    risk_signals: tuple[str, ...]
    # Kept apart from `risk_signals` because they are not a risk: an ad asking
    # to trade is in perfect condition. What they cast doubt on is the *parts
    # list*, which is a different failure and gets a different published flag.
    trade_signals: tuple[str, ...] = ()

    @property
    def is_confident(self) -> bool:
        """Whether this ad can skip the model entirely.

        Requires all four: something was found, the type is unambiguous,
        nothing in the text needs judgement, and no phrase suggests a named
        part is wanted rather than owned. Any doubt and the answer is no.
        """
        return (
            bool(self.components)
            and self.item_type is not None
            and not self.risk_signals
            and not self.trade_signals
        )


def _canonical(text: str) -> str:
    return " ".join(text.split()).upper()


def find_components(text: str) -> tuple[ComponentVerdict, ...]:
    """Pull recognisable hardware out of free text.

    Deduplicated by canonical name: sellers repeat the model in the title and
    again in the body, and counting it twice would tell the index there were
    two graphics cards.
    """
    found: dict[str, ComponentVerdict] = {}

    def add(ctype: ComponentTypeEnum, name: str) -> None:
        key = f"{ctype.value}:{_canonical(name)}"
        found.setdefault(key, ComponentVerdict(ctype, _canonical(name)))

    for match in _GPU.finditer(text):
        add(ComponentTypeEnum.GPU, match.group(1))
    for match in _CPU_INTEL.finditer(text):
        add(ComponentTypeEnum.CPU, f"{match.group(1)}-{match.group(2)}")
    for match in _CPU_AMD.finditer(text):
        add(ComponentTypeEnum.CPU, f"Ryzen {match.group(1)} {match.group(2)}")
    for match in _RAM_GEN.finditer(text):
        size = _ram_capacity_near(text, match.start(), match.end())
        if size is not None:
            add(ComponentTypeEnum.RAM, f"{size}GB DDR{match.group(1)}")
    for match in _STORAGE.finditer(text):
        # One alternation, two orderings — whichever branch fired supplies the
        # groups; the other three are None.
        size, unit, medium = (
            (match.group(1), match.group(2), match.group(3))
            if match.group(1)
            else (match.group(5), match.group(6), match.group(4))
        )
        add(ComponentTypeEnum.STORAGE, f"{size}{unit.upper()} {_storage_medium(medium)}")
    for match in _PSU_WATTS.finditer(text):
        if _has_psu_context(text, match.start(), match.end()):
            add(ComponentTypeEnum.PSU, f"{match.group(1)}W PSU")

    return tuple(found.values())


def _storage_medium(raw: str) -> str:
    return "NVMe" if raw.lower().replace(".", "") == "m2" else raw.upper()


def _has_psu_context(text: str, start: int, end: int) -> bool:
    """Whether a power-supply word sits close enough to a wattage to vouch for it."""
    window = text[max(0, start - _PSU_WINDOW) : end + _PSU_WINDOW]
    return _PSU_WORD.search(window) is not None


def find_risk_signals(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(signal for signal in RISK_SIGNALS if signal in lowered)


def find_trade_signals(text: str) -> tuple[str, ...]:
    """Phrases suggesting a named part is wanted rather than owned."""
    lowered = text.lower()
    return tuple(signal for signal in TRADE_SIGNALS if signal in lowered)


def find_damage_markers(*texts: str | None) -> tuple[str, ...]:
    """Markers saying the item does not work, across every text given.

    Variadic because the callers hold the evidence in pieces — title, body, and
    the model's own risk flags — and any one of them is enough. Ad 13 says
    "DEFEKT" in its title, "helt defekt skick" in its body and "defekt" in its
    flags; a caller that happened to have only one of the three should still
    reach the same answer.
    """
    lowered = "\n".join(text.lower() for text in texts if text)
    return tuple(marker for marker in DAMAGE_MARKERS if marker in lowered)


def classify(text: str, components: tuple[ComponentVerdict, ...]) -> ItemTypeEnum | None:
    """Decide item type, or return None meaning "ask the model".

    Returning None is the common and correct outcome. This only answers the two
    cases that are genuinely unambiguous: a build word plus a CPU and a GPU, or
    exactly one component and no build language anywhere.
    """
    kinds = {component.component_type for component in components}
    mentions_build = _BUILD_WORDS.search(text) is not None

    if mentions_build and {ComponentTypeEnum.CPU, ComponentTypeEnum.GPU} <= kinds:
        return ItemTypeEnum.FULL_BUILD

    if len(components) == 1 and not mentions_build:
        return ItemTypeEnum.SINGLE_COMPONENT

    # Two or three parts with no build language is the genuinely ambiguous case
    # — a CPU+motherboard bundle and a stripped-down PC read almost identically.
    return None


def extract(title: str, description: str | None = "") -> Extraction:
    """Read an ad deterministically."""
    text = f"{title}\n{description or ''}"
    components = find_components(text)
    return Extraction(
        components=components,
        item_type=classify(text, components),
        risk_signals=find_risk_signals(text),
        trade_signals=find_trade_signals(text),
    )
