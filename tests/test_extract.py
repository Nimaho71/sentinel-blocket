"""Deterministic extraction tests.

Structured around the asymmetry that shapes the module: being unsure is cheap
(one LLM call), being confidently wrong is expensive (a corrupt price index).
So most of these assert that doubtful input yields `None` rather than a guess,
and the sharpest one is the regression where a whole computer was confidently
filed as a bare graphics card.
"""

from __future__ import annotations

import pytest

from db.models import ComponentTypeEnum, ItemTypeEnum
from pipeline.extract import extract, find_components, find_risk_signals


def kinds(extraction) -> set[ComponentTypeEnum]:
    return {c.component_type for c in extraction.components}


def names(extraction) -> set[str]:
    return {c.component_name for c in extraction.components}


# --- Component recognition ---------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("RTX 3080", "RTX 3080"),
        ("rtx3080", "RTX3080"),
        ("GeForce RTX 3060 Ti 8GB", "RTX 3060 TI"),
        ("Radeon RX 6800 XT", "RX 6800 XT"),
        ("GTX 1080 Ti", "GTX 1080 TI"),
    ],
)
def test_graphics_cards_are_recognised(text, expected):
    assert expected in names(extract(text))


def test_a_ti_suffix_is_not_dropped():
    """The nesting problem that made `ilike` wrong in the index.

    If the base pattern claimed the prefix first, "RTX 3060 Ti" would be read as
    "RTX 3060" — a materially cheaper card.
    """
    assert names(extract("RTX 3060 Ti")) == {"RTX 3060 TI"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("i7-10700", "I7-10700"),
        ("Intel i5 12400F", "I5-12400F"),
        ("i9-14900KS", "I9-14900KS"),
        ("Ryzen 7 5800X", "RYZEN 7 5800X"),
        ("Ryzen5 5600", "RYZEN 5 5600"),
    ],
)
def test_processors_are_recognised(text, expected):
    assert expected in names(extract(text))


@pytest.mark.parametrize(
    "variants",
    [
        ("Ryzen 5 5600", "Ryzen5 5600", "ryzen  5  5600", "RYZEN 5 5600"),
        ("i5-12400F", "i5 12400F", "I5  12400f"),
        ("RTX 3080", "rtx 3080", "RTX  3080"),
    ],
)
def test_spelling_variants_canonicalise_to_one_name(variants):
    """The property that makes the market index work.

    Two spellings of one part must not become two entries — that halves both
    sample counts and makes each median look thinner than the evidence really
    is. Found by "Ryzen5 5600" and "Ryzen 5 5600" disagreeing.
    """
    canonical = {next(iter(names(extract(v)))) for v in variants}

    assert len(canonical) == 1, f"{variants} produced {canonical}"


def test_memory_is_recognised():
    assert "32GB DDR4" in names(extract("32 GB DDR4 3200MHz"))


def test_a_part_repeated_in_title_and_body_is_counted_once():
    """Sellers restate the model in the description. Counting it twice would
    tell the index there were two graphics cards."""
    extraction = extract("RTX 3080 säljes", "Säljer min RTX 3080, fint skick")

    assert len(extraction.components) == 1


def test_unknown_hardware_is_simply_not_found():
    """Silence, not a guess. A hallucinated part poisons the index."""
    assert extract("Snurrig grunka till datorn").components == ()


# --- Classification: the expensive direction ---------------------------------


def test_a_complete_pc_is_not_filed_as_a_bare_graphics_card():
    """Regression from the live corpus.

    "Komplett Gaming Dator Setup" contains no literal "komplett dator" — the
    word "Gaming" sits between them — so a phrase list missed it and a whole
    computer was confidently classified as a single GPU. Priced against a GPU
    median, that ad would top the leaderboard forever.
    """
    extraction = extract("Komplett Gaming Dator Setup – RTX 3060")

    assert extraction.item_type is not ItemTypeEnum.SINGLE_COMPONENT


@pytest.mark.parametrize(
    "title",
    [
        "Komplett Gaming Dator Setup – RTX 3060",
        "Gaming PC med RTX 3060",
        "Speldator RTX 3060",
        "Stationär dator RTX 3060",
        "Snygg rig med RTX 3060",
    ],
)
def test_build_language_prevents_a_confident_single_component(title):
    """Whatever else it decides, it must never say "just a GPU" here."""
    assert extract(title).item_type is not ItemTypeEnum.SINGLE_COMPONENT


def test_a_full_build_needs_both_a_cpu_and_a_gpu():
    extraction = extract("Kraftfull speldator RTX 3060 | i7-10700 | 32Gb DDR4")

    assert extraction.item_type is ItemTypeEnum.FULL_BUILD
    assert extraction.is_confident
    assert kinds(extraction) >= {ComponentTypeEnum.CPU, ComponentTypeEnum.GPU}


def test_build_language_without_a_cpu_asks_the_model():
    """A "gaming PC" ad listing only a GPU is genuinely unclear — it may be a
    prebuilt whose CPU went unmentioned, or a GPU pulled from one."""
    assert extract("Gamingdator RTX 3060").item_type is None


def test_a_lone_component_is_classified_confidently():
    extraction = extract("ASUS Dual RTX 3060 Ti OC V2 (8GB)")

    assert extraction.item_type is ItemTypeEnum.SINGLE_COMPONENT
    assert extraction.is_confident


def test_a_bundle_without_build_language_asks_the_model():
    """CPU + motherboard and a stripped-down PC read almost identically."""
    assert extract("Ryzen 7 5800X och RTX 3080 säljes").item_type is None


def test_nothing_found_is_never_confident():
    assert not extract("Diverse datorprylar").is_confident


# --- Risk signals send an ad to the model regardless -------------------------


@pytest.mark.parametrize(
    "phrase",
    ["ej testad", "i befintligt skick", "defekt", "inget kvitto", "mining", "för delar"],
)
def test_risk_language_is_detected(phrase):
    assert find_risk_signals(f"RTX 3080. {phrase}. Ring för info") != ()


def test_a_risky_ad_is_never_confident_however_cleanly_it_parses():
    """The specs parse perfectly; that is exactly the trap.

    Judging "ej testad" is what language models are good at and regexes are bad
    at, so a clean parse must not be allowed to short-circuit that.
    """
    extraction = extract("ASUS RTX 3060 Ti", "Ej testad, säljes i befintligt skick")

    assert extraction.components
    assert extraction.item_type is ItemTypeEnum.SINGLE_COMPONENT
    assert not extraction.is_confident


def test_a_clean_ad_carries_no_risk_signals():
    assert extract("ASUS RTX 3060 Ti", "Nyskick, kvitto finns.").is_confident


# --- Shape -------------------------------------------------------------------


def test_empty_input_is_safe():
    extraction = extract("", None)

    assert extraction.components == ()
    assert extraction.item_type is None
    assert not extraction.is_confident


def test_extraction_reads_the_description_as_well_as_the_title():
    assert find_components("Bra dator") == ()
    assert names(extract("Bra dator", "Innehåller RTX 3080")) == {"RTX 3080"}


# --- Storage and power supplies ----------------------------------------------
#
# Added in Phase 5, once the build profiles gave these numbers something to be
# tested against. The module docstring had deliberately left them out until
# then: an unvalidated pattern for an ambiguous number is how an index gets
# quietly poisoned.


def test_storage_is_read_in_either_word_order():
    # Names are upper-cased by `_canonical`, as everywhere else in this module.
    assert names(extract("Samsung 970 EVO 1TB NVMe")) >= {"1TB NVME"}
    assert names(extract("SSD 500GB till salu")) >= {"500GB SSD"}


def test_m2_and_nvme_are_the_same_medium():
    assert names(extract("1TB M.2")) == names(extract("1TB NVMe"))


def test_a_bare_capacity_is_not_a_drive():
    """The guard that makes the storage pattern safe.

    "500GB" appears in ads about phones, cameras and cloud plans, and in PC ads
    it is as often a passing mention as a thing being sold. Requiring the medium
    word costs a few real matches and prevents a category of nonsense.
    """
    assert find_components("Säljer 500GB") == ()


def test_vram_is_not_mistaken_for_storage():
    """"RTX 3060 12GB" is a graphics card, not a twelve-gigabyte drive."""
    assert names(extract("RTX 3060 12GB")) == {"RTX 3060"}


def test_a_wattage_needs_a_power_supply_word_to_vouch_for_it():
    """The single most treacherous number in a hardware ad.

    Identical digits mean a monitor's rating, the card's *recommended* PSU size,
    or an actual power supply for sale. Only the surrounding words distinguish
    them, so only the surrounding words are trusted.
    """
    assert names(extract("Corsair 750W nätaggregat")) >= {"750W PSU"}
    assert names(extract("Seasonic 650W")) >= {"650W PSU"}

    # A recommended PSU size mentioned in a graphics-card ad is not a PSU.
    assert names(extract("RTX 3080 säljes, kräver 750W")) == {"RTX 3080"}


def test_psu_context_must_be_nearby_not_merely_present():
    """A PSU word three sentences away does not vouch for a wattage."""
    text = "Nätaggregat medföljer. " + "x" * 200 + " Skärmen drar 450W vid full ljusstyrka."
    assert not any(c.component_name.endswith("PSU") for c in find_components(text))


# --- Older cards --------------------------------------------------------------


def test_three_digit_model_numbers_are_recognised():
    """The RX 570 and 580 are among the most-traded used cards here.

    A four-digit-only pattern matched none of them — they fell through to the
    raw-name fallback, and every wording became its own index entry.
    """
    assert names(extract("Radeon RX 580 8GB")) == {"RX 580"}
    assert names(extract("GTX 970 säljes")) == {"GTX 970"}


class TestTradeIntentIsNotAPartsList:
    """A used-marketplace listing routinely names hardware it does not contain.

    The live failure: a HP ZBook 17 G5 laptop, whose actual GPU is a Quadro
    P4200, said "Främst intresserad av RTX 3080 Ti eller bättre" — what the
    seller wanted in trade. Deterministic extraction recorded an RTX 3080 Ti as
    a component, valued the machine 5 500 kr above its contents, and published
    it as the best deal on the board.
    """

    _TITLE = "HP ZBook 17 G5 – i7-8850H / 32GB RAM / Quadro P4200 8GB / 512GB SSD"
    _BODY = (
        "Säljer min HP ZBook 17 G5 i bra skick. En rejäl och kraftfull "
        "workstation-laptop som passar bra för CAD, 3D och rendering.\n\n"
        "Intresserad av grafikkort + am4 processor.\n"
        "Främst intresserad av RTX 3080 Ti eller bättre.\n"
    )

    def test_the_zbook_never_settles_deterministically(self):
        extraction = extract(self._TITLE, self._BODY)

        assert extraction.trade_signals, "the trade phrasing must be seen"
        assert not extraction.is_confident, (
            "an ad naming parts it wants must go to the model, not be valued "
            "from a regex read of its wishlist"
        )

    def test_trade_intent_is_not_reported_as_a_risk(self):
        """A seller asking to trade is not describing a fault. Keeping the two
        apart is what stops 'wants an RTX 3080' being published as a warning
        about the machine's condition."""
        extraction = extract(self._TITLE, self._BODY)

        assert extraction.trade_signals
        assert not extraction.risk_signals

    def test_an_ordinary_listing_still_skips_the_model(self):
        """The control. Over-matching costs a model call on every clean ad in
        the corpus, so the guard has to stay off ads that never mention a
        trade — otherwise this 'fix' quietly disables the fast path."""
        extraction = extract(
            "ASUS Dual RTX 3060 Ti OC V2 (8GB)",
            "Nyskick, kvitto finns. Fungerar perfekt.",
        )

        assert extraction.trade_signals == ()
        assert extraction.is_confident

    @pytest.mark.parametrize(
        "phrase",
        [
            "Kan tänka mig att byta mot ps5 pro.",
            "Byte mot laptop eller 27000 vid snabb affär.",
            "Säljes eller bytes mot gaming-laptop.",
            "4080 eller liknande kort i byte + pengar.",
            "Främst intresserad av RTX 3080 Ti eller bättre.",
        ],
    )
    def test_real_trade_phrasings_are_caught(self, phrase):
        """Every one of these is quoted from the live corpus."""
        extraction = extract("Gaming PC RTX 3060 / Ryzen 5 5600", f"Fin dator. {phrase}")

        assert extraction.trade_signals, f"missed trade intent in: {phrase}"
        assert not extraction.is_confident

    @pytest.mark.parametrize(
        "phrase",
        [
            "Hör av dig om du vill ha fler bilder.",
            "Söker du en välbalanserad och högpresterande dator?",
            "Kringutrustning medföljer om köparen önskar.",
            "Säljer inför uppgradering till nytt kort.",
            "Går att uppgradera till Windows 11.",
            "Inga skambud eller byten, tack.",
        ],
    )
    def test_ordinary_sales_copy_is_not_mistaken_for_a_trade(self, phrase):
        """The half that keeps this from being useless.

        Each phrase below was a candidate signal, and each was measured against
        the live corpus and rejected — "vill ha" alone matched 262 ads, almost
        all of them offering more photographs. A signal list that fires on
        ordinary sales copy does not weakly help, it stops being a signal.

        "Inga byten" is the sharpest case: the seller is *refusing* trades, so
        the parts list is the most trustworthy kind there is. Flagging it would
        invert the meaning entirely.
        """
        extraction = extract(
            "Gaming PC RTX 3060 / Ryzen 5 5600", f"Fin dator, allt fungerar. {phrase}"
        )

        assert extraction.trade_signals == (), f"false positive on: {phrase}"
        assert extraction.is_confident
