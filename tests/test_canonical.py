"""Canonical-name tests.

Every case in the first test is a real string that appeared in the live index,
fragmenting one graphics card into five separate entries.
"""

from __future__ import annotations

import pytest

from pipeline.canonical import canonicalize, is_laptop_listing


def test_the_real_fragmentation_from_the_live_index_is_fixed():
    """These five names were five index rows for one card.

    `n=9` was being reported as `n=5` plus four singletons — and singletons read
    as "low confidence", so the system was disclaiming knowledge of a part it
    had genuinely seen nine times.
    """
    observed = [
        "rtx 3070",
        "Asus Dual GeForce RTX 3070 OC",
        "ASUS GeForce RTX 3070 DUAL V2",
        "Gigabyte GeForce RTX 3070",
        "RTX 3070 Asus TUF",
    ]

    assert {canonicalize(n) for n in observed} == {"rtx 3070"}


def test_a_ti_variant_is_still_a_different_part():
    """The line that must not be crossed. Model names nest, the longer one is
    the pricier part, and over-merging corrupts the index silently."""
    assert canonicalize("Gigabyte GeForce RTX 3070 Ti") != canonicalize("Gigabyte GeForce RTX 3070")
    assert canonicalize("Gigabyte GeForce RTX 3070 Ti") == "rtx 3070 ti"


@pytest.mark.parametrize(
    "variants,expected",
    [
        (["Ryzen 7 5800X", "AMD Ryzen7 5800X", "ryzen  7  5800x"], "ryzen 7 5800x"),
        (["i5-12400F", "Intel Core i5 12400F", "I5  12400f"], "i5-12400f"),
        (["RX 6800 XT", "AMD Radeon RX6800XT"], None),
        (["32GB DDR4 3200MHz", "32 GB DDR4"], "32gb ddr4"),
    ],
)
def test_variants_collapse_to_one_key(variants, expected):
    keys = {canonicalize(v) for v in variants}
    assert len(keys) == 1, f"{variants} fragmented into {keys}"
    if expected:
        assert keys == {expected}


def test_an_unrecognised_part_still_gets_a_stable_key():
    """No model pattern matches, so it will not merge with reworded sightings —
    but it must not collapse to empty and collide with everything else."""
    assert canonicalize("  Corsair   RM750x  ") == "corsair rm750x"
    assert canonicalize("Corsair RM750x") == canonicalize("corsair rm750x")


def test_empty_input_is_safe():
    assert canonicalize("") == ""


def test_both_extraction_paths_agree():
    """The writer and the reader must produce identical keys.

    `ai/batch_processor.py` canonicalises on write and `pipeline/index.py`
    matches on read. If these two ever disagree the index silently returns n=0
    for everything — no error, just an index that stopped working.
    """
    from pipeline.extract import extract

    ai_style = "Asus Dual GeForce RTX 3070 OC"
    regex_style = next(iter(extract("Asus Dual GeForce RTX 3070 OC").components)).component_name

    assert canonicalize(ai_style) == canonicalize(regex_style)


# --- Phase 5 additions --------------------------------------------------------


def test_three_digit_models_canonicalise():
    assert canonicalize("Sapphire Radeon RX 580 Nitro+ 8GB") == "rx 580"
    assert canonicalize("MSI GTX 970 Gaming") == "gtx 970"


def test_four_digit_reading_still_wins_where_one_exists():
    """`{3,4}` must not start truncating the cards that already worked."""
    assert canonicalize("RTX 3080 Ti") == "rtx 3080 ti"
    assert canonicalize("RX 6800 XT") == "rx 6800 xt"


def test_storage_collapses_to_capacity_and_medium():
    assert canonicalize("Samsung 980 Pro 1TB NVMe") == "1tb nvme"
    assert canonicalize("SSD 500GB Kingston") == "500gb ssd"
    assert canonicalize("1TB M.2") == canonicalize("1TB NVMe")


def test_psu_collapses_to_wattage():
    assert canonicalize("Corsair RM750 750W 80+ Gold") == "750w psu"
    assert canonicalize("650 W nätaggregat") == "650w psu"


def test_a_card_is_not_read_as_a_power_supply():
    """The PSU rule is last for exactly this reason — a bare wattage is the
    least specific thing in the list and would claim anything containing one."""
    assert canonicalize("RTX 3080 320W") == "rtx 3080"


def test_storage_and_psu_agree_across_both_extraction_paths():
    from pipeline.extract import extract

    for text in ("1TB NVMe", "Corsair 750W nätaggregat"):
        regex_name = next(iter(extract(text).components)).component_name
        assert canonicalize(text) == canonicalize(regex_name)


class TestLaptopGpusAreADifferentPart:
    """Nvidia gives the mobile and desktop card the same number. They are not
    the same silicon and do not trade at the same price.

    The live failure: an Acer Predator Helios laptop reached the top of the
    published board claiming 22 700 kr below parts, because its "5090" matched
    the single desktop RTX 5090 observation in the index at 38 000 kr.
    """

    def test_the_acer_that_caused_this(self):
        """Its title says "5090" and never says "laptop". What gives it away is
        the mobile CPU beside it — an i9 275HX cannot go in a desktop."""
        title = "Acer Predator Helios 16” 5090/I9 275HX/64G/2TB"
        assert is_laptop_listing(title, "rtx 5090", "intel ultra 9 275hx")

        laptop = is_laptop_listing(title, "rtx 5090", "intel ultra 9 275hx")
        assert canonicalize("RTX 5090", laptop=laptop) == "rtx 5090 laptop"

    def test_the_mobile_card_never_merges_with_the_desktop_one(self):
        assert canonicalize("RTX 5090", laptop=True) != canonicalize("RTX 5090")

    def test_desktop_cards_are_completely_unaffected(self):
        """The control, and the one that matters most. Every existing index key
        must survive unchanged, or this 'fix' silently re-fragments the whole
        index it was written to protect."""
        for name in ("RTX 3070", "RTX 4070 Ti Super", "GTX 1060", "RX 6800 XT"):
            assert canonicalize(name, laptop=False) == canonicalize(name)
            assert "laptop" not in canonicalize(name)

    def test_only_the_gpu_takes_the_suffix(self):
        """A mobile CPU already says so in its own model number, and laptop RAM
        and storage are the same parts at the same prices. Suffixing those would
        fragment the index for nothing."""
        assert canonicalize("i7-8850H", laptop=True) == "i7-8850h"
        assert canonicalize("32GB DDR4", laptop=True) == "32gb ddr4"
        assert canonicalize("512GB SSD", laptop=True) == "512gb ssd"

    def test_a_name_that_says_laptop_needs_no_help_from_the_caller(self):
        assert canonicalize("RTX 4080 Laptop GPU") == "rtx 4080 laptop"
        assert canonicalize("RTX 3070 Max-Q") == "rtx 3070 laptop"

    def test_the_suffix_is_not_applied_twice(self):
        assert canonicalize("RTX 4080 Laptop GPU", laptop=True) == "rtx 4080 laptop"

    @pytest.mark.parametrize(
        "title",
        [
            "MSI Stealth 15 OLED gaminglaptop 15,6\"",      # one word, not two
            "Lenovo Legion 7 pro RTX 5080 & Ryzen 9 9955HX3D",  # AMD mobile + 3D
            "HP ZBook 17 G5 – i7-8850H / Quadro P4200",
            "Dell Precision 5560 15,6\" UHD i9-11950H 32GB",
        ],
    )
    def test_real_laptops_from_the_corpus_are_detected(self, title):
        assert is_laptop_listing(title)

    @pytest.mark.parametrize(
        "title",
        [
            "Gamingdator – i9-12900KF / RTX 4070 Ti / 32 GB RAM",
            "Egenbyggd speldator i7-6700K / GTX 970 / 16 GB RAM",
            "Nybyggd gaming dator ryzen 7 5800x rtx 3080 32gb vit.",
            "MSI Codex XE Gamingdator – RTX 2080 8GB / i7-8700K",
        ],
    )
    def test_real_desktops_from_the_corpus_are_not(self, title):
        assert not is_laptop_listing(title)

    def test_the_description_is_never_consulted(self):
        """Measured: "laptop" appears in 292 descriptions, and the first one
        read belonged to an "Uppgraderad stationär" — a desktop whose seller
        merely mentioned laptops. Only the title and the parts list are asked.
        """
        desktop = "HP Omen 30L 3080Ti | R7-5800x | Uppgraderad stationär"
        assert not is_laptop_listing(desktop)

        body = "Säljer även min gamla laptop separat, hör av dig."
        assert not is_laptop_listing(desktop), "the title alone decides"
        assert is_laptop_listing(body), "but the same words in a title would count"
