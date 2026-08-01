"""Canonical-name tests.

Every case in the first test is a real string that appeared in the live index,
fragmenting one graphics card into five separate entries.
"""

from __future__ import annotations

import pytest

from pipeline.canonical import canonicalize


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
