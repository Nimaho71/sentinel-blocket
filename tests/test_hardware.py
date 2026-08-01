"""Tests for the hardware reference table.

The table itself is data and a test cannot tell whether an entry is *correct* —
only a datasheet can. What these tests protect is the logic around it: that a
stated capacity beats an assumed one, that an unknown card reads as unknown
rather than as zero, and that the ambiguous models are handled pessimistically.
Those are the parts that can be wrong in a way that costs money.
"""

from __future__ import annotations

import pytest

from db.models import ComponentTypeEnum
from pipeline.canonical import canonicalize
from pipeline.hardware import (
    AMBIGUOUS_VRAM,
    GPU_SPECS,
    gpu_tdp_w,
    gpu_vendor,
    gpu_vram_gb,
    is_vram_ambiguous,
    possible_vram,
    psu_watts,
    ram_capacity_gb,
    storage_capacity_gb,
    vram_from_text,
)
from pipeline.profiles import machine_spec


class TestTableIntegrity:
    def test_every_key_is_already_canonical(self):
        """A key the canonicaliser would rewrite is a key that never matches.

        This is the failure mode that would be invisible in production: the
        lookup simply returns None for every card, every VRAM check reads
        "unknown", and the profile silently stops filtering.
        """
        for name in GPU_SPECS:
            assert canonicalize(name) == name, f"{name!r} is not in canonical form"

    def test_ambiguous_models_are_all_in_the_spec_table(self):
        assert set(AMBIGUOUS_VRAM) <= set(GPU_SPECS)

    def test_ambiguous_entries_record_the_smaller_variant(self):
        """The pessimistic-assumption rule, enforced rather than documented.

        If a future edit sets the RTX 3060 to 12 GB, an 8 GB card starts
        clearing a 12 GB floor — and somebody buys the wrong card.
        """
        for name, variants in AMBIGUOUS_VRAM.items():
            assert GPU_SPECS[name].vram_gb == min(variants), name


class TestVramFromText:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("RTX 3060 12GB", 12),
            ("Säljer RTX 3060 12 GB, fint skick", 12),
            ("RTX 3080-10GB", 10),
            ("rx 6700 xt 12gb", 12),
        ],
    )
    def test_reads_a_stated_capacity(self, text, expected):
        assert vram_from_text(text) == expected

    def test_ignores_capacities_that_are_not_attached_to_a_gpu(self):
        """The bug this exists to prevent: reporting a 32 GB graphics card.

        System RAM and drive capacities live in the same sentence as the GPU in
        almost every full-build ad, and an unanchored `\\d+GB` pattern picks up
        whichever comes first.
        """
        assert vram_from_text("RTX 3070, 32GB RAM, 1TB SSD") != 32
        assert vram_from_text("Intel i5-12400, 32GB RAM, 1TB SSD") is None

    def test_ignores_implausible_capacities(self):
        assert vram_from_text("RTX 3080 999GB") is None

    def test_no_capacity_stated(self):
        assert vram_from_text("Säljer mitt RTX 3080") is None


class TestGpuVram:
    def test_stated_capacity_beats_the_table(self):
        """The seller is holding the card; the table is not."""
        assert gpu_vram_gb("rtx 3060") == 8  # pessimistic table default
        assert gpu_vram_gb("rtx 3060", text_hint="RTX 3060 12GB") == 12

    def test_unknown_card_is_unknown_not_zero(self):
        """`None` and `0` diverge at the first comparison, and only one is right."""
        assert gpu_vram_gb("radeon vii") is None
        assert gpu_vram_gb("radeon vii") != 0

    def test_unambiguous_card_is_not_flagged(self):
        assert not is_vram_ambiguous("rtx 3070")
        assert is_vram_ambiguous("rtx 3060")

    def test_tdp_lookup(self):
        assert gpu_tdp_w("rtx 3080") == 320
        assert gpu_tdp_w("nonexistent") is None


class TestCapacityParsing:
    @pytest.mark.parametrize(
        "canonical,expected",
        [("32gb ddr4", 32), ("16gb ddr5", 16), ("rtx 3080", None)],
    )
    def test_ram(self, canonical, expected):
        assert ram_capacity_gb(canonical) == expected

    @pytest.mark.parametrize(
        "canonical,expected",
        [("1tb ssd", 1000), ("500gb ssd", 500), ("2tb hdd", 2000), ("1tb nvme", 1000)],
    )
    def test_storage(self, canonical, expected):
        assert storage_capacity_gb(canonical) == expected

    def test_psu(self):
        assert psu_watts("750w psu") == 750
        assert psu_watts("32gb ddr4") is None


class TestStatedVramMustBePlausible:
    """Regression: the published leaderboard's number-one AI workstation.

    *"Nybyggd gaming dator ryzen 7 5800x rtx 3080 32gb vit."* — 14 500 kr, read
    as a 32 GB graphics card, and therefore the cheapest VRAM on the market at
    453 kr/GB. The 32 GB is system memory.

    Two earlier guards on `_VRAM_NEAR_MODEL` did not catch it: one rejects a
    comma before the number, the other a hardware noun after it, and this ad has
    neither. The third attempt at reading Swedish word order would have lost to
    a fourth phrasing, so the check moved to the one place that cannot be
    rephrased — an RTX 3080 has 10 or 12 GB, and that is written down.
    """

    def test_the_live_example(self):
        assert gpu_vram_gb(
            "rtx 3080", text_hint="Nybyggd gaming dator ryzen 7 5800x rtx 3080 32gb vit."
        ) == 10

    @pytest.mark.parametrize(
        "hint,expected",
        [
            # Believed: a real variant of an ambiguous card.
            ("RTX 3060 12GB", 12),
            ("RTX 3060 8GB", 8),
            # Rejected: not a configuration this card was sold in.
            ("RTX 3060 16GB", 8),
            ("RTX 3070 16GB", 8),
            # Not stated at all: the pessimistic variant, as before.
            ("Speldator RTX 3060", 8),
        ],
    )
    def test_plausibility(self, hint, expected):
        canonical = "rtx 3060" if "3060" in hint else "rtx 3070"
        assert gpu_vram_gb(canonical, text_hint=hint) == expected

    def test_an_unknown_card_still_trusts_the_seller(self):
        """`()` from the table means no opinion, not a veto.

        Discarding a stated figure because a hand-written table has not heard of
        the card would make the table's gaps actively harmful, rather than
        merely incomplete.
        """
        assert gpu_vram_gb("rtx 9090", text_hint="RTX 9090 24GB") == 24

    def test_possible_vram_covers_both_tables(self):
        assert possible_vram("rtx 3060") == (8, 12)   # ambiguous
        assert possible_vram("rtx 3070") == (8,)      # single, from GPU_SPECS
        assert possible_vram("rtx 9090") == ()        # unknown


class TestAssumedFlagAfterRejection:
    def test_a_rejected_statement_still_counts_as_assumed(self):
        """The caveat has to fire even though the ad *did* state a number.

        Otherwise the 3080 above is reported as 10 GB with full confidence, when
        what actually happened is that we overrode the seller and guessed.
        """
        spec = machine_spec(
            [(ComponentTypeEnum.GPU, "rtx 3080")],
            text_hint="ryzen 7 5800x rtx 3080 32gb vit",
        )
        assert spec.vram_gb == 10
        assert spec.vram_is_assumed is True

    def test_a_believed_statement_is_not_assumed(self):
        spec = machine_spec(
            [(ComponentTypeEnum.GPU, "rtx 3060")], text_hint="RTX 3060 12GB"
        )
        assert spec.vram_gb == 12
        assert spec.vram_is_assumed is False


class TestGpuVendor:
    """Which vendor made a card, for the `ai-workstation` CUDA requirement.

    Read off the model prefix rather than stored per row, because the naming
    schemes are vendor-exclusive: no AMD card is called "RTX", no Nvidia card
    is called "RX". That makes a prefix as reliable as a column, and unable to
    fall out of step with `GPU_SPECS` when a row is added.
    """

    @pytest.mark.parametrize(
        "name, vendor",
        [
            ("rtx 3060", "nvidia"),
            ("rtx 4090", "nvidia"),
            ("gtx 1660", "nvidia"),
            ("rx 7800 xt", "amd"),
            ("rx 6600", "amd"),
            ("arc a770", "intel"),
        ],
    )
    def test_recognised_schemes(self, name, vendor):
        assert gpu_vendor(name) == vendor

    def test_an_unknown_name_is_none_not_a_guess(self):
        """None means "unproven", which is what the CUDA check needs to hear.

        Most cards on the market are Nvidia. Defaulting to that would make the
        requirement pass on exactly the listings nobody could verify.
        """
        assert gpu_vendor("quadro p5000") is None
        assert gpu_vendor("") is None

    def test_every_card_in_the_spec_table_has_a_vendor(self):
        """A row nobody can attribute would silently fail the CUDA requirement."""
        unattributed = [name for name in GPU_SPECS if gpu_vendor(name) is None]
        assert unattributed == []
