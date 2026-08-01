"""Tests for build profiles.

The property under test throughout is the three-state one: met, unmet, and
*unknown* must stay distinct. Collapsing unknown into unmet is the mistake that
would quietly hide most of the market, and it is the kind of mistake that makes
every test still pass — the code would just return fewer results.
"""

from __future__ import annotations

import pytest

from db.models import ComponentTypeEnum as CT
from pipeline.profiles import (
    AI_WORKSTATION,
    DEFAULT_UPGRADE_SEK,
    GAMING,
    HOME_SERVER,
    PROFILES,
    evaluate,
    machine_spec,
    rank_key,
    resolve,
)


class TestMachineSpec:
    def test_reads_a_full_build(self):
        spec = machine_spec(
            [
                (CT.GPU, "rtx 3080"),
                (CT.CPU, "ryzen 5 5600x"),
                (CT.RAM, "16gb ddr4"),
                (CT.STORAGE, "1tb nvme"),
                (CT.PSU, "750w psu"),
            ]
        )
        assert spec.gpu == "rtx 3080"
        assert spec.vram_gb == 10
        assert spec.ram_gb == 16
        assert spec.storage_gb == 1000
        assert spec.psu_w == 750

    def test_ram_and_storage_sum_but_vram_does_not(self):
        """Two sticks are more memory. Two cards are not a bigger card.

        A mining rig listing eight 8 GB cards must not read as a 64 GB
        workstation — which is exactly what summing would produce, and it would
        top the `ai-workstation` leaderboard while doing it.
        """
        spec = machine_spec(
            [
                (CT.RAM, "16gb ddr4"),
                (CT.RAM, "16gb ddr4"),
                (CT.STORAGE, "1tb ssd"),
                (CT.STORAGE, "2tb hdd"),
                (CT.GPU, "rtx 3070"),
                (CT.GPU, "rtx 3070"),
            ]
        )
        assert spec.ram_gb == 32
        assert spec.storage_gb == 3000
        assert spec.vram_gb == 8

    def test_multiple_gpus_take_the_largest_not_the_first(self):
        spec = machine_spec([(CT.GPU, "rtx 3070"), (CT.GPU, "rtx 3090")])
        assert spec.vram_gb == 24

    def test_absent_parts_are_none_not_zero(self):
        """The distinction the whole module rests on."""
        spec = machine_spec([(CT.GPU, "rtx 3080")])
        assert spec.ram_gb is None
        assert spec.storage_gb is None

    def test_stated_vram_overrides_the_pessimistic_default(self):
        assert machine_spec([(CT.GPU, "rtx 3060")]).vram_gb == 8
        spec = machine_spec([(CT.GPU, "rtx 3060")], text_hint="RTX 3060 12GB OC")
        assert spec.vram_gb == 12
        assert not spec.vram_is_assumed

    def test_ambiguous_card_without_a_stated_capacity_is_flagged(self):
        spec = machine_spec([(CT.GPU, "rtx 3060")])
        assert spec.vram_is_assumed


class TestAiWorkstation:
    def test_the_case_the_profile_exists_for(self):
        """A 3070 is the better *deal*; a 3060 12GB is the better *buy*.

        This is the thesis of the whole module. If this test ever passes with
        the 3070 viable, the VRAM floor has stopped working.
        """
        good = evaluate(
            AI_WORKSTATION,
            machine_spec(
                [(CT.GPU, "rtx 3060"), (CT.RAM, "32gb ddr4")],
                text_hint="RTX 3060 12GB",
            ),
            3000,
        )
        bad = evaluate(
            AI_WORKSTATION,
            machine_spec([(CT.GPU, "rtx 3070"), (CT.RAM, "32gb ddr4")]),
            2800,
        )
        assert good.verdict == "fits"
        assert bad.verdict == "unfit"
        assert "12 GB VRAM" in bad.blocked_by

    def test_soldered_shortfall_blocks_but_ram_shortfall_only_costs(self):
        """Upgradeable and unupgradeable failures are not the same failure."""
        fit = evaluate(
            AI_WORKSTATION,
            machine_spec([(CT.GPU, "rtx 3090"), (CT.RAM, "16gb ddr4")]),
            9000,
        )
        assert fit.verdict == "fits-after-upgrade"
        assert fit.is_viable
        assert fit.upgrade_cost_sek == DEFAULT_UPGRADE_SEK["32gb ddr4"]
        assert fit.effective_price_sek == 9000 + DEFAULT_UPGRADE_SEK["32gb ddr4"]

    def test_unstated_ram_is_unverified_not_rejected(self):
        """Most ads state the GPU and nothing else. Those must survive."""
        fit = evaluate(AI_WORKSTATION, machine_spec([(CT.GPU, "rtx 3090")]), 8000)
        assert fit.verdict == "unverified"
        assert fit.is_viable
        assert "32 GB system RAM" in fit.unknown
        assert fit.unmet == ()
        assert fit.upgrade_cost_sek == 0

    def test_unknown_gpu_does_not_silently_pass_the_floor(self):
        fit = evaluate(AI_WORKSTATION, machine_spec([(CT.GPU, "quadro p5000")]), 4000)
        assert "12 GB VRAM" in fit.unknown
        assert fit.verdict == "unverified"

    def test_observed_upgrade_price_beats_the_fallback(self):
        """Same discipline as valuation: prefer a counted number to an assumed one."""
        spec = machine_spec([(CT.GPU, "rtx 3090"), (CT.RAM, "16gb ddr4")])
        fit = evaluate(AI_WORKSTATION, spec, 9000, upgrade_prices={"32gb ddr4": 450})
        assert fit.upgrade_cost_sek == 450

    def test_metric_is_kr_per_gb_of_vram_including_the_upgrade(self):
        spec = machine_spec([(CT.GPU, "rtx 3090"), (CT.RAM, "32gb ddr4")])
        fit = evaluate(AI_WORKSTATION, spec, 12000)
        assert fit.metric_value == pytest.approx(500.0)  # 12000 / 24 GB

    def test_vram_caveat_is_surfaced_not_hidden(self):
        fit = evaluate(AI_WORKSTATION, machine_spec([(CT.GPU, "rtx 3060")]), 2500)
        assert fit.vram_caveat is not None
        assert "8 GB or 12 GB" in fit.vram_caveat


class TestCudaRequirement:
    """A Radeon is not a cheaper way to buy the same capability.

    Live output, `ai-workstation` board: an **RX 7800 XT 16 GB** machine at
    10 000 kr ranked second, on 625 kr/GB VRAM — the best figure on the board
    bar one listing. Three more Radeons sat in the same top ten.

    The metric was doing exactly what it was written to do. VRAM is what bounds
    a local model, the market underprices VRAM, and 16 GB of it for 10 000 kr is
    genuinely cheap. It is also unusable for the profile's stated purpose: the
    toolchain is CUDA-bound, and ROCm on a consumer Radeon is partial at best.

    So the profile was recommending, at the top of its own leaderboard, hardware
    that cannot do the one thing the profile is named after.
    """

    def test_a_radeon_is_unfit_however_cheap_its_vram(self):
        spec = machine_spec(
            [(CT.GPU, "rx 7800 xt"), (CT.RAM, "32gb ddr4")],
            text_hint="Speldator | 32GB DDR4 | RX 7800 XT 16 GB",
        )
        fit = evaluate(AI_WORKSTATION, spec, 10000)
        assert fit.verdict == "unfit"
        assert "Nvidia GPU" in fit.blocked_by
        # The VRAM floor is cleared — it is the vendor and nothing else.
        assert "12 GB VRAM" in fit.met

    def test_the_equivalent_geforce_still_qualifies(self):
        """The control: same memory, same money, different vendor."""
        spec = machine_spec(
            [(CT.GPU, "rtx 4070 ti super"), (CT.RAM, "32gb ddr4")],
            text_hint="RTX 4070 Ti Super 16GB",
        )
        assert evaluate(AI_WORKSTATION, spec, 10000).verdict == "fits"

    def test_an_unrecognised_card_is_unknown_not_assumed_nvidia(self):
        """Absence of evidence, again. Most cards are Nvidia; that is not proof."""
        fit = evaluate(AI_WORKSTATION, machine_spec([(CT.GPU, "quadro p5000")]), 4000)
        assert "Nvidia GPU" not in fit.blocked_by
        assert "Nvidia GPU" in fit.unknown

    def test_gaming_is_indifferent_to_vendor(self):
        """The requirement belongs to one profile, not to the pipeline."""
        spec = machine_spec(
            [(CT.GPU, "rx 7800 xt"), (CT.RAM, "16gb ddr4")],
            text_hint="RX 7800 XT 16 GB",
        )
        assert evaluate(GAMING, spec, 5000).verdict == "fits"


class TestGaming:
    def test_lower_vram_floor_admits_what_the_workstation_rejects(self):
        """The same listing, two profiles, two answers — the point of profiles."""
        spec = machine_spec([(CT.GPU, "rtx 3070"), (CT.RAM, "16gb ddr4")])
        assert evaluate(GAMING, spec, 2800).verdict == "fits"
        assert evaluate(AI_WORKSTATION, spec, 2800).verdict == "unfit"

    def test_has_no_metric_on_purpose(self):
        """Gaming is ranked on market delta alone. See the profile's rationale."""
        assert GAMING.metric is None
        fit = evaluate(GAMING, machine_spec([(CT.GPU, "rtx 3080")]), 4000)
        assert fit.metric_value is None


class TestHomeServer:
    def test_a_gaming_card_is_a_liability_here(self):
        spec = machine_spec([(CT.GPU, "rtx 3080"), (CT.STORAGE, "2tb hdd")])
        fit = evaluate(HOME_SERVER, spec, 5000)
        assert fit.verdict == "unfit"
        assert "GPU under 100 W" in fit.blocked_by

    def test_low_power_card_with_storage_qualifies(self):
        spec = machine_spec([(CT.GPU, "gtx 1050"), (CT.STORAGE, "2tb hdd")])
        assert evaluate(HOME_SERVER, spec, 3000).verdict == "fits"

    def test_metric_is_kr_per_terabyte(self):
        spec = machine_spec([(CT.GPU, "gtx 1050"), (CT.STORAGE, "2tb hdd")])
        fit = evaluate(HOME_SERVER, spec, 3000)
        assert fit.metric_value == pytest.approx(1500.0)


class TestRankKey:
    def test_qualifying_machines_outrank_upgradeable_ones(self):
        fits = evaluate(
            AI_WORKSTATION,
            machine_spec([(CT.GPU, "rtx 3090"), (CT.RAM, "32gb ddr4")]),
            9000,
        )
        needs_ram = evaluate(
            AI_WORKSTATION,
            machine_spec([(CT.GPU, "rtx 3090"), (CT.RAM, "16gb ddr4")]),
            8000,
        )
        assert rank_key(fits, -500, 1.0) < rank_key(needs_ram, -2000, 1.0)

    def test_a_bad_fit_does_not_flatter_an_overpriced_listing(self):
        """The arithmetic trap that a multiplicative score would fall into.

        `delta_sek` is negative when a listing is cheap. Multiplying it by a
        fitness factor below 1 correctly penalises good deals — and
        simultaneously *rewards* overpriced ones, because scaling a positive
        delta toward zero moves it up the leaderboard. One operator, two
        opposite meanings, which is why ranking is a tuple.
        """
        overpriced = evaluate(
            AI_WORKSTATION,
            machine_spec([(CT.GPU, "rtx 3070"), (CT.RAM, "32gb ddr4")]),
            8000,
        )
        good = evaluate(
            AI_WORKSTATION,
            machine_spec(
                [(CT.GPU, "rtx 3060"), (CT.RAM, "32gb ddr4")], text_hint="RTX 3060 12GB"
            ),
            3000,
        )
        # The unfit listing sorts last regardless of how its delta scales.
        assert rank_key(good, -200, 1.0) < rank_key(overpriced, +3000, 0.3)

    def test_a_listing_above_the_index_loses_to_one_below_it(self):
        """Regression from the live `ai-workstation` board.

        `metric` is a continuous float, so it essentially never ties — which
        meant the delta term behind it was a tiebreak that never fired, and the
        index had no influence on the ordering at all. A 19 000 kr RTX 3090
        machine ranked fourth on kr/GB VRAM while sitting 3 976 kr *above* the
        market it was being measured against.

        Here the overpriced machine has the *better* metric and must still lose.
        """
        overpriced = evaluate(
            AI_WORKSTATION,
            machine_spec([(CT.GPU, "rtx 3090"), (CT.RAM, "32gb ddr4")]),
            13000,
        )
        keener = evaluate(
            AI_WORKSTATION,
            machine_spec(
                [(CT.GPU, "rtx 3060"), (CT.RAM, "32gb ddr4")],
                text_hint="RTX 3060 12GB",
            ),
            9000,
        )
        assert overpriced.metric_value < keener.metric_value  # better kr/GB VRAM
        assert rank_key(keener, -1870, 1.0) < rank_key(overpriced, +3976, 1.0)

    def test_an_unknown_delta_is_not_treated_as_overpriced(self):
        """Three states here too: no index coverage is not evidence of a markup."""
        fit = evaluate(
            AI_WORKSTATION,
            machine_spec([(CT.GPU, "rtx 3090"), (CT.RAM, "32gb ddr4")]),
            9000,
        )
        assert rank_key(fit, None, 1.0) < rank_key(fit, +1, 1.0)

    def test_the_metric_still_decides_between_two_honest_prices(self):
        """The delta term is one coarse boolean, not a replacement for ranking.

        Both of these are below the index, so the profile's own metric — the
        mispricing it exists to exploit — is what separates them.
        """
        more_vram = evaluate(
            AI_WORKSTATION,
            machine_spec([(CT.GPU, "rtx 3090"), (CT.RAM, "32gb ddr4")]),
            12000,
        )
        less_vram = evaluate(
            AI_WORKSTATION,
            machine_spec(
                [(CT.GPU, "rtx 3060"), (CT.RAM, "32gb ddr4")],
                text_hint="RTX 3060 12GB",
            ),
            9000,
        )
        assert rank_key(more_vram, -100, 1.0) < rank_key(less_vram, -2000, 1.0)


class TestProfileRegistry:
    def test_resolve_falls_back_rather_than_raising(self):
        """An unknown `?profile=` in a URL must not 500 a page."""
        assert resolve("nonsense").id == AI_WORKSTATION.id
        assert resolve(None).id == AI_WORKSTATION.id
        assert resolve("gaming").id == "gaming"

    def test_every_profile_has_a_rationale_and_unique_id(self):
        ids = [p.id for p in PROFILES]
        assert len(ids) == len(set(ids))
        for profile in PROFILES:
            assert profile.rationale.strip()
            assert profile.requirements

    def test_declared_upgrade_parts_all_have_a_fallback_price(self):
        """A requirement claiming to be upgradeable with no price is a silent zero.

        It would report "fits-after-upgrade, +0 kr" — indistinguishable from a
        machine that needs nothing.
        """
        for profile in PROFILES:
            for key in profile.upgrade_price_keys():
                assert key in DEFAULT_UPGRADE_SEK, f"{profile.id}: {key}"


class TestCrossLanguageContract:
    """The profile ids exist in two languages, and nothing else checks that.

    `web/src/types.ts` declares `ProfileId` as a string-literal union, and the
    published snapshot keys its per-deal fit map by exactly those strings. If
    Python renames a profile, TypeScript still compiles — it just receives an
    object whose keys match none of its declared ids, and every card silently
    renders with no fit panel at all.

    A regex over the type declaration is crude, but it is the only thing
    standing between a rename here and a dashboard that quietly stops working.
    """

    def test_typescript_profile_ids_match_the_python_ones(self):
        import re
        from pathlib import Path

        types_ts = Path(__file__).resolve().parents[1] / "web" / "src" / "types.ts"
        source = types_ts.read_text(encoding="utf-8")

        match = re.search(r"export type ProfileId\s*=\s*([^;]+);", source)
        assert match, "ProfileId union not found in web/src/types.ts"

        declared = set(re.findall(r"'([^']+)'", match.group(1)))
        assert declared == {p.id for p in PROFILES}
