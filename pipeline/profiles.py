"""Build profiles — "is this a good deal" answered for a specific *purpose*.

### Why the generic version of this question is not worth answering

Up to here the pipeline computes one thing: asking price versus the sum of
observed part medians. That is a real number and it is auditable, but it only
says whether a listing is cheap *relative to what the market charges for the
same parts*. It cannot say whether the machine is any use.

An RTX 3070 selling 800 kr under its median is an excellent deal and a bad
purchase for someone running local language models, because 8 GB of VRAM will
not hold the weights. An RTX 3060 at exactly its median is a worse deal and a
better buy, because it has 12 GB. No amount of price arithmetic reaches that
conclusion, and this module is where it gets made.

### The thesis, stated plainly

**The used market prices GPUs for gaming, because that is what almost everyone
buys them for.** Frame rates set the price. VRAM barely moves it.

That mispricing is the entire opportunity: a card whose value for running local
models comes mostly from memory capacity is priced as though memory capacity
were a rounding error. So the `gaming` profile does *not* get its own clever
metric — for gaming, the market already is the benchmark, and the best buy is
simply the best price-versus-index deal that clears a playable VRAM floor.
`ai-workstation` needs a different metric precisely because the market is not
pricing the thing that profile cares about.

### Three states, not two

A requirement is met, unmet, or **unknown**, and collapsing the third into the
second would be the same class of error as the coverage bug in
`pipeline/index.py`: presenting an absence of evidence as evidence of absence.

If the parts list never mentioned RAM, this module does not report "fails the
32 GB requirement". It reports that it could not tell — which is a prompt to
read the listing, not a reason to discard it. Given how many ads state the GPU
and nothing else, treating silence as failure would hide most of the market.

### Upgradeable shortfalls are not disqualifying

A 7 000 kr prebuilt with 16 GB of RAM does not fail the `ai-workstation`
profile; it costs about 700 kr more than it looks like it does. Soldered VRAM
is a different matter — no amount of money turns an 8 GB card into a 12 GB one
without replacing it.

So requirements carry an upgrade cost where a fix exists, and the profile
reports an *effective* price. That is the comparison a buyer actually faces, and
it is the one that reorders the leaderboard: a cheaper machine needing 700 kr of
RAM can genuinely beat a dearer one that ships correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from db.models import ComponentTypeEnum
from pipeline.hardware import (
    gpu_tdp_w,
    gpu_vendor,
    gpu_vram_gb,
    is_vram_ambiguous,
    psu_watts,
    ram_capacity_gb,
    storage_capacity_gb,
    vram_from_text,
    vram_variants,
)

# Fallback prices for the upgrades a profile knows how to cost, used only when
# the market index has nothing to say. Deliberately on the high side: a
# too-cheap upgrade estimate flatters a listing, which is the direction that
# loses money. Callers pass `upgrade_prices` from the index to replace these
# with observed figures — see `upgrade_price_keys`.
DEFAULT_UPGRADE_SEK: dict[str, int] = {
    "32gb ddr4": 800,
    "1tb ssd": 700,
}


@dataclass(frozen=True, slots=True)
class MachineSpec:
    """What the pipeline believes a listing physically contains.

    Every field is optional, and `None` means "not stated" rather than "absent".
    That distinction is the whole reason this is a dataclass of `None`-ables
    rather than a dict of numbers with zero defaults.
    """

    gpu: str | None = None
    gpu_vendor: str | None = None
    vram_gb: int | None = None
    vram_is_assumed: bool = False
    gpu_tdp_w: int | None = None
    cpu: str | None = None
    ram_gb: int | None = None
    storage_gb: int | None = None
    psu_w: int | None = None


def machine_spec(
    parts: Iterable[tuple[ComponentTypeEnum, str]], *, text_hint: str = ""
) -> MachineSpec:
    """Fold a listing's canonical parts into one spec.

    `text_hint` is the ad's title (and description, where still held), consulted
    only for a stated VRAM capacity. It is the seller's own word on which
    variant of an ambiguous card they have, and it outranks the lookup table.

    Where a machine has several of something the larger wins: two sticks of RAM
    are more RAM, two drives are more storage. Two graphics cards are *not* a
    bigger graphics card, so the GPU slot takes the one with the most memory
    rather than summing — a mining rig full of 8 GB cards must not read as a
    64 GB workstation.
    """
    gpu = gpu_vram = gpu_tdp = None
    cpu = None
    ram_total = storage_total = 0
    psu_max = 0
    saw_ram = saw_storage = saw_psu = False

    # Did the seller state a capacity anywhere? If so, nothing below is a guess.
    stated_vram = vram_from_text(text_hint)

    for component_type, name in parts:
        if component_type is ComponentTypeEnum.GPU:
            vram = gpu_vram_gb(name, text_hint=text_hint)
            if gpu is None or (vram is not None and (gpu_vram is None or vram > gpu_vram)):
                gpu, gpu_vram = name, vram
                gpu_tdp = gpu_tdp_w(name)
        elif component_type is ComponentTypeEnum.CPU:
            cpu = cpu or name
        elif component_type is ComponentTypeEnum.RAM:
            capacity = ram_capacity_gb(name)
            if capacity:
                saw_ram = True
                ram_total += capacity
        elif component_type is ComponentTypeEnum.STORAGE:
            capacity = storage_capacity_gb(name)
            if capacity:
                saw_storage = True
                storage_total += capacity
        elif component_type is ComponentTypeEnum.PSU:
            watts = psu_watts(name)
            if watts:
                saw_psu = True
                psu_max = max(psu_max, watts)

    return MachineSpec(
        gpu=gpu,
        gpu_vendor=gpu_vendor(gpu) if gpu else None,
        vram_gb=gpu_vram,
        # A figure is "assumed" only when the table supplied it *and* the model
        # name genuinely covers more than one memory configuration. Everything
        # else is either the seller's own statement or an unambiguous fact.
        #
        # Compared against the *resolved* figure rather than against
        # `stated_vram is None`, because a stated capacity is no longer always
        # believed: `gpu_vram_gb` rejects one the model was never sold with
        # ("rtx 3080 32gb", where the 32 is system memory) and falls back to the
        # table. Such a listing has a stated figure and is nonetheless assumed,
        # and the earlier test could not express that.
        vram_is_assumed=(
            gpu is not None and is_vram_ambiguous(gpu) and gpu_vram != stated_vram
        ),
        gpu_tdp_w=gpu_tdp,
        cpu=cpu,
        ram_gb=ram_total if saw_ram else None,
        storage_gb=storage_total if saw_storage else None,
        psu_w=psu_max if saw_psu else None,
    )


# `True` met, `False` unmet, `None` cannot tell. See the module docstring.
Check = Callable[[MachineSpec], "bool | None"]


@dataclass(frozen=True, slots=True)
class Requirement:
    """One hard condition, and what it costs to fix if it fails."""

    label: str
    why: str
    check: Check
    # Which component this requirement is *about*.
    #
    # Needed because "unknown" has two different correct meanings depending on
    # the question being asked. When ranking a whole machine, an unreadable RAM
    # figure means "the ad did not say", and rejecting on that would hide most
    # of the market. When *assembling* a basket, picking a stick of RAM whose
    # capacity we cannot read to satisfy a 32 GB floor is not caution — it is an
    # assertion we have no evidence for.
    #
    # Live output made the difference concrete: the AI-workstation basket
    # selected "server ram ddr4 ecc" for its 32 GB slot. The name matched no
    # capacity pattern, so the requirement read `unknown`, so it passed.
    slot: ComponentTypeEnum | None = None
    # Index key for the part that resolves this shortfall, or None when nothing
    # you can buy afterwards fixes it (VRAM is soldered to the board).
    upgrade_part: str | None = None

    def upgrade_cost(self, prices: Mapping[str, int] | None = None) -> int | None:
        if self.upgrade_part is None:
            return None
        if prices and self.upgrade_part in prices:
            return prices[self.upgrade_part]
        return DEFAULT_UPGRADE_SEK.get(self.upgrade_part)


@dataclass(frozen=True, slots=True)
class Metric:
    """A named ratio a profile is ranked by, computed from real numbers only."""

    label: str
    unit: str
    compute: Callable[[MachineSpec, int], "float | None"]


def _at_least(getter: Callable[[MachineSpec], "int | None"], floor: int) -> Check:
    def check(spec: MachineSpec) -> bool | None:
        value = getter(spec)
        return None if value is None else value >= floor

    return check


def _at_most(getter: Callable[[MachineSpec], "int | None"], ceiling: int) -> Check:
    def check(spec: MachineSpec) -> bool | None:
        value = getter(spec)
        return None if value is None else value <= ceiling

    return check


def _is_cuda_capable(spec: MachineSpec) -> bool | None:
    """Whether the card can run the local-LLM toolchain at all.

    Three states, as everywhere else here: no GPU named is `None` ("the ad did
    not say"), a recognised Radeon is `False`, and only a recognised GeForce is
    `True`. An unrecognised name stays unknown rather than being assumed
    Nvidia — the common case is not evidence.
    """
    if spec.gpu is None or spec.gpu_vendor is None:
        return None
    return spec.gpu_vendor == "nvidia"


def _kr_per_gb_vram(spec: MachineSpec, price_sek: int) -> float | None:
    if not spec.vram_gb:
        return None
    return price_sek / spec.vram_gb


def _kr_per_tb_storage(spec: MachineSpec, price_sek: int) -> float | None:
    if not spec.storage_gb:
        return None
    return price_sek / (spec.storage_gb / 1000)


@dataclass(frozen=True, slots=True)
class Profile:
    id: str
    label: str
    rationale: str
    requirements: tuple[Requirement, ...]
    # None means "ranked purely on price versus the market index". That is the
    # right answer for `gaming` and the wrong one for `ai-workstation`; see the
    # module docstring.
    metric: Metric | None = None

    @staticmethod
    def describes_part(slot: ComponentTypeEnum, spec: MachineSpec) -> bool:
        """Whether the part's defining quantity could be read at all.

        Separate from `confirms_part` because a profile with no requirement for
        a slot would otherwise accept anything for it: `all()` over an empty set
        is `True`, so the check passed vacuously.

        That is not hypothetical. The `home-server` profile has no RAM floor, so
        after `confirms_part` was tightened it *still* put "server ram ddr4 ecc"
        in its basket at 499 kr — a stick of unknown capacity, and ECC memory
        that would not have posted in a consumer board. Quoting a price for a
        part whose size nobody knows is not a basket, it is a guess with a
        receipt.

        A part is describable when the quantity that defines it parses out.
        Slots with no such quantity — CPU, motherboard, case — are accepted on
        the strength of having a canonical name at all.
        """
        readable = {
            ComponentTypeEnum.GPU: spec.vram_gb,
            ComponentTypeEnum.RAM: spec.ram_gb,
            ComponentTypeEnum.STORAGE: spec.storage_gb,
            ComponentTypeEnum.PSU: spec.psu_w,
        }
        return slot not in readable or readable[slot] is not None

    def confirms_part(
        self, slot: ComponentTypeEnum, spec: MachineSpec
    ) -> bool:
        """Whether this profile can *positively vouch* for one part.

        Strict where `evaluate` is permissive, and deliberately so — the two
        answer different questions.

        `evaluate` asks "should this listing be shown to someone shopping for
        this profile?", and there the honest answer to an unstated RAM figure is
        "we could not tell, go and look". Here the question is "may this part be
        put in a basket that claims to meet the profile's floors?", and an
        unverifiable part is not a qualifying part. Every requirement *about*
        this slot must come back positively true.
        """
        return all(
            requirement.check(spec) is True
            for requirement in self.requirements
            if requirement.slot is slot
        )

    def upgrade_price_keys(self) -> tuple[str, ...]:
        """Index keys this profile would like current medians for."""
        return tuple(
            r.upgrade_part for r in self.requirements if r.upgrade_part is not None
        )


AI_WORKSTATION = Profile(
    id="ai-workstation",
    label="AI workstation",
    rationale=(
        "Running language models locally is bounded by VRAM before anything "
        "else: a model that does not fit in memory does not run slowly, it does "
        "not run. 12 GB is the floor for a comfortable 7B–14B at reasonable "
        "quantisation. The used market prices cards on frame rates, so memory "
        "capacity is the thing most reliably underpriced — which is what makes "
        "kr per GB the metric worth sorting on."
    ),
    requirements=(
        Requirement(
            label="Nvidia GPU",
            why=(
                "The local-model toolchain is CUDA-bound in practice. ROCm "
                "support on consumer Radeons is partial and worse on Windows, "
                "so a Radeon is not a cheaper way to buy the same capability."
            ),
            check=_is_cuda_capable,
            slot=ComponentTypeEnum.GPU,
            upgrade_part=None,  # a different card, not an upgrade
        ),
        Requirement(
            label="12 GB VRAM",
            why="Below this, useful models will not fit in memory.",
            check=_at_least(lambda s: s.vram_gb, 12),
            slot=ComponentTypeEnum.GPU,
            upgrade_part=None,  # soldered; only a different card fixes it
        ),
        Requirement(
            label="32 GB system RAM",
            why="Datasets, notebooks and a VM or two will thrash 16 GB.",
            check=_at_least(lambda s: s.ram_gb, 32),
            slot=ComponentTypeEnum.RAM,
            upgrade_part="32gb ddr4",
        ),
    ),
    metric=Metric("VRAM cost", "kr/GB VRAM", _kr_per_gb_vram),
)

GAMING = Profile(
    id="gaming",
    label="Gaming",
    rationale=(
        "No bespoke metric here, on purpose. Gaming is what the used market "
        "prices for, so the market is already the benchmark and the best buy is "
        "simply the largest discount to the index that still clears a playable "
        "memory floor. Inventing a performance score would mean shipping a "
        "benchmark table nobody validated, to rank listings the price data "
        "already ranks correctly."
    ),
    requirements=(
        Requirement(
            label="8 GB VRAM",
            why="Below 8 GB, current titles force texture settings down.",
            check=_at_least(lambda s: s.vram_gb, 8),
            slot=ComponentTypeEnum.GPU,
            upgrade_part=None,
        ),
        Requirement(
            label="16 GB system RAM",
            why="The practical minimum for modern titles.",
            check=_at_least(lambda s: s.ram_gb, 16),
            slot=ComponentTypeEnum.RAM,
            upgrade_part="32gb ddr4",
        ),
    ),
    metric=None,
)

HOME_SERVER = Profile(
    id="home-server",
    label="Home server / NAS",
    rationale=(
        "A machine that runs continuously is judged on the bill it generates "
        "and the storage it holds, not on peak throughput. A 300 W gaming card "
        "idling in a cupboard is a liability rather than a feature, so a TDP "
        "ceiling is a genuine requirement and not a preference."
    ),
    requirements=(
        Requirement(
            label="1 TB storage",
            why="Anything less is not worth dedicating a machine to.",
            check=_at_least(lambda s: s.storage_gb, 1000),
            slot=ComponentTypeEnum.STORAGE,
            upgrade_part="1tb ssd",
        ),
        Requirement(
            label="GPU under 100 W",
            why="A high-TDP card runs up the electricity bill 24/7.",
            check=_at_most(lambda s: s.gpu_tdp_w, 100),
            slot=ComponentTypeEnum.GPU,
            upgrade_part=None,
        ),
    ),
    metric=Metric("Storage cost", "kr/TB", _kr_per_tb_storage),
)

PROFILES: tuple[Profile, ...] = (AI_WORKSTATION, GAMING, HOME_SERVER)
PROFILES_BY_ID: dict[str, Profile] = {p.id: p for p in PROFILES}
DEFAULT_PROFILE_ID = AI_WORKSTATION.id


@dataclass(frozen=True, slots=True)
class ProfileFit:
    """How a listing measures up against one profile."""

    profile_id: str
    met: tuple[str, ...]
    unmet: tuple[str, ...]
    unknown: tuple[str, ...]
    # Sum of the fixable shortfalls. Zero when nothing needs fixing, and zero
    # also when the only shortfalls are unfixable — read `blocked_by` for that.
    upgrade_cost_sek: int
    blocked_by: tuple[str, ...]
    listed_price_sek: int
    metric_value: float | None
    metric_label: str | None
    metric_unit: str | None
    vram_caveat: str | None

    @property
    def effective_price_sek(self) -> int:
        """Asking price plus what it costs to make the machine qualify.

        The number a buyer is actually choosing between. A 6 500 kr box needing
        800 kr of RAM competes at 7 300 kr, and should be ranked there.
        """
        return self.listed_price_sek + self.upgrade_cost_sek

    @property
    def verdict(self) -> str:
        if self.blocked_by:
            return "unfit"
        if self.upgrade_cost_sek:
            return "fits-after-upgrade"
        if self.unknown:
            return "unverified"
        return "fits"

    @property
    def is_viable(self) -> bool:
        """Whether to show it at all. "Unverified" counts — see the docstring.

        An ad that simply did not mention its RAM is the common case, not a
        suspicious one, and hiding those would hide most of the market.
        """
        return not self.blocked_by

    def describe(self) -> str:
        parts = [f"{self.profile_id}: {self.verdict}"]
        if self.blocked_by:
            parts.append("blocked by " + ", ".join(self.blocked_by))
        if self.upgrade_cost_sek:
            parts.append(f"+{self.upgrade_cost_sek} kr to qualify")
        if self.unknown:
            parts.append("unverified: " + ", ".join(self.unknown))
        if self.metric_value is not None:
            parts.append(f"{self.metric_value:.0f} {self.metric_unit}")
        return " | ".join(parts)


def evaluate(
    profile: Profile,
    spec: MachineSpec,
    listed_price_sek: int,
    *,
    upgrade_prices: Mapping[str, int] | None = None,
) -> ProfileFit:
    """Score one listing against one profile.

    `upgrade_prices` maps an index key ("32gb ddr4") to a current median. Passing
    it makes the upgrade estimate observed rather than assumed, which is the
    same discipline applied to valuation everywhere else in the pipeline —
    `DEFAULT_UPGRADE_SEK` is only the fallback for a part the index has not seen.
    """
    met: list[str] = []
    unmet: list[str] = []
    unknown: list[str] = []
    blocked_by: list[str] = []
    upgrade_total = 0

    for requirement in profile.requirements:
        result = requirement.check(spec)
        if result is True:
            met.append(requirement.label)
            continue
        if result is None:
            unknown.append(requirement.label)
            continue

        unmet.append(requirement.label)
        cost = requirement.upgrade_cost(upgrade_prices)
        if cost is None:
            blocked_by.append(requirement.label)
        else:
            upgrade_total += cost

    metric_value = None
    if profile.metric is not None:
        metric_value = profile.metric.compute(spec, listed_price_sek + upgrade_total)

    caveat = None
    if spec.vram_is_assumed and spec.gpu:
        options = " or ".join(f"{v} GB" for v in vram_variants(spec.gpu))
        caveat = (
            f"{spec.gpu.upper()} ships as {options}; the listing does not say "
            f"which, so the smaller was assumed."
        )

    return ProfileFit(
        profile_id=profile.id,
        met=tuple(met),
        unmet=tuple(unmet),
        unknown=tuple(unknown),
        upgrade_cost_sek=upgrade_total,
        blocked_by=tuple(blocked_by),
        listed_price_sek=listed_price_sek,
        metric_value=metric_value,
        metric_label=profile.metric.label if profile.metric else None,
        metric_unit=profile.metric.unit if profile.metric else None,
        vram_caveat=caveat,
    )


def rank_key(fit: ProfileFit, delta_sek: int | None, confidence: float) -> tuple:
    """Sort key for a profile-filtered leaderboard. Lower sorts first.

    Deliberately a tuple of independent terms rather than one blended score.
    The single opaque 0–100 `deal_score` was deleted in Phase 2 because nobody
    could say why one listing outranked another, and a weighted sum here would
    quietly reintroduce it.

    Multiplying by fit would also be *arithmetically* wrong, not merely opaque.
    `delta_sek` is negative for a good deal, so scaling it by a fitness factor
    below 1 makes good deals look worse — correct — but simultaneously makes
    overpriced listings look better. One multiplication, two opposite meanings.

    So: qualifying machines first, then anything priced above the index, then
    the profile's own metric, then confidence-weighted market delta.

    ### Why "above the index" outranks the metric

    It did not, and the live board showed the cost. `metric` is a continuous
    float and so essentially never ties, which meant the delta term — the whole
    output of the index — was a tiebreak that never fired. The `ai-workstation`
    board ranked a 19 000 kr RTX 3090 machine fourth on kr/GB VRAM alone while
    it sat 3 976 kr *above* the market, and a Radeon at +2 001 kr second.

    Sorting on kr/GB VRAM is still right: it is the mispricing this profile
    exists to exploit. But "cheapest per GB" and "overpaying for those GB" are
    not in tension — a machine can be both, and when it is, the buyer is paying
    over the odds for capacity available cheaper elsewhere. One coarse boolean
    ahead of the metric says so without blending two scales into one number.

    Deliberately a *boolean*, not the delta itself. Ordering by delta directly
    would rank a 500 kr saving on a 3 000 kr box above a 400 kr saving on a
    20 000 kr one and call that a better fit for the profile, which is the
    metric's job to judge, not the index's.

    `delta_sek is None` counts as not-overpriced. Unknown is not a fault — the
    same three-state rule the requirements follow.
    """
    tier = 0 if fit.verdict == "fits" else 1 if fit.verdict != "unfit" else 2
    above_index = 1 if (delta_sek is not None and delta_sek > 0) else 0
    metric = fit.metric_value if fit.metric_value is not None else float("inf")
    value = (delta_sek if delta_sek is not None else 0) * confidence
    return (tier, above_index, metric, value)


def resolve(profile_id: str | None) -> Profile:
    """Look up a profile, falling back to the default rather than raising."""
    return PROFILES_BY_ID.get(profile_id or "", PROFILES_BY_ID[DEFAULT_PROFILE_ID])


def describe_all() -> Sequence[dict]:
    """Profile metadata for the UI's selector."""
    return [
        {
            "id": p.id,
            "label": p.label,
            "rationale": p.rationale,
            "requirements": [
                {"label": r.label, "why": r.why, "upgradeable": r.upgrade_part is not None}
                for r in p.requirements
            ],
            "metric": p.metric.label if p.metric else None,
            "metricUnit": p.metric.unit if p.metric else None,
        }
        for p in PROFILES
    ]
