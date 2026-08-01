# Buying guide

**What to actually buy, and why.** This is the deliverable the rest of the
project exists to produce: a first desktop, 5 000–10 000 kr, that has to run
local language models, play games, survive a data-science course, and host a
couple of VMs for security work.

Every number below comes from the snapshot in
[`web/public/deals.json`](../web/public/deals.json), generated **2026-07-30**
from **1 029 live Blocket listings**. Prices move; the reasoning does not.
Re-run `python -m pipeline.run --stage publish` and the tables change — the
argument for *how to choose* is the part worth keeping.

---

## The short answer

> **Buy a used prebuilt with an RTX 3060 12 GB and at least 32 GB of RAM, from a
> company rather than a private seller, for 6 000–7 000 kr. Keep 2 500 kr back
> for a monitor, keyboard and mouse.**

The current best match in the snapshot:

| | |
|---|---|
| **Listing** | Gaming PC med RTX 3060 12GB — HP Z4 G4 Xeon / 64 GB ECC / 512 GB SSD / Win 11 Pro |
| **Price** | **6 495 kr** · Bromma · *Bra skick* |
| **Parts value** | 8 365 kr → **1 870 kr under** the sum of its parts |
| **VRAM cost** | **541 kr/GB** — the lowest of 28 published full builds |
| **Verdict** | `fits` — 12 GB VRAM ✓, 32 GB RAM ✓ (it has 64) |
| **Seller** | ITA Store — a shop, not a private seller. This matters; see [below](#buy-from-a-company). |

It is not a bargain because it is cheap. Cheaper machines are everywhere. It is
the pick because it is the only one under 7 000 kr that clears *both* hard
requirements at once, and because buying it from a company converts a used
purchase into one with three years of legal recourse.

---

## Why VRAM is the only axis that matters

A language model that does not fit in video memory does not run slowly. It does
not run. Everything else — core count, clock speed, memory bandwidth — decides
*how fast*, and only after the model has somewhere to live.

Roughly, at 4-bit quantisation:

| VRAM | What fits |
|---|---|
| 8 GB | 7–8B models, short context. Tight. |
| **12 GB** | **13–14B comfortably, 7B with room for a long context.** |
| 16 GB | 14B with headroom, 20B+ at a squeeze. |
| 24 GB | 30B class. |

12 GB is the floor where the machine stops feeling like a demo. That is the
number the `ai-workstation` profile enforces, and it is deliberately a **block**
rather than a penalty: RAM can be added later, soldered VRAM cannot. No amount
of money turns an 8 GB card into a 12 GB one.

### The market disagrees, and that is the whole opportunity

Used GPUs are priced on **frame rates**, because that is what almost everyone
buys them for. Memory capacity barely moves the price. Here is what that looks
like in observed Swedish sale prices from this snapshot:

| Card | VRAM | Median | n | **kr per GB** |
|---|---:|---:|---:|---:|
| **RTX 3060** | **12 GB** | 2 590 kr | 9 | **216 kr** ⭐ |
| RTX 3060 Ti | 8 GB | 2 500 kr | 7 | 313 kr |
| RTX 3070 | 8 GB | 2 797 kr | 10 | 350 kr |
| RTX 4060 | 8 GB | 2 850 kr | 6 | 356 kr |
| RTX 3080 | 10 GB | 4 290 kr | 11 | 429 kr |
| RTX 3080 Ti | 12 GB | 5 350 kr | 6 | 446 kr |
| RTX 3090 | 24 GB | 11 249 kr | 4 | 469 kr |

Read the top two rows again. **An RTX 3060 Ti costs slightly less than an RTX
3060 and has a third less memory.** The Ti is the better gaming card and the
market prices it accordingly — it is faster, so it is worth more, so the plain
3060 gets discounted for being the weaker product. For running models, the
"weaker" card is 45 % cheaper per gigabyte than anything else on the board.

That inversion is not a bug in the market. Gamers are right to prefer the Ti.
It is only mispriced *relative to what you want*, and that is the only kind of
edge available to someone with 10 000 kr and no inside information.

The same holds for whole machines. In the 5 000–9 000 kr band this snapshot
contains noticeably more 3060 **Ti** builds than plain 3060 builds, at
overlapping prices. Most buyers filter for the faster card. You should filter
against it.

---

## Why not the alternatives

**RTX 3070 / 3060 Ti (8 GB) at ~2 800 kr** — the sweet spot for gaming and a
dead end for everything else. 8 GB caps you at 7–8B models today and will age
badly; model sizes are not shrinking. Skip.

**RTX 3080 (10 GB) at ~4 290 kr, whole machines at 9 500–10 000 kr** — the
tempting one, and the closest call in this guide. It is roughly twice the gaming
card, and 10 GB genuinely does run 7–8B models well. Two things rule it out:
14B-class models do not fit, and a 9 500 kr machine consumes the entire budget
with nothing left for a screen. **If you decide you care about gaming more than
you care about 14B models, this is the honest alternative** — the snapshot holds
eight such machines at 9 500–10 000 kr. Just make that trade
deliberately rather than by drifting toward the faster card.

**The 5 800 kr machine Basket Mode names as "best prebuilt"** — a private
listing in Alnö whose entire description is *"Tjenare, kollar vad folk ger"*
("hi, seeing what people offer"). It has the right card and 16 GB of RAM, so the
profile scores it 5 800 kr including the memory upgrade, and on paper it beats
the recommendation by 695 kr. It is a private sale with no fixed price, no
stated condition and no recourse. 695 kr is not what that difference is worth.

**RTX 3090 (24 GB) at ~11 249 kr** — the correct card for this workload and
outside the budget on its own, before a machine to put it in. Worth knowing it
exists as the eventual upgrade.

**A laptop** — every laptop in the snapshot is either over budget or a worse
machine at the same price, and mobile GPUs are meaningfully slower than the
desktop cards they share a name with. (The tool currently prices them against
desktop medians, which makes two laptops look like enormous bargains. They are
not. This is a known open bug, noted in the repo.)

---

## Prebuilt or build it yourself?

This is the question Basket Mode exists to answer, and the answer is
counter-intuitive enough that it was worth building the feature to find out.

| | Total | Sellers |
|---|---:|---:|
| Cheapest parts basket meeting the profile | **6 299 kr** | 4 |
| Best prebuilt meeting the profile | **5 800 kr** | 1 |

**The basket loses — and it loses by more than the table shows**, because the
6 299 kr excludes the motherboard, PSU and case. Those are excluded because they
are poorly represented on Blocket and the index cannot price them honestly,
which is the right call for the tool and a trap for the reader. Add them and the
build lands around 8 000 kr.

So the real comparison is roughly **8 000 kr, four strangers, four meetups, no
warranty, and an assembly you have never done** against **6 495 kr, one shop,
one transaction, and a working Windows install.**

Buy the prebuilt. Build the second machine, when you know what you want and a
mistake costs you a weekend rather than the whole budget.

One honest caveat about that basket: its cheapest CPU is an i7-7700K from 2017,
bundled with a motherboard and RAM. Four cores in 2026 is thin for VMs. The
basket is a price floor, not a recommendation.

---

## Buy from a company

The single highest-value line in this guide, and it has nothing to do with
hardware.

| | Private seller | Company (*näringsidkare*) |
|---|---|---|
| Governing law | Köplagen | **Konsumentköplagen (2022:260)** |
| Right to complain | 2 years, you must prove the fault existed at sale | **3 years** |
| Burden of proof | On you, immediately | **On the seller for the first 2 years** |
| *"Säljes i befintligt skick"* | Effectively binding | **Cannot override the law** |

A private seller can hand you a machine that dies in three weeks and you have
essentially no recourse. A shop cannot — a fault appearing within two years is
*presumed* to have been there when you bought it, and the seller has to show
otherwise. That protection is worth several hundred kronor of price premium on
a used computer, and the recommended listing has it for free.

Two things it does **not** give you: there is no 14-day *ångerrätt* when you
collect in person (that applies to distance selling), and "used" does lower the
standard of what counts as a fault. It is still an enormous asymmetry.

*This is a plain-language summary, not legal advice.*

---

## Before you pay

The pipeline reads text. Text lies, or is simply incomplete. Four things to
settle in person:

1. **Confirm the card is the 12 GB RTX 3060, not the 8 GB one.** Both exist and
   listings rarely say which. Open a terminal on the machine and run `dxdiag`,
   or check Task Manager → Performance → GPU. If it says 8 GB, walk. This is the
   one check that invalidates the entire purchase, and the tool flags it as a
   caveat on every ambiguous 3060 for exactly that reason.
2. **Ask what the Xeon actually is.** The listing says *"Xeon som motsvarar en
   i7-8700"*, which is marketing, not a part number. The extractor recorded
   `i7-8700` from that sentence — it is a comparison, not the CPU. An HP Z4 G4
   takes Xeon W-series chips; ask for the exact model and look it up.
3. **The 64 GB is ECC registered memory.** Excellent for VMs, and it will *not*
   transfer to an ordinary consumer desktop later. Fine — but it means the RAM
   is part of this machine's value, not part of your next one's.
4. **Watch it boot, from cold, in front of you.** Load a game or run a stress
   test for ten minutes and put a hand on the exhaust. Ask about the PSU age. A
   workstation chassis is a good sign here: HP built the Z4 to run all day.

Pay by Swish or card, in person, after it boots. Never in advance, never by
bank transfer to a stranger, and be actively suspicious of any seller who wants
to ship a computer they will not let you see running.

---

## The budget, honestly

You do not own a monitor, keyboard or mouse.

| | |
|---:|---|
| 6 495 kr | The machine |
| ~1 500 kr | 1440p monitor, used |
| ~500 kr | Keyboard and mouse |
| **~8 500 kr** | **Total, inside budget** |

That leaves roughly 1 500 kr of slack, which is the right amount to hold back
for the thing that always comes up — a cable, a Windows licence, a drive that
turns out to be smaller than advertised.

Do not spend the slack on a better GPU. Spend it on nothing, and keep it.

---

## Where this goes next

The 12 GB card runs 7–14B models well and will feel limiting in about two years.
The upgrade path from here, in order of value per krona:

1. **Nothing, for a year.** Learn what actually constrains you before buying to
   fix an imagined problem.
2. **A second SSD** when 512 GB fills up, which it will — model weights are
   enormous. Around 1 500 kr for 1 TB NVMe, and the cheapest real improvement
   available.
3. **A 24 GB card** — RTX 3090, ~11 000 kr today and falling. This is the real
   upgrade, and it is why PSU headroom and a workstation chassis matter now.
4. **The NAS**, as a separate low-power box. Do not compromise the desktop to
   make it dual-purpose; the `home-server` profile scores an entirely different
   kind of machine, and it correctly rejects every build in this guide.

---

## Regenerating this

```bash
python -m pipeline.run                       # scrape, evaluate, index, publish
python -m pipeline.run --stage basket --budget 10000
```

The leaderboard sorted by `ai-workstation` is this guide's first section,
recomputed against whatever is for sale tonight. The rest — the kr-per-GB
argument, company-versus-private, keeping money back for a monitor — is
independent of the snapshot, and is most of what actually decides whether the
purchase was a good one.
