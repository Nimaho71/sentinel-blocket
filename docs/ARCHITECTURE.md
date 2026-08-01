# Architecture

## The shape

```
LOCAL MACHINE (nightly)
  sources/     Blocket JSON API  ──rate limited, circuit broken──┐
                                                                 ▼
  pipeline/    ingest → persist → extract → evaluate → index → publish
                                     │         │
                                     │         └── ai/  Gemini: risk + classification
                                     └───────────── regex: specs (no network)
                                                                 │
  db/          sentinel_blocket.db  (SQLite, full fidelity, never leaves)
                                                                 │
                                     redact + cap to top-N ──────┘
                                                                 ▼
  web/public/deals.json  ──git push──▶  Vercel  ──▶  static React site
```

The public site is a **dumb static reader**. It holds no secrets, calls no API,
and talks to no database.

---

## Why SQLite

The honest answer is that the scrape has to run on this machine anyway, and once
that is true, most of the usual database reasoning stops applying.

Anti-bot systems score datacenter IP ranges heavily — AWS, GCP, Azure, Hetzner,
Fly and Render all publish theirs. A consumer VPN exit is usually *worse*: shared
by thousands of people and well known. A Swedish residential connection is the
best available option here, and it is free. So the pipeline runs locally.

Given a single-writer, single-machine workload, SQLite is not a compromise:

| | |
|---|---|
| **Concurrency needed** | One writer, sequentially. The rate limiter is the bottleneck, not the database — at 2 s/request nothing is contending for anything. |
| **Data volume** | Thousands of rows. Postgres starts paying off around the point where the working set stops fitting in RAM; this is four orders of magnitude away. |
| **Operational cost** | Zero. No server, no connection string, no backup story, no credentials to leak. A file. |
| **Portability** | The SQLAlchemy code is dialect-agnostic. Moving to Postgres or Turso later is a URL change, not a rewrite. |

It also removes an entire category of security problem. There is no hosted
database, so there is no hosted database to secure, and no cloud credentials in
Vercel's environment to leak.

**When this stops being right:** multiple writers, or a hosted API that serves
live queries. Neither is in scope. If it ever is, the migration is Fly.io +
Turso (libSQL, SQLite-compatible), and the SQLAlchemy models move unchanged.

---

## The two guarantees the structure exists to enforce

**1. Personal data cannot reach the published file, because it never enters the
system.** Blocket's detail payload contains GPS coordinates and a seller id.
`sources/base.py` has no field for either, so they are dropped at the boundary
rather than filtered at publish. `db/models.py` has no column for them either.
A redaction bug downstream cannot leak what was never ingested — two independent
layers would both have to be wrong.

**2. Secrets cannot reach the browser, because the browser never needs one.**
The frontend reads a JSON file. There is no serverless function, so there is no
environment to hold a key. `vite.config.ts` carries a comment explaining why the
`define` block was removed and must not come back.

---

## Where the value calculation lives, and why it is split

The weakest part of the original design was asking Gemini what a used RTX 3080
sells for in Sweden. That produces a fluent, undated, unfalsifiable number
recalled from training data.

The responsibilities are now split by what each side is actually good at:

| | Owns | Because |
|---|---|---|
| **Arithmetic** | Price. Median of observed listings, with sample count. | Reproducible, auditable, and improves as data accumulates. |
| **Gemini** | Risk flags, classification, confidence. | *"Ej testad"*, *"inget kvitto"*, *"säljes i befintligt skick"* are judgement calls that regexes read badly and language models read well. |

Ranking multiplies the price delta by the AI's confidence, so a mathematically
good deal the model distrusts sinks — without the model ever inventing a price.

A third tier sits underneath: `pipeline/extract.py` resolves specs by regex
where the text is unambiguous, which settles ~35% of real ads with no model call
at all. Not primarily for quota (two Flash Lite pools give ~10 000 ads/day
against a corpus in the hundreds) but for **determinism** — `"RTX 3080"` must
parse to the same string every run, or the market index quietly disagrees with
itself.

---

## Profiles: "good deal" is not a property of a listing

The valuation above answers one question — is this cheap relative to what the
market charges for the same parts? That number is real and auditable, and on
its own it is not enough to buy anything with. It cannot say whether the machine
is any *use*.

An RTX 3070 selling 800 kr under its median is an excellent deal and a bad
purchase for running local language models, because 8 GB of VRAM will not hold
the weights. An RTX 3060 12 GB at exactly its median is the worse deal and the
better buy. No amount of price arithmetic reaches that conclusion.

### The thesis

**The used market prices GPUs for gaming, because that is what almost everyone
buys them for.** Frame rates set the price; VRAM barely moves it. That
mispricing is the entire opportunity, and it is why the profiles are not
cosmetic filters:

| Profile | Hard floors | Ranked by | Why that metric |
|---|---|---|---|
| `ai-workstation` | 12 GB VRAM, 32 GB RAM | **kr per GB VRAM** | The market underprices the thing this profile depends on. |
| `gaming` | 8 GB VRAM, 16 GB RAM | market delta only | Gaming *is* what the market prices, so the market is already the benchmark. |
| `home-server` | 1 TB storage, GPU under 100 W | **kr per TB** | A machine that runs 24/7 is judged on its electricity bill and its capacity. |

`gaming` having no bespoke metric is a decision, not an omission. Inventing a
performance score would mean shipping a benchmark table nobody validated, in
order to re-rank listings the price data already ranks correctly.

### Three states, because "we could not tell" is not "no"

A requirement is met, unmet, or **unknown**. Most Blocket ads state the graphics
card and nothing else, so an ad that never mentions its RAM must not be reported
as *failing* a 32 GB requirement — collapsing those two would hide most of the
market behind a filter that still looked like it was working. This is the same
error as the coverage bug in the index, where unpriced parts were counted as
free.

### Upgradeable shortfalls cost money; soldered ones disqualify

A 6 400 kr machine with 16 GB of RAM is not cheaper than a 6 800 kr machine with
32 GB. Requirements carry an upgrade cost where a fix exists — sourced from the
market index, falling back to a deliberately high estimate — and everything
ranks and filters on the resulting **effective price**. VRAM has no such path:
no purchase turns an 8 GB card into a 12 GB one.

### Ranking is a tuple, not a score

Fit tier, then the profile's metric, then confidence-weighted market delta.
Deliberately not a weighted sum: the opaque 0–100 `deal_score` was deleted from
this project because nobody could say why one listing outranked another.

Blending would also be arithmetically wrong. `delta_sek` is negative for a good
deal, so multiplying it by a fitness factor below 1 correctly penalises bargains
*and simultaneously rewards overpriced listings*, because scaling a positive
delta toward zero moves it up the board. One operator, two opposite meanings.

### Basket Mode: the same query, inverted

The index values a prebuilt by summing what its parts sell for. Reverse it and
the same data answers build-versus-buy: given every component listing live right
now, what is the cheapest set that satisfies the profile?

Two honesty constraints do most of the work here:

- **Priced from live listings, not medians.** A basket must be something a
  person could go and buy this afternoon. This makes the figure genuinely
  volatile — one cheap card selling moves it hundreds of kronor — and the seller
  count is shown beside it, because a 1 100 kr saving spread across four
  strangers in four towns is not the same offer as 1 100 kr in one transaction.
- **The exclusions travel with the number.** A basket of CPU, GPU, RAM and a
  drive is not a computer; it still needs a motherboard, a PSU and a case, and
  those are the three things sellers itemise worst. Omitting them silently would
  tilt every comparison toward building by exactly the amount omitted, so
  `excludedSlots` is part of the published contract and an incomplete basket
  reports **no** saving rather than a flattering one.

### Where the specs come from

Prices are counted; specs are looked up. `pipeline/hardware.py` is a static
table, and that is defensible for exactly the reason a price table would not be:
an RTX 3070 has 8 GB, it always had 8 GB, and a wrong entry is a one-line fix
rather than a silent drift.

Several model names cover more than one card — the RTX 3060 shipped as 8 GB
*and* 12 GB, and it is the single most important card for this use case. Two
rules resolve it: believe a capacity the seller states, and otherwise **assume
the smaller variant**. Guessing high admits a card that does not qualify;
guessing low withholds one that might. Only the first way costs money. The
assumption is shown to the user rather than hidden.

---

## Filtering: server-side, and locally as well

Both, deliberately.

Blocket's endpoint accepts `price_from` / `price_to`, which the `blocket_api`
wrapper does not expose for general goods (only for cars, boats and
motorcycles). `sources/blocket.py` passes them anyway. The gain is not just
fewer results — a filtered page comes back *full* of in-budget ads rather than
partly full, so one request yields more usable data. Fewer requests for the same
coverage is simultaneously faster and lighter on Blocket.

`pipeline/ingest.py` re-checks every price locally regardless. This is not
belt-and-braces pedantry: while probing the endpoint, `price_max` — a completely
plausible parameter name — turned out to be **silently ignored**, returning
everything unfiltered with a `200`. On an undocumented API an unsupported filter
fails *open*. The local bound is what keeps a silent upstream change from
becoming a silent pipeline change.

### A tension worth naming: the budget filter and the index disagree

Filtering ingest tightly to a 10 000 kr budget would bias the market index.
A median computed only over cheap listings sits low, which makes every ad look
like a bad deal — the filter would corrupt the very number used to judge.

So the two are separated:

- **Ingest** stays broad (300–40 000 kr) so the index sees the real distribution.
- **Presentation** filters to budget, at the point of ranking and publishing.

Server-side bounds are still used to skip the genuinely absurd, which costs the
index nothing.

---

## Failure handling

Every stage distinguishes *"nothing to do"* from *"something is wrong"*. The
original pipeline did not, and stalled at 120 of 474 ads for months without
anything reporting it: a failed batch returned `[]`, which is also what an empty
batch returns, so the caller hit `continue` and the ads were dropped forever.

- **Sources** raise `RateLimited` / `CircuitOpen` rather than returning empty.
- **Circuit breaker** halts after 3 consecutive failures. A challenge means the
  request rate is unwelcome; pushing through escalates a soft signal into a hard
  block. **Solving or bypassing a CAPTCHA is never attempted.**
- **The AI stage** dead-letters every failure to `ai_failures` with an attempt
  count, so a stall is queryable and bounded rather than invisible and permanent.
- **Transactions** are per batch. One bad batch cannot roll back the good ones.

---

## Retention, and why deleting data makes the system *more* accurate

Sold ads are purged after ~90 days; description text is dropped ~7 days after
extraction. That is affordable because price observations are recorded against a
canonical part, never against an ad — so the listing disappears and the data
point it contributed does not.

Privacy and accuracy usually pull against each other. Here one design satisfies
both, which is the main reason it is built this way.

## Publishing: the one output that leaves the machine

Everything above runs locally against a database of real listings written by real
people. `pipeline/publish.py` produces `web/public/deals.json`, which is
committed to a public repository and served from a CDN. That asymmetry shapes the
whole stage: a mistake anywhere else is a wrong number on a dashboard one person
looks at, and a mistake here is a stranger's phone number in a git history that
cannot be rewritten.

So redaction is not a filter applied at the end. It is the construction method.

### Allowlist, not blocklist

Every published field is named explicitly in the builder. Nothing is copied
wholesale from an ORM object, nothing is `dict(**row)`, nothing is
`model_dump()`. That inversion is what makes the guarantee survive schema
change: adding a column to `Ad` cannot leak it, because nothing in the publisher
would ever read it.

A blocklist has the opposite property. It protects against the fields you thought
of on the day you wrote it, and stops protecting the moment someone adds one you
did not — which is precisely when the protection was needed.

### Three independent layers

| Layer | Where | Catches |
|---|---|---|
| Construction | `pipeline/publish.py` | Anything never named as publishable |
| Verification | `pipeline/redact.py` | Contact details, image URLs, coordinates, postal codes, suspicious *field names* |
| Commit & CI | `.githooks/pre-commit`, `.github/workflows/ci.yml` | The bytes actually committed |

`pipeline/redact.py` is a separate module for two reasons, and both are the
point. It shares no code with the builder, so it is a genuine second opinion
rather than the same assumption asserted twice. And it imports nothing outside
the standard library — no SQLAlchemy, no models — so CI can verify the committed
artifact on a bare Python image without being able to build a snapshot at all.
**A check that always runs beats a better one that sometimes does not.**

The tests feed it deliberately poisoned input and require it to fail. A safety
check that has only ever seen clean data is not known to work; it is known to be
silent, which is also what it would be if it were broken.

### Why the cap is per profile *and* per category

Capping the file to a top-N per category needs a sort order, and the sort order
is profile-dependent — that is the entire premise of the profiles module. Cap on
one profile's ranking and the other two get a leaderboard pre-filtered by
criteria they disagree with: a selector that looks like it works and does not.

So the snapshot carries the **union of each profile's own top-N per category**.
At most three times the rows, a few hundred kilobytes, and every profile is
complete to its stated depth. The alternative — three separate files that can
disagree about the same listing — is worse in every dimension except size.

The cap itself is load-bearing rather than cosmetic: it is the part of the
*Innoweb v Wegener* argument in `ETHICS.md` that distinguishes a leaderboard from
a copy of someone else's collection. `assert_capped` enforces it as an invariant
so that it cannot quietly drift into "everything" while still passing every other
check.

### The deployed site holds nothing

Vercel serves a static bundle with **no environment variables and no serverless
function**. There is no place to put an API key and nothing that could call
Gemini even if a key existed. `vercel.json` sets a CSP of `connect-src 'self'`,
which makes that structural rather than merely true today — the browser will
refuse a request to any third-party origin.

That is the permanent fix for the bug Phase 0 opened with, where `vite.config.ts`
injected `GEMINI_API_KEY` into the browser bundle through `define`. CI greps the
built output for the key pattern on every push, and the CSP means that even a
successful injection could not be used.

### Excluded from publication, and why

Two categories of listing are withheld rather than merely ranked low, because
both would land at the **top** of a leaderboard that sorts by cheapness:

- **Implausibly cheap.** An asking price below 20% of the parts it claims to
  contain is not an asking price — it is a financing figure, a starting bid, or
  bait. Live example: an "i9-13900 / RTX 4070 Ti / 64GB DDR5" machine at 395 kr.
- **Inventory listings.** An ad naming six graphics cards and five processors is
  a dealer's stock, and summing all of it against one price produces a
  spectacular fictional bargain.

Both guards already existed in `pipeline/basket.py`. The first real run of the
publisher put the 395 kr machine at position one, because a guard that exists is
not a guard that is applied and the two call sites are the two least able to
share code. That failure was invisible to 366 passing tests: the snapshot was
well-formed, fully redacted, correctly capped, and wrong at the very top.

**Reading real output remains the primary bug-finding discipline of this
project.** The same first run also surfaced `"rtx 3080 32gb"` being parsed as a
32 GB graphics card, a card extracted twice and counted twice, and a want-to-buy
ad presented as a bargain. None of them had a failing test; all of them have one
now, with the live example quoted in the docstring.
