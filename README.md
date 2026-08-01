# Sentinel

**A used-hardware deal engine for Blocket.se.** It scrapes listings, extracts
specifications with an LLM, prices them against an index it builds from observed
sale prices, and ranks them against *what you are actually trying to build*.

I wrote it because I needed to buy my first computer — 5 000–10 000 kr, for
running local language models, gaming, and university data-science work — and I
could not tell a good listing from a bad one.

📊 **[Live leaderboard](https://sentinel-blocket.vercel.app)** ·
⚖️ **[Ethics & legal reasoning](ETHICS.md)** ·
🏗️ **[Architecture](docs/ARCHITECTURE.md)** ·
🛒 **[Buying guide](docs/BUYING-GUIDE.md)**

---

## The idea

Three ideas hold the project up. Each of them started as a bug.

### 1. Math owns value, the AI owns risk

The first version asked Gemini for `estimated_market_value_sek` and got a
confident number invented from memory — not Sweden-specific, not current, and
impossible to check. It was the weakest part of the project.

The responsibilities are now split along the line of what each side is good at:

- **Deterministic arithmetic owns price.** Every component sighting is appended
  to a `price_observations` table; value is a recency-weighted *median* over a
  trailing window. Fully auditable, and always reported with its sample size:
  *"RTX 3080 median 3 200 kr (n=31)."*
- **The LLM owns extraction and risk.** Reading "säljes i befintligt skick" and
  concluding *untested, no receipt* is exactly what a language model is for.
  Reading a price out of its own weights is not.

The model never emits a price. That is the invariant.

### 2. Absence of evidence is not evidence of absence

The single most productive idea in the codebase, and it fixed three separate
bugs.

Most Blocket ads state the graphics card and nothing else. So *"this machine has
too little RAM"* and *"the listing never mentioned its RAM"* must be different
outcomes. Collapse them and you hide most of the market behind a filter that
looks like it is working. Requirements are therefore three-state — met, unmet,
**unknown** — and the same principle governs the valuation: if too few of a
build's parts can be priced, the delta is `None` rather than a wrong number.

That last one was a real bug. Every full build was coming back "overpriced by
2 000–4 000 kr", uniformly — which is not what a real market looks like. Only
the GPU had an index entry, so "sum of parts" quietly meant "sum of the one part
we knew", and the CPU, RAM and storage were each counted as worth zero. A
systematic one-directional bias, presented as a confident figure, on the exact
number the leaderboard sorted by.

### 3. "A good deal" is not a property of a listing

An RTX 3070 selling 800 kr under its median is an excellent *deal* and a bad
*purchase* for running local models, because 8 GB of VRAM will not hold the
weights. An RTX 3060 at exactly its median is a worse deal and a better buy.

The used market prices GPUs for **gaming**, because that is what almost everyone
buys them for. Frame rates set the price; VRAM barely moves it. That mispricing
is the entire opportunity, and it is why the `ai-workstation` profile ranks on
**kronor per GB of VRAM** while `gaming` deliberately has no bespoke metric — for
gaming, the market already *is* the benchmark.

Profiles also price their own shortfalls. A 6 500 kr machine with 16 GB of RAM
does not fail; it costs 800 kr more than it looks like it does, and competes at
7 300 kr. Soldered VRAM is different — no amount of money turns an 8 GB card into
a 12 GB one — so that blocks, and says so.

---

## Architecture

The insight that simplified everything: **the scrape has to run on my laptop
anyway.** Datacenter IP ranges are published and heavily scored against; consumer
VPN exits are worse. A home Swedish residential connection is the best available
option and it is free. Given that, there is no reason to host a server or a
database at all.

```
LOCAL (laptop, nightly)
  ingest → evaluate → index → basket → publish → janitor
     ↓
  sentinel_blocket.db   full fidelity, never leaves the machine
     ↓
  redact + cap to top-N per category
     ↓
  web/public/deals.json ──git push──▶ Vercel ──▶ static React site
```

The deployed site is a **dumb static reader**. No serverless function, no API
key in Vercel's environment, no cold start, no attack surface. "Live" means
updated by the nightly local run.

| Layer | Choice | Why |
|---|---|---|
| Source | Blocket's own JSON API | v1 was a Playwright stealth crawler. Finding the underlying API made it ~20× faster, removed the CAPTCHA surface entirely, and is *lighter on Blocket's infrastructure*. The crawler is in `legacy/`. |
| Database | SQLite + async SQLAlchemy + Alembic | Single-writer, local, zero-ops. Migrates to Turso unchanged if it ever needs hosting. |
| LLM | Gemini Flash Lite, free tier | Two documented 500-RPD pools, rotated. See [quota](#quota). |
| Frontend | React 19 + Vite + Tailwind 4 | Static build. |
| Hosting | Vercel | Free, and it serves one JSON file. |

Full reasoning in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY
git config core.hooksPath .githooks
alembic upgrade head

python -m pipeline.run                        # full nightly pass
python -m pipeline.run --stage publish        # rebuild deals.json only
python -m pipeline.run --stage basket --budget 8000
python -m pipeline.run --dry-run              # scrape and report, write nothing

cd web && npm install && npm run dev          # dashboard on :3000
```

Stages are separately runnable, and every one of them reports counts. That is
not a convenience feature — see [the bug that shaped the pipeline](#the-bug-that-shaped-everything).

### Quota

The binding constraint on the free tier is **requests per day**, and it is
wildly uneven across models. Three things keep usage inside it:

1. **Do the easy work deterministically.** Regex plus a parts dictionary settles
   about half of all ads; only genuinely ambiguous listings reach the model.
   Biggest multiplier by far, and it makes the system more testable too.
2. **Rotate across pools Google documents as separate.** Two 500-RPD Flash Lite
   pools is ~1 000 RPD, and at batch size 10 that is far more than needed.
3. **Never re-analyse.** Every ad carries a content hash of `(title,
   description, price)`; an unchanged re-scrape costs nothing.

**On multiple API keys: no.** Creating extra accounts to multiply free quota
violates Google's terms, risks the account, and would be plainly visible in a
public repository. Pool rotation gets the same throughput legitimately.

---

## Safety

This scrapes a live marketplace and handles other people's personal data, so
those constraints are enforced in code rather than promised in prose.

- **No CAPTCHA is ever solved or bypassed.** If challenged, the correct response
  is to slow down or stop. No exceptions.
- **Personal data is never ingested.** No column exists for a seller name, phone
  number, postal code or GPS coordinate — and the API returns the last two.
- **Ad text is scrubbed before it reaches Gemini.** Enforced at the network
  boundary, so a new caller cannot bypass it. A scan of 103 real ads found a
  contact email sitting in plain text.
- **Descriptions are deleted ~7 days after extraction.** The specs are kept, the
  prose is not. `price_observations` holds no reference to the ad it came from,
  so purging listings costs nothing in accuracy — privacy and a better index
  turned out to be the same feature.
- **The published file is built from an allowlist**, then verified by an
  independent zero-dependency checker (`pipeline/redact.py`) that CI runs
  against the committed bytes. It is tested against deliberately poisoned input,
  because a safety check that has only seen clean data is not known to work.

The full legal reasoning — Blocket's terms, the EU database right and *Innoweb
v Wegener*, GDPR, and why `robots.txt` is **deliberately not enforced** rather
than performatively checked — is in **[ETHICS.md](ETHICS.md)**, along with a
takedown contact.

---

## The bug that shaped everything

The database held 474 ads and 120 evaluations. Nothing reported the other 354 as
missing, because three ordinary-looking things went wrong together:

1. `analyze_ads_batch` returned `[]` on any error — indistinguishable from a
   legitimately empty result.
2. The caller responded with a bare `continue`. The batch was dropped with no
   record it had ever been attempted.
3. The next run re-selected the same ads, because the query only asked whether
   an evaluation row existed, and failed the same way. Forever.

**A permanent failure that reports itself as ordinary progress is the hardest
kind to find,** and none of the three parts looked wrong on its own.

The fix was structural rather than a patch: failures raise, every failure is
recorded against its ad, attempts are counted and bounded, and every stage
reports how much it could not do. A stall is now loud.

That lesson is the reason this project has 394 tests and a habit of reading real
output. Nearly every significant bug — a 395 kr "i9-13900 / RTX 4070 Ti"
machine topping the leaderboard, a dealer's 999 kr inventory listing valued as
one computer, `"rtx 3080 32gb"` parsed as a 32 GB graphics card — was found by
looking at what the pipeline actually printed, not by a failing test. Each then
got a named regression test with the live example quoted in its docstring.

---

## Status

Working end to end against ~1 029 live listings. The market index is still
young: 95 distinct parts observed, 11 of them with enough samples (n≥5) to price
confidently. The rest honestly report *"low confidence, n=2"* rather than
inventing a number, and depth compounds with every nightly run.

It has also answered the question it was built for. The finding, in one line:
**an RTX 3060 12 GB costs 216 kr per GB of VRAM against 313 kr for the 3060 Ti,
because the used market prices cards on frame rates and the Ti is the better
gaming card.** The full reasoning, the current pick, and the parts-basket
comparison are in **[docs/BUYING-GUIDE.md](docs/BUYING-GUIDE.md)**.
