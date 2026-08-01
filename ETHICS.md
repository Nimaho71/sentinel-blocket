# Ethics, law, and what this project does about them

This document exists because the project scrapes a live commercial marketplace,
processes strangers' personal data, and publishes a derived artifact on the open
internet. Each of those deserves a straight answer rather than a disclaimer.

Nothing here is legal advice. It is a record of the reasoning, so that a
disagreement can be with the argument rather than with a silence.

**Contact / takedown:** open an issue on this repository, or email the address in
the GitHub profile. Any removal request — from Blocket or from an individual
seller — will be honoured without argument and without requiring it to be
justified. See [Takedown](#takedown) below.

---

## 1. What the tool actually does

A nightly job on a personal laptop queries Blocket's own JSON endpoints for
used-PC listings, extracts hardware specifications, prices them against a
locally-built index of observed sale prices, and writes a small ranked summary
to a static file. A public web page reads that file. Nothing else is deployed;
there is no server, no hosted database, and no public API.

Roughly one request every 1.5–2.0 seconds, a few thousand requests a night, from
one Swedish residential connection, scheduled for the small hours when the
marginal load on Blocket is lowest.

---

## 2. Blocket's terms of service

Blocket's terms prohibit automated access. This project performs automated
access. That is a plain statement of fact and this document will not dress it up
as anything else.

What follows is why the project exists anyway, and what it does to keep the
imposition proportionate.

**Terms of service are a contract, not a criminal statute.** Breaching them is a
civil matter between the parties, and the remedy available to Blocket is to
block the client or to ask for the project to stop. Both are entirely reasonable
outcomes and this project is built to accept either immediately (see
[Halting](#5-halting-and-the-standing-rule-on-challenges)).

**No access control is circumvented.** The endpoints used are the same
unauthenticated public JSON APIs that Blocket's own website calls to render
pages that anyone can view without an account. No login is bypassed, no paywall
is defeated, no rate limit is evaded, and — the standing rule of the project —
**no CAPTCHA or bot challenge is ever solved or worked around.** This matters
legally as well as ethically: circumventing a technical access control is the
line that separates "against the terms" from a much more serious category, and
this project does not go near it.

**The load is negligible and deliberately so.** One request every 1.75 seconds
is several orders of magnitude below what Blocket serves continuously, and the
sweep is scheduled overnight so even that lands when there is most headroom to
absorb it. An earlier
version of this project used a headless Chromium crawler that rendered whole
pages; it was retired partly because the JSON API is *lighter on Blocket's
infrastructure*, not only because it was easier to work with. Choosing the
gentler method when the heavier one also worked is the substance of the claim to
be acting in good faith.

**It sends traffic to Blocket.** Every listing published links back to the
original ad. The site is a pointer, not a substitute — a reader who finds
something here goes to Blocket to buy it.

### Why `robots.txt` is not enforced, and this is not an oversight

`robots.txt` was read and considered. It is not enforced, and pretending
otherwise would be worse than saying so.

It is a convention for search-engine crawlers, published under no statutory
authority, and it carries no independent legal force in the EU. It also almost
certainly disallows the paths this tool needs, so adding a check would mean
either ignoring the result — a purely performative gesture — or abandoning the
project while claiming compliance.

A documented, reasoned deviation is more honest than a checkbox. The politeness
measures below are the substance; `robots.txt` compliance would have been the
appearance.

---

## 3. The EU database right, and why this is a leaderboard

Blocket's collection of listings is protected in the EU by the *sui generis*
database right (Directive 96/9/EC). The case that matters is
**[*Innoweb v Wegener* (C-202/12)](https://curia.europa.eu/juris/liste.jsf?num=C-202/12)**,
in which the CJEU held that a "dedicated meta search engine" — one that passes a
user's query through to another site's database in real time and returns
essentially the whole of its content — infringes that right.

The distinction the Court drew is between *re-utilising the collection* and
*referring to individual items in it*. This project is built on the second side
of that line, on purpose:

| *Innoweb* infringed by… | This project instead… |
|---|---|
| Answering arbitrary user queries in real time | Serves a **static file**, refreshed nightly. No query reaches Blocket from a visitor. |
| Exposing substantially the whole collection | Publishes a **top-10 per category**, currently ~90 of ~1 029 listings. |
| Providing a search interface over another's database | Has **no free-text search box** and **no pagination** through the collection. |
| Substituting for the original site | **Links every entry back** to the original ad. |

The cap is not a UI preference. It is the load-bearing part of this argument,
which is why `pipeline/publish.py::assert_capped` enforces it as an invariant
and CI fails the build if it is ever exceeded. A "top ten per category" that
quietly grew into "everything" would dismantle the reasoning above while still
passing every other check.

The published data is also **derived, not copied**: canonical part names,
computed price deltas, sample counts, and profile verdicts. The one thing
reproduced close to verbatim is the listing title, which is needed to identify
which ad is being referred to.

---

## 4. Personal data (GDPR)

Ad descriptions are written by private individuals and routinely contain phone
numbers, email addresses and names. Blocket's JSON API additionally returns
**postal codes and GPS coordinates** — more precise location data than the
public web page ever shows.

Four measures, applied at four different layers:

**Never ingested.** `sources/blocket.py` does not read coordinates, postal codes
or seller identity out of the API response. `db/models.py` has no column for
any of them. A leak would require both layers to be wrong at once.

**Scrubbed before leaving the machine.** `ai/scrub.py` removes phone numbers,
emails and URLs from description text before it is sent to Gemini. This is
enforced at the network boundary, not at the call sites, so a new caller cannot
bypass it. A scan of 103 freshly ingested ads found a contact email sitting in
plain text in a description — this is not a theoretical risk.

**Discarded after use.** `raw_description` is deleted roughly seven days after
extraction has read it (`pipeline/janitor.py`). The specs are kept; the prose is
not. `price_observations` holds **no reference to the ad it came from**, so
deleting listings costs nothing in accuracy — data minimisation and a better
price index turn out to be the same design.

**Never published.** `pipeline/publish.py` builds the public file from an
explicit allowlist of fields, then an independent check re-reads the finished
JSON and refuses it if it finds contact details, image URLs, coordinates,
postal codes, or a field name suggesting any of those.

For the local tool, GDPR's household exemption (Art. 2(2)(c)) very likely
applies anyway. The measures above are belt-and-braces, and they are cheap.

### The free-tier trade, stated plainly

This project uses Gemini's **free tier**, under which Google may use submitted
content to improve their products. The paid tier does not. That is the trade
being made in exchange for zero cost, and it is the single strongest reason the
PII scrub is not optional: without it, strangers' contact details would be going
into a training corpus they never agreed to.

Only ad *text* is sent, and only after scrubbing. No images, no location, no
seller identity.

**On multiple API keys: no.** Creating extra projects or accounts to multiply
free quota violates Google's terms. The project instead rotates across model
pools that Google documents as having separate quotas — the sanctioned way to
get the same throughput.

---

## 5. Halting, and the standing rule on challenges

- One consistent, honest `User-Agent`. No browser impersonation, no rotating
  fingerprints, no residential proxies.
- ~1 request per 1.75 seconds with jitter (1.5 s floor). Never bursts.
- Scheduled overnight, and bounded by a wall-clock budget rather than running
  until it runs out of work.
- Exponential backoff on 403/429/503; `Retry-After` is honoured.
- Circuit breaker: three consecutive failures halt the run, persist state, and
  cool down. A partial sweep reports itself as partial rather than as success.
  A sweep cut short by its own time budget says so too, and the next run starts
  where the list was left rather than repeating the same prefix.

**The standing rule, which has no exceptions: never solve or bypass a
challenge.** If Blocket ever starts serving CAPTCHAs to this client, the correct
response is to slow down or stop. Not to defeat it. Solving services cost money,
are an unwinnable arms race, and are the one thing that unambiguously reads as
circumventing an access control rather than merely breaching a term.

An earlier version of this project contained bot-detection evasion: a
Playwright crawler with `playwright-stealth`, a rotating User-Agent pool, and
simulated mouse movement. It was taken out of service and nothing in the
running pipeline uses it.

The code itself is still in the tree, under `legacy/`, and that is deliberate.
Deleting it would make this document unverifiable — a claim to have stopped
doing something, backed by an empty directory. `legacy/README.md` sets out what
each piece did and why it was dropped, including that the User-Agent pool was
*counterproductive* as well as dishonest: it announced Firefox while running
Chromium, which is a louder bot signal than sending nothing at all. The history
of the retirement is deliberately not rewritten.

---

## 6. What is published, and what is not

| Published | Withheld |
|---|---|
| Listing title (scrubbed of contact details) | The description, in full |
| Price, city, seller-declared condition | Postal code, GPS coordinates |
| Canonical part names | Seller name, phone number, email |
| Index medians and sample counts | Image URLs of any kind |
| Profile verdicts and the model's risk notes | The raw listing record |
| A link back to the original ad | Anything not named in this table |

Enforced by three independent layers, because the cost of this particular
failure is unbounded and irreversible:

1. **Construction.** Every published field is named explicitly in
   `pipeline/publish.py`. Nothing is copied wholesale from an ORM object, so
   adding a database column cannot leak it.
2. **Verification.** `assert_redacted` re-reads the finished JSON and shares no
   code with the builder. `tests/test_publish.py` feeds it deliberately
   poisoned input and requires it to fail — a safety check that has only ever
   seen clean data is not known to work.
3. **Commit and CI.** `.githooks/pre-commit` blocks the obvious cases locally;
   GitHub Actions runs the full check on every push.

Never committed: `sentinel_blocket.db` (real ad data), `.env` (API key). Both
are gitignored and both are blocked by the pre-commit hook.

---

## Takedown

If you are Blocket, or an individual whose listing appears here, and you want
this stopped or your data removed:

**Open an issue on this repository, or email the address on the GitHub profile.**

It will be actioned without argument, without asking for justification, and
without requiring a formal notice. A request from Blocket will stop the scraping
entirely, not merely remove a listing. This is a personal learning project and a
portfolio piece; no part of it is worth a dispute with anyone.
