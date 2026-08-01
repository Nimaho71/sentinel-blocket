# legacy/ — superseded code, kept for the record

Nothing in this directory runs. It is retained because the *reason* it was
replaced is more interesting than the code itself, and because deleting it would
erase the decision.

## What was here

`scraper/` and `orchestrator.py` were v1: a Playwright browser driving
blocket.se, with `playwright-stealth`, a rotating User-Agent pool, simulated
mouse movement, and CAPTCHA detection.

## Why it was replaced

Blocket's own website is served by JSON endpoints. Talking to those directly
(via the `blocket-api` package, see `sources/blocket.py`) is better on every
axis that matters:

| | JSON endpoints | Playwright crawler |
|---|---|---|
| Throughput | 53 ads in **0.4 s** | ~1 ad per 8 s, plus page render |
| Load on Blocket | one small JSON response | full page render, all subresources |
| CAPTCHA surface | none | real |
| Memory | ~50 MB | Chromium, ~500 MB+ |
| Breaks when | endpoints change | CSS *or* endpoints change |

Those throughput numbers are measured, not estimated.

The decisive argument is the second row. Rendering a full page to read a
listing costs Blocket far more than requesting the JSON their own frontend
requests. The lighter-touch option was also the faster and more reliable one.

## What was worth keeping

`crawler.py` parsed JSON-LD first and fell back to DOM selectors only when that
failed. That instinct — prefer the structured payload a site already publishes,
treat markup scraping as the fallback — is exactly what led to replacing the
crawler entirely. It is the same idea, taken one step further.

## What was deliberately dropped

The User-Agent pool was actively harmful. It contained a Firefox UA while
launching Chromium, so the browser announced one engine while its JS runtime,
`navigator.plugins`, WebGL renderer and TLS fingerprint all said another. That
mismatch is a louder bot signal than sending no stealth at all. It also re-rolled
per run against a fixed residential IP, and a fingerprint that mutates from one
address is itself a tell.

CAPTCHA handling detected challenges and then carried on. The replacement is a
circuit breaker (`sources/throttle.py`): skip one failure, halt after three
consecutive. A challenge means the request rate is unwelcome; pushing through
collects nothing and escalates a soft signal into a hard block.

**Solving or bypassing a CAPTCHA was never implemented and will not be.** See
`ETHICS.md`.
