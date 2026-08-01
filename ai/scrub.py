"""Strip personal data from ad text before it leaves this machine.

### Why this is not optional

Free-tier Gemini may use submitted content to improve Google's products. Ad
descriptions are typed by hand by private individuals and routinely contain
phone numbers and email addresses. Sending those is a transfer of a stranger's
contact details to a US processor and into a training corpus — a thing they did
not agree to and cannot undo.

This is not hypothetical. A scan of 103 freshly ingested ads found a contact
email sitting in plain text in a description.

Three benefits, one mechanism: no personal data leaves the machine, GDPR
exposure drops, and the prompt gets shorter — which matters because TPM, not
RPM, is the binding quota constraint.

### The tension that shapes every pattern here

Over-scrubbing destroys the specs the pipeline exists to read. Hardware names
are *full of* digit runs: "Ryzen 7 5800X", "DDR4 3200MHz", "i5 12400F",
"850W". A careless phone-number pattern eats them and the extractor silently
gets worse — a failure that shows up as poor results, not as an error.

So the patterns are deliberately narrow, and `tests/test_scrub.py` asserts on
both directions: personal data is removed, **and** hardware names survive
untouched. The second half of that is the half that is easy to forget.

### Measured against 474 real descriptions

Run over the whole local corpus, the scrubber removed 5 URLs and destroyed
exactly one hardware name — `rtx4070ti`, which appeared only inside an ASUS
product-page link and so was taken by the URL rule rather than a digit rule.
That is a real cost, accepted knowingly: it is one spec in 474 ads, the model
name was also in that ad's title, and the alternative is parsing URLs to decide
which are personal, which is a worse trade. Recorded here so the next person to
read `_URL` knows it was measured rather than assumed.
"""

from __future__ import annotations

import re

EMAIL_TOKEN = "[email]"
PHONE_TOKEN = "[phone]"
URL_TOKEN = "[link]"

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")

# URLs. Removed because they are an exfiltration path for anything the other
# patterns miss (a "contact me here" link to a profile page) and because a wall
# of query parameters is pure token cost.
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

# Swedish mobile and landline numbers.
#
# Anchored on a leading `+46` or `0`, which is what keeps it off hardware names:
# no GPU, CPU or RAM model starts with a zero. Requires at least 8 further
# digits, so a 4-digit model number cannot reach the threshold. Separators are
# allowed between groups because sellers write numbers every possible way
# (`070-123 45 67`, `+46 70 1234567`, `0701234567`).
_PHONE = re.compile(
    r"""
    (?<![\d\w])                 # not mid-token: won't fire inside "i5-12400F"
    (?: \+46 | 0 )              # country code or national trunk prefix
    [\s\-.()]{0,2}
    (?: \d [\s\-.()]{0,2} ){8,11}
    \d
    (?![\d\w])
    """,
    re.VERBOSE,
)

# A "call me" number written without any prefix at all, e.g. "ring 707112233".
# Kept separate and strict — nine or more bare digits in a row is never a part
# number in this domain, but eight might be.
_BARE_LONG_NUMBER = re.compile(r"(?<![\d\w])\d{9,}(?![\d\w])")


def scrub(text: str | None) -> str:
    """Remove contact details from free text.

    Order matters: emails are removed before phone numbers, because an address
    like `070123456@example.com` would otherwise have its local part eaten by
    the phone pattern and leave a broken fragment behind.
    """
    if not text:
        return ""

    text = _EMAIL.sub(EMAIL_TOKEN, text)
    text = _URL.sub(URL_TOKEN, text)
    text = _PHONE.sub(PHONE_TOKEN, text)
    text = _BARE_LONG_NUMBER.sub(PHONE_TOKEN, text)
    return text


def contains_pii(text: str | None) -> bool:
    """Whether anything the scrubber removes is still present.

    Used as an assertion at the network boundary rather than as a filter — if
    this is ever true at that point, something bypassed `scrub` and the right
    response is to fail loudly, not to quietly clean it up.
    """
    if not text:
        return False
    return any(p.search(text) for p in (_EMAIL, _URL, _PHONE, _BARE_LONG_NUMBER))
