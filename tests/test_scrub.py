"""PII scrub tests.

Half of this file removes personal data. The other half proves the scrubber
does **not** eat hardware names — and that half is the one that matters more in
practice, because over-scrubbing has no symptom. Nothing raises; the extractor
just quietly gets worse at its job.

The Swedish number formats below are the ones sellers actually use. They are
here because the real-ad corpus contained *no* phone numbers at all, so without
synthetic cases the phone pattern would be entirely unexercised — a regex that
never fires is indistinguishable from a regex that cannot fire.
"""

from __future__ import annotations

import pytest

from ai.scrub import EMAIL_TOKEN, PHONE_TOKEN, URL_TOKEN, contains_pii, scrub


@pytest.mark.parametrize(
    "number",
    [
        "070-123 45 67",
        "0701234567",
        "070 123 45 67",
        "+46 70 123 45 67",
        "+46701234567",
        "073-123 45 67",
        "08-123 456 78",  # Stockholm landline
        "0701-234567",
    ],
)
def test_swedish_phone_formats_are_removed(number):
    scrubbed = scrub(f"Ring mig på {number} för snabb affär")

    assert PHONE_TOKEN in scrubbed
    assert not any(ch.isdigit() for ch in scrubbed), scrubbed


def test_email_is_removed():
    scrubbed = scrub("Maila info@teknikfronten.se vid frågor")

    assert EMAIL_TOKEN in scrubbed
    assert "@" not in scrubbed


def test_a_real_address_from_the_live_corpus_is_removed():
    """Verbatim shape of the address found in a freshly scraped ad."""
    assert "teknikfronten" not in scrub("Kontakt: info@teknikfronten.se")


def test_urls_are_removed():
    assert URL_TOKEN in scrub("Mer info på https://example.se/min-profil?id=99")
    assert URL_TOKEN in scrub("se www.blocket.se/annons/123")


def test_email_is_removed_before_phone_can_fragment_it():
    """`0701234567@example.com` — the phone pattern would otherwise eat the
    local part and leave a broken `[phone]@example.com` behind."""
    scrubbed = scrub("kontakt 0701234567@example.com tack")

    assert scrubbed.count(EMAIL_TOKEN) == 1
    assert "example.com" not in scrubbed


# --- The other direction: specs must survive. -------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "RTX 3080",
        "RTX 3060 Ti",
        "RTX4070",
        "GTX 1080 Ti",
        "RX 6800 XT",
        "Ryzen 7 5800X",
        "Ryzen 5 5600G",
        "i5-12400F",
        "i7 12700K",
        "i9-14900KS",
        "32GB DDR4 3200MHz",
        "16 GB DDR5 6000 MHz",
        "850W PSU",
        "2TB NVMe SSD",
        "512GB M.2",
        "B550M moderkort",
        "Samsung 970 EVO Plus 1TB",
        "240Hz skärm",
    ],
)
def test_hardware_names_survive_the_scrub(spec):
    """Over-scrubbing is silent. These are the strings the pipeline exists to read."""
    text = f"Säljer en dator med {spec} i mycket bra skick."

    assert spec in scrub(text), f"scrub destroyed {spec!r}"


def test_a_full_realistic_listing_survives_intact():
    text = (
        "Gamingdator säljes. Ryzen 5 5600X, RTX 3060 Ti 8GB, 16GB DDR4 3200MHz, "
        "1TB NVMe, 650W Corsair, B550 moderkort. Köpt 2021, kvitto finns. "
        "Ring 070-123 45 67 eller maila mig@example.se"
    )
    scrubbed = scrub(text)

    for spec in ["Ryzen 5 5600X", "RTX 3060 Ti", "16GB DDR4 3200MHz", "1TB", "650W", "B550"]:
        assert spec in scrubbed
    assert PHONE_TOKEN in scrubbed and EMAIL_TOKEN in scrubbed
    assert "070" not in scrubbed and "example.se" not in scrubbed


def test_a_purchase_year_is_not_mistaken_for_a_phone_number():
    assert "2021" in scrub("Köpt 2021, använd sparsamt")


def test_a_price_in_the_body_survives():
    assert "8500" in scrub("Nypris 8500 kr, säljes för 5000")


# --- The boundary assertion. -------------------------------------------------


def test_contains_pii_detects_what_scrub_removes():
    """These two must agree, since `contains_pii` is the tripwire asserting
    that nothing bypassed `scrub` on the way to the network."""
    dirty = "Ring 070-123 45 67 eller maila a@b.se, se www.x.se"

    assert contains_pii(dirty)
    assert not contains_pii(scrub(dirty))


def test_clean_hardware_text_is_not_flagged():
    assert not contains_pii("RTX 3080, Ryzen 7 5800X, 32GB DDR4 3200MHz, 850W")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_input_is_handled(value):
    assert scrub(value).strip() == ""
    assert not contains_pii(value)
