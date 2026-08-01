"""Quota ledger tests.

The single most important test in this file is
`test_the_ledger_survives_a_restart`. The limiter it replaces was a module
global, so every `python -m pipeline.run` began believing it had a full day's
quota — and RPD is the constraint that actually binds on the free tier. That
bug had no symptom until a 429 arrived, and then looked like Google's fault.

Times are injected rather than mocked. The daily reset is a real boundary at a
real timezone with real DST, and `now=` lets the tests stand either side of it
without waiting or patching the clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from ai.quota import MODELS, ModelQuota, QuotaExhausted, QuotaLedger

PACIFIC = ZoneInfo("US/Pacific")
PRIMARY = MODELS[0].name
SECONDARY = MODELS[1].name
ESCALATION = MODELS[2].name


@pytest.fixture
def ledger(tmp_path):
    with QuotaLedger(tmp_path / "quota.db") as led:
        yield led


def test_a_fresh_ledger_has_full_quota(ledger):
    assert ledger.used_today(PRIMARY) == 0
    assert ledger.remaining_today(PRIMARY) == 500


def test_requests_are_counted(ledger):
    for _ in range(3):
        ledger.record(PRIMARY)

    assert ledger.used_today(PRIMARY) == 3
    assert ledger.remaining_today(PRIMARY) == 497


def test_pools_are_counted_separately(ledger):
    """The premise of the whole rotation strategy."""
    ledger.record(PRIMARY)

    assert ledger.used_today(PRIMARY) == 1
    assert ledger.used_today(SECONDARY) == 0


def test_the_ledger_survives_a_restart(tmp_path):
    """The bug this module exists to fix.

    Two `QuotaLedger` objects over one file stand in for two runs of the
    pipeline. The in-memory global this replaces would report zero here — and
    then cheerfully spend a second full day's quota.
    """
    path = tmp_path / "quota.db"

    with QuotaLedger(path) as first_run:
        for _ in range(120):
            first_run.record(PRIMARY)

    with QuotaLedger(path) as second_run:
        assert second_run.used_today(PRIMARY) == 120
        assert second_run.remaining_today(PRIMARY) == 380


def test_rotation_moves_to_the_second_pool_when_the_first_is_spent(ledger):
    for _ in range(500):
        ledger.record(PRIMARY)

    assert ledger.remaining_today(PRIMARY) == 0
    assert ledger.choose_model().name == SECONDARY


def test_exhausting_every_primary_pool_raises(ledger):
    for model in (PRIMARY, SECONDARY):
        for _ in range(500):
            ledger.record(model)

    with pytest.raises(QuotaExhausted, match="out of daily quota"):
        ledger.choose_model()


def test_the_exhaustion_message_says_what_was_used(ledger):
    """A limit that reports "quota exceeded" and nothing else is a limit you
    debug by guessing."""
    for model in (PRIMARY, SECONDARY):
        for _ in range(500):
            ledger.record(model)

    with pytest.raises(QuotaExhausted) as exc:
        ledger.choose_model()

    assert f"{PRIMARY}=500/500" in str(exc.value)
    assert "Pacific" in str(exc.value)


def test_escalation_models_are_never_chosen_by_default(ledger):
    """20 RPD is a reserve, not a workhorse. Spending it on routine batches
    means having none left for the ads that actually needed a better model."""
    assert ledger.choose_model().escalation_only is False
    assert ledger.choose_model(escalate=True).name == ESCALATION


def test_escalation_quota_is_separate_from_the_primary_pool(ledger):
    for _ in range(500):
        ledger.record(PRIMARY)

    assert ledger.remaining_today(PRIMARY) == 0
    assert ledger.remaining_today(ESCALATION) == 20


# --- The daily reset boundary ------------------------------------------------


def test_quota_resets_at_pacific_midnight_not_local_midnight():
    """Google's free-tier day rolls over at midnight US/Pacific.

    Counting from local (Swedish) midnight would free quota up to nine hours
    early and produce 429s that look like the API misbehaving.
    """
    before = datetime(2026, 3, 15, 6, 0, tzinfo=PACIFIC)  # 06:00 Pacific
    after_reset = datetime(2026, 3, 16, 0, 30, tzinfo=PACIFIC)  # just past midnight

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with QuotaLedger(f"{tmp}/q.db") as led:
            for _ in range(10):
                led.record(PRIMARY, now=before.astimezone(timezone.utc))

            assert led.used_today(PRIMARY, now=before.astimezone(timezone.utc)) == 10
            assert led.used_today(PRIMARY, now=after_reset.astimezone(timezone.utc)) == 0


def test_requests_earlier_the_same_pacific_day_still_count(ledger):
    morning = datetime(2026, 3, 15, 1, 0, tzinfo=PACIFIC).astimezone(timezone.utc)
    evening = datetime(2026, 3, 15, 23, 0, tzinfo=PACIFIC).astimezone(timezone.utc)

    ledger.record(PRIMARY, now=morning)

    assert ledger.used_today(PRIMARY, now=evening) == 1


def test_the_reset_handles_a_dst_transition(ledger):
    """US DST begins 8 March 2026. A hardcoded -8 offset would be an hour out
    for most of the year, and mis-count right at the boundary."""
    just_after_dst = datetime(2026, 3, 9, 0, 30, tzinfo=PACIFIC).astimezone(timezone.utc)
    before_dst_midnight = datetime(2026, 3, 8, 23, 0, tzinfo=PACIFIC).astimezone(timezone.utc)

    ledger.record(PRIMARY, now=before_dst_midnight)

    # 23:00 on the 8th and 00:30 on the 9th are different Pacific days.
    assert ledger.used_today(PRIMARY, now=before_dst_midnight) == 1
    assert ledger.used_today(PRIMARY, now=just_after_dst) == 0


# --- Housekeeping ------------------------------------------------------------


def test_old_rows_are_prunable(ledger):
    old = datetime.now(timezone.utc) - timedelta(days=30)
    ledger.record(PRIMARY, now=old)
    ledger.record(PRIMARY)

    assert ledger.prune(older_than_days=7) == 1
    assert ledger.used_today(PRIMARY) == 1


def test_last_call_at_tracks_the_most_recent_request(ledger):
    assert ledger.last_call_at(PRIMARY) is None

    stamp = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
    ledger.record(PRIMARY, now=stamp)

    assert ledger.last_call_at(PRIMARY) == stamp


def test_an_unknown_model_has_no_quota(ledger):
    """Fail closed. A typo in a model name must not read as unlimited."""
    assert ledger.remaining_today("gemini-does-not-exist") == 0


def test_report_shows_every_pool(ledger):
    ledger.record(PRIMARY)
    report = ledger.report()

    for model in MODELS:
        assert model.name in report
    assert f"{PRIMARY}: 1/500" in report


# --- Spacing -----------------------------------------------------------------


@pytest.mark.parametrize("rpm,floor", [(15, 4.0), (5, 12.0)])
def test_request_spacing_leaves_headroom_under_the_rpm_limit(rpm, floor):
    """Pacing exactly at the limit trips 429s: our minute window and the
    server's are not aligned, so requests land either side of a boundary."""
    interval = ModelQuota("m", rpm=rpm, rpd=100).min_interval_s

    assert interval > floor
    assert interval < floor * 1.5
