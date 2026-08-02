"""Free-tier quota accounting and model rotation.

### Why the ledger is on disk

The previous limiter was a module global. It enforced a spacing between requests
within one process and forgot everything on exit — so the **daily** cap was never
tracked at all. Run the pipeline five times in an afternoon and the fifth run
believes it has a full day's quota left. RPD is the binding constraint on the
free tier, so the one limit that mattered was the one not being counted.

The ledger is therefore a SQLite table, one row per request. Restart-safe by
construction: the count is a `SELECT`, not a variable.

### Why rotation across models is legitimate

Gemini 3.1 Flash Lite and 3.5 Flash Lite have **separate 500-RPD pools** on one
key. Using both is 1 000 requests/day, which at batch size 10 is far more than
this project needs. That is model rotation within the terms of one account.

Creating extra accounts or keys to multiply a free quota is not the same thing —
it violates Google's terms, risks the account, and would be plainly visible in a
public repo. It is not done here, and `MODELS` below is the reason it does not
need to be.

Full Flash models are held back for escalation: 20 RPD each, spent only on ads
Flash Lite could not parse. Twenty batches of failures a day is more headroom
than a healthy pipeline should ever want.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelQuota:
    """Published free-tier limits for one model.

    `rpd` is the number that actually binds. `rpm` sets the request spacing.
    """

    name: str
    rpm: int
    rpd: int
    tpm: int = 250_000
    escalation_only: bool = False

    @property
    def min_interval_s(self) -> float:
        """Seconds between requests to stay under RPM, with a safety margin.

        The 10% is not superstition: the server's minute window and ours are not
        aligned, so pacing exactly at the limit puts requests either side of a
        boundary and trips 429s at a nominally legal rate.
        """
        return (60.0 / self.rpm) * 1.1


# Ordered by preference. The two Flash Lite pools are drained first, in order;
# escalation models are only ever chosen explicitly.
#
# Pinned IDs, never the `-latest` aliases, for two reasons. The alias can be
# repointed by Google without notice, which would silently change both cost and
# output quality. More importantly the whole rotation strategy rests on these
# being *separate* quota pools — and if an alias resolved to the same underlying
# model as a pinned entry, the two rows here would be one pool wearing two names
# and the ledger's arithmetic would be quietly wrong.
#
# Limits are from Nils's own quota page, not from the general documentation:
# free-tier allowances vary per account.
MODELS: tuple[ModelQuota, ...] = (
    ModelQuota("gemini-3.1-flash-lite", rpm=15, rpd=500),
    ModelQuota("gemini-3.5-flash-lite", rpm=15, rpd=500),
    ModelQuota("gemini-3.6-flash", rpm=5, rpd=20, escalation_only=True),
    ModelQuota("gemini-3.5-flash", rpm=5, rpd=20, escalation_only=True),
)

MODELS_BY_NAME = {m.name: m for m in MODELS}


class QuotaExhausted(RuntimeError):
    """Every eligible model is out of daily quota."""


class QuotaLedger:
    """Persistent record of requests made, for RPM and RPD enforcement.

    Deliberately its own SQLite file and its own plain `sqlite3` connection
    rather than a table in the main async database. Quota is spent at the moment
    a request is *attempted*, including attempts that fail or that happen while
    the main transaction later rolls back — so it must not share a transaction
    with anything. A rolled-back batch does not give the quota back.
    """

    def __init__(self, path: str | Path = "quota_ledger.db") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id        INTEGER PRIMARY KEY,
                model     TEXT NOT NULL,
                called_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_requests_model_time ON requests (model, called_at)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> QuotaLedger:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- accounting ---------------------------------------------------------

    def record(self, model: str, *, now: datetime | None = None) -> None:
        """Log a request. Called before the call is made, not after.

        Recording up front means a request that times out or errors still counts
        — because Google counted it. Recording on success would drift under
        exactly the conditions where accuracy matters most.
        """
        now = now or datetime.now(timezone.utc)
        self._conn.execute(
            "INSERT INTO requests (model, called_at) VALUES (?, ?)",
            (model, now.isoformat()),
        )
        self._conn.commit()

    def used_today(self, model: str, *, now: datetime | None = None) -> int:
        """Requests since the most recent quota reset.

        Google's free-tier daily quota resets at midnight **Pacific time**, not
        UTC and not local. Counting from local midnight would free up quota
        hours early and produce 429s that look inexplicable.
        """
        now = now or datetime.now(timezone.utc)
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM requests WHERE model = ? AND called_at >= ?",
            (model, _last_reset(now).isoformat()),
        )
        return cursor.fetchone()[0]

    def remaining_today(self, model: str, *, now: datetime | None = None) -> int:
        quota = MODELS_BY_NAME.get(model)
        if quota is None:
            return 0
        return max(0, quota.rpd - self.used_today(model, now=now))

    def remaining_requests_today(self, *, now: datetime | None = None) -> int:
        """Requests left today across the pools ordinary evaluation may use.

        Escalation models are excluded on purpose. Their 20 RPD exists to give
        a *failed* batch somewhere to go; spending it on routine work would
        empty the reserve before the ads that need it arrive.
        """
        return sum(
            self.remaining_today(model.name, now=now)
            for model in MODELS
            if not model.escalation_only
        )

    def last_call_at(self, model: str) -> datetime | None:
        cursor = self._conn.execute(
            "SELECT MAX(called_at) FROM requests WHERE model = ?", (model,)
        )
        row = cursor.fetchone()[0]
        return datetime.fromisoformat(row) if row else None

    def prune(self, *, older_than_days: int = 7, now: datetime | None = None) -> int:
        """Drop rows no longer needed for any window. Unbounded growth for a
        counter that only ever looks back one day would be silly."""
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=older_than_days)).isoformat()
        cursor = self._conn.execute("DELETE FROM requests WHERE called_at < ?", (cutoff,))
        self._conn.commit()
        return cursor.rowcount

    # -- selection ----------------------------------------------------------

    def choose_model(
        self, *, escalate: bool = False, now: datetime | None = None
    ) -> ModelQuota:
        """Pick the best model with quota left.

        Drains the first Flash Lite pool before touching the second rather than
        balancing across them. Draining in order means a quota problem shows up
        as "pool one is empty" — a legible state — instead of both pools sitting
        half-full with no clear reason why.
        """
        now = now or datetime.now(timezone.utc)
        candidates = [m for m in MODELS if m.escalation_only == escalate]

        for model in candidates:
            if self.remaining_today(model.name, now=now) > 0:
                return model

        raise QuotaExhausted(
            f"All {'escalation' if escalate else 'primary'} models are out of daily "
            f"quota. Used: "
            + ", ".join(
                f"{m.name}={self.used_today(m.name, now=now)}/{m.rpd}" for m in candidates
            )
            + ". Quota resets at midnight US/Pacific."
        )

    def report(self, *, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        return " | ".join(
            f"{m.name}: {self.used_today(m.name, now=now)}/{m.rpd}" for m in MODELS
        )


def _last_reset(now: datetime) -> datetime:
    """Most recent midnight in US/Pacific, as UTC.

    `zoneinfo` handles the DST shift; a fixed -8 offset would be an hour wrong
    for most of the year and silently mis-count quota near the boundary.
    """
    from zoneinfo import ZoneInfo

    pacific = ZoneInfo("US/Pacific")
    local_now = now.astimezone(pacific)
    midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.astimezone(timezone.utc)
