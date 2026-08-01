"""Persistence stage: `AdRecord`s from `ingest` into the database.

### Why this module exists rather than a `session.add` inside the ingest loop

`AsyncSession` is not concurrency-safe. The old `orchestrator.py` opened one
session and handed it to five tasks under `asyncio.gather`, each committing
independently — two coroutines interleaving on one connection, with whichever
commit landed first flushing the others' half-built state. It mostly worked,
which is the worst property a bug like that can have.

The rule here is **one session, one owner, one transaction**. Records are
written in batches from a single coroutine. If concurrency is ever added, each
task takes its own session from the factory; sessions are never shared.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from sqlalchemy.ext.asyncio import async_sessionmaker

from db.crud import upsert_ad
from sources.base import AdRecord

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50


@dataclass
class PersistStats:
    written: int = 0
    batches_failed: int = 0

    def summary(self) -> str:
        text = f"persisted {self.written} ads"
        if self.batches_failed:
            text += f" ({self.batches_failed} batches rolled back)"
        return text


def _chunks(items: Sequence[AdRecord], size: int) -> Iterable[Sequence[AdRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def persist_records(
    session_factory: async_sessionmaker,
    records: Sequence[AdRecord],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PersistStats:
    """Write records in batches, one transaction per batch.

    Batching rather than committing per ad: a commit per ad on SQLite means an
    fsync per ad. Batching rather than one giant transaction: a single bad
    record would otherwise roll back the entire sweep.
    """
    stats = PersistStats()

    for batch in _chunks(records, batch_size):
        async with session_factory() as session:
            try:
                for record in batch:
                    await upsert_ad(session, record)
                await session.commit()
                stats.written += len(batch)
            except Exception as exc:  # noqa: BLE001 — batch isolation is the point
                await session.rollback()
                stats.batches_failed += 1
                # Logged loudly and counted. A dropped batch that nothing
                # reports is how the pipeline lost track of 354 ads.
                logger.error("Batch of %d ads failed to persist: %s", len(batch), exc)

    logger.info(stats.summary())
    return stats
