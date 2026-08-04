"""Stage-orchestration tests.

These exist for one class of bug that unit tests structurally cannot reach:
mistakes about *which thread* something runs on. `stage_ingest` drives a
blocking, synchronous sweep on a worker thread while the database writes happen
on the event loop, and getting that boundary wrong does not raise — it hangs.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import pipeline.run as run_module
from pipeline.ingest import IngestStats
from sources.base import AdRecord


def _record(external_id: str) -> AdRecord:
    return AdRecord(
        source="blocket",
        external_id=external_id,
        url=f"https://blocket.se/annons/{external_id}",
        title="RTX 3080 Gaming PC",
        description="Fungerar bra.",
        price_sek=6000,
        category_path=("Datorer",),
        attributes={},
        city="Solna",
        listed_at=None,
        image_count=1,
    )


@pytest.fixture
def stubbed_stage(monkeypatch, session_factory):
    """`stage_ingest` with the network and the source removed, nothing else."""
    written: list[str] = []

    async def fake_persist(_factory, records, **_kw):
        written.extend(r.external_id for r in records)

    import db.crud

    async def no_known(*_a, **_k):
        return set()

    monkeypatch.setattr(db.crud, "known_external_ids", no_known)
    monkeypatch.setattr(run_module, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(run_module, "persist_records", fake_persist)
    class _Source:
        name = "blocket"

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(run_module, "BlocketSource", _Source)
    monkeypatch.setattr(run_module, "pc_search_targets", lambda: [])
    return written


async def test_the_final_batch_does_not_deadlock_the_event_loop(
    stubbed_stage, monkeypatch
):
    """The bug this file was created for.

    The first version drained the leftover records with
    `run_coroutine_threadsafe(...).result()` *after* awaiting the worker thread
    — so on the event loop itself, waiting for a coroutine only that loop could
    schedule. A live run hung at 0% CPU straight after a clean 77/77 sweep and
    held its lock until it was killed. Nothing raised, nothing logged.

    A partial batch is the normal case: any sweep whose record count is not an
    exact multiple of PERSIST_EVERY ends with one.
    """
    def fake_ingest(_source, _targets, *, on_record=None, **_kw):
        for i in range(run_module.PERSIST_EVERY + 3):  # one full batch, then 3
            on_record(_record(str(i)))
        return [], IngestStats()

    monkeypatch.setattr(run_module, "ingest", fake_ingest)

    # Driven from its own thread with its own loop, and joined with a timeout.
    # `asyncio.wait_for` cannot guard this: a deadlocked loop never gets to run
    # the timer that would cancel it, so the buggy version hangs the entire test
    # session instead of failing one test. Verified by reverting the fix — the
    # suite stopped rather than reported.
    result: dict = {}

    def drive() -> None:
        result["ok"] = asyncio.run(run_module.stage_ingest(run_module.RunOptions()))

    worker = threading.Thread(target=drive, daemon=True)
    worker.start()
    worker.join(timeout=15)

    assert not worker.is_alive(), (
        "stage_ingest deadlocked: the final batch must be awaited on the loop, "
        "not handed back to it with run_coroutine_threadsafe().result()"
    )
    assert result["ok"]
    assert len(stubbed_stage) == run_module.PERSIST_EVERY + 3, (
        "every record must be written, including the trailing partial batch"
    )


async def test_records_are_written_during_the_sweep_not_after(
    stubbed_stage, monkeypatch
):
    """The whole point of incremental persistence: work already fetched is in
    the database before the sweep ends, so a power cut costs one batch."""
    seen_mid_sweep: list[int] = []

    def fake_ingest(_source, _targets, *, on_record=None, **_kw):
        for i in range(run_module.PERSIST_EVERY * 2):
            on_record(_record(str(i)))
            if i == run_module.PERSIST_EVERY:
                seen_mid_sweep.append(len(stubbed_stage))
        return [], IngestStats()

    monkeypatch.setattr(run_module, "ingest", fake_ingest)
    await asyncio.wait_for(
        run_module.stage_ingest(run_module.RunOptions()), timeout=10
    )

    assert seen_mid_sweep[0] == run_module.PERSIST_EVERY, (
        "the first batch must already be persisted while the sweep is running"
    )


async def test_a_dry_run_writes_nothing(stubbed_stage, monkeypatch):
    def fake_ingest(_source, _targets, *, on_record=None, **_kw):
        assert on_record is None, "a dry run must not be handed a writer"
        return [_record("1")], IngestStats()

    monkeypatch.setattr(run_module, "ingest", fake_ingest)
    await asyncio.wait_for(
        run_module.stage_ingest(run_module.RunOptions(dry_run=True)), timeout=10
    )

    assert stubbed_stage == []
