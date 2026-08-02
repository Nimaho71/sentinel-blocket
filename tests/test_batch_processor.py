"""Evaluation-stage tests.

This file exists mainly for one property: **a failed batch must leave a trace.**
The production database ended up with 474 ads and 120 evaluations because a
failure looked exactly like an empty result and the batch was dropped with a
bare `continue`. Nothing here mocks that away — the tests drive the real code
against a real database and assert the failure is recorded, counted, bounded,
and cleared on recovery.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

import ai.batch_processor as batch_processor
from ai.gemini_client import MAX_DESCRIPTION_CHARS, BatchResult, GeminiError
from ai.schemas import AdVerdict, ComponentVerdict
from db.crud import ads_needing_evaluation, content_hash
from db.models import Ad, AiEvaluation, AiFailure, ComponentTypeEnum, ItemTypeEnum


@pytest.fixture
def stub_gemini(monkeypatch):
    """Replace the network call with a scripted one.

    Records the payloads it was sent so tests can assert on what would have
    crossed the wire — that is where the PII scrub will be checked in Phase 3.
    """

    class Stub:
        def __init__(self):
            self.calls: list[list[dict]] = []
            self.behaviour = None

        def returns(self, fn):
            self.behaviour = fn

        async def __call__(self, payload, *, model=None, escalate=False):
            self.calls.append(list(payload))
            return BatchResult(
                verdicts=self.behaviour(list(payload)), model=model or "stub-model"
            )

    stub = Stub()
    monkeypatch.setattr(batch_processor, "analyze_ads_batch", stub)
    return stub


def verdict_for(entry: dict, **overrides) -> AdVerdict:
    kwargs = {
        "ad_id": entry["ad_id"],
        "item_type": ItemTypeEnum.FULL_BUILD,
        "confidence": 0.9,
        "risk_flags": ("ej testad",),
        "reasoning": "Looks fine.",
        "components": (
            ComponentVerdict(ComponentTypeEnum.GPU, "RTX 3080 Asus TUF"),
        ),
    }
    kwargs.update(overrides)
    return AdVerdict(**kwargs)


async def seed(session_factory, ad_factory, count: int) -> list[int]:
    async with session_factory() as session:
        ads = [ad_factory(str(i)) for i in range(count)]
        session.add_all(ads)
        await session.commit()
        return [ad.id for ad in ads]


async def test_successful_batch_persists_verdict_and_components(
    session_factory, ad_factory, stub_gemini
):
    await seed(session_factory, ad_factory, 2)
    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])

    stats = await batch_processor.process_unscored_ads(session_factory, batch_size=10)

    assert (stats.considered, stats.evaluated, stats.failed) == (2, 2, 0)

    async with session_factory() as session:
        evaluations = (await session.execute(select(AiEvaluation))).scalars().all()
        assert len(evaluations) == 2
        assert all(e.confidence == 0.9 for e in evaluations)
        assert all(e.flags == ["ej testad"] for e in evaluations)


async def test_canonical_name_is_written_for_the_market_index(
    session_factory, ad_factory, stub_gemini
):
    """Without this the index has nothing to join on and every median is empty.

    The model returns "RTX 3080 Asus TUF"; the deterministic path returns
    "RTX 3080". Both must land on the same key, or one card becomes two index
    entries and each reports half the evidence it actually has.
    """
    await seed(session_factory, ad_factory, 1)
    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])

    await batch_processor.process_unscored_ads(session_factory)

    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
        await session.refresh(ad, ["extracted_components"])
        # Board partner and cooler name are marketing; the silicon sets the price.
        assert ad.extracted_components[0].canonical_name == "rtx 3080"
        assert ad.extracted_components[0].component_name == "RTX 3080 Asus TUF"


async def test_a_failed_batch_is_recorded_not_silently_dropped(
    session_factory, ad_factory, stub_gemini
):
    """The core regression. Previously: `continue`, and the ads vanished."""
    await seed(session_factory, ad_factory, 3)

    def boom(payload):
        raise GeminiError("503 Service Unavailable")

    stub_gemini.returns(boom)

    stats = await batch_processor.process_unscored_ads(session_factory, batch_size=3)

    assert stats.evaluated == 0
    assert stats.failed == 3
    # The count reaching the caller is what makes a stall visible at all.
    assert "3 failed" in stats.summary()

    async with session_factory() as session:
        failures = (await session.execute(select(AiFailure))).scalars().all()
        assert len(failures) == 3
        assert all(f.attempts == 1 for f in failures)
        assert all("503" in f.last_error for f in failures)


async def test_repeated_failures_stop_being_retried(
    session_factory, ad_factory, stub_gemini
):
    """Bounded retries.

    An ad that fails forever must not consume a slot on every future run — that
    turns one poisoned listing into a permanent tax on throughput.
    """
    await seed(session_factory, ad_factory, 1)

    def boom(payload):
        raise GeminiError("400 invalid content")

    stub_gemini.returns(boom)

    for expected_attempts in (1, 2, 3):
        stats = await batch_processor.process_unscored_ads(session_factory)
        assert stats.considered == 1, "should still be eligible"
        async with session_factory() as session:
            failure = (await session.execute(select(AiFailure))).scalars().one()
            assert failure.attempts == expected_attempts

    # Fourth run: three strikes reached, the ad drops out of the queue.
    stats = await batch_processor.process_unscored_ads(session_factory)
    assert stats.considered == 0


async def test_recovery_clears_the_dead_letter_row(
    session_factory, ad_factory, stub_gemini
):
    """A transient outage must not leave a permanent mark."""
    await seed(session_factory, ad_factory, 1)

    stub_gemini.returns(lambda payload: (_ for _ in ()).throw(GeminiError("503")))
    await batch_processor.process_unscored_ads(session_factory)

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(session_factory)

    assert stats.evaluated == 1
    async with session_factory() as session:
        assert (await session.execute(select(AiFailure))).scalars().first() is None


async def test_an_ad_omitted_from_the_reply_counts_as_failed(
    session_factory, ad_factory, stub_gemini
):
    """A partial answer is a partial failure, and must be reported as one.

    This is the original bug in miniature: the model answers for four of five
    ads, four rows appear, and nothing anywhere mentions the fifth.
    """
    await seed(session_factory, ad_factory, 3)
    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload[:2]])

    stats = await batch_processor.process_unscored_ads(session_factory, batch_size=3)

    assert stats.evaluated == 2
    assert stats.failed == 1

    async with session_factory() as session:
        failure = (await session.execute(select(AiFailure))).scalars().one()
        assert "omitted" in failure.last_error


async def test_one_bad_batch_does_not_lose_the_good_ones(
    session_factory, ad_factory, stub_gemini
):
    """Batch isolation: the reason each batch owns its own transaction."""
    await seed(session_factory, ad_factory, 4)

    def fail_only_the_first_batch(payload):
        if not stub_gemini.calls[:-1]:  # this call is already recorded
            raise GeminiError("503")
        return [verdict_for(e) for e in payload]

    stub_gemini.returns(fail_only_the_first_batch)

    stats = await batch_processor.process_unscored_ads(session_factory, batch_size=2)

    assert stats.evaluated == 2
    assert stats.failed == 2


async def test_evaluated_ads_are_not_re_sent(session_factory, ad_factory, stub_gemini):
    """Quota is the binding constraint; re-analysing settled ads wastes it."""
    await seed(session_factory, ad_factory, 2)
    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])

    await batch_processor.process_unscored_ads(session_factory)
    stats = await batch_processor.process_unscored_ads(session_factory)

    assert stats.considered == 0
    assert len(stub_gemini.calls) == 1


async def test_a_price_change_makes_an_ad_eligible_again(
    session_factory, ad_factory, stub_gemini
):
    """The old `~Ad.ai_evaluation.has()` check could not express this.

    A seller cutting the price is the single most interesting event this whole
    system watches for, and it was invisible: an evaluation row existed, so the
    ad was never looked at again.
    """
    [ad_id] = await seed(session_factory, ad_factory, 1)
    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    await batch_processor.process_unscored_ads(session_factory)

    async with session_factory() as session:
        ad = await session.get(Ad, ad_id)
        ad.price_sek = 4000
        ad.content_hash = content_hash(ad.title, ad.raw_description, 4000)
        await session.commit()

    async with session_factory() as session:
        assert len(await ads_needing_evaluation(session)) == 1

    stats = await batch_processor.process_unscored_ads(session_factory)
    assert stats.evaluated == 1


async def test_re_evaluation_replaces_components_rather_than_appending(
    session_factory, ad_factory, stub_gemini
):
    """Otherwise a re-analysed ad is double-counted by the market index."""
    [ad_id] = await seed(session_factory, ad_factory, 1)
    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    await batch_processor.process_unscored_ads(session_factory)

    async with session_factory() as session:
        ad = await session.get(Ad, ad_id)
        ad.content_hash = "changed"
        await session.commit()

    await batch_processor.process_unscored_ads(session_factory)

    async with session_factory() as session:
        ad = await session.get(Ad, ad_id)
        await session.refresh(ad, ["extracted_components"])
        assert len(ad.extracted_components) == 1


async def test_an_unambiguous_ad_never_reaches_the_model(
    session_factory, ad_factory, stub_gemini
):
    """~35% of the real corpus is settled this way — deterministically, and for
    free. The point is not the saving so much as that "RTX 3080" parses to the
    same string on every run, which a stochastic model does not guarantee."""
    async with session_factory() as session:
        session.add(ad_factory("1", title="ASUS Dual RTX 3060 Ti OC V2 (8GB)",
                               description="Nyskick, kvitto finns."))
        await session.commit()

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(session_factory)

    assert stats.evaluated == 1
    assert stats.resolved_locally == 1
    assert stub_gemini.calls == [], "no model call should have been made"

    async with session_factory() as session:
        evaluation = (await session.execute(select(AiEvaluation))).scalars().one()
    assert evaluation.model == batch_processor.DETERMINISTIC_MODEL
    assert evaluation.item_type is ItemTypeEnum.SINGLE_COMPONENT


async def test_local_verdicts_are_marked_less_confident_than_reviewed_ones(
    session_factory, ad_factory, stub_gemini
):
    """Nothing read these listings. Ranking multiplies delta by confidence, so
    they must not outrank an ad a model actually looked at and approved."""
    async with session_factory() as session:
        session.add(ad_factory("1", title="ASUS Dual RTX 3060 Ti OC V2 (8GB)",
                               description="Nyskick, kvitto finns."))
        await session.commit()

    await batch_processor.process_unscored_ads(session_factory)

    async with session_factory() as session:
        evaluation = (await session.execute(select(AiEvaluation))).scalars().one()
    assert evaluation.confidence == batch_processor.DETERMINISTIC_CONFIDENCE
    assert evaluation.confidence < 1.0


async def test_an_ad_with_risk_language_still_goes_to_the_model(
    session_factory, ad_factory, stub_gemini
):
    """Its specs parse perfectly, which is exactly the trap. Judging "ej testad"
    is the thing regexes are bad at and models are good at."""
    async with session_factory() as session:
        session.add(ad_factory("1", title="ASUS Dual RTX 3060 Ti OC V2 (8GB)",
                               description="Ej testad, säljes i befintligt skick."))
        await session.commit()

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(session_factory)

    assert stats.resolved_locally == 0
    assert len(stub_gemini.calls) == 1


async def test_local_resolution_can_be_turned_off(session_factory, ad_factory, stub_gemini):
    """Keeps the model path testable in isolation, and gives a way to A/B the
    two if the deterministic rules are ever suspected of being too lax."""
    async with session_factory() as session:
        session.add(ad_factory("1", title="ASUS Dual RTX 3060 Ti OC V2 (8GB)",
                               description="Nyskick."))
        await session.commit()

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(
        session_factory, use_deterministic=False
    )

    assert stats.resolved_locally == 0
    assert len(stub_gemini.calls) == 1


async def test_descriptions_are_truncated_before_being_sent(
    session_factory, ad_factory, stub_gemini
):
    """TPM, not RPM, is the real ceiling. Specs live at the top of an ad."""
    async with session_factory() as session:
        ad = ad_factory("1", description="x" * 9000)
        session.add(ad)
        await session.commit()

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    await batch_processor.process_unscored_ads(session_factory)

    sent = stub_gemini.calls[0][0]
    assert len(sent["description"]) == MAX_DESCRIPTION_CHARS


# --------------------------------------------------------------- quota sizing
#
# The stage used to take a fixed 200 ads per run. On a measured production run
# that spent 18 of 1 000 available requests while 2 235 scraped ads sat
# unevaluated — and because the price index is built from `canonical_name`,
# which only evaluation writes, those ads were invisible to every valuation on
# the site. The cap now comes from the quota, and these tests pin the parts of
# that which are easy to get subtly wrong.


@pytest.fixture
def quota(monkeypatch):
    """Pin the day's remaining model quota to an exact number of requests."""

    class FakeLedger:
        def __init__(self):
            self.requests = 1_000

        def remaining_requests_today(self, *, now=None):
            return self.requests

    ledger = FakeLedger()
    monkeypatch.setattr(batch_processor, "affordable_today",
                        lambda batch_size=batch_processor.DEFAULT_BATCH_SIZE: (
                            int(ledger.requests * (1 - batch_processor.QUOTA_RESERVE))
                            * batch_size
                        ))
    return ledger


async def test_the_whole_backlog_is_taken_not_a_fixed_two_hundred(
    session_factory, ad_factory, stub_gemini, quota
):
    """The regression that matters. 250 > the old hardcoded 200, so this test
    fails against the previous default while passing against a quota that can
    comfortably afford all of them."""
    await seed(session_factory, ad_factory, 250)

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(session_factory)

    assert stats.evaluated == 250
    assert stats.deferred == 0

    async with session_factory() as session:
        remaining = await ads_needing_evaluation(session, limit=None)
    assert remaining == [], "nothing should be left waiting when quota allows"


async def test_free_local_resolution_is_never_rationed_by_quota(
    session_factory, ad_factory, stub_gemini, quota
):
    """The subtle half of this change.

    The deterministic pass is regex over text already on disk — it spends no
    quota whatsoever. Capping the database *fetch* by quota, as a naive fix
    would, rations the free work along with the paid work. With zero requests
    left, every locally-resolvable ad must still be resolved.
    """
    quota.requests = 0
    async with session_factory() as session:
        session.add_all([
            ad_factory(str(i), title=f"ASUS Dual RTX 3060 Ti OC V{i} (8GB)",
                       description="Nyskick, kvitto finns.")
            for i in range(30)
        ])
        await session.commit()

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(session_factory)

    assert stats.resolved_locally == 30
    assert stats.evaluated == 30
    assert stub_gemini.calls == [], "no quota, and none was needed"
    assert stats.deferred == 0


async def test_ads_beyond_the_quota_are_deferred_and_counted(
    session_factory, ad_factory, stub_gemini, quota
):
    """A backlog is not a failure, and must not be reported as one — nor
    hidden. `deferred` is the number that makes a partial run legible."""
    quota.requests = 5  # 5 * 0.8 = 4 batches = 40 ads
    await seed(session_factory, ad_factory, 60)

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(session_factory)

    assert stats.evaluated == 40
    assert stats.deferred == 20
    assert stats.failed == 0, "deferred work is not failed work"
    assert "20 deferred" in stats.summary()

    async with session_factory() as session:
        remaining = await ads_needing_evaluation(session, limit=None)
    assert len(remaining) == 20, "the rest stay eligible for the next run"


async def test_exhausted_quota_defers_everything_rather_than_erroring(
    session_factory, ad_factory, stub_gemini, quota
):
    quota.requests = 0
    await seed(session_factory, ad_factory, 12)

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(session_factory)

    assert stats.deferred == 12
    assert stats.failed == 0
    assert stub_gemini.calls == []


async def test_an_explicit_limit_still_overrides_the_quota_sizing(
    session_factory, ad_factory, stub_gemini, quota
):
    """`--limit` remains a testing cap. It has to keep working, or a quick
    `--limit 5` run would quietly spend the whole day's quota."""
    await seed(session_factory, ad_factory, 40)

    stub_gemini.returns(lambda payload: [verdict_for(e) for e in payload])
    stats = await batch_processor.process_unscored_ads(session_factory, limit=10)

    assert stats.evaluated == 10
    assert len(stub_gemini.calls) == 1
