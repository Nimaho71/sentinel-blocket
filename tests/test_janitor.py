"""Janitor tests.

The bug being guarded against is a politeness bug, and politeness bugs do not
throw. The old code built a task per active ad and handed all 474 to
`asyncio.gather` — it worked perfectly, and looked like a small denial-of-service
from a single residential IP. So the test asserts on *how* the requests were
made, not just on the result.
"""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import select

from db.models import Ad, StatusEnum
from pipeline.janitor import MIN_INTERVAL_S, check_liveness, enforce_retention
from pipeline.persist import persist_records
from tests.test_persistence import record


class RecordingTransport(httpx.AsyncBaseTransport):
    """Serves scripted responses and records the order requests arrived in."""

    def __init__(self, handler):
        self.handler = handler
        self.in_flight = 0
        self.max_in_flight = 0
        self.urls: list[str] = []

    async def handle_async_request(self, request):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.urls.append(str(request.url))
        try:
            return self.handler(request)
        finally:
            self.in_flight -= 1


@pytest.fixture
def sleeps():
    """Captures the pacing without spending it."""
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    fake_sleep.recorded = recorded
    return fake_sleep


def client_for(handler) -> tuple[httpx.AsyncClient, RecordingTransport]:
    transport = RecordingTransport(handler)
    return httpx.AsyncClient(transport=transport), transport


async def seed_active(session_factory, count: int) -> None:
    await persist_records(session_factory, [record(str(i)) for i in range(count)])


async def test_requests_are_sequential_and_spaced(session_factory, sleeps):
    """Never a burst. This is the actual fix for bug #9."""
    await seed_active(session_factory, 5)
    client, transport = client_for(lambda r: httpx.Response(200))

    async with client:
        stats = await check_liveness(session_factory, client=client, sleep=sleeps)

    assert stats.checked == 5
    assert transport.max_in_flight == 1, "requests must not overlap"
    assert len(sleeps.recorded) == 5
    assert all(s >= MIN_INTERVAL_S for s in sleeps.recorded)


async def test_a_removed_ad_is_marked_sold(session_factory, sleeps):
    await seed_active(session_factory, 1)
    client, _ = client_for(lambda r: httpx.Response(404))

    async with client:
        stats = await check_liveness(session_factory, client=client, sleep=sleeps)

    assert stats.marked_sold == 1
    async with session_factory() as session:
        ad = (await session.execute(select(Ad))).scalars().one()
    assert ad.status is StatusEnum.SOLD


async def test_a_redirect_to_the_homepage_means_removed(session_factory, sleeps):
    """Blocket's actual behaviour for a taken-down listing."""
    await seed_active(session_factory, 1)

    def handler(request):
        if request.url.path != "/":
            return httpx.Response(302, headers={"Location": "https://www.blocket.se/"})
        return httpx.Response(200)

    client, _ = client_for(handler)
    async with client:
        stats = await check_liveness(session_factory, client=client, sleep=sleeps)

    assert stats.marked_sold == 1


async def test_a_network_error_never_marks_an_ad_sold(session_factory, sleeps):
    """A timeout says something about the network, not about the listing.

    Treating "could not reach" as "gone" would delete live ads every time the
    connection wobbled — and because retention purges SOLD ads, that loss is
    permanent.
    """
    await seed_active(session_factory, 2)

    def handler(request):
        raise httpx.ConnectTimeout("timed out")

    client, _ = client_for(handler)
    async with client:
        stats = await check_liveness(session_factory, client=client, sleep=sleeps)

    assert stats.marked_sold == 0
    async with session_factory() as session:
        ads = (await session.execute(select(Ad))).scalars().all()
    assert all(ad.status is StatusEnum.ACTIVE for ad in ads)


async def test_a_server_error_is_inconclusive_not_sold(session_factory, sleeps):
    await seed_active(session_factory, 1)
    client, _ = client_for(lambda r: httpx.Response(503))

    async with client:
        stats = await check_liveness(session_factory, client=client, sleep=sleeps)

    assert stats.marked_sold == 0


async def test_persistent_failures_halt_the_run(session_factory, sleeps):
    """Stop asking. A far end that has stopped answering is not persuaded by
    more requests, and continuing is how a soft signal becomes a hard block."""
    await seed_active(session_factory, 50)

    def handler(request):
        raise httpx.ConnectError("refused")

    client, transport = client_for(handler)
    async with client:
        stats = await check_liveness(session_factory, client=client, sleep=sleeps)

    assert stats.halted_early
    assert len(transport.urls) < 50, "must not work through the whole list"


async def test_unchecked_ads_keep_their_place_in_the_queue(session_factory, sleeps):
    """After an early halt, the skipped ads must be checked *first* next time.

    Stamping every target as checked would send them to the back of an
    oldest-first queue, so a repeatedly-halting run would starve the same ads
    forever.
    """
    await seed_active(session_factory, 20)

    async def timestamps() -> dict[int, object]:
        async with session_factory() as session:
            rows = (await session.execute(select(Ad.id, Ad.last_checked_at))).all()
        return dict(rows)

    before = await timestamps()

    def handler(request):
        raise httpx.ConnectError("refused")

    client, _ = client_for(handler)
    async with client:
        stats = await check_liveness(session_factory, client=client, sleep=sleeps)

    assert stats.halted_early and stats.checked == 0
    assert await timestamps() == before, "no ad was reached, so none may be stamped"


async def test_retention_purges_long_sold_ads(session_factory):
    from datetime import datetime, timedelta, timezone

    await seed_active(session_factory, 2)
    async with session_factory() as session:
        ads = (await session.execute(select(Ad))).scalars().all()
        ads[0].status = StatusEnum.SOLD
        ads[0].last_checked_at = datetime.now(timezone.utc) - timedelta(days=200)
        ads[1].status = StatusEnum.SOLD  # sold, but recently
        await session.commit()

    stats = await enforce_retention(session_factory, sold_retention_days=90)

    assert stats.purged == 1
    async with session_factory() as session:
        remaining = (await session.execute(select(Ad))).scalars().all()
    assert len(remaining) == 1


class TestPublishedAdsAreVerifiedFirst:
    """Oldest-first alone is the fair order and the wrong one.

    Live: 4 083 active ads against a budget of 500 a night is a full pass every
    eight nights, and the corpus grows faster than the janitor retires it. Two
    listings on the published board had already gone — one 404, one marked
    *Såld* — and would have stayed up for another week while the janitor
    re-checked thousands of ads nobody could see.
    """

    async def test_a_published_ad_jumps_the_queue(self, session_factory, sleeps):
        await seed_active(session_factory, 20)
        async with session_factory() as session:
            rows = (await session.execute(
                select(Ad.id, Ad.url).order_by(Ad.last_checked_at.asc())
            )).all()
        # Last in an oldest-first queue, so it would not be reached at limit=3.
        last_id, last_url = rows[-1]

        def handler(request):
            return httpx.Response(200)

        client, transport = client_for(handler)
        async with client:
            await check_liveness(
                session_factory, limit=3, published_ids=[last_id],
                client=client, sleep=sleeps,
            )

        assert len(transport.urls) == 3
        # Compared by URL, not by id: the url carries `external_id`, which is a
        # different number, and an earlier version of this test asserted the two
        # against each other and failed on a correct result.
        assert transport.urls[0] == last_url, (
            "the ad a visitor can see must be checked before the rotation"
        )

    async def test_the_rest_of_the_budget_still_goes_to_the_rotation(
        self, session_factory, sleeps
    ):
        """Prioritising must not become skipping. The 4 000 ads nobody sees
        still need to be retired, or the corpus never converges."""
        await seed_active(session_factory, 20)
        async with session_factory() as session:
            ids = [r[0] for r in (await session.execute(select(Ad.id))).all()]

        def handler(request):
            return httpx.Response(200)

        client, transport = client_for(handler)
        async with client:
            stats = await check_liveness(
                session_factory, limit=5, published_ids=[ids[-1]],
                client=client, sleep=sleeps,
            )

        assert stats.checked == 5
        assert len({u for u in transport.urls}) == 5, "no ad checked twice"

    async def test_published_ads_are_not_checked_twice(self, session_factory, sleeps):
        """A published ad that is *also* the oldest must not consume two slots
        of a budget the whole point of which is that it is scarce."""
        await seed_active(session_factory, 6)
        async with session_factory() as session:
            ids = [r[0] for r in (await session.execute(select(Ad.id))).all()]
        oldest = ids[0]

        def handler(request):
            return httpx.Response(200)

        client, transport = client_for(handler)
        async with client:
            stats = await check_liveness(
                session_factory, limit=4, published_ids=[oldest],
                client=client, sleep=sleeps,
            )

        assert stats.checked == 4
        assert len(set(transport.urls)) == 4

    async def test_no_published_ids_behaves_exactly_as_before(
        self, session_factory, sleeps
    ):
        """A first run, before anything has been published, is not a failure."""
        await seed_active(session_factory, 5)

        def handler(request):
            return httpx.Response(200)

        client, transport = client_for(handler)
        async with client:
            stats = await check_liveness(
                session_factory, limit=3, client=client, sleep=sleeps
            )

        assert stats.checked == 3
        assert len(transport.urls) == 3


class TestSoldAdsAreDetectedNotJustDeletedOnes:
    """A sold listing stays at its URL and answers 200. Only a *deleted* one
    404s, and sellers rarely delete.

    Measured against live data: an ad plainly marked "Såld" returned 200 to a
    HEAD and was therefore treated as still for sale, so it sat on the
    published board indefinitely. The status is in the page body.
    """

    # Quoted from the live page: the state arrives as escaped JSON in a script
    # tag, so the flag really does look like this.
    SOLD_BODY = r'...on\":\"\"}],\"disposed\":true,\"disposedText\":\"Såld\",\"adVi...'
    LIVE_BODY = r'...on\":\"\"}],\"disposed\":false,\"adViewTypeLabel\":\"...'

    async def test_a_sold_ad_is_marked_sold(self, session_factory, sleeps):
        await seed_active(session_factory, 1)

        def handler(request):
            return httpx.Response(200, text=self.SOLD_BODY)

        client, _ = client_for(handler)
        async with client:
            stats = await check_liveness(session_factory, client=client, sleep=sleeps)

        assert stats.marked_sold == 1
        assert stats.inconclusive == 0

    async def test_a_live_ad_is_left_alone(self, session_factory, sleeps):
        await seed_active(session_factory, 1)

        def handler(request):
            return httpx.Response(200, text=self.LIVE_BODY)

        client, _ = client_for(handler)
        async with client:
            stats = await check_liveness(session_factory, client=client, sleep=sleeps)

        assert stats.marked_sold == 0
        assert stats.inconclusive == 0

    async def test_an_unrecognisable_page_is_inconclusive_never_sold(
        self, session_factory, sleeps
    ):
        """If Blocket restyles the page the marker disappears. Guessing "still
        live" fossilises the board; guessing "sold" deletes real listings. It
        must be neither, and it must be counted."""
        await seed_active(session_factory, 3)

        def handler(request):
            return httpx.Response(200, text="<html>totally redesigned</html>")

        client, _ = client_for(handler)
        async with client:
            stats = await check_liveness(session_factory, client=client, sleep=sleeps)

        assert stats.marked_sold == 0
        assert stats.inconclusive == 3
        assert "3 status unreadable" in stats.summary()

    async def test_a_deleted_ad_is_still_caught(self, session_factory, sleeps):
        """The old 404 path must keep working — this replaces nothing."""
        await seed_active(session_factory, 1)

        def handler(request):
            return httpx.Response(404)

        client, _ = client_for(handler)
        async with client:
            stats = await check_liveness(session_factory, client=client, sleep=sleeps)

        assert stats.marked_sold == 1
