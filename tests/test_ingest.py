"""Ingest-stage tests.

The behaviours worth protecting here are the ones whose absence caused the
original pipeline to stall silently: cheap pre-filtering before expensive calls,
skipping known ads, and halting rather than looping when the source refuses us.
"""

from __future__ import annotations

import pytest

from pipeline.ingest import (
    IngestStats,
    SearchTarget,
    ingest,
    is_candidate,
    pc_search_targets,
    rotate_targets,
)
from sources.base import AdRecord, AdSummary, CircuitOpen, SourceError


def summary(ad_id: str, price: int = 3000, title: str = "RTX 3060") -> AdSummary:
    return AdSummary(
        source="test",
        external_id=ad_id,
        url=f"https://example.test/{ad_id}",
        title=title,
        price_sek=price,
    )


def record_for(s: AdSummary) -> AdRecord:
    return AdRecord(
        source=s.source,
        external_id=s.external_id,
        url=s.url,
        title=s.title,
        description="a description",
        price_sek=s.price_sek,
    )


class FakeSource:
    """Scriptable source. Counts detail calls so we can assert on cost."""

    name = "fake"

    def __init__(self, pages=None, search_error=None, detail_error=None, missing=()):
        self.pages = pages if pages is not None else {}
        self.search_error = search_error
        self.detail_error = detail_error
        self.missing = set(missing)
        self.detail_calls: list[str] = []
        self.search_calls: list[tuple[str, int]] = []
        self.price_bounds: list[tuple[int | None, int | None]] = []

    def search(self, query, *, page=1, price_from=None, price_to=None):
        self.search_calls.append((query, page))
        self.price_bounds.append((price_from, price_to))
        if self.search_error:
            raise self.search_error
        return self.pages.get((query, page), [])

    def fetch_detail(self, s):
        self.detail_calls.append(s.external_id)
        if self.detail_error:
            raise self.detail_error
        if s.external_id in self.missing:
            return None
        return record_for(s)


# --------------------------------------------------------------------- filtering


@pytest.mark.parametrize(
    "price,expected",
    [(299, False), (300, True), (3000, True), (40_000, True), (40_001, False)],
)
def test_price_bounds(price, expected):
    assert is_candidate(summary("1", price=price)) is expected


def test_untitled_ads_are_rejected():
    assert not is_candidate(summary("1", title=""))


# ------------------------------------------------------------------- cost control


def test_detail_is_only_fetched_for_candidates():
    """The whole point of pre-filtering: don't pay for ads we'd discard."""
    src = FakeSource(pages={("q", 1): [
        summary("cheap", price=50),
        summary("good", price=3000),
        summary("expensive", price=99_000),
    ]})

    records, stats = ingest(src, [SearchTarget("q")])

    assert src.detail_calls == ["good"], "only the in-range ad should cost a request"
    assert stats.filtered_out == 2
    assert [r.external_id for r in records] == ["good"]


def test_known_ads_are_skipped_before_detail():
    src = FakeSource(pages={("q", 1): [summary("old"), summary("new")]})

    records, stats = ingest(src, [SearchTarget("q")], is_known=lambda i: i == "old")

    assert src.detail_calls == ["new"]
    assert stats.already_known == 1
    assert [r.external_id for r in records] == ["new"]


def test_removed_listings_are_counted_not_returned():
    src = FakeSource(pages={("q", 1): [summary("gone"), summary("live")]}, missing={"gone"})

    records, stats = ingest(src, [SearchTarget("q")])

    assert [r.external_id for r in records] == ["live"]
    assert stats.detail_missing == 1


def test_multiple_pages_are_swept():
    src = FakeSource(pages={("q", 1): [summary("a")], ("q", 2): [summary("b")]})

    records, _ = ingest(src, [SearchTarget("q", max_pages=2)])

    assert [r.external_id for r in records] == ["a", "b"]


# ------------------------------------------------------------------ failure modes


def test_open_circuit_halts_the_run_immediately():
    """A blocked run must stop, not grind through every remaining target."""
    src = FakeSource(search_error=CircuitOpen("cooling down"))

    records, stats = ingest(src, [SearchTarget("a"), SearchTarget("b"), SearchTarget("c")])

    assert stats.halted_early is True
    assert records == []
    assert len(src.search_calls) == 1, "must not keep trying after the breaker opens"


def test_circuit_open_during_detail_halts_but_keeps_partial_progress():
    """Work already done is not thrown away when the breaker trips mid-run."""

    class HalfwaySource(FakeSource):
        def fetch_detail(self, s):
            self.detail_calls.append(s.external_id)
            if len(self.detail_calls) > 1:
                raise CircuitOpen("tripped")
            return record_for(s)

    src = HalfwaySource(pages={("q", 1): [summary("a"), summary("b"), summary("c")]})

    records, stats = ingest(src, [SearchTarget("q")])

    assert [r.external_id for r in records] == ["a"], "first record should be retained"
    assert stats.halted_early is True


def test_ordinary_search_failure_continues_to_next_target():
    """One bad query shouldn't abort an otherwise healthy sweep."""

    class FlakySource(FakeSource):
        def search(self, query, *, page=1, price_from=None, price_to=None):
            self.search_calls.append((query, page))
            if query == "bad":
                raise SourceError("upstream hiccup")
            return [summary("ok")]

    src = FlakySource()

    records, stats = ingest(src, [SearchTarget("bad"), SearchTarget("good")])

    assert stats.failures == 1
    assert [r.external_id for r in records] == ["ok"]


def test_detail_failure_does_not_lose_other_ads():
    class OneBadDetail(FakeSource):
        def fetch_detail(self, s):
            self.detail_calls.append(s.external_id)
            if s.external_id == "bad":
                raise SourceError("detail blew up")
            return record_for(s)

    src = OneBadDetail(pages={("q", 1): [summary("bad"), summary("fine")]})

    records, stats = ingest(src, [SearchTarget("q")])

    assert [r.external_id for r in records] == ["fine"]
    assert stats.failures == 1


# ----------------------------------------------------------------- incremental IO


def test_on_record_fires_as_records_arrive():
    """Lets the caller persist incrementally and survive a mid-run halt."""
    src = FakeSource(pages={("q", 1): [summary("a"), summary("b")]})
    seen: list[str] = []

    ingest(src, [SearchTarget("q")], on_record=lambda r: seen.append(r.external_id))

    assert seen == ["a", "b"]


# ------------------------------------------------------------------------ targets


def test_default_targets_are_mostly_component_queries():
    """Index quality depends on component sample counts, so they should dominate."""
    targets = pc_search_targets()
    components = [t for t in targets if t.component_only]

    assert len(components) > len(targets) - len(components)
    assert len({t.query for t in targets}) == len(targets), "no duplicate queries"


def test_component_queries_are_paged_deeply_enough_to_feed_the_index():
    """The sweep that starved the market index.

    Component queries ran one page deep while build queries ran two, which is
    backwards: a build query finds one machine to look at, a component query is
    a price sample. The result was 214 observations across 95 distinct parts —
    71 of them with exactly one observation, only 11 clearing
    `MIN_CONFIDENT_SAMPLES`. Every delta the site published rested on that.
    """
    targets = pc_search_targets()
    components = [t for t in targets if t.component_only]

    assert all(t.max_pages >= 8 for t in components), "component ceiling too low"
    # A ceiling rather than a quota: paging stops on a short page, so a thin
    # query costs one request no matter how high this is set. What it must not
    # do is truncate the broad queries, which were still returning full pages
    # at page 4 under the old fixed depth.
    assert all(t.max_pages >= 8 for t in targets if not t.component_only)


def test_the_sweep_covers_the_used_market_not_the_current_one():
    """Model coverage, checked as a property rather than as a fixed list.

    The corpus is a *used* market: cards Nvidia stopped selling in 2020 are the
    median listing, not obsolete stock. An earlier sweep named only 30- and
    40-series parts and no i9 or Ryzen 9 at all, so the index had nothing to say
    about most of what is actually for sale.
    """
    queries = {t.query.lower() for t in pc_search_targets()}

    for expected in ("rtx 2060", "gtx 1660", "i9 13900k", "ryzen 9 5900x"):
        assert expected in queries, f"{expected} missing from the sweep"
    # Intel sockets were absent entirely while AM4 and AM5 were both swept.
    assert any("lga" in q for q in queries)


def test_stats_summary_flags_early_halt():
    assert "HALTED EARLY" in IngestStats(halted_early=True).summary()
    assert "HALTED EARLY" not in IngestStats().summary()


def test_price_bounds_are_pushed_to_the_source():
    """Server-side filtering means a page returns more usable ads per request.

    Measured against the live API: `price_from=3000` still returned a full 53
    results, but the cheapest was 3 100 kr instead of 900 kr. Same request, more
    in-budget ads — better for us and lighter on Blocket at once.
    """
    source = FakeSource({("gpu", 1): [summary("1")]})

    ingest(source, [SearchTarget("gpu")], min_price=1000, max_price=9000)

    assert source.price_bounds == [(1000, 9000)]


def test_out_of_budget_ads_are_still_filtered_locally():
    """The server-side filter must not be trusted on its own.

    Probing the live endpoint turned up the dangerous failure mode: `price_max`
    — a completely plausible parameter name — is *silently ignored* and returns
    everything with a 200. An unsupported filter on an unofficial API fails
    open, so a source that quietly stops filtering must not become a pipeline
    that quietly stops filtering.
    """
    ignores_bounds = FakeSource(
        {("gpu", 1): [summary("cheap", price=50), summary("ok", price=5000),
                      summary("silly", price=99_000)]}
    )

    records, stats = ingest(
        ignores_bounds, [SearchTarget("gpu")], min_price=1000, max_price=9000
    )

    assert stats.filtered_out == 2
    assert ignores_bounds.detail_calls == ["ok"]


# --------------------------------------------------------------- time budget


class FakeClock:
    """A clock that only advances when a *request* is made.

    Reading it is free, which is the property that matters: the real limiter
    spends time on requests, not on checking whether the budget is gone. An
    earlier version of this fake advanced on every read, so the deadline checks
    consumed their own budget and the sweep issued zero requests — the test
    failing for a reason that could never happen in production.

    Deliberately not `time.monotonic` with real sleeps: the assertion is about
    which requests were issued, and a test that takes real minutes to prove a
    time budget works is a test nobody runs.
    """

    def __init__(self, step: float = 1.0):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += self.step


class TimedSource(FakeSource):
    """A source whose requests take time, like the rate-limited one does."""

    def __init__(self, clock: FakeClock, **kwargs):
        super().__init__(**kwargs)
        self.clock = clock

    def search(self, query, *, page=1, price_from=None, price_to=None):
        self.clock.tick()
        return super().search(
            query, page=page, price_from=price_from, price_to=price_to
        )

    def fetch_detail(self, s):
        self.clock.tick()
        return super().fetch_detail(s)


class TestTimeBudget:
    """A sweep bounded by wall-clock rather than by how much work is left.

    Time is the only unit that can be predicted. A request count cannot be
    turned into a duration without knowing how many ads will turn out to be new
    — which is precisely the unknown. "Stop after ninety minutes" is a promise
    the scheduler can keep.
    """

    def test_a_deadline_stops_the_sweep(self):
        clock = FakeClock(step=1.0)
        src = TimedSource(clock, pages={
            ("a", 1): [], ("b", 1): [], ("c", 1): [], ("d", 1): [],
        })
        targets = [SearchTarget(q) for q in "abcd"]

        _, stats = ingest(src, targets, max_duration_s=2.0, clock=clock)

        assert stats.stopped_at_deadline
        assert not stats.is_complete_sweep
        assert len(src.search_calls) == 2, "kept searching past the deadline"
        assert "OUT OF TIME" in stats.summary()

    def test_an_unbounded_run_sweeps_everything(self):
        """The control: without a budget nothing is skipped."""
        clock = FakeClock(step=1000.0)
        src = TimedSource(clock, pages={
            ("a", 1): [summary("a1")],
            ("b", 1): [summary("b1")],
        })
        targets = [SearchTarget("a"), SearchTarget("b")]

        records, stats = ingest(src, targets, clock=clock)

        assert not stats.stopped_at_deadline
        assert stats.is_complete_sweep
        assert stats.targets_swept == stats.targets_total == 2
        assert len(records) == 2

    def test_the_deadline_never_truncates_a_record(self):
        """The budget decides whether to *start* work, never interrupts it.

        A run that stopped mid-fetch would persist half a listing, which is a
        worse outcome than overshooting the budget by one request.
        """
        clock = FakeClock(step=1.0)
        src = TimedSource(
            clock, pages={("a", 1): [summary("a1"), summary("a2"), summary("a3")]}
        )
        saved: list[str] = []

        records, stats = ingest(
            src,
            [SearchTarget("a")],
            on_record=lambda r: saved.append(r.external_id),
            max_duration_s=3.0,
            clock=clock,
        )

        assert stats.stopped_at_deadline
        # Whatever was fetched is whole, and every fetch produced a record.
        assert saved == [r.external_id for r in records]
        assert len(src.detail_calls) == len(records)
        assert 0 < len(records) < 3, "should have fetched some but not all"

    def test_a_partial_sweep_is_not_reported_as_complete(self):
        """`is_complete_sweep` is the honest-reporting half of the budget.

        A time-capped run that claimed success would look exactly like a full
        one, which is how the index starved in the first place: every night
        "worked" and the tail of the list was never reached.
        """
        clock = FakeClock(step=1.0)
        src = TimedSource(clock, pages={("a", 1): [], ("b", 1): []})
        _, stats = ingest(
            src,
            [SearchTarget("a"), SearchTarget("b")],
            max_duration_s=1.0,
            clock=clock,
        )
        assert stats.targets_swept < stats.targets_total
        assert not stats.is_complete_sweep


class TestRotation:
    """Without rotation a time-capped sweep starves the end of the list.

    This is the non-obvious half of the time budget. `ingest` walks targets in
    order and stops when the budget is gone, so an unrotated capped run sweeps
    the same prefix every night — "grafikkort", the Intel sockets and the PSU
    queries would sit permanently unswept while every run reported success.
    """

    def test_rotation_moves_the_starting_point(self):
        targets = [SearchTarget("a"), SearchTarget("b"), SearchTarget("c")]
        assert [t.query for t in rotate_targets(targets, 0)] == ["a", "b", "c"]
        assert [t.query for t in rotate_targets(targets, 1)] == ["b", "c", "a"]
        assert [t.query for t in rotate_targets(targets, 2)] == ["c", "a", "b"]

    def test_rotation_wraps_rather_than_dropping(self):
        """Every query survives every rotation; only the order changes."""
        targets = [SearchTarget(q) for q in "abcde"]
        for offset in range(12):
            rotated = rotate_targets(targets, offset)
            assert len(rotated) == len(targets)
            assert {t.query for t in rotated} == {t.query for t in targets}

    def test_rotation_is_safe_on_an_empty_list(self):
        assert rotate_targets([], 3) == []

    def test_consecutive_capped_runs_reach_the_tail(self):
        """The property that makes a bounded sweep honest over time.

        Each run covers two of five queries. Rotating by the number already
        swept means the whole list is covered in three runs, rather than the
        first two queries being swept three times and the last three never.
        """
        targets = [SearchTarget(q) for q in "abcde"]
        pages = {(q, 1): [] for q in "abcde"}
        swept: set[str] = set()

        offset = 0
        for _ in range(3):
            clock = FakeClock(step=1.0)
            src = TimedSource(clock, pages=pages)
            _, stats = ingest(
                src, targets, max_duration_s=2.0, rotate=offset, clock=clock
            )
            swept |= {q for q, _ in src.search_calls}
            offset += stats.targets_swept

        assert swept == set("abcde")

    def test_without_rotation_the_tail_is_never_reached(self):
        """The control, and the bug rotation exists to prevent.

        Three identical capped runs at offset 0 sweep the same two queries three
        times over. Every run reports success; three of five queries have never
        been issued.
        """
        targets = [SearchTarget(q) for q in "abcde"]
        pages = {(q, 1): [] for q in "abcde"}
        swept: set[str] = set()

        for _ in range(3):
            clock = FakeClock(step=1.0)
            src = TimedSource(clock, pages=pages)
            ingest(src, targets, max_duration_s=2.0, rotate=0, clock=clock)
            swept |= {q for q, _ in src.search_calls}

        assert swept == {"a", "b"}
        assert not swept & {"c", "d", "e"}


class TestPagingStopsWhenTheResultsDo:
    """Fixed paging was wrong in both directions at once.

    Measured on a real sweep: 14 of 41 search requests — 34% — came back empty,
    because queries like "moderkort AM4" have 27 listings and were being asked
    for four pages of them. At the same time "SSD 1TB", "NVMe 1TB", "DDR5 32GB"
    and "moderkort" were all still returning *full* pages at page 4 and being
    cut off mid-stream. No single constant is right, because the queries are
    not alike.
    """

    def test_a_full_page_is_followed_by_another(self):
        src = FakeSource(pages={
            ("q", 1): [summary(f"a{i}") for i in range(50)],
            ("q", 2): [summary(f"b{i}") for i in range(50)],
            ("q", 3): [summary("c")],
        })

        records, stats = ingest(src, [SearchTarget("q", max_pages=12)])

        assert stats.searched == 3
        assert len(records) == 101

    def test_a_short_page_ends_the_query(self):
        """53, 20, then stop — "nätaggregat 750W" exactly. The third request
        that used to follow returned nothing, twice."""
        src = FakeSource(pages={
            ("q", 1): [summary(f"a{i}") for i in range(53)],
            ("q", 2): [summary(f"b{i}") for i in range(20)],
            ("q", 3): [summary("never-reached")],
        })

        records, stats = ingest(src, [SearchTarget("q", max_pages=12)])

        assert stats.searched == 2, "a short page means there is no next page"
        assert "never-reached" not in [r.external_id for r in records]

    def test_an_empty_page_ends_the_query(self):
        src = FakeSource(pages={
            ("q", 1): [summary(f"a{i}") for i in range(53)],
            ("q", 2): [],
            ("q", 3): [summary("never-reached")],
        })

        _, stats = ingest(src, [SearchTarget("q", max_pages=12)])

        assert stats.searched == 2

    def test_a_thin_query_costs_one_wasted_request_not_three(self):
        """The "moderkort AM4" shape: 27 results, then nothing.

        Page one cannot be judged — there is no page size to compare against
        yet — so one empty page is still requested. That is the honest cost of
        not hardcoding the API's page size, and it is one request rather than
        the three the old fixed depth spent.
        """
        src = FakeSource(pages={("q", 1): [summary(f"a{i}") for i in range(27)]})

        _, stats = ingest(src, [SearchTarget("q", max_pages=12)])

        assert stats.searched == 2

    def test_the_ceiling_still_binds(self):
        """A query that never runs short must not page forever."""
        src = FakeSource(pages={
            ("q", n): [summary(f"p{n}-{i}") for i in range(50)] for n in range(1, 30)
        })

        _, stats = ingest(src, [SearchTarget("q", max_pages=6)])

        assert stats.searched == 6

    def test_the_page_size_is_learned_not_assumed(self):
        """Deliberately not hardcoded to 53.

        Pinning the API's page size means that the day Blocket changes it, every
        query stops after page one and each run still reports success — a silent
        halving of coverage, which is this project's recurring failure mode.
        Here a page size of 8 works exactly as 53 does.
        """
        src = FakeSource(pages={
            ("q", 1): [summary(f"a{i}") for i in range(8)],
            ("q", 2): [summary(f"b{i}") for i in range(8)],
            ("q", 3): [summary("c")],
            ("q", 4): [summary("never-reached")],
        })

        _, stats = ingest(src, [SearchTarget("q", max_pages=12)])

        assert stats.searched == 3


class TestAnAdIsFetchedOncePerRun:
    """`is_known` is a snapshot taken before the sweep, so it cannot know about
    an ad this run has just fetched.

    Measured mid-run on live data: 927 detail requests for 759 distinct ads,
    18% of them repeats — and that was before the sweep reached the GPU
    queries, where "grafikkort" overlaps every model-specific query.
    """

    def test_the_same_ad_under_two_queries_costs_one_request(self):
        src = FakeSource(pages={
            ("gpu", 1): [summary("shared"), summary("only-a")],
            ("grafikkort", 1): [summary("shared"), summary("only-b")],
        })

        records, stats = ingest(
            src, [SearchTarget("gpu"), SearchTarget("grafikkort")]
        )

        assert stats.detail_fetched == 3
        assert stats.duplicates_skipped == 1
        assert sorted(r.external_id for r in records) == ["only-a", "only-b", "shared"]

    def test_the_duplicate_is_counted_not_merely_avoided(self):
        """The number measures how much the query list overlaps itself, which
        is the only way to tell whether adding a query bought new coverage or
        re-bought the old."""
        src = FakeSource(pages={
            ("a", 1): [summary("x")],
            ("b", 1): [summary("x")],
            ("c", 1): [summary("x")],
        })

        _, stats = ingest(src, [SearchTarget(q) for q in ("a", "b", "c")])

        assert stats.duplicates_skipped == 2
        assert "2 dup" in stats.summary()

    def test_a_failed_fetch_is_not_retried_under_the_next_query(self):
        """The request was paid for whether or not it returned anything.
        Retrying it spends a second one to get the same nothing."""
        class AlwaysFails(FakeSource):
            def fetch_detail(self, s):
                self.detail_calls.append(s.external_id)
                raise SourceError("detail blew up")

        src = AlwaysFails(pages={
            ("a", 1): [summary("bad")],
            ("b", 1): [summary("bad")],
        })

        _, stats = ingest(src, [SearchTarget("a"), SearchTarget("b")])

        assert stats.failures == 1, "the failure must not be repeated"
        assert stats.duplicates_skipped == 1

    def test_a_removed_listing_is_not_re_requested(self):
        src = FakeSource(
            pages={("a", 1): [summary("gone")], ("b", 1): [summary("gone")]},
            missing={"gone"},
        )

        _, stats = ingest(src, [SearchTarget("a"), SearchTarget("b")])

        assert stats.detail_missing == 1
        assert stats.duplicates_skipped == 1

    def test_ads_already_in_the_database_are_still_counted_separately(self):
        """`already_known` and `duplicates_skipped` answer different questions —
        one is "we had it yesterday", the other "we fetched it ten minutes ago".
        Conflating them would hide either a stale corpus or a redundant query."""
        src = FakeSource(pages={
            ("a", 1): [summary("old"), summary("new")],
            ("b", 1): [summary("old"), summary("new")],
        })

        _, stats = ingest(
            src, [SearchTarget("a"), SearchTarget("b")], is_known={"old"}.__contains__
        )

        assert stats.already_known == 2
        assert stats.duplicates_skipped == 1
        assert stats.detail_fetched == 1
