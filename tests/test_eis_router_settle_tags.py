"""The four T11.33 settle-relay tags, read off a step and into ``measurements``.

``analysis/eis/router.py`` is the **only** route these four values have to
``record_measurement``: ``core/production_read.py`` stamps them onto the
production step's tags, and this is where the tags become columns. The same
funnel ``fixture_id`` / ``electrode_mode`` / ``sample_uuid`` already use
(``[a206]``), so what is pinned here is that funnel's newest four keys.

**Why a real ``DataStore`` rather than a fake.** ``EISResultRouter.handle``
catches every exception and returns ``None`` — a routing failure must not fail a
step that physically succeeded. That makes *"it did not raise"* a vacuous
assertion (``SUBAGENT_RULES.md`` §3.1e): a test could pass while the row was
never written. So every test here asserts on the **recorded row**, which can only
exist if the whole path ran. The spy wraps rather than replaces
``record_measurement``, so the kwargs are observed *and* the insert really
happens.

**Why a new file rather than an addition to ``test_result_router_golden.py``.**
That file is another session's and currently dirty (``CLAUDE.md`` §6, "prefer a
new file to a contested edit"). It also pins the row shape as a whole, which is a
different question from the one asked here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis.router import EISResultRouter, RouterContext
from softae.core.data_store import DataStore
from softae.workflows.workflow_model import WorkflowStep

# The 5-column [f, |Z|, phase, Z', -Z''] layout `EISResult.from_raw` documents as
# the contract shared by AsyncESPico and MockESPico. Small and deterministic: no
# fit is requested (no `circuit_model` param), so the numbers only have to be a
# well-formed spectrum, not a physical one.
_F = np.logspace(5, 0, 12)
_ZREAL = np.full(12, 4000.0)
_ZIMG = np.linspace(-50.0, -900.0, 12)
RAW = np.column_stack([_F, np.abs(_ZREAL + 1j * _ZIMG),
                       np.degrees(np.arctan2(_ZIMG, _ZREAL)), _ZREAL, -_ZIMG])

SETTLE_TAGS = {
    "certification": "settled",
    "well_verdict": "rate_quiet",
    "rate_per_hour": "0.0042",
    "upper_bound_per_hour": "0.0191",
}


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "project")
    yield ds
    ds.close()


class _Spy:
    """Wraps ``record_measurement``, recording its kwargs and still inserting."""

    def __init__(self, store: DataStore) -> None:
        self.calls: list[dict] = []
        self._real = store.record_measurement
        store.record_measurement = self  # type: ignore[assignment]

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._real(*args, **kwargs)

    @property
    def only(self) -> dict:
        assert len(self.calls) == 1, f"expected one row, got {len(self.calls)}"
        return self.calls[0]


def _step(**tags: str) -> WorkflowStep:
    return WorkflowStep(
        name="production_measure_eis_ch21",
        instrument="pico1",
        method="sendscript_getdata",
        params={"mscrpath": "f.mscr", "outdir": "out", "chan": 21},
        tags={"channel": "21", "measurement": "primary", **tags},
    )


async def _route(store: DataStore, step: WorkflowStep):
    """Run the router over *step*, returning (spy, result)."""
    run_id = store.start_run("settle_tag_route")
    spy = _Spy(store)
    ctx = RouterContext(data_store=store, run_id=run_id, manager=None,
                        auto_fit=False)
    result = await EISResultRouter().handle(step, RAW, ctx)
    return spy, result


# ── All four present ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_all_four_settle_tags_reach_record_measurement_typed(store):
    """The two words stay strings; the two rates are parsed to floats.

    ``WorkflowStep.with_tags`` is typed ``**extra: str``, so the numbers cross
    the boundary as text and this is where they become numbers — the same
    ``float()`` the ``nominal`` tag beside them already takes.
    """
    spy, result = await _route(store, _step(**SETTLE_TAGS))

    assert result is not None, "the row was not recorded at all"
    call = spy.only
    assert call["certification"] == "settled"
    assert call["well_verdict"] == "rate_quiet"
    assert call["rate_per_hour"] == pytest.approx(0.0042)
    assert call["upper_bound_per_hour"] == pytest.approx(0.0191)
    assert isinstance(call["rate_per_hour"], float)
    assert isinstance(call["upper_bound_per_hour"], float)


@pytest.mark.asyncio
async def test_router_settle_tags_are_readable_back_off_the_stored_row(store):
    """The seam all the way to the column, not just to the call.

    A kwarg the store silently ignored would satisfy the test above and store
    nothing — "could not check, and said fine" (``SUBAGENT_RULES.md`` §3.1a).
    """
    _, result = await _route(store, _step(**SETTLE_TAGS))
    assert result is not None

    rows = store._conn.execute(
        "SELECT certification, well_verdict, rate_per_hour, upper_bound_per_hour "
        "FROM measurements"
    ).fetchall()
    assert len(rows) == 1
    certification, well_verdict, rate, bound = rows[0]
    assert certification == "settled"
    assert well_verdict == "rate_quiet"
    assert rate == pytest.approx(0.0042)
    assert bound == pytest.approx(0.0191)


# ── None present ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_no_settle_tags_records_all_four_as_none(store):
    """A read not taken under a settle phase says so, and does not say 'settled'.

    Present-only: an absent tag must stay NULL rather than become a plausible
    word, which is the distinction these columns exist to make.
    """
    spy, result = await _route(store, _step())

    assert result is not None
    call = spy.only
    assert call["certification"] is None
    assert call["well_verdict"] is None
    assert call["rate_per_hour"] is None
    assert call["upper_bound_per_hour"] is None


@pytest.mark.asyncio
async def test_router_a_board_word_without_a_well_verdict_records_just_the_word(
    store,
):
    """The deviation criterion's shape: a board word, no per-well rate.

    Under the shipped default ``criterion="deviation"`` the tracker never
    computes a ``RateCheck``, so ``SettleOutcome.by_channel`` is empty and the
    production step carries ``certification`` alone. That is the common case, not
    an edge one.
    """
    spy, _ = await _route(store, _step(certification="ceiling"))

    call = spy.only
    assert call["certification"] == "ceiling"
    assert call["well_verdict"] is None
    assert call["rate_per_hour"] is None
    assert call["upper_bound_per_hour"] is None


# ── Malformed ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_a_non_numeric_rate_tag_records_none_and_still_writes_the_row(
    store,
):
    """A junk number must not cost the measurement its row.

    The assertion that matters is that the row was **written** — ``handle``
    swallows every exception and returns ``None``, so *"it did not raise"* would
    pass even if the ``float()`` had exploded and taken the insert with it
    (``SUBAGENT_RULES.md`` §3.1e). The word beside it survives intact, because
    one unparseable number is not a reason to discard a verdict that parsed.
    """
    spy, result = await _route(store, _step(
        certification="settled", well_verdict="rate_quiet",
        rate_per_hour="not-a-number", upper_bound_per_hour="0.02"))

    assert result is not None, "a malformed tag cost the measurement its row"
    call = spy.only
    assert call["rate_per_hour"] is None
    assert call["upper_bound_per_hour"] == pytest.approx(0.02)
    assert call["certification"] == "settled"
    assert call["well_verdict"] == "rate_quiet"


@pytest.mark.asyncio
async def test_router_an_empty_rate_tag_records_none_rather_than_zero(store):
    """``float("")`` raises ``ValueError``; zero would be a fabricated slope."""
    spy, _ = await _route(store, _step(rate_per_hour="", certification="settled"))

    call = spy.only
    assert call["rate_per_hour"] is None
