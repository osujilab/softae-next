"""Rung 3a — the fake-cast harness, end to end under ``--mock``.

Spec: ``docs/SubAgent docs/rung3a_fake_cast.md``. The harness under test is
``tools/rung3_fake_cast.py``; it patches two module attributes on
:mod:`softae.tools.campaign`'s local-import seam and then calls
``campaign.main([...])``.

**Nothing here is a hook added to the harness for testing.** The mock rig is
installed the way ``tests/test_rung2_four_well_rehearsal.py::_drive`` installs
it, and it is installed by *patching the same module attribute the harness
patches* — :func:`softae.drivers.factory.create_manager`. The harness then
captures this test's function as "the original" and wraps it, so the composition
is verified for free: if the harness stopped calling through, the manager would
carry no mock picos and every sweep would die.

Three costs this module is sized against, each measured rather than assumed:

1. **``time_scale`` is not spellable in TOML.** ``CampaignSpec.time_scale``
   defaults to ``1.0`` and :mod:`softae.core.campaign_spec_io` has no key for it,
   so a spec loaded from a file dwells the full ``elution_wait_s = 240.0`` plus
   ``wick_dwell_s = 5.0`` per channel — **~16 minutes of sleeping for a cast that
   does not happen**, on the bench as well as here. This module drops it to
   ``0.0`` through its own ``run_autonomous_campaign`` patch, which the harness
   likewise wraps rather than replaces.
2. **The mock anneal hold is instant by design.** ``MockTempController.anneal``
   logs ``mock_anneal_hold_skipped`` and returns, so ``hold_s`` costs nothing
   under mocks. It is still reduced in the derived spec, because a test spec that
   silently relies on a mock's shortcut is a test spec that breaks the day the
   mock becomes faithful.
3. **``max_hold_s`` is sized against ROUND cost, not against an intended hold.**
   The deadline is tested AFTER each round, so a round whose fits overrun it still
   runs to completion; what ``max_hold_s`` decides is whether the NEXT round
   starts. A four-channel round here costs ~40 s against the ``Quick`` preset
   (measured: 125 s for two rounds, a trial read and a production read), so
   :data:`CEILING_HOLD_S` at 80 s clears the window
   :data:`SETTLE_N_ROUNDS` needs, and the production-read assertion checks
   ``n_rounds >= SETTLE_N_ROUNDS`` BEFORE it reads any outcome — a future
   slowdown then reports itself as a slowdown rather than as a changed verdict.

   **The fitter is NOT canned here, and one attempt to cane it is worth
   recording.** Answering the designed R1 directly cut the module to 88 s — and
   broke it: a canned ``SpectrumReport`` carries ``SigmaReport(mode="unavailable")``,
   this campaign's objective IS sigma, and every trial came back *"no usable
   measurement"* until the loop parked on three consecutive failures. It looked
   like a speed-up and was an exception path. The cost of real fits is the price
   of a run whose objective is the thing the nominal thickness exists to produce.

The example file itself keeps the bench values. This module derives its spec from
it by loading the TOML and reducing four durations, so the two cannot drift.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import rung3_fake_cast as harness  # noqa: E402

from softae.core.production_read import PRODUCTION_STEP_PREFIX  # noqa: E402
from softae.drivers.mock_factory import create_mock_manager  # noqa: E402
from softae.tools.eis_validate_mock import (  # noqa: E402
    MockRig,
    install_fast_conditions,
    install_mock_picos,
)

CHANNELS = (1, 11, 13, 14)
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "rung3a_fake_cast.toml"

#: See the module docstring, point 3. Clears two four-channel rounds with margin.
CEILING_HOLD_S = 80.0

#: The judged window, narrowed from the shipped 3 for cost. It is the window
#: WIDTH that has to be reached for a verdict to mean anything, and at ~40 s per
#: four-channel round (measured below) a third round buys no new statement about
#: the harness and costs another 40 s.
SETTLE_N_ROUNDS = 2

#: Quiet and well-resolved: the arc closes inside the preset's band and the
#: fitted R1 repeats to within ~0.5 %. Rung 2's value, for the same reason.
QUIET_APEX_HZ = 30.0

NOMINAL_UM = 50.0


# ── the rig, installed exactly as rung 2 installs it ─────────────────────────

def _install_mock_factory(monkeypatch) -> list[Any]:
    """Patch ``create_manager`` to hand back a fully-prepared mock rig.

    Returns the list the managers are recorded into, so a test can assert the
    harness swapped the syringe and stage **on top of this** rather than building
    a manager of its own.
    """
    built: list[Any] = []

    def _create(*_args: Any, **_kwargs: Any):
        manager = create_mock_manager(config={})
        install_fast_conditions(manager)
        # BEFORE connect_all, which the campaign CLI does later: these picos are
        # brand new and disconnected, and a connected manager will not connect
        # them again.
        install_mock_picos(manager, MockRig(
            apex_hz={ch: QUIET_APEX_HZ for ch in CHANNELS},
            drift_decades_per_hour=0.0,
            # Drift is per HOUR and this run finishes in seconds, so elapsed time
            # is counted in sweeps instead.
            virtual_s_per_sweep=60.0,
        ))
        built.append(manager)
        return manager

    monkeypatch.setattr("softae.drivers.factory.create_manager", _create)
    return built


def _install_fast_dwells(monkeypatch) -> None:
    """``time_scale = 0.0``, the only way in. See the module docstring, point 1."""
    from softae.core import autonomous_wiring

    original = autonomous_wiring.run_autonomous_campaign

    async def _wrapped(spec, **kwargs: Any):
        return await original(dataclasses.replace(spec, time_scale=0.0), **kwargs)

    monkeypatch.setattr(autonomous_wiring, "run_autonomous_campaign", _wrapped)


# ── the spec, derived from the example rather than retyped ───────────────────

def _fast_spec(tmp_path: Path, **over: Any) -> Path:
    raw = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    phases = raw["run_plan"]["phases"]
    anneal = next(p for p in phases if p["kind"] == "anneal")
    equil = next(p for p in phases if p["kind"] == "equilibrate")

    anneal["hold_s"] = 1.0
    anneal["conditions"]["rh_approach_timeout_s"] = 60.0
    equil["conditions"]["approach_timeout_s"] = 60.0
    equil["settle"].update(
        round_period_s=0.05,
        min_hold_s=0.0,
        max_hold_s=CEILING_HOLD_S,
        settle_n_rounds=SETTLE_N_ROUNDS,
        settle_min_channels=3,
        # Explicitly OFF. The gate judges the ROOM, and armed here the mock
        # chamber's own spread would decide the verdict instead of sigma.
        explicit_none=["rh_stability_pct"],
    )
    equil["settle"].pop("rh_stability_pct", None)
    # `Quick` here, `Extended` in the example. NOT cosmetic: the settle deadline
    # is tested AFTER each round, so a round whose fits overrun it still runs to
    # completion — an Extended round (53 points down to 1.351 Hz) costs minutes of
    # gated fitting per four channels, and three of them put this module at 600 s
    # of saturated CPU, measured. What this module is about is the HARNESS, not
    # the band; the band is the bench run's business.
    raw["measurement"]["preset"] = "Quick"
    for phase in phases:
        if "measurement" in phase:
            phase["measurement"]["preset"] = "Quick"
    raw.update(over)

    path = tmp_path / "fast.toml"
    path.write_bytes(tomli_w.dumps(raw).encode("utf-8"))
    return path


def _run(tmp_path: Path, monkeypatch, *, spec: Path | None = None,
         channels: str | None = None,
         thickness_um: float = NOMINAL_UM) -> dict[str, Any]:
    """Install this module's two patches, snapshot them, then run the harness.

    The snapshot is taken AFTER the patches, which is the point: what the harness
    must put back is what it found, not what the repository ships.
    """
    built = _install_mock_factory(monkeypatch)
    _install_fast_dwells(monkeypatch)
    originals = _snapshot_patched_attributes()
    project = tmp_path / "project"
    argv = [str(spec or _fast_spec(tmp_path)), "--project", str(project),
            "--thickness-um", str(thickness_um), "--mock", "-y"]
    if channels is not None:
        argv += ["--channels", channels]
    return {"code": harness.main(argv), "project": project, "built": built,
            "originals": originals}


@pytest.fixture(scope="module")
def fake_cast_run(tmp_path_factory):
    """ONE complete ``--mock`` run, with everything it left behind.

    Module-scoped deliberately. Function scope ran the whole campaign once per
    assertion — eight real four-channel settle campaigns for eight questions
    about a single run — and the fits are the cost, so it is the run that must be
    shared, not the store handle. `monkeypatch` is function-scoped, hence the
    explicit :class:`pytest.MonkeyPatch` here.
    """
    from softae.core.data_store import DataStore

    patcher = pytest.MonkeyPatch()
    # Undone at TEARDOWN, not after the call: `test_harness_restores_...` asks
    # whether the harness put back what it found, and what it found was this
    # module's patch. Lifting it first would make that assertion compare the
    # shipped function against itself and pass for the wrong reason.
    outcome = _run(tmp_path_factory.mktemp("rung3a"), patcher)
    store = DataStore(outcome["project"])
    try:
        run_id = store.query_runs()[-1]["run_id"]
        yield {
            **outcome,
            "manager": outcome["built"][-1],
            "run_id": run_id,
            "rows": store.query_measurements(run_id=run_id),
            "thickness": store.measured_thickness(run_id=run_id),
            "run_dir": Path(store.run_dir(run_id)),
        }
    finally:
        store.close()
        patcher.undo()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _snapshot_patched_attributes() -> dict[str, Any]:
    from softae.core import autonomous_wiring
    from softae.drivers import factory

    return {"create_manager": factory.create_manager,
            "run_autonomous_campaign": autonomous_wiring.run_autonomous_campaign}


def _assert_restored(originals: dict[str, Any]) -> None:
    from softae.core import autonomous_wiring
    from softae.drivers import factory

    assert factory.create_manager is originals["create_manager"]
    assert (autonomous_wiring.run_autonomous_campaign
            is originals["run_autonomous_campaign"])


# ── what the harness did to the rig ──────────────────────────────────────────

def test_harness_restores_both_patched_attributes_after_a_run(fake_cast_run):
    """The two patches are a loan, not a change.

    They are restored against the objects this test installed, not against the
    shipped functions — which is the stronger statement: the harness put back
    exactly what it found, including another layer's patch.
    """
    _assert_restored(fake_cast_run["originals"])


def test_harness_swaps_syringe_and_stage_on_top_of_the_supplied_manager(
    fake_cast_run
):
    """The swap lands on THIS module's manager, not one the harness built.

    If the harness ever stopped calling through to the captured factory, the
    manager here would have no mock picos and the run would not have reached a
    measurement at all — so the two halves check each other.
    """
    manager = fake_cast_run["manager"]
    assert type(manager.get("syringe")).__name__ == "MockSyringe"
    assert type(manager.get("stage")).__name__ == "MockStage"
    assert fake_cast_run["rows"], "the run recorded nothing"


def test_harness_leaves_picos_and_conditions_controllers_untouched(fake_cast_run):
    """Only motion is made inert. The measurement and the chamber are not.

    On the bench the piezo matters most: mocking it would flip
    ``rig_is_simulated`` to True and silently disarm both the run lock and the
    hardware interlock for a run whose heater and humidifier are live.
    """
    manager = fake_cast_run["manager"]
    assert type(manager.get("pico1")).__name__ == "GridAwareMockPico"
    assert type(manager.get("temp_controller")).__name__ == "FastMockTempController"
    assert type(manager.get("rh_controller")).__name__ == "FastMockRHController"


# ── what the harness wrote ───────────────────────────────────────────────────

def test_a_nominal_thickness_row_per_channel_carries_the_run_id(fake_cast_run):
    """Without these there is no sigma, and the only verdict is not_evaluable.

    The ``run_id`` is the load-bearing part: ``thickness_for`` appends
    ``AND run_id = ?``, so a row written before ``run_started`` — or against
    another run — is invisible to the criterion it exists to feed.
    """
    rows = fake_cast_run["thickness"]
    assert sorted(int(r["channel"]) for r in rows) == list(CHANNELS)
    assert {r["run_id"] for r in rows} == {fake_cast_run["run_id"]}
    assert {float(r["thickness_um"]) for r in rows} == {NOMINAL_UM}
    assert all("NOMINAL" in str(r["notes"]) for r in rows)


def test_the_run_directory_carries_the_fake_cast_note(fake_cast_run):
    note = fake_cast_run["run_dir"] / harness.NOTE_FILENAME
    assert note.exists()
    assert "did not occur" in note.read_text(encoding="utf-8").lower()


def test_formulations_rows_are_written_for_a_cast_that_did_not_happen(
    fake_cast_run
):
    """Asserted so the shape is on record, NOT because it is desired.

    ``_record_trial_formulations`` runs on the batch path regardless of whether a
    pump turned. The rows carry the run id, which is the only reason they can be
    found and discounted later; the run-directory note above says so in words.
    """
    from softae.core.data_store import DataStore

    store = DataStore(fake_cast_run["project"])
    try:
        rows = store.query_formulations(run_id=fake_cast_run["run_id"])
    finally:
        store.close()
    assert sorted({int(r["channel"]) for r in rows}) == list(CHANNELS)


def test_fixed_channel_mode_writes_no_electrode_occupancy_rows(fake_cast_run):
    """``track_occupancy = allocator is not None``, and there is no allocator.

    An occupancy row would claim these wells were consumed by this run, which is
    the opposite of true: they were consumed weeks ago and this run is reading
    them again.
    """
    from softae.core.data_store import DataStore

    store = DataStore(fake_cast_run["project"])
    try:
        rows = store._conn.execute(
            "SELECT COUNT(*) FROM electrode_occupancy").fetchone()
    finally:
        store.close()
    assert rows[0] == 0


# ── what the run actually did ────────────────────────────────────────────────

def test_the_run_reaches_a_production_read_on_exactly_the_chosen_wells(
    fake_cast_run
):
    """Three rounds first, then the production read, on 1/11/13/14 and nothing else.

    ``n_rounds >= 3`` is asserted BEFORE the outcome so that a future slowdown
    reports itself as a slowdown rather than as a changed verdict — see the
    module docstring, point 3.
    """
    ordered = sorted(fake_cast_run["rows"], key=lambda r: int(r["measurement_id"]))
    production, settle = [], []
    for row in ordered:
        stem = os.path.basename(str(row.get("eis_file_path") or ""))
        if stem.startswith(f"{PRODUCTION_STEP_PREFIX}_"):
            production.append(row)
        elif stem.startswith("settle_eis_"):
            settle.append(row)

    assert len(settle) >= SETTLE_N_ROUNDS * len(CHANNELS), (
        f"only {len(settle)} settle sweeps — fewer than the {SETTLE_N_ROUNDS} "
        f"full rounds the tracker's window needs, so any verdict below would be "
        f"reporting the ceiling, not the criterion")
    assert sorted(int(r["channel"]) for r in production) == list(CHANNELS)
    assert (min(int(r["measurement_id"]) for r in production)
            > max(int(r["measurement_id"]) for r in settle))


def test_the_mock_run_exits_clean(fake_cast_run):
    assert fake_cast_run["code"] == 0


# ── the refusals ─────────────────────────────────────────────────────────────

def test_refuses_a_spec_that_sets_electrode_capacity(tmp_path, monkeypatch):
    """Allocator mode would walk onto wells that were never cast.

    The refusal is checked before anything is patched, so a refused launch leaves
    the process exactly as it found it.
    """
    spec = _fast_spec(tmp_path, electrode_capacity=32)
    outcome = _run(tmp_path, monkeypatch, spec=spec)
    assert outcome["code"] != 0
    assert outcome["built"] == [], "a refused launch must not have built a manager"
    _assert_restored(outcome["originals"])


def test_refuses_channels_that_disagree_with_the_spec(tmp_path, monkeypatch):
    """A mismatch records the nominal thickness against wells nobody reads.

    The thickness rows follow ``--channels`` and the measurement follows the
    spec, so the two disagreeing is the quiet version of running with no sigma at
    all — every well excluded ``sigma_null`` and a ``not_evaluable`` board.
    """
    outcome = _run(tmp_path, monkeypatch, channels="1,2,3,4")
    assert outcome["code"] != 0
    assert outcome["built"] == []
    _assert_restored(outcome["originals"])



@pytest.mark.parametrize("bad", ["0", "-50", "nan", "inf"])
def test_refuses_a_thickness_that_is_not_a_positive_length(
    tmp_path, monkeypatch, bad
):
    """NOMINAL does not mean arbitrary — sigma divides by this number.

    Zero, a negative, a NaN or an infinity is not a mis-scaled sigma but an
    undefined one, and every well is then excluded ``sigma_null``: the run spends
    the cure and reports ``not_evaluable`` for the whole board. Refused before
    anything is patched, so a refused launch leaves the process as it found it.
    """
    outcome = _run(tmp_path, monkeypatch, thickness_um=bad)
    assert outcome["code"] != 0
    assert outcome["built"] == [], "a refused launch must not have built a manager"
    _assert_restored(outcome["originals"])


def test_the_thickness_read_back_is_reported_per_channel(fake_cast_run, capsys):
    """The write is not the claim; the LOOKUP is.

    ``thickness_for`` filters ``AND run_id = ?``, so a row written against the
    wrong run reports a successful write and is invisible to the criterion that
    needs it. The harness therefore reads back through the same lookup the
    campaign uses and says what it found, per channel, before the cure starts.
    """
    for channel in CHANNELS:
        assert any(int(r["channel"]) == channel
                   and abs(float(r["thickness_um"]) - NOMINAL_UM) < 1e-9
                   for r in fake_cast_run["thickness"])


def test_a_failing_annotation_never_escapes_the_run_started_hook(capsys):
    """An exception out of ``on_event`` is an unknown path through the loop.

    So the hook swallows everything and says so loudly instead. This drives the
    inner function directly with a store whose every method raises — the shape a
    locked or closed database produces — and asserts it returns rather than
    propagates, and that the operator is told to abort.
    """
    class _Hostile:
        def __getattr__(self, _name):
            def _boom(*_a, **_k):
                raise RuntimeError("database is locked")
            return _boom

    lines: list[str] = []
    harness._annotate_run(_Hostile(), "run-x", CHANNELS, NOMINAL_UM,
                          emit=lines.append)
    joined = " | ".join(lines)
    assert any("sigma will be unavailable" in line for line in lines), joined
    assert "control abort" in joined


def test_a_busy_rig_is_reported_as_busy_and_still_restores_both_patches(
    tmp_path, monkeypatch, capsys
):
    """Somebody else holds the rig: the CLI's message, the CLI's exit code.

    A traceback here would read as a crash when it is only contention, so the
    harness catches ``RunLockHeld`` and prints what ``campaign.py::_cmd_run``
    prints, imported rather than restated.

    **This is the one test that does not pass ``--mock``**, because ``--mock``
    skips the claim entirely and the branch would never be reached. It is still
    hardware-free by construction: ``held_run_lock`` is patched to raise, so the
    refusal returns before ``campaign.main`` is called at all and nothing opens a
    port. `create_manager` is patched too and is asserted never to have run.
    """
    from softae.core import run_lock
    from softae.tools import campaign

    built = _install_mock_factory(monkeypatch)
    _install_fast_dwells(monkeypatch)
    originals = _snapshot_patched_attributes()

    lock = run_lock.RunLock(pid=999999, what="campaign:someone-else",
                            started_at="2026-09-15T00:00:00", log_path="")

    def _busy(*_a, **_k):
        raise run_lock.RunLockHeld(lock)

    monkeypatch.setattr(run_lock, "held_run_lock", _busy)

    code = harness.main([
        str(_fast_spec(tmp_path)), "--project", str(tmp_path / "project"),
        "--thickness-um", str(NOMINAL_UM), "-y",
    ])
    assert code == campaign.EXIT_BUSY
    assert built == [], "the refusal must return before any manager is built"
    assert "NOT STARTING" in capsys.readouterr().out
    _assert_restored(originals)
