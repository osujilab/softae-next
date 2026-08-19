"""The scout at its call sites — and, first, its absence at them.

Two flags are landing in the same package at the same time: this wiring, which
changes **which spectrum is acquired**, and a fitter pre-gate, which changes **how
a spectrum is fit**. They have to be flippable independently, or a later movement
in fit quality has two candidate causes instead of one. So the load-bearing tests
here are the *characterization* ones: with ``[eis.scout] enabled = false`` each
call site writes the same ``.mscr`` bytes, in the same call sequence, that it
wrote before this module existed.

The second load-bearing group is the economics. The sweep the operator asked for
runs first and **is** the measurement whenever it is adequate, so the tests that
matter most after inertness are the ones asserting that an adequate spectrum is
measured exactly once. A scout that quietly doubled every acquisition would pass
a naive "did it plan?" test and fail the rig.

The flag-off baselines are built by calling :func:`eis_run_mscrbuild` directly
with the arguments the pre-wiring code passed, rather than by pasting expected
bytes — a paste would freeze a formatting decision that belongs to the emitter,
which is exactly what ``eis_run_mscrbuild`` must be free to keep and the four
stopwatched timing anchors depend on.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis.scout import ScoutDecision, ScoutSettings
from softae.core.eis_scout_scripts import (
    SEGMENTED_DURATION_BASIS,
    SEGMENTED_SWEEP_TAG,
    SWEEP_ROLE_FOLLOW_UP,
    SWEEP_ROLE_SCOUT,
    WIDER_PRESET_SWEEP_TAG,
    ScoutPlanner,
    segmented_params,
)
from softae.drivers.mscr_library import eis_run_mscrbuild

from tests.test_eis_arc_closure import arc_with_blocking_tail, semicircle

#: `Quick`, the DEFAULT_PRESET and the intended scout sweep.
PRESET = {"f_hi": 200_000, "f_lo_mHz": 6_475, "npts": 27, "mv_ac": 10, "mv_dc": 0}
#: One rung wider — what an `extend_low` verdict steps to.
STANDARD = {"f_hi": 200_000, "f_lo_mHz": 3_912, "npts": 34}
#: The bottom of the ladder: nothing configured reaches lower.
WIDEST = {**PRESET, "f_lo_mHz": 228, "npts": 39}


# ── Helpers ──────────────────────────────────────────────────────────────────


def raw_from(f: np.ndarray, z_imag_neg: np.ndarray, r: float = 1.0e4):
    """A ``sendscript_getdata`` return in the five-column extractdata contract."""
    z_real = np.full(np.asarray(f).size, r)
    z = z_real - 1j * np.asarray(z_imag_neg)
    return [np.asarray(f), np.abs(z), np.angle(z, deg=True),
            z_real, np.asarray(z_imag_neg)]


def preset_bytes(tmp_path: Path, channel: int, params: dict = PRESET) -> str:
    """What the pre-wiring build wrote for *channel* — the baseline, not a paste."""
    reference = tmp_path / f"reference_ch{channel}_{params['f_lo_mHz']}.mscr"
    eis_run_mscrbuild(
        str(reference),
        mux_ch=channel,
        mVac=params.get("mv_ac", 10),
        f_hi=params.get("f_hi", 200_000),
        f_lo=params.get("f_lo_mHz", 100),
        npts=params.get("npts", 20),
        mVdc=params.get("mv_dc", 0),
    )
    return reference.read_text()


class FakePico:
    """Returns a canned spectrum and remembers which script it was handed."""

    def __init__(self, raw, output_dir):
        self._raw = raw
        self._output_dir = str(output_dir)
        self.scripts_seen: list[str] = []

    def sendscript_getdata(self, script_path, output_dir, channel):
        self.scripts_seen.append(Path(script_path).read_text())
        return self._raw


class FakeManager:
    def __init__(self, pico):
        self._pico = pico

    def get(self, _name):
        return self._pico


@pytest.fixture
def calls(monkeypatch):
    """Every ``.mscr`` build either call site makes, in order."""
    import softae.drivers.mscr_library as lib

    seen: list[tuple[str, dict]] = []
    real_run, real_seg = lib.eis_run_mscrbuild, lib.eis_segmented_mscrbuild

    def run_spy(filename, mux_ch, **kw):
        seen.append(("preset", {"mux_ch": mux_ch, **kw}))
        return real_run(filename, mux_ch, **kw)

    def seg_spy(filename, mux_ch, segments, **kw):
        seen.append(("segmented", {"mux_ch": mux_ch, "segments": tuple(segments), **kw}))
        return real_seg(filename, mux_ch, segments, **kw)

    monkeypatch.setattr(lib, "eis_run_mscrbuild", run_spy)
    monkeypatch.setattr(lib, "eis_segmented_mscrbuild", seg_spy)
    return seen


def manual_worker(manager, *, scout, params=None):
    from softae.gui.tabs.tab_manual import _ManualEisWorker

    return _ManualEisWorker(
        manager, None, channels=[3], eis_params=dict(params or PRESET),
        preset_label="Quick", auto_fit=False, fit_model="simpleSalt",
        auto_save=False, scout=scout,
    )


def sweep(channels=(1, 2), params=None):
    from softae.analysis.arrhenius import ArrheniusSweepConfig
    from softae.workflows.temp_eis_sweep import ArrheniusSweep

    config = ArrheniusSweepConfig(
        channels=list(channels), T_start=25.0, T_stop=35.0, T_step=10.0,
        dwell_s=0.0, eis_params=dict(params or PRESET),
    )
    return ArrheniusSweep(config, manager=None)


def planner(enabled: bool, actuate: bool, *, manual=None, **kw) -> ScoutPlanner:
    """*manual* stands in for the tab's checkbox; ``None`` defers to *actuate*."""
    return ScoutPlanner(
        settings=ScoutSettings(enabled=enabled, actuate=actuate, **kw),
        site="test", actuate=manual)


def measure(tmp_path, spectrum, *, scout, params=None):
    """One ``_measure_one`` call; returns ``(pico, payload)``."""
    pico = FakePico(raw_from(*spectrum), tmp_path)
    worker = manual_worker(FakeManager(pico), scout=scout, params=params)
    return pico, worker._measure_one(3, "pico1", None)


# Spectra whose verdicts are pinned by tests/test_eis_scout.py; named here so a
# test reads as the case it is about rather than as a pair of magic numbers.
ADEQUATE = arc_with_blocking_tail(f_apex=1.0e3)            # -> "ok"
#: Arc still climbing at the floor: `extend_low` with NO apex — widen blindly.
TRUNCATED = semicircle(f_peak=2.0, f_lo=20.0)
#: Apex measured and prominent, but only 0.67 decades under it: `extend_low`
#: WITH an apex — the one case a piecewise grid beats every preset.
UNDER_SERVED = semicircle(f_peak=30.0, f_lo=6.475)
NO_INTERIOR_APEX = semicircle(f_peak=1.0e6, f_lo=20.0)     # -> "no_arc"
UNREADABLE = (np.array([2.0e5, 1.0e3]), np.array([1.0, 2.0]))   # -> "no_data"


# ── The characterization: with the flag off, nothing moved ───────────────────


class TestTheManualTabIsUnchangedWithTheFlagOff:
    """This is the pin. It fails the moment the off path stops being the old one."""

    def test_measure_one_writes_the_bytes_the_preset_build_wrote(
            self, qapp, tmp_path, calls):
        pico, _ = measure(tmp_path, ADEQUATE, scout=planner(False, False))

        assert pico.scripts_seen == [preset_bytes(tmp_path, 3)]

    def test_measure_one_makes_exactly_one_preset_build_and_no_segmented_one(
            self, qapp, tmp_path, calls):
        measure(tmp_path, ADEQUATE, scout=planner(False, False))

        assert [kind for kind, _ in calls] == ["preset"]
        assert calls[0][1] == {"mux_ch": 3, "mVac": 10, "f_hi": 200_000,
                               "f_lo": 6_475, "npts": 27, "mVdc": 0}

    def test_the_recorded_params_are_the_caller_s_own_dict(self, qapp, tmp_path):
        pico = FakePico(raw_from(*ADEQUATE), tmp_path)
        worker = manual_worker(FakeManager(pico), scout=planner(False, False))

        payload = worker._measure_one(3, "pico1", None)

        assert payload["eis_result"].eis_params is worker._eis_params
        assert "eis_scout_verdict" not in payload["eis_result"].eis_params

    def test_a_worker_with_no_planner_at_all_still_measures(self, qapp, tmp_path):
        # The GUI always supplies one, but `scout` defaults to None and the
        # off-path must not depend on an object being there to decline.
        pico, _ = measure(tmp_path, ADEQUATE, scout=None)

        assert pico.scripts_seen == [preset_bytes(tmp_path, 3)]

    def test_a_global_actuate_cannot_switch_this_tab_on(self, qapp, tmp_path, calls):
        """The manual tab's own control is the authority there, full stop.

        This tab is where uncharacterised samples get measured, and the planner
        assumes one arc. A deployment-wide flag must not be able to start planning
        sweeps around whichever arc happens to be tallest on somebody's two-arc
        stack — that decision belongs to the person who put it on the board.
        """
        # Global actuate ON, tab checkbox OFF, and on the verdict that would
        # otherwise earn a second sweep.
        pico, payload = measure(tmp_path, TRUNCATED,
                                scout=planner(True, True, manual=False))

        assert [kind for kind, _ in calls] == ["preset"]
        assert pico.scripts_seen == [preset_bytes(tmp_path, 3)]
        # Observing is a separate question and stays on: a verdict recorded
        # against a manual measurement is useful even on a strange sample.
        assert payload["eis_result"].eis_params["eis_scout_verdict"] == "extend_low"
        assert payload["eis_result"].eis_params["eis_sweep_role"] == SWEEP_ROLE_SCOUT

    def test_global_actuate_with_the_tab_off_and_observing_off_is_byte_identical(
            self, qapp, tmp_path, calls):
        pico = FakePico(raw_from(*TRUNCATED), tmp_path)
        worker = manual_worker(FakeManager(pico),
                               scout=planner(False, True, manual=False))

        payload = worker._measure_one(3, "pico1", None)

        assert [kind for kind, _ in calls] == ["preset"]
        assert pico.scripts_seen == [preset_bytes(tmp_path, 3)]
        assert payload["eis_result"].eis_params is worker._eis_params

    def test_the_tab_checkbox_alone_is_enough_to_switch_it_on(
            self, qapp, tmp_path, calls):
        # The converse, and the reason `observing` follows `planning`: with the
        # shipped `enabled = false` a tab-only switch would otherwise be dead.
        measure(tmp_path, TRUNCATED, scout=planner(False, False, manual=True))

        assert [kind for kind, _ in calls] == ["preset", "preset"]


class TestTheTemperatureSweepIsUnchangedWithTheFlagOff:
    def test_build_channel_scripts_writes_the_preset_bytes_per_channel(
            self, tmp_path, monkeypatch, calls):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        arrhenius = sweep(channels=(1, 2))
        arrhenius._scout_planner = planner(False, False)

        arrhenius._build_channel_scripts()

        for ch in (1, 2):
            written = (tmp_path / f"softae_ch{ch}.mscr").read_text()
            assert written == preset_bytes(tmp_path, ch)
        assert [kind for kind, _ in calls] == ["preset", "preset"]

    def test_the_recorded_params_stay_the_caller_s_own_dict(self, tmp_path):
        arrhenius = sweep(channels=(1,))
        arrhenius._scout_planner = planner(False, False)

        assert arrhenius._eis_params_for(1) is arrhenius.config.eis_params

    def test_a_missing_mscr_library_is_still_survivable(self, tmp_path, monkeypatch):
        # The pre-wiring code swallowed ImportError so the sweep could be
        # exercised without the driver package; the refactor keeps that.
        import builtins

        real_import = builtins.__import__

        def refuse(name, *a, **kw):
            if name == "softae.drivers.mscr_library":
                raise ImportError(name)
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", refuse)
        arrhenius = sweep(channels=(1,))
        arrhenius._scout_planner = planner(False, False)

        arrhenius._build_channel_scripts()  # must not raise


# ── Observing: enabled, actuate off ──────────────────────────────────────────


class TestObservingChangesTheRecordAndNotTheSweep:
    def test_the_verdict_lands_on_the_row_it_was_drawn_from(self, qapp, tmp_path):
        _, payload = measure(tmp_path, ADEQUATE, scout=planner(True, False))

        params = payload["eis_result"].eis_params
        assert params["eis_scout_verdict"] == "ok"
        assert params["eis_sweep_role"] == SWEEP_ROLE_SCOUT
        assert params["eis_scout_apex_hz"] > 0.0

    def test_observing_alone_never_takes_a_second_sweep(self, qapp, tmp_path, calls):
        # Even on the verdict that WOULD earn a follow-up under `actuate`.
        pico, _ = measure(tmp_path, TRUNCATED, scout=planner(True, False))

        assert [kind for kind, _ in calls] == ["preset"]
        assert pico.scripts_seen == [preset_bytes(tmp_path, 3)]

    def test_one_channel_s_verdict_never_lands_on_another_channel_s_row(
            self, qapp, tmp_path):
        # `eis_params` is one dict shared by every channel of a manual run. A
        # verdict stamped on the shared dict would leave the first channel's
        # saved result claiming the last channel's conclusion — provenance that
        # looks trustworthy and is not, which is the defect this wiring is
        # supposed to be closing rather than opening.
        pico = FakePico(raw_from(*ADEQUATE), tmp_path)
        worker = manual_worker(FakeManager(pico), scout=planner(True, False))
        first = worker._measure_one(3, "pico1", None)["eis_result"]

        pico._raw = raw_from(*UNREADABLE)
        second = worker._measure_one(7, "pico1", None)["eis_result"]

        assert first.eis_params["eis_scout_verdict"] == "ok"
        assert second.eis_params["eis_scout_verdict"] == "no_data"
        assert "eis_scout_verdict" not in worker._eis_params


# ── Planning: the sweep already taken is usually the measurement ─────────────


class TestAnAdequateSweepIsNeverReMeasured:
    """The economics. A scout that doubled every acquisition would be a defect."""

    @pytest.mark.parametrize(
        "spectrum, verdict",
        [(ADEQUATE, "ok"), (NO_INTERIOR_APEX, "no_arc"), (UNREADABLE, "no_data")],
    )
    def test_one_sweep_and_only_one(self, qapp, tmp_path, calls, spectrum, verdict):
        pico, payload = measure(tmp_path, spectrum, scout=planner(True, True))

        assert payload["eis_result"].eis_params["eis_scout_verdict"] == verdict
        assert [kind for kind, _ in calls] == ["preset"]
        assert pico.scripts_seen == [preset_bytes(tmp_path, 3)]

    def test_the_accepted_sweep_is_recorded_as_the_scout_sweep(self, qapp, tmp_path):
        _, payload = measure(tmp_path, ADEQUATE, scout=planner(True, True))

        params = payload["eis_result"].eis_params
        assert params["eis_sweep_role"] == SWEEP_ROLE_SCOUT
        assert "eis_scout_sweep_s" not in params      # there was no second sweep
        assert params["npts"] == PRESET["npts"]

    def test_a_shoulder_below_the_prominence_cut_is_not_worth_a_second_sweep(
            self, qapp, tmp_path, calls):
        scout = planner(True, True, apex_prominence_min=0.99)
        _, payload = measure(tmp_path, ADEQUATE, scout=scout)

        assert payload["eis_result"].eis_params["eis_scout_verdict"] == "no_arc"
        assert [kind for kind, _ in calls] == ["preset"]


class TestAnInadequateSweepEarnsAWiderOne:
    def test_a_truncated_arc_steps_one_rung_down_the_preset_ladder(
            self, qapp, tmp_path, calls):
        pico, _ = measure(tmp_path, TRUNCATED, scout=planner(True, True))

        assert [kind for kind, _ in calls] == ["preset", "preset"]
        assert pico.scripts_seen[0] == preset_bytes(tmp_path, 3)
        assert pico.scripts_seen[1] == preset_bytes(tmp_path, 3, {**PRESET, **STANDARD})

    def test_one_rung_not_straight_to_the_widest(self, qapp, tmp_path, calls):
        # Quick -> Longest is +499 s on a spectrum that may close at Standard's
        # floor, and the follow-up gets its own verdict, so the ladder can be
        # walked again if it has to be.
        measure(tmp_path, TRUNCATED, scout=planner(True, True))

        assert calls[1][1]["f_lo"] == STANDARD["f_lo_mHz"]

    def test_the_row_records_the_follow_up_and_what_triggered_it(
            self, qapp, tmp_path):
        _, payload = measure(tmp_path, TRUNCATED, scout=planner(True, True))

        params = payload["eis_result"].eis_params
        assert params["eis_sweep"] == WIDER_PRESET_SWEEP_TAG
        assert params["eis_sweep_role"] == SWEEP_ROLE_FOLLOW_UP
        assert params["eis_preset"] == "Standard"
        assert params["eis_scout_trigger_verdict"] == "extend_low"
        assert (params["f_lo_mHz"], params["npts"]) == (
            STANDARD["f_lo_mHz"], STANDARD["npts"])
        # A real preset grid, so its stopwatch anchor still applies.
        assert params["eis_duration_basis"] == "measured"

    def test_the_discarded_sweep_s_cost_stays_in_the_record(self, qapp, tmp_path):
        # The superseded spectrum is not stored, so without this the acquisition
        # would look cheaper than it was.
        _, payload = measure(tmp_path, TRUNCATED, scout=planner(True, True))

        assert "eis_scout_sweep_s" in payload["eis_result"].eis_params

    def test_the_follow_up_keeps_the_operator_s_excitation(self, qapp, tmp_path, calls):
        loud = {**PRESET, "mv_ac": 25, "mv_dc": 5}
        _, payload = measure(tmp_path, TRUNCATED, scout=planner(True, True),
                             params=loud)

        # The follow-up widens the band; it does not re-specify the excitation.
        assert calls[1][1]["mVac"] == 25 and calls[1][1]["mVdc"] == 5
        assert payload["eis_result"].eis_params["mv_ac"] == 25

    def test_at_the_bottom_of_the_ladder_the_sweep_taken_stands(
            self, qapp, tmp_path, calls):
        pico, payload = measure(tmp_path, TRUNCATED, scout=planner(True, True),
                                params=WIDEST)

        assert [kind for kind, _ in calls] == ["preset"]
        assert payload["eis_result"].eis_params["eis_sweep_role"] == SWEEP_ROLE_SCOUT

    def test_the_operator_s_own_params_are_not_edited_on_the_way_past(
            self, qapp, tmp_path):
        pico = FakePico(raw_from(*TRUNCATED), tmp_path)
        worker = manual_worker(FakeManager(pico), scout=planner(True, True))

        worker._measure_one(3, "pico1", None)

        assert worker._eis_params == PRESET


class TestASegmentedFollowUpNeedsAnObservedApex:
    """The one case a piecewise grid beats every preset, driven end to end.

    A segmented sweep can only be laid around an apex the scout has *seen*. An
    arc found with too little band under it is exactly that: the apex is a
    measurement, the sweep just stopped too soon below it. Everything else —
    ``no_arc``, ``no_data``, an open arc with no interior maximum — widens
    blindly by one preset rung instead, because there is no centre to lay a grid
    around and this module does not invent one.
    """

    def under_served_apex(self, apex_hz: float = 1.0e3) -> ScoutDecision:
        return ScoutDecision("extend_low", apex_hz, 0.2, 0.5, "open")

    def test_a_live_verdict_reaches_the_segmented_branch(
            self, qapp, tmp_path, calls):
        # UNDER_SERVED is a real spectrum through the real decision — no injected
        # ScoutDecision anywhere in this test.
        pico, payload = measure(tmp_path, UNDER_SERVED, scout=planner(True, True))

        assert [kind for kind, _ in calls] == ["preset", "segmented"]
        assert payload["eis_result"].eis_params["eis_sweep"] == SEGMENTED_SWEEP_TAG
        assert pico.scripts_seen[1].count("meas_loop_eis") >= 2

    def test_the_plan_is_a_plausible_multi_segment_grid(self, qapp, tmp_path):
        _, payload = measure(tmp_path, UNDER_SERVED, scout=planner(True, True))
        params = payload["eis_result"].eis_params

        segments = params["eis_segments"]
        assert len(segments) >= 2
        # Descending, non-overlapping, and the dense band straddles the apex.
        assert all(a > b for a, b, _ in segments)
        assert all(segments[i][0] < segments[i - 1][1]
                   for i in range(1, len(segments)))
        apex = params["eis_scout_apex_hz"]
        assert segments[1][0] > apex > segments[1][1]

    def test_the_follow_up_reaches_below_the_sweep_that_asked_for_it(
            self, qapp, tmp_path):
        # A follow-up may never be narrower. Here that is arithmetic rather than
        # structure: `extend_low` means the apex sits under one decade of band,
        # and the plan puts its floor a decade below the apex.
        _, payload = measure(tmp_path, UNDER_SERVED, scout=planner(True, True))

        floor_hz = payload["eis_result"].eis_params["f_lo_mHz"] / 1000.0
        assert floor_hz < PRESET["f_lo_mHz"] / 1000.0

    def test_the_row_carries_the_same_provenance_the_preset_follow_up_does(
            self, qapp, tmp_path):
        _, payload = measure(tmp_path, UNDER_SERVED, scout=planner(True, True))
        params = payload["eis_result"].eis_params

        assert params["eis_sweep_role"] == SWEEP_ROLE_FOLLOW_UP
        assert params["eis_scout_trigger_verdict"] == "extend_low"
        assert "eis_scout_sweep_s" in params
        assert params["npts"] == sum(n for _, _, n in params["eis_segments"])

    def test_the_stored_segments_are_the_emitted_ones(self, tmp_path, calls):
        scout = planner(True, True)
        path = str(tmp_path / "ch3.mscr")

        params = scout.build_follow_up(path, 3, PRESET, self.under_served_apex())

        emitted = [list(seg) for seg in calls[0][1]["segments"]]
        assert emitted == [[a, b, n] for a, b, n in params["eis_segments"]]

    def test_duration_provenance_is_extrapolated_and_says_why(self, tmp_path):
        scout = planner(True, True)

        params = scout.build_follow_up(
            str(tmp_path / "ch3.mscr"), 3, PRESET, self.under_served_apex())

        # A per-sample grid can never match an anchor. The interlock is not
        # defeated here; the reason string is what separates "structural" from
        # "somebody edited a preset".
        assert params["eis_duration_basis"] == SEGMENTED_DURATION_BASIS
        assert "per sample" in params["eis_duration_basis_reason"]

    @pytest.mark.parametrize(
        "spectrum", [ADEQUATE, TRUNCATED, NO_INTERIOR_APEX, UNREADABLE])
    def test_a_verdict_with_no_observed_apex_never_reaches_it(
            self, qapp, tmp_path, calls, spectrum):
        # The withholding still holds where it should: these four have no apex
        # that was both measured and prominent, so none of them may be planned
        # around. TRUNCATED still widens — by one preset rung, blindly.
        measure(tmp_path, spectrum, scout=planner(True, True))

        assert "segmented" not in [kind for kind, _ in calls]

    def test_an_unrenderable_plan_falls_back_to_the_wider_preset(
            self, tmp_path, monkeypatch, calls):
        import softae.drivers.mscr_library as lib

        def explode(*_a, **_kw):
            raise ValueError("no exact literal")

        monkeypatch.setattr(lib, "eis_segmented_mscrbuild", explode)
        scout = planner(True, True)

        params = scout.build_follow_up(
            str(tmp_path / "ch3.mscr"), 3, PRESET, self.under_served_apex())

        assert params["eis_sweep"] == WIDER_PRESET_SWEEP_TAG
        assert [kind for kind, _ in calls] == ["preset"]


# ── Nothing survives an acquisition unit ─────────────────────────────────────


class TestNoPlanOutlivesTheMeasurementThatMadeIt:
    def test_the_planner_remembers_nothing_between_calls(self, tmp_path):
        # Structural, not a policy the planner has to remember to apply: a
        # decision is consumed by the measurement that produced it. `None` here
        # stands for "a later unit, with no decision of its own yet".
        from softae.analysis.eis_data import EISResult

        scout = planner(True, True)
        f, y = ADEQUATE
        scout.observe(3, EISResult.from_raw(raw_from(f, y), channel=3))

        assert scout.build_follow_up(str(tmp_path / "ch3.mscr"), 3, PRESET, None) is None
        assert not (tmp_path / "ch3.mscr").exists()

    def test_the_planner_carries_no_per_channel_state_to_go_stale(self):
        import dataclasses

        # The rule is "no plan survives an acquisition unit". Enforcing it by
        # having nowhere to put one is stronger than remembering to clear a
        # cache at every boundary — and there are three boundaries here (Run
        # press, sweep, RH setpoint) that no single call site owns.
        # Named exhaustively rather than by exclusion, so adding a field is a
        # decision somebody has to make here rather than something that slips in.
        # All three are set at construction and describe *configuration*; none
        # can hold a plan, a decision, or anything keyed by channel.
        assert [f.name for f in dataclasses.fields(ScoutPlanner)] == [
            "settings", "site", "actuate"]


class TestTheTemperatureSweepObservesOnly:
    def test_actuate_cannot_change_a_script_on_this_path(
            self, tmp_path, monkeypatch, calls):
        # The workflow holds every measurement, so the only granularity this
        # path could offer is per-sweep — which is exactly the cross-boundary
        # carry the always-replan rule forbids.
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        arrhenius = sweep(channels=(1,))
        arrhenius._scout_planner = planner(True, True)

        arrhenius._build_channel_scripts()

        assert [kind for kind, _ in calls] == ["preset"]
        assert (tmp_path / "softae_ch1.mscr").read_text() == preset_bytes(tmp_path, 1)

    def test_observing_stamps_the_row_without_editing_the_configuration(
            self, tmp_path):
        from softae.analysis.eis_data import EISResult

        arrhenius = sweep(channels=(1,))
        arrhenius._scout_planner = planner(True, False)

        f, y = ADEQUATE
        row = EISResult.from_raw(
            raw_from(f, y), channel=1, eis_params=arrhenius._eis_params_for(1))
        arrhenius._scout.observe(1, row)

        assert row.eis_params["eis_scout_verdict"] == "ok"
        assert arrhenius.config.eis_params == PRESET


# ── Failure degrades to the sweep already taken ──────────────────────────────


class TestFailureNeverAbortsAMeasurement:
    def test_a_raising_decision_still_returns_a_spectrum(
            self, qapp, tmp_path, monkeypatch, calls):
        import softae.core.eis_scout_scripts as wiring

        def explode(*_a, **_kw):
            raise RuntimeError("arc guard blew up")

        monkeypatch.setattr(wiring, "scout_decision", explode)
        _, payload = measure(tmp_path, ADEQUATE, scout=planner(True, True))

        assert payload["eis_result"].npts > 0
        assert [kind for kind, _ in calls] == ["preset"]

    def test_an_unreadable_preset_ladder_leaves_the_sweep_taken_standing(
            self, qapp, tmp_path, monkeypatch, calls):
        import softae.config.loader as loader

        monkeypatch.setattr(loader, "eis_presets", lambda: (_ for _ in ()).throw(
            RuntimeError("config gone")))
        _, payload = measure(tmp_path, TRUNCATED, scout=planner(True, True))

        assert [kind for kind, _ in calls] == ["preset"]
        assert payload["eis_result"].npts > 0


# ── Provenance shape ─────────────────────────────────────────────────────────


class TestSegmentedParams:
    SEGMENTS = ((200_000.0, 10_000.0, 10), (9_500.0, 100.0, 24))

    def test_the_scalar_triple_is_the_aggregate_over_bands(self):
        params = segmented_params(PRESET, self.SEGMENTS)
        assert params["npts"] == 34
        assert params["f_hi"] == 200_000.0
        assert params["f_lo_mHz"] == 100_000.0

    def test_the_preset_s_own_triple_does_not_survive(self):
        # The whole point: a row must not claim a sweep that never ran.
        params = segmented_params(PRESET, self.SEGMENTS)
        assert params["npts"] != PRESET["npts"]
        assert params["f_lo_mHz"] != PRESET["f_lo_mHz"]
        assert params["eis_sweep"] == SEGMENTED_SWEEP_TAG

    def test_amplitude_settings_are_carried_through_untouched(self):
        params = segmented_params({**PRESET, "mv_ac": 25}, self.SEGMENTS)
        assert params["mv_ac"] == 25 and params["mv_dc"] == 0

    def test_it_does_not_mutate_the_caller_s_dict(self):
        base = dict(PRESET)
        segmented_params(base, self.SEGMENTS)
        assert base == PRESET
