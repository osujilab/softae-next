"""Storage, CLI and wiring for the thickness harness.

The wiring test is the one that matters most: ``resolve_thickness_cm`` has ranked
``profilometry`` above ``predicted`` since E0, and until this harness existed *nothing
could supply it*. A top precedence tier that no caller can reach reads as finished from
inside the module, which is why it survived several passes unnoticed.
"""

from __future__ import annotations

import pytest

from softae.core.data_store import DataStore
from softae.core.thickness_series import plan_series
from softae.tools import thickness as cli


@pytest.fixture
def store(tmp_path):
    s = DataStore(str(tmp_path))
    yield s
    s.close()


class TestStorage:
    def test_a_plan_round_trips_through_the_database(self, store):
        plan = plan_series([100, 150, 200, 250], range(1, 17), seed=4,
                           plan_id="geo-1", created_at="2026-08-06")
        store.record_thickness_plan(plan)

        back = store.thickness_plan("geo-1")
        assert back is not None
        assert back.assignment == plan.assignment
        assert back.achieved_correlation == pytest.approx(plan.achieved_correlation)

    def test_an_absent_plan_is_none_not_an_error(self, store):
        assert store.thickness_plan("nope") is None

    def test_a_plan_needs_an_id(self, store):
        plan = plan_series([100, 200, 300, 400], range(1, 9), seed=1)
        with pytest.raises(ValueError, match="plan_id"):
            store.record_thickness_plan(plan)

    def test_replanning_replaces_rather_than_duplicating(self, store):
        for seed in (1, 2):
            store.record_thickness_plan(
                plan_series([100, 150, 200, 250], range(1, 17), seed=seed,
                            plan_id="geo-1"))
        assert len(store.thickness_plans()) == 1

    def test_measurements_append_because_remeasuring_is_a_second_observation(self,
                                                                            store):
        store.record_thickness(7, 148.2, plan_id="geo-1")
        store.record_thickness(7, 151.9, plan_id="geo-1")
        rows = store.measured_thickness(plan_id="geo-1")
        assert len(rows) == 2
        # ...and the most recent is what the lookup takes.
        assert store.thickness_for(7, plan_id="geo-1") == pytest.approx(151.9)

    def test_an_unmeasured_channel_is_none_not_a_substituted_nominal(self, store):
        store.record_thickness(7, 148.2)
        assert store.thickness_for(9) is None

    def test_provenance_fields_survive(self, store):
        store.record_thickness(3, 149.0, plan_id="p", run_id="r", level_um=150.0,
                               uncertainty_um=2.5, instrument="Dektak XT",
                               operator="po", notes="edge bead avoided")
        row = store.measured_thickness(plan_id="p")[0]
        assert row["level_um"] == 150.0
        assert row["uncertainty_um"] == 2.5
        assert row["instrument"] == "Dektak XT"
        assert row["measured_at"]

    def test_nan_uncertainty_becomes_null(self, store):
        store.record_thickness(3, 149.0, uncertainty_um=float("nan"))
        assert store.measured_thickness()[0]["uncertainty_um"] is None

    def test_filters_do_not_leak_across_plans(self, store):
        store.record_thickness(1, 100.0, plan_id="a")
        store.record_thickness(1, 200.0, plan_id="b")
        assert store.thickness_for(1, plan_id="a") == pytest.approx(100.0)
        assert store.thickness_for(1, plan_id="b") == pytest.approx(200.0)


class TestObjectiveWiring:
    """Closing the Unreachable path."""

    def test_a_measured_thickness_outranks_the_twins_prediction(self, store):
        from softae.core.autonomous_wiring import make_thickness_lookup

        store.start_run("run1") if hasattr(store, "start_run") else None
        store.record_thickness(5, 148.2, run_id="run1")

        class _Store:
            def __init__(self, inner):
                self._inner = inner

            def thickness_for(self, ch, **kw):
                return self._inner.thickness_for(ch, **kw)

            def predicted_thickness_um(self, run_id, channel):
                return 150.0

        reading = make_thickness_lookup(_Store(store), "run1")(5)
        assert reading is not None
        assert reading.method == "profilometry"
        assert reading.um == pytest.approx(148.2)

    def test_without_a_measurement_it_falls_back_to_predicted(self, store):
        from softae.core.autonomous_wiring import make_thickness_lookup

        class _Store:
            def thickness_for(self, ch, **kw):
                return None

            def predicted_thickness_um(self, run_id, channel):
                return 150.0

        reading = make_thickness_lookup(_Store(), "run1")(5)
        assert reading is not None and reading.method == "predicted"

    def test_a_store_without_the_new_method_still_works(self):
        """A DataStore predating the harness must not break the objective."""
        from softae.core.autonomous_wiring import make_thickness_lookup

        class _Old:
            def predicted_thickness_um(self, run_id, channel):
                return 150.0

        reading = make_thickness_lookup(_Old(), "run1")(5)
        assert reading is not None and reading.method == "predicted"

    def test_the_source_reaches_the_cell_constant(self):
        """The point of carrying the method at all — σ's provenance must be right."""
        from softae.analysis.eis.geometry import cell_constant_for_sample
        from softae.core.autonomous_wiring import ThicknessReading, _thickness_parts

        um, method = _thickness_parts(ThicknessReading(148.2, "profilometry"))
        cell = cell_constant_for_sample(**{f"{method}_um": um})
        assert cell is not None
        assert cell.thickness_method == "profilometry"

    def test_a_bare_float_is_still_read_as_the_twins_prediction(self):
        """Backward compatibility: that is what the parameter always meant."""
        from softae.core.autonomous_wiring import _thickness_parts

        assert _thickness_parts(150.0) == (150.0, "predicted")
        assert _thickness_parts(None) == (None, "unavailable")


class TestAThicknessWithoutItsAreaIsNotAThickness:
    """P.11 — the σ objective must refuse a quotient with an unknown denominator.

    A thickness is ``final_volume_uL / area_mm2 * 1000``. The 4-stripe board's
    deposit area moved from 4.0 mm² to 18.704 mm² on 2026-08-07, so rows written
    either side of that differ by 4.676× **in the same column, with the same
    units**, and a formulation row does not record its board. P.7 started writing
    the denominator beside the quotient; until this, nothing read it — so a
    pre-correction row still arrived at the objective tagged ``"predicted"`` with
    full confidence, which is verbatim the corruption ``make_thickness_lookup``'s
    own docstring says it exists to prevent.
    """

    def _store(self, *, um, area, method="predicted", measured=None):
        """A store speaking the P.11 reader, standing in for a real DataStore."""
        from softae.core.data_store import PredictedThicknessRecord

        class _Store:
            def thickness_for(self, ch, **kw):
                return measured

            def predicted_thickness_um(self, run_id, channel):
                return um

            def predicted_thickness_record(self, run_id, channel):
                if um is None:
                    return None
                return PredictedThicknessRecord(um=um, area_mm2=area, method=method)

        return _Store()

    def test_a_thickness_with_no_recorded_area_is_unavailable_because_its_basis_is_unknown(
            self):
        """NULL area means *unknown basis*, and unknown basis means no reading.

        `None` is not a degraded answer here -- it is the correct one, and every
        layer below already absorbs it: `_thickness_parts` yields
        `(None, "unavailable")`, `_sigma_from_eis_raw` then omits the `*_um` kwarg,
        `cell_constant_for_sample` returns `None`, and `analyze_spectrum` reports σ
        unavailable instead of computing one from a nominal. Handing the number
        through would instead steer the campaign by a σ that is wrong by up to
        4.676× while every step reports success.
        """
        from softae.core.autonomous_wiring import make_thickness_lookup

        store = self._store(um=150.0, area=None, method=None)
        assert make_thickness_lookup(store, "run1")(5) is None

    def test_a_thickness_recorded_with_its_area_is_returned_normally(self):
        """The guard must cost nothing to a row that carries its own provenance.

        Every campaign cast since P.7 records both, so this is the ordinary path.
        A guard that also suppressed these would silence σ for the whole system
        rather than for the rows whose basis is genuinely unknown.
        """
        from softae.core.autonomous_wiring import make_thickness_lookup

        reading = make_thickness_lookup(self._store(um=150.0, area=18.7038), "run1")(5)
        assert reading is not None
        assert reading.method == "predicted"
        assert reading.um == pytest.approx(150.0)

    def test_a_null_area_is_never_rescaled_because_we_cannot_know_which_board_it_came_from(
            self):
        """Refusing beats correcting: a rescale would invent a number.

        4.676× is the ratio *for the 4-stripe board*. A NULL-area row may equally
        have been cast on a board the correction never touched, in which case its
        thickness was already right and rescaling would break it. Since the row does
        not say which, no factor is defensible -- so the lookup must yield nothing
        rather than any of 150.0, 32.1 (150/4.676) or 701.4 (150×4.676).
        """
        from softae.core.autonomous_wiring import make_thickness_lookup

        reading = make_thickness_lookup(self._store(um=150.0, area=None), "run1")(5)
        assert reading is None

    def test_a_store_without_the_new_method_keeps_the_old_behaviour_so_stubs_do_not_break(
            self):
        """A store with no area concept is not a store hiding an area.

        Reaching the new reader through `getattr` -- the pattern `thickness_for`
        already uses four lines above -- means a stub that implements only
        `predicted_thickness_um` keeps working unchanged. That is right, not merely
        convenient: such a stub has no areas at all, and a test exercising something
        else should not be forced to grow one to keep its thickness.
        """
        from softae.core.autonomous_wiring import make_thickness_lookup

        class _Old:
            def predicted_thickness_um(self, run_id, channel):
                return 150.0

        assert not hasattr(_Old(), "predicted_thickness_record")
        reading = make_thickness_lookup(_Old(), "run1")(5)
        assert reading is not None
        assert reading.um == pytest.approx(150.0)
        assert reading.method == "predicted"

    def test_profilometry_still_outranks_a_predicted_reading(self):
        """The ladder is unchanged: the guard applies to the predicted tier only.

        A measured film has an area only in the trivial sense -- the profilometer
        traced the actual deposit, so no denominator was assumed. Withholding it
        because the *twin's* row lacks an area would discard the one tier
        `resolve_thickness_cm` ranks highest, on account of a defect in the tier
        below it.
        """
        from softae.core.autonomous_wiring import make_thickness_lookup

        store = self._store(um=150.0, area=None, method=None, measured=148.2)
        reading = make_thickness_lookup(store, "run1")(5)
        assert reading is not None
        assert reading.method == "profilometry"
        assert reading.um == pytest.approx(148.2)

    def test_withholding_a_thickness_warns_so_an_operator_learns_why_sigma_went_quiet(
            self):
        """Silence is the failure mode this guard could introduce.

        The campaign keeps running and simply stops reporting σ, which from outside
        is indistinguishable from a twin that never spoke. The warning names the
        channel and says the area was never recorded, so the reason is in the log
        rather than reconstructed from the database months later.
        """
        import structlog

        from softae.core.autonomous_wiring import make_thickness_lookup

        with structlog.testing.capture_logs() as logs:
            make_thickness_lookup(self._store(um=150.0, area=None), "run1")(5)

        withheld = [e for e in logs
                    if e["event"] == "thickness_withheld_area_never_recorded"]
        assert withheld, "an operator must be told which channel lost its thickness"
        assert withheld[0]["channel"] == 5
        assert withheld[0]["log_level"] == "warning"


class TestCLI:
    def _run(self, tmp_path, *argv):
        return cli.main([*argv, "--project", str(tmp_path)])

    def test_plan_then_record_then_check_is_clean(self, tmp_path, capsys):
        assert self._run(tmp_path, "plan", "--levels", "100,150,200,250",
                         "--channels", "1-16", "--id", "geo-1", "--seed", "5") == 0
        out = capsys.readouterr().out
        assert "adequate for a geometry series" in out
        assert "Cast list" in out

        s = DataStore(str(tmp_path))
        try:
            plan = s.thickness_plan("geo-1")
            assert plan is not None
            for ch, level in plan.as_rows():
                self._run(tmp_path, "record", "--plan", "geo-1",
                          "--channel", str(ch), "--um", str(level + 1.0))
        finally:
            s.close()
        capsys.readouterr()

        assert self._run(tmp_path, "check", "--plan", "geo-1") == 0
        assert "within the" in capsys.readouterr().out

    def test_check_exits_nonzero_on_a_confounded_series(self, tmp_path, capsys):
        """A script must not be able to ignore this."""
        for ch, level in zip(range(27, 33), [200, 200, 150, 150, 100, 100]):
            self._run(tmp_path, "record", "--channel", str(ch), "--um", str(level),
                      "--level", str(level))
        capsys.readouterr()

        assert self._run(tmp_path, "check") == cli.EXIT_CONFOUNDED
        out = capsys.readouterr().out
        assert "CONFOUNDED" in out
        assert "F12" in out

    def test_recording_flags_a_large_deviation_from_the_planned_level(self, tmp_path,
                                                                     capsys):
        self._run(tmp_path, "plan", "--levels", "100,150,200,250",
                  "--channels", "1-16", "--id", "geo-1", "--seed", "5")
        s = DataStore(str(tmp_path))
        try:
            ch, level = s.thickness_plan("geo-1").as_rows()[0]
        finally:
            s.close()
        capsys.readouterr()

        self._run(tmp_path, "record", "--plan", "geo-1", "--channel", str(ch),
                  "--um", str(level * 0.5))
        assert "Worth a second look" in capsys.readouterr().out

    def test_an_impossible_plan_fails_with_a_reason(self, tmp_path, capsys):
        assert self._run(tmp_path, "plan", "--levels", "100,200",
                         "--channels", "1-2") == cli.EXIT_FAILED
        assert "no assignment" in capsys.readouterr().err

    def test_a_two_level_plan_warns_it_cannot_answer_e5(self, tmp_path, capsys):
        assert self._run(tmp_path, "plan", "--levels", "100,250",
                         "--channels", "1-16", "--seed", "3") == 0
        assert "NOT sufficient for a geometry series" in capsys.readouterr().out

    def test_check_with_nothing_recorded_says_so(self, tmp_path, capsys):
        assert self._run(tmp_path, "check") == cli.EXIT_FAILED
        assert "No thickness measurements" in capsys.readouterr().err

    def test_list_shows_plans_and_measurements(self, tmp_path, capsys):
        self._run(tmp_path, "plan", "--levels", "100,150,200,250",
                  "--channels", "1-8", "--id", "geo-1", "--seed", "2")
        self._run(tmp_path, "record", "--channel", "1", "--um", "101.5")
        capsys.readouterr()

        self._run(tmp_path, "list", "--plans")
        assert "geo-1" in capsys.readouterr().out
        self._run(tmp_path, "list")
        assert "101.50" in capsys.readouterr().out

    def test_channel_syntax_matches_the_other_tools(self):
        assert cli._parse_channels("1, 3-6") == [1, 3, 4, 5, 6]
        assert cli._parse_channels("5,5,5") == [5]
