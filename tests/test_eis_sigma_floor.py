"""The sweep floor as a **conductivity** floor, and the guardrail that states it.

Run ``20260811T023757Z_equilibration_characterization`` took 1440 spectra at
``Quick`` (200 kHz → 20 Hz) and **476 of them never closed their arc** — the −Z″
peak sat on the lowest measured point, so R₁ came from extrapolating the
high-frequency limb. Against 929 closed arcs that extrapolation measures a 60.9 %
median *overestimate* at only 1.5× past the apex (175 % with a full CPE fit): a
systematic bias, not scatter, and therefore not something an error bar can absorb.

The apex is predictable — ``f_peak = 1/(2*pi*R*C_cell)``, and C_cell is a property
of the **board** (0.09 nF median while R moved 109×) — so a frequency floor is a
conductivity floor and can be computed rather than tuned. ``Quick`` now sweeps to
7 Hz for that reason.

These tests pin the computation, the plan line that states its reach, and the
thing most likely to be quietly undone: ``Quick``'s stopwatch anchor timed the
*20 Hz* sweep and had to be retired with the floor. Restoring it without re-timing
would put the word MEASURED back on a projection of half a night.
"""

from __future__ import annotations

import math

import pytest

from softae.config import loader
from softae.core.autonomous_wiring import CampaignSpec
from softae.core.eis_scripts import (
    CELL_CAPACITANCE_F,
    EISParams,
    sigma_floor_S_per_cm,
)
from softae.core.measurement_spec import MeasurementSpec
from softae.core.preflight import (
    EIS_MEASURED_S_PER_CHANNEL,
    eis_duration_basis,
    project_campaign,
)
from softae.core.task_catalog import TaskCatalog
from softae.tools.equilibration import _cmd_plan, build_config, build_parser
from softae.workflows.equilibration import DEFAULT_EIS_PRESET

#: The 4-stripe board's nominal geometry, in the argument order the helper takes.
BOARD = (0.2, 0.02, 0.2)

#: The same three terms as CLI flags, for the plan-level tests.
GEOMETRY_FLAGS = ("--electrode-l-cm", "0.2", "--electrode-t-cm", "0.02",
                  "--electrode-w-cm", "0.2")


def _args(*argv):
    return build_parser().parse_args(list(argv))


@pytest.fixture()
def project(monkeypatch, tmp_path):
    """Point the default store at a temporary project directory.

    ``plan`` opens the store to answer the thickness note, so without this the
    reach tests would read (and create rows in) the real project.
    """
    monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
    return tmp_path


# ── The helper ───────────────────────────────────────────────────────────────

class TestSigmaFloor:
    def test_floor_at_seven_hz_reaches_two_e_minus_seven(self):
        """The number the 7 Hz floor was chosen to buy.

        σ_min = 2π·f_lo·C_cell·L/(t·w) — measured C, not fitted, so this is a
        prediction the board makes about a material nobody has run yet. The
        settled σ of the source run spanned ~1e-6 to 1.4e-4, so 7 Hz carries about
        a decade of headroom below anything this rig has seen.
        """
        assert sigma_floor_S_per_cm(7.0, *BOARD) == pytest.approx(2.0e-7, rel=0.05)

    def test_floor_scales_linearly_with_frequency(self):
        """The whole design rule in one assertion: to reach a tenth the
        conductivity, take a tenth the floor.

        It is what makes ``--f-lo-mHz`` usable — an operator holding a material two
        decades less conductive needs no model, only division.
        """
        base = sigma_floor_S_per_cm(7.0, *BOARD)
        assert sigma_floor_S_per_cm(70.0, *BOARD) == pytest.approx(10 * base)
        assert sigma_floor_S_per_cm(0.7, *BOARD) == pytest.approx(base / 10)

    def test_floor_is_the_closed_form_of_f_peak_and_sigma(self):
        """Derived, not curve-fitted: eliminating R between the two definitions.

        A sample sitting exactly at the floor must peak exactly at f_lo. The round
        trip catches a transposed geometry term, which a magnitude assertion on one
        number would not.
        """
        f_lo, (L, t, w) = 7.0, BOARD
        sigma = sigma_floor_S_per_cm(f_lo, L, t, w)
        R = L / (sigma * t * w)
        assert 1.0 / (2 * math.pi * R * CELL_CAPACITANCE_F) == pytest.approx(f_lo)

    @pytest.mark.parametrize("kwargs", [
        {"L_cm": None}, {"t_cm": None}, {"w_cm": None}, {"f_lo_hz": None},
        {"L_cm": 0.0}, {"t_cm": 0.0}, {"w_cm": 0.0}, {"f_lo_hz": 0.0},
        {"t_cm": -0.02},
    ])
    def test_a_missing_or_nonpositive_term_refuses_rather_than_defaults(self, kwargs):
        """No number is better than a number computed from a geometry nobody gave.

        Falling back to a nominal board would quote a reach that is right about the
        physics and wrong about *this* run — the exact failure the reach exists to
        prevent. Same posture ``resolve_thickness_cm`` takes for thickness.
        """
        call = {"f_lo_hz": 7.0, "L_cm": BOARD[0], "t_cm": BOARD[1], "w_cm": BOARD[2]}
        call.update(kwargs)
        assert sigma_floor_S_per_cm(**call) is None


# ── The preset carries the floor; the flag only deviates from it ─────────────

class TestPresetFloor:
    def test_the_quick_preset_sweeps_to_seven_hz(self):
        """The operator decision of 2026-08-14: the measured floor is the DEFAULT,
        not an opt-in. ``Quick`` is ``DEFAULT_PRESET`` and what the equilibration
        runs take, so this is the reach the system has unless told otherwise."""
        assert loader.eis_presets()["Quick"]["f_lo_mHz"] == 7_000
        assert EISParams.from_preset("Quick").f_lo_mHz == 7_000

    def test_the_default_presets_reach_is_two_e_minus_seven(self):
        """The preset and the reach are one statement, checked end to end rather
        than as two constants that could drift apart."""
        f_lo_hz = EISParams.from_preset(DEFAULT_EIS_PRESET).f_lo_mHz / 1000.0
        assert sigma_floor_S_per_cm(f_lo_hz, *BOARD) == pytest.approx(2.0e-7,
                                                                     rel=0.05)

    def test_the_other_presets_floors_are_untouched(self):
        """Only ``Quick`` moved. ``Standard`` and ``Extended`` are still the sweeps
        their stopwatch anchors were read from."""
        assert EISParams.from_preset("Standard").f_lo_mHz == 4_000
        assert EISParams.from_preset("Extended").f_lo_mHz == 1_200

    def test_quicks_other_parameters_did_not_move_with_the_floor(self):
        params = EISParams.from_preset("Quick")
        assert (params.npts, params.f_hi, params.mv_ac) == (25, 200_000, 10)


class TestFloorFlag:
    def test_the_flag_defaults_to_none_and_passes_no_override(self):
        """Default ``None``, deliberately not 7000.

        With ``Quick`` already at 7 Hz, a numeric default would make every ordinary
        run look like an override and trip the EXTRAPOLATED-because-overridden
        branch for a reason that is not true.
        """
        assert _args("plan").f_lo_mHz is None
        assert build_config(_args("plan")).eis_f_lo_mHz is None

    def test_an_ordinary_run_resolves_its_floor_from_the_preset(self):
        assert build_config(_args("plan")).eis_params().f_lo_mHz == 7_000

    def test_a_supplied_floor_reaches_the_sweep_parameters_as_an_override(self):
        """A flag that does not reach ``EISParams`` is a flag that does nothing.

        Checked through ``build_config`` rather than on the namespace, because the
        namespace value is not what the ``.mscr`` files are built from.
        """
        config = build_config(_args("plan", "--f-lo-mHz", "500"))
        assert config.eis_f_lo_mHz == 500
        assert config.eis_params().f_lo_mHz == 500
        # One key overridden, not the sweep replaced.
        assert config.eis_params().npts == EISParams.from_preset("Quick").npts

    def test_a_zero_floor_is_refused_because_it_reaches_nothing(self):
        with pytest.raises(ValueError, match="not a frequency"):
            build_config(_args("plan", "--f-lo-mHz", "0"))

    def test_the_floor_travels_in_the_saved_plan(self, tmp_path):
        """A design value absent from the plan file is one that reverts on ``run``
        -- the 2026-08-10 defect this module exists to prevent."""
        from softae.tools.equilibration import load_plan, write_plan

        path = write_plan(_args("plan", "--f-lo-mHz", "500"), tmp_path / "p.toml")
        assert load_plan(path)["f_lo_mHz"] == 500


# ── The plan header ──────────────────────────────────────────────────────────

class TestPlanReach:
    def test_the_header_states_the_reach_the_supplied_geometry_buys(
            self, project, capsys):
        assert _cmd_plan(_args("plan", "--channels", "1-12",
                               *GEOMETRY_FLAGS)) == 0
        out = capsys.readouterr().out
        assert "sigma reach:  f_lo 7 Hz -> arcs close for sigma >~ 2.0e-07 S/cm" in out
        assert "at L=0.2 t=0.02 w=0.2 cm" in out

    def test_without_geometry_the_header_says_unavailable_rather_than_a_number(
            self, project, capsys):
        """The point of the change: a plan that cannot name its reach says so.

        Printing a nominal-board number here would tell an operator running a
        two-decades-less-conductive material that the floor reaches it.
        """
        assert _cmd_plan(_args("plan", "--channels", "1-12")) == 0
        out = capsys.readouterr().out
        assert "sigma reach:  unavailable, geometry not supplied" in out
        reach_block = out.split("settle:")[0].split("sigma reach:")[1]
        assert "S/cm" not in reach_block

    def test_the_reach_tracks_the_floor_the_operator_typed(self, project, capsys):
        """Not a constant in a print statement: the line is computed from the
        plan's own f_lo, so a tenth of the floor is visibly a tenth of the
        reach."""
        assert _cmd_plan(_args("plan", "--channels", "1-12", "--f-lo-mHz", "700",
                               *GEOMETRY_FLAGS)) == 0
        assert "sigma >~ 2.0e-08 S/cm" in capsys.readouterr().out


# ── The anchor that had to go with the floor ─────────────────────────────────

class TestRetiredAnchor:
    def test_quick_has_no_stopwatch_anchor_and_reports_extrapolated(self):
        """Both halves pinned together, on purpose.

        ``EIS_MEASURED_S_PER_CHANNEL['Quick'] = 10.47`` timed the *20 Hz* sweep.
        Restoring that entry without re-timing the 7 Hz one would make
        ``eis_duration_basis`` answer "measured" for a sweep costing roughly twice
        as much — a whole night mis-projected with a stopwatch's authority. Pinning
        only one half would let the other be edited back in isolation.
        """
        assert "Quick" not in EIS_MEASURED_S_PER_CHANNEL
        assert eis_duration_basis("Quick") == "extrapolated"

    def test_the_still_valid_anchors_survived(self):
        """``Standard`` and ``Extended`` were not touched: their floors did not
        move, so their stopwatch readings still describe the sweeps they timed."""
        assert EIS_MEASURED_S_PER_CHANNEL["Standard"] == pytest.approx(40.85)
        assert EIS_MEASURED_S_PER_CHANNEL["Extended"] == pytest.approx(115.2)
        assert eis_duration_basis("Standard") == "measured"
        assert eis_duration_basis("Extended") == "measured"

    def test_the_default_plan_says_its_duration_is_extrapolated(
            self, project, capsys):
        """Pinned so nobody "fixes" the notice away.

        It is not noise: it is the difference between a night budgeted from a
        stopwatch and one budgeted from a model asked about a sweep below its
        lowest anchor. The equilibration plan does not pass through
        ``project_campaign``, so it has to say this itself.
        """
        assert _cmd_plan(_args("plan", "--channels", "1-12")) == 0
        out = capsys.readouterr().out
        assert "! EXTRAPOLATED" in out
        assert "has never been timed on this rig" in out

    def test_a_timed_preset_at_its_own_floor_carries_no_such_notice(
            self, project, capsys):
        """The converse, so the notice stays a signal rather than boilerplate."""
        assert _cmd_plan(_args("plan", "--channels", "1-12",
                               "--preset", "Standard")) == 0
        assert "EXTRAPOLATED" not in capsys.readouterr().out

    def test_overriding_a_timed_presets_floor_downgrades_it_too(
            self, project, capsys):
        """The other route to the same exposure: the preset is anchored, but this
        is no longer the sweep that was timed."""
        assert _cmd_plan(_args("plan", "--channels", "1-12", "--preset", "Standard",
                               "--f-lo-mHz", "700")) == 0
        out = capsys.readouterr().out
        assert "! EXTRAPOLATED: --f-lo-mHz 700" in out


class TestCampaignProjectionWarning:
    """``preflight.project_campaign`` — the campaign path's version of the same
    downgrade, and the code the retired anchor was reasoned about against."""

    @pytest.fixture(scope="class")
    def catalog(self) -> TaskCatalog:
        return TaskCatalog.load_toml(loader.tasks_toml_path())

    def _spec(self, **measurement) -> CampaignSpec:
        return CampaignSpec(
            name="floor", channels=(21, 22), pcb_name="SoftAE_EIS_4Stripe",
            parameter_space={
                "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
                "vol_p1": {"type": "float", "low": 5.0, "high": 30.0}},
            vol_params=("vol_p0", "vol_p1"), pump_ids=(0, 1), two_phase=True,
            budget=20,
            measurement=MeasurementSpec(modality="eis", **measurement))

    def test_the_default_preset_now_projects_as_extrapolated(self, catalog):
        """No override needed any more: ``Quick`` lost its anchor with its floor,
        so the campaign path reaches the same warning by the honest route."""
        warnings = project_campaign(self._spec(preset="Quick"),
                                    catalog=catalog).warnings
        assert any("EXTRAPOLATED" in w for w in warnings), warnings

    def test_a_timed_preset_keeps_its_measured_basis(self, catalog):
        warnings = project_campaign(self._spec(preset="Standard"),
                                    catalog=catalog).warnings
        assert not any("EXTRAPOLATED" in w for w in warnings), warnings

    def test_an_override_downgrades_even_a_timed_preset(self, catalog):
        warnings = project_campaign(
            self._spec(preset="Standard", overrides={"f_lo_mHz": 700}),
            catalog=catalog).warnings
        assert any("EXTRAPOLATED" in w for w in warnings), warnings
