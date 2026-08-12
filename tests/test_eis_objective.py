"""The campaign objective (E1.5) — which metric a campaign is steered by, and why.

There are two legitimate campaign *modes*, and neither is a degraded version of the
other. A **composition** campaign plans formulations and maps them to volumes, so the
twin knows what was cast, can predict a dry thickness, and conductivity is both
available and the quantity actually wanted. A **volume** campaign explores raw pump
volumes: exploration is easier and feasibility is native (a volume limit is just a
bound), with composition resolved post hoc — but without stock identity there is no
elution and hence no dry thickness, so σ is *impossible* rather than merely absent, and
mean |Z| is the honest objective.

``[eis] objective = "auto"`` follows the mode. Three things are pinned here. That the
metric is *derived* from what the campaign can actually deliver. That the optimisation
*direction* is derived from the metric rather than chosen alongside it — minimising |Z|
and maximising σ are the same goal, so a free-standing direction field is an invitation
to spend a whole budget hunting the worst conductor on the board while every step
reports success. And that a pinned objective a campaign cannot honour is **refused**,
never silently swapped for the other one.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.core.autonomous_wiring import (
    OBJECTIVE_DIRECTION,
    CampaignSpec,
    _scalar_from_eis_raw,
    _sigma_from_eis_raw,
    _trial_objective_kind,
    objective_kind,
    resolve_direction,
    resolve_objective,
)
from softae.errors import CampaignError
from tests.eis_synthetic import pure_series_rc, reference_spectrum
from tests.test_autonomous_composition import SPACE as COMPOSITION_SPACE
from tests.test_autonomous_composition import _context


def _raw(f: np.ndarray, Z: np.ndarray) -> list[np.ndarray]:
    """The 5-column array an EIS step returns: ``[f, |Z|, phase, Z', -Z'']``."""
    return [np.column_stack([f, np.abs(Z), np.degrees(np.angle(Z)),
                             Z.real, -Z.imag])]


class _Settings:
    def __init__(self, objective: str, engine: str = "legacy"):
        self.objective = objective
        self.engine = engine


def _volume_spec(**over) -> CampaignSpec:
    """Volume mode: raw pump volumes, no composition, so no dry thickness."""
    base = dict(
        name="vol_only", channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space={"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
        vol_params=("vol_p0",), pump_ids=(0,),
    )
    base.update(over)
    return CampaignSpec(**base)


def _composition_spec(**over) -> CampaignSpec:
    """Composition mode: the twin knows the stocks, so it can predict a thickness."""
    base = dict(
        name="comp", channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=COMPOSITION_SPACE,
        formulation=_context(),
        pump_ids=(0, 1, 2),
    )
    base.update(over)
    return CampaignSpec(**base)


class TestObjectiveSelection:
    def test_the_shipped_default_defers_to_the_campaign_mode(self):
        # "auto" is not itself an objective — it is the instruction to derive one.
        assert objective_kind() == "auto"

    def test_a_pinned_objective_is_read_verbatim(self):
        assert objective_kind(_Settings("mean_abs_z")) == "mean_abs_z"
        assert objective_kind(_Settings("sigma")) == "sigma"

    def test_the_two_objectives_require_opposite_directions(self):
        # Lower impedance and higher conductivity are the same goal.
        assert OBJECTIVE_DIRECTION["mean_abs_z"] == "minimize"
        assert OBJECTIVE_DIRECTION["sigma"] == "maximize"

    def test_the_objective_is_not_tied_to_the_analysis_engine(self):
        # WHICH metric a campaign is steered by and WHICH engine computes it are
        # separate keys: `[eis] objective` answers the first, `[eis] engine` the
        # second. Pinning σ on the legacy engine is a coherent instruction — σ is
        # still what is optimised, it is just computed the legacy way (T2.6b).
        assert objective_kind(_Settings("sigma", engine="legacy")) == "sigma"


class TestModeAwareResolution:
    def test_a_composition_campaign_resolves_to_conductivity_and_maximises_it(self):
        spec = _composition_spec()
        kind, reason = resolve_objective(spec)
        assert kind == "sigma"
        assert "thickness" in reason
        assert resolve_direction(spec) == ("maximize", "sigma")

    def test_a_volume_campaign_resolves_to_impedance_and_minimises_it(self):
        spec = _volume_spec()
        kind, reason = resolve_objective(spec)
        assert kind == "mean_abs_z"
        assert "volume mode" in reason
        assert resolve_direction(spec) == ("minimize", "mean_abs_z")

    def test_the_direction_is_derived_rather_than_carried_alongside_the_metric(self):
        # The spec's own field defaults to "auto" precisely so the two cannot drift
        # apart — a free-standing direction is what made this bug possible.
        assert CampaignSpec.objective == "auto"

    def test_an_explicit_direction_that_agrees_is_honoured(self):
        assert resolve_direction(_volume_spec(objective="minimize"))[0] == "minimize"
        assert resolve_direction(
            _composition_spec(objective="maximize"))[0] == "maximize"

    def test_an_explicit_direction_that_contradicts_the_metric_is_refused(self):
        # Not a degraded run: this campaign would spend its entire budget finding the
        # *most* resistive material on the board while reporting healthy progress.
        with pytest.raises(CampaignError, match="contradicts"):
            resolve_direction(_volume_spec(objective="maximize"))
        with pytest.raises(CampaignError, match="contradicts"):
            resolve_direction(_composition_spec(objective="minimize"))


class TestPinnedObjectives:
    """A pin is honoured or refused — never quietly swapped for the other metric."""

    def test_pinning_sigma_on_a_volume_campaign_is_refused(self):
        with pytest.raises(CampaignError, match="volume mode"):
            resolve_objective(_volume_spec(), settings=_Settings("sigma"))

    def test_the_refusal_names_both_ways_out(self):
        with pytest.raises(CampaignError) as excinfo:
            resolve_objective(_volume_spec(), settings=_Settings("sigma"))
        message = str(excinfo.value)
        assert "formulation context" in message
        assert "mean_abs_z" in message

    def test_pinning_impedance_on_a_composition_campaign_is_allowed(self):
        # σ is available but not compulsory — mean |Z| stays selectable for
        # comparison runs against the historical objective.
        kind, reason = resolve_objective(
            _composition_spec(), settings=_Settings("mean_abs_z"))
        assert kind == "mean_abs_z"
        assert "pinned" in reason
        assert resolve_direction(
            _composition_spec(), settings=_Settings("mean_abs_z"))[0] == "minimize"


class TestTrialKindIsSettledOnce:
    """One metric per trial. Mixing two with opposite directions inside one average
    would be worse than either — the number would mean nothing at all."""

    def test_a_resolved_kind_passes_straight_through(self):
        assert _trial_objective_kind("sigma", has_thickness=False) == "sigma"
        assert _trial_objective_kind("mean_abs_z", has_thickness=True) == "mean_abs_z"

    def test_auto_never_escapes_as_an_objective(self):
        assert _trial_objective_kind("auto", has_thickness=True) == "sigma"
        assert _trial_objective_kind("auto", has_thickness=False) == "mean_abs_z"

    def test_an_unresolved_call_infers_the_mode_from_thickness_availability(self):
        # Direct calls (demos, examples) have no spec to consult. A dry thickness
        # exists exactly when the twin knew what was cast, so it *is* the mode signal.
        assert _trial_objective_kind(None, has_thickness=True) == "sigma"
        assert _trial_objective_kind(None, has_thickness=False) == "mean_abs_z"


class TestSigmaExtractor:
    def test_a_spectrum_with_no_per_sample_thickness_yields_no_objective(self):
        # Conductivity divides by thickness. Absent is never guessed (P7.2's posture),
        # so the trial is reported unmeasured rather than scored on a nominal.
        f, Z = reference_spectrum()
        assert _sigma_from_eis_raw(_raw(f, Z)) is None

    def test_a_spectrum_the_gates_reject_yields_no_objective(self):
        # A pure series RC contains no parallel conduction at any frequency; fitting
        # it would manufacture an R1 that σ = K/R happily consumes.
        f, Z = pure_series_rc()
        assert _sigma_from_eis_raw(_raw(f, Z)) is None

    def test_a_malformed_result_yields_none_rather_than_raising(self):
        assert _sigma_from_eis_raw([np.array([1.0, 2.0, 3.0])]) is None
        assert _sigma_from_eis_raw(None) is None

    def test_a_broken_analysis_path_does_not_discard_the_measurement(self, monkeypatch):
        # The safeguard must never be the reason data is lost.
        import softae.analysis.eis.engine as engine

        def _boom(*a, **k):
            raise RuntimeError("analysis exploded")

        monkeypatch.setattr(engine, "analyze_spectrum", _boom)
        f, Z = reference_spectrum()
        assert _sigma_from_eis_raw(_raw(f, Z)) is None      # declines, does not raise


class TestThicknessIsRequiredForSigma:
    def test_a_channel_with_a_recorded_thickness_yields_a_conductivity(self):
        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)
        value = _scalar_from_eis_raw(_raw(f, Z), channel=5, thickness_um=150.0,
                                     kind="sigma")
        assert value is not None and value > 0

    def test_a_channel_without_one_is_unmeasured_rather_than_scored_on_a_nominal(self):
        # Under a σ campaign a missing thickness is *unmeasured*, and specifically not
        # mean |Z|: handing a maximiser an impedance for one channel would score the
        # worst conductor in the trial as its best result.
        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)
        assert _scalar_from_eis_raw(_raw(f, Z), channel=5, kind="sigma") is None

    def test_the_same_channel_is_perfectly_measurable_under_a_volume_campaign(self):
        # The measurement is not deficient — mean |Z| needs no thickness. This is the
        # difference between "impossible" and "missing".
        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)
        value = _scalar_from_eis_raw(_raw(f, Z), channel=5, kind="mean_abs_z")
        assert value is not None and value > 0

    def test_a_missing_result_is_still_unmeasured_not_zero(self):
        # Telling the optimizer 0.0 for a measurement that never happened makes the
        # surrogate confident about a point nobody observed.
        assert _scalar_from_eis_raw(None) is None

    def test_a_trace_too_short_to_analyse_is_declined_where_mean_abs_z_averaged_it(self):
        # The legacy objective happily returned |Z| = 5 for a single-point "spectrum".
        # σ cannot gate or fit one, so it reports unmeasured rather than handing the
        # optimizer a number derived from nothing.
        arr = np.array([[1.0, 2.0, 0.5, 3.0, 4.0]])
        assert _scalar_from_eis_raw([arr], channel=1, thickness_um=150.0,
                                    kind="sigma") is None


class TestOneSigmaEverywhere:
    """T2.6b — the campaign objective and the GUI must report the SAME conductivity.

    User ruling, 2026-08-09 (TASKS.md T2.6b, mail ``[a23]``): *"the GUI and objective
    should report the same conductivity; nothing changes about the casting nor
    measurement between them."* The objective used to pass ``engine="gated"`` at its
    ``analyze_spectrum`` call while every GUI fit site left the keyword unset and so
    followed ``[eis] engine`` — which ships ``"legacy"``. One spectrum, two numbers,
    and nothing on screen said so.

    The fix adopts the GUI's own mechanism rather than a second resolver: **omit the
    keyword** and let ``analyze_spectrum`` read ``[eis] engine``. These tests pin the
    agreement under BOTH settings, and the third one proves the agreement is not
    vacuous — the two engines really do return different σ for the same spectrum, so
    a test that only ever ran one of them would prove nothing.
    """

    @staticmethod
    def _use_engine(monkeypatch, name: str) -> None:
        """Make ``[eis] engine`` read ``name`` for every consumer of the settings."""
        from softae.analysis.eis.settings import eis_settings

        monkeypatch.setattr("softae.analysis.eis.engine.eis_settings",
                            lambda *a, **k: eis_settings({"engine": name}))

    @staticmethod
    def _gui_sigma(f: np.ndarray, Z: np.ndarray, *, channel: int,
                   thickness_um: float) -> float:
        """σ exactly as a GUI surface computes it: build a cell, analyse, read σ.

        `gui_cell` + `analyze_spectrum` with ``engine`` unset + `report_sigma` is the
        route every migrated GUI fit site takes (P.20 Stage A). The geometry comes
        from the objective's own cell so that the ONLY difference left between the two
        paths is the engine — which is the thing the ruling is about.
        """
        from softae.analysis.eis.engine import analyze_spectrum
        from softae.analysis.eis.geometry import cell_constant_for_sample
        from softae.analysis.eis_data import EISResult
        from softae.gui.eis_sigma import gui_cell, report_sigma

        L, t, w = cell_constant_for_sample(
            predicted_um=thickness_um,
            re_connection="bridged_by_sample").as_legacy_triple()
        eis = EISResult.from_arrays(channel=channel, f=f, z_real=Z.real,
                                    z_imag_neg=-Z.imag)
        report = analyze_spectrum(eis, cell=gui_cell(L, t, w),
                                  re_connection="bridged_by_sample")
        return report_sigma(report)

    def test_the_objective_and_the_gui_agree_under_the_shipped_legacy_engine(
            self, monkeypatch):
        self._use_engine(monkeypatch, "legacy")
        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)

        objective = _sigma_from_eis_raw(_raw(f, Z), channel=5, thickness_um=150.0)

        assert objective is not None
        assert objective == pytest.approx(
            self._gui_sigma(f, Z, channel=5, thickness_um=150.0), rel=1e-12)

    def test_the_objective_and_the_gui_agree_under_the_gated_engine(self, monkeypatch):
        self._use_engine(monkeypatch, "gated")
        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)

        objective = _sigma_from_eis_raw(_raw(f, Z), channel=5, thickness_um=150.0)

        assert objective is not None
        assert objective == pytest.approx(
            self._gui_sigma(f, Z, channel=5, thickness_um=150.0), rel=1e-12)

    def test_flipping_the_engine_moves_the_number_so_the_agreement_is_not_vacuous(
            self, monkeypatch):
        # The positive control. Both engines agreeing across the two surfaces is only
        # meaningful if the config reaches the objective at all — and it does: the
        # legacy fitter and the gated one differ by more than a factor of two on this
        # spectrum. That gap IS the divergence the user ruled out; it was previously
        # invisible because the objective always took one side of it.
        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)

        self._use_engine(monkeypatch, "legacy")
        legacy = _sigma_from_eis_raw(_raw(f, Z), channel=5, thickness_um=150.0)
        self._use_engine(monkeypatch, "gated")
        gated = _sigma_from_eis_raw(_raw(f, Z), channel=5, thickness_um=150.0)

        assert legacy is not None and gated is not None
        assert legacy != pytest.approx(gated, rel=1e-3)

    def test_the_objective_names_no_engine_at_its_call_site(self, monkeypatch):
        # The mechanism, not just the outcome. Resolving `[eis] engine` here and
        # passing the answer back in would agree with the GUI today and drift the
        # moment the resolution rule changes in one place only; omitting the keyword
        # means there is nothing to keep in step.
        import softae.analysis.eis.engine as engine

        real = engine.analyze_spectrum
        seen: list[dict] = []

        def spy(*args, **kwargs):
            seen.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(engine, "analyze_spectrum", spy)
        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)
        _sigma_from_eis_raw(_raw(f, Z), channel=5, thickness_um=150.0)

        assert seen, "the σ extractor did not reach analyze_spectrum at all"
        assert all("engine" not in kwargs for kwargs in seen)


class TestAggregateObjective:
    def test_only_channels_with_a_thickness_contribute_to_the_average(self):
        from softae.core.autonomous_wiring import (
            eis_impedance_objective,
            make_thickness_lookup,
        )

        class _Store:
            def predicted_thickness_um(self, run_id, channel):
                return 150.0 if channel == 5 else None

        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)
        raw = _raw(f, Z)
        lookup = make_thickness_lookup(_Store(), "run1")

        one = eis_impedance_objective({"measure_eis_ch5": raw}, {},
                                      thickness_for=lookup, kind="sigma")
        both = eis_impedance_objective(
            {"measure_eis_ch5": raw, "measure_eis_ch6": raw}, {},
            thickness_for=lookup, kind="sigma")
        assert one is not None
        assert both == pytest.approx(one), (
            "ch6 has no thickness, so it must not dilute the average with a zero")

    def test_a_trial_where_nothing_has_a_thickness_is_unmeasured(self):
        from softae.core.autonomous_wiring import eis_impedance_objective

        f, Z = reference_spectrum()
        assert eis_impedance_objective(
            {"measure_eis_ch5": _raw(f, Z)}, {}, kind="sigma") is None

    def test_one_trial_is_averaged_in_one_metric_even_when_thickness_is_patchy(self):
        # The hazard the settle-once rule exists for: were the metric decided per
        # channel, ch6 would contribute a mean |Z| of order 10³ Ω to an average of
        # conductivities of order 10⁻⁴ S/cm and swamp it entirely.
        from softae.core.autonomous_wiring import (
            eis_impedance_objective,
            make_thickness_lookup,
        )

        class _Store:
            def predicted_thickness_um(self, run_id, channel):
                return 150.0 if channel == 5 else None

        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=3)
        raw = _raw(f, Z)
        value = eis_impedance_objective(
            {"measure_eis_ch5": raw, "measure_eis_ch6": raw}, {},
            thickness_for=make_thickness_lookup(_Store(), "run1"))
        assert value is not None and value < 1.0, (
            "a σ of order 1e-4 must not be averaged with an impedance of order 1e3")

    def test_a_broken_thickness_store_declines_rather_than_crashing_the_trial(self):
        from softae.core.autonomous_wiring import make_thickness_lookup

        class _Store:
            def predicted_thickness_um(self, run_id, channel):
                raise RuntimeError("db gone")

        assert make_thickness_lookup(_Store(), "run1")(5) is None
