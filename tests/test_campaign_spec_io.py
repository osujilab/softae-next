"""The decode-time pinning check in :mod:`softae.core.campaign_spec_io`.

The rest of that module's coverage lives in ``tests/test_campaign_cli.py``,
which was written when the loader and the CLI arrived together; this file holds
only the refusal added with :mod:`softae.core.pinned_recipe`, so the defect and
its controls sit in one place rather than being appended to an 800-line file
about something else.

**What the refusal is for.** ``composition_axes.build_targets_from_axes``
substitutes an axis's ``low`` when a suggestion does not carry it, deliberately:
a solver on a stale bound is recoverable, a campaign dying mid-round on a
``KeyError`` is not. That is right at run time and silent at load time — a spec
declaring a searched axis it never listed in ``parameter_space`` casts every
trial at the corner of its declared box and reports a search. The GUI's axes
editor is refused that shape by ``validate_axes``; a TOML-loaded campaign
reached no check at all until this one.
"""

from __future__ import annotations

import pytest
import structlog

from softae.core.campaign_spec_io import (
    SpecLoadError,
    load_campaign_spec,
    spec_from_dict,
)
from tests.support.fixture_catalog import (
    EXAMPLES,
    example_path,
    use_default_chemistry,
)

#: A legacy volume-mode spec — no composition context, nothing to pin.
MINIMAL = {
    "name": "c",
    "parameter_space": {"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
}

RATIO_AXIS = {"kind": "molar_ratio", "a": "EO", "b": "Li",
              "low": 5.0, "high": 40.0, "basis": "volume"}
PINNED_RATIO_AXIS = {**RATIO_AXIS, "low": 20.0, "high": 20.0}
PINNED_SILICA_AXIS = {"kind": "dried_fraction", "a": "SiO2", "b": "",
                      "low": 0.1, "high": 0.1, "basis": "volume"}

RATIO_PARAM = {"type": "float", "low": 5.0, "high": 40.0}
REPLICATE_PARAM = {"type": "int", "low": 1, "high": 4}


@pytest.fixture
def stub_catalogs(monkeypatch):
    """Stock names resolve without a data root — the seam's stated purpose."""
    from softae.core import campaign_spec_fields as fields
    from softae.core.formulation import ChemicalCatalog, Solution, SolutionCatalog

    sol = SolutionCatalog()
    sol.add(Solution(name="PEO stock"))
    chem = ChemicalCatalog()
    monkeypatch.setattr(fields, "catalogs", lambda: (chem, sol))
    return chem, sol


def _payload(axes, space):
    return {
        "name": "comp",
        "budget": 4,
        "parameter_space": space,
        "general_formulation": {
            "stocks": ["PEO stock"],
            "pump_assignment": {"PEO stock": 0},
            "target_deposition_uL": 4.5,
            "axes": axes,
        },
    }


class TestPinningRefusal:

    def test_a_searched_axis_absent_from_parameter_space_is_refused(
        self, stub_catalogs
    ):
        """THE DEFECT. Before the decode hook this loaded and ran, silently."""
        payload = _payload([RATIO_AXIS], {"replicate": REPLICATE_PARAM})

        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict(payload, source="<defect>")

        assert "ratio_EO_Li" in str(exc.value)
        assert "lower bound" in str(exc.value)

    def test_a_searched_axis_listed_in_parameter_space_still_decodes(
        self, stub_catalogs
    ):
        """The control: the check must not refuse an ordinary search spec."""
        payload = _payload([RATIO_AXIS], {"ratio_EO_Li": RATIO_PARAM})

        spec = spec_from_dict(payload)

        assert set(spec.parameter_space) == {"ratio_EO_Li"}
        assert spec.general_formulation.axes[0].high == 40.0

    def test_a_fully_pinned_recipe_decodes_with_only_a_replicate_parameter(
        self, stub_catalogs
    ):
        """The shape the check exists to permit, not to refuse."""
        payload = _payload([PINNED_RATIO_AXIS, PINNED_SILICA_AXIS],
                           {"replicate": REPLICATE_PARAM})

        spec = spec_from_dict(payload)

        assert all(a.is_fixed for a in spec.general_formulation.axes)
        assert set(spec.parameter_space) == {"replicate"}

    def test_a_pinned_axis_beside_a_searched_one_is_not_itself_required(
        self, stub_catalogs
    ):
        """Only *searched* axes need a parameter; a pinned one must not."""
        payload = _payload([RATIO_AXIS, PINNED_SILICA_AXIS],
                           {"ratio_EO_Li": RATIO_PARAM})

        spec = spec_from_dict(payload)

        assert len(spec.general_formulation.axes) == 2

    def test_a_volume_mode_spec_is_unaffected_by_the_pinning_check(self):
        """Every spec that loaded before this hook must still load."""
        assert spec_from_dict(MINIMAL).name == "c"


class TestLoadEvent:

    def _event(self, payload):
        with structlog.testing.capture_logs() as logs:
            spec_from_dict(payload)
        return next(e for e in logs if e["event"] == "campaign_spec_loaded")

    def test_a_pinned_recipe_is_recorded_as_pinned(self, stub_catalogs):
        event = self._event(_payload([PINNED_RATIO_AXIS, PINNED_SILICA_AXIS],
                                     {"replicate": REPLICATE_PARAM}))

        assert event["recipe_pinned"] is True
        assert event["n_searched_axes"] == 0

    def test_a_search_spec_is_not_recorded_as_pinned(self, stub_catalogs):
        event = self._event(_payload([RATIO_AXIS, PINNED_SILICA_AXIS],
                                     {"ratio_EO_Li": RATIO_PARAM}))

        assert event["recipe_pinned"] is False
        assert event["n_searched_axes"] == 1

    def test_a_volume_mode_spec_is_not_recorded_as_pinned(self):
        """"Nothing to pin" and "pinned deliberately" must not share a token."""
        event = self._event(MINIMAL)

        assert event["recipe_pinned"] is False
        assert event["n_searched_axes"] == 0


# ─────────────────────── the campaign-level [conditions] baseline (T11.28) ───────────────────────

class TestTheConditionsFieldIsRegistered:
    """``conditions`` crosses the file boundary like ``run_plan`` does.

    **The seam, stated.** The loader's legal-key set is
    ``{f.name for f in dataclass_fields(CampaignSpec)}``, and ``CampaignSpec``
    lives in ``core/autonomous_wiring.py`` -- another session's file. Until the
    one-line field lands there, a top-level ``[conditions]`` block is refused as
    an unknown field, so the round trip is exercised here through the registered
    codec rather than through ``spec_from_dict``. The two halves commit together.
    """

    def test_spec_io_conditions_is_registered_as_an_object_field(self):
        from softae.core.campaign_spec_fields import OBJECT_FIELDS
        from softae.core.campaign_spec_run_plan import baseline_conditions_codec

        assert OBJECT_FIELDS["conditions"] is baseline_conditions_codec()

    def test_spec_io_conditions_block_round_trips(self):
        from softae.core.campaign_spec_fields import OBJECT_FIELDS
        from softae.core.phase_setpoints import PhaseSetpoints

        codec = OBJECT_FIELDS["conditions"]
        table = {"name": "baseline", "temp_setpoint_C": 25.0,
                 "rh_setpoint_pct": 40.0}
        setpoints = codec.decode(table)

        assert setpoints == PhaseSetpoints("baseline", temp_setpoint_C=25.0,
                                           rh_setpoint_pct=40.0)
        assert codec.encode(setpoints) == table
        assert codec.decode(codec.encode(setpoints)) == setpoints

    def test_spec_io_conditions_reaches_the_loader_when_the_spec_declares_it(self):
        """Records WHICH state the seam is in, rather than leaving it silent.

        This is a statement about ``CampaignSpec``, not about the codec: the
        moment the dataclass declares ``conditions``, the loader carries it and
        this test is replaced by a round trip through ``spec_from_dict``.
        """
        from dataclasses import fields as dataclass_fields

        from softae.core.autonomous_wiring import CampaignSpec

        declared = any(f.name == "conditions"
                       for f in dataclass_fields(CampaignSpec))
        if declared:
            spec = spec_from_dict({**MINIMAL, "conditions": {
                "name": "baseline", "temp_setpoint_C": 25.0}})
            assert spec.conditions.temp_setpoint_C == 25.0
            return
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "conditions": {"name": "baseline"}})
        assert "unknown field(s) ['conditions']" in str(exc.value)


# ────────────────────────────── the [piezo] codec ────────────────────────────
#
# ``piezo`` was the last field ``_UNSUPPORTED`` refused outright, in the shape
# ``run_plan`` was in before it got a codec: ``PiezoPlan`` is wired end-to-end in
# Python and built from config by the HT tab, so a *file* was the only surface
# that could not ask for it — and no file-driven campaign ever actuated the piezo.
# The codec lives in ``campaign_spec_fields`` (six flat primitives, no module of
# its own), so its tests live beside the other ``OBJECT_FIELDS`` round trips here.

PIEZO_PCB = {"grid": [8, 4], "spacing_mm": [10, 10]}


def _piezo_codec():
    from softae.core.campaign_spec_fields import OBJECT_FIELDS

    return OBJECT_FIELDS["piezo"]


def _piezo_catalog():
    """Enough catalog to compile a single-drop cast with the piezo enabled."""
    from softae.core.task_catalog import Task, TaskCatalog

    cat = TaskCatalog()
    for name in ("startup_flush_full", "single_drop_simul", "final_flush"):
        cat.add(Task(name=name, instrument="liquid_handler", method=name,
                     params={"x": 0, "y": 0, "vols": [1, 1, 1],
                             "disp_rate": 75}))
    cat.add(Task(name="piezo_channel_a_on", instrument="piezo",
                 method="set_channel", params={"channel": "A", "enabled": True}))
    cat.add(Task(name="piezo_channel_a_off", instrument="piezo",
                 method="set_channel", params={"channel": "A", "enabled": False}))
    cat.add(Task(name="piezo_standby", instrument="piezo", method="standby",
                 params={}))
    return cat


class TestThePiezoFieldIsRegistered:

    def test_spec_io_piezo_is_no_longer_refused_outright(self):
        """The whole defect, in one assertion: a file may now name it."""
        from softae.core.campaign_spec_fields import OBJECT_FIELDS
        from softae.core.campaign_spec_io import _UNSUPPORTED

        assert "piezo" not in _UNSUPPORTED
        assert "piezo" in OBJECT_FIELDS

    def test_spec_io_piezo_keys_cover_every_field_but_event_params(self):
        """The seam that makes a future ``PiezoPlan`` field go red, not silent.

        ``_PIEZO_KEYS`` is written out rather than derived, so a field added to
        the dataclass would be refused as an unknown key until the codec learns
        it. That is the intended direction — but only if somebody is told. This
        test is the telling.
        """
        from dataclasses import fields as dataclass_fields

        from softae.core.campaign_spec_fields import _PIEZO_KEYS
        from softae.core.deposition_recipe import PiezoPlan

        declared = {f.name for f in dataclass_fields(PiezoPlan)}
        assert declared == set(_PIEZO_KEYS) | {"event_params"}


class TestThePiezoRoundTrip:

    def test_spec_io_piezo_default_plan_writes_only_what_was_chosen(self):
        """An enabled plan at every other default is one key, not six."""
        from softae.core.deposition_recipe import PiezoPlan

        codec, plan = _piezo_codec(), PiezoPlan(enabled=True)

        assert codec.encode(plan) == {"enabled": True}
        assert codec.decode(codec.encode(plan)) == plan

    def test_spec_io_piezo_every_field_non_default_round_trips(self):
        from softae.core.deposition_recipe import PiezoPlan

        codec = _piezo_codec()
        plan = PiezoPlan(
            enabled=True, on_task="ch_b_on", off_task="ch_b_off",
            standby_task="ch_b_standby", event_task="piezo_liquid_event",
            elution_scope="all_elution")

        table = codec.encode(plan)

        assert table == {
            "enabled": True, "on_task": "ch_b_on", "off_task": "ch_b_off",
            "standby_task": "ch_b_standby", "event_task": "piezo_liquid_event",
            "elution_scope": "all_elution"}
        assert codec.decode(table) == plan

    def test_spec_io_piezo_a_spec_carrying_a_plan_is_reported_complete(self):
        """The write half: a file may now stand in for a piezo campaign."""
        from softae.core.campaign_spec_io import (
            spec_to_dict,
            spec_toml_completeness,
        )

        spec = spec_from_dict({**MINIMAL, "piezo": {
            "enabled": True, "event_task": "piezo_liquid_event"}})

        assert spec_to_dict(spec)["piezo"] == {
            "enabled": True, "event_task": "piezo_liquid_event"}
        assert spec_toml_completeness(spec).complete


class TestThePiezoRefusals:

    def test_spec_io_piezo_event_params_encodes_as_unrepresentable(self):
        """The `anneal_params` refusal, on the encode side."""
        from softae.core.campaign_spec_fields import UNREPRESENTABLE
        from softae.core.deposition_recipe import PiezoPlan

        plan = PiezoPlan(enabled=True, event_params={"frequency_hz": 525.0})

        assert _piezo_codec().encode(plan) is UNREPRESENTABLE

    def test_spec_io_piezo_event_params_in_the_file_is_refused(self):
        """And on the decode side, naming what to do instead."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "piezo": {
                "enabled": True, "event_params": {"frequency_hz": 525.0}}})

        assert "event_params" in str(exc.value)
        assert "event_task" in str(exc.value)

    def test_spec_io_piezo_an_unknown_key_is_refused_with_the_legal_set(self):
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "piezo": {"enabled": True,
                                                 "elution_scopes": "deposit"}})

        message = str(exc.value)
        assert "unknown key(s) ['elution_scopes']" in message
        assert "'elution_scope'" in message and "'standby_task'" in message

    def test_spec_io_piezo_an_illegal_elution_scope_is_refused(self):
        """``PiezoPlan`` accepts any string, and the engine then runs the
        narrower scope in silence — so the codec owns this refusal."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "piezo": {"enabled": True,
                                                 "elution_scope": "all"}})

        message = str(exc.value)
        assert "unknown elution_scope 'all'" in message
        assert "'deposit'" in message and "'all_elution'" in message

    def test_spec_io_piezo_a_non_boolean_enabled_is_refused(self):
        """``enabled = "false"`` is truthy in Python: a plan that would actuate."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "piezo": {"enabled": "false"}})

        assert "'enabled' must be true or false" in str(exc.value)

    def test_spec_io_piezo_an_empty_task_name_is_refused(self):
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "piezo": {"enabled": True,
                                                 "on_task": "  "}})

        assert "'on_task' must be the name of a catalog task" in str(exc.value)


class TestThePiezoAbsentAndEmptyTable:
    """What each of the two silences actually produces, pinned rather than assumed.

    They are **not** the same object — ``CampaignSpec.piezo`` defaults to ``None``
    and ``OBJECT_FIELDS`` decode runs only when the key is present at all, so an
    empty table decodes to a default ``PiezoPlan`` instead. What matters is that
    they are the same *experiment*: the engine asks ``piezo is not None and
    piezo.enabled``, so both actuate nothing, and the second test is the one that
    says so rather than trusting the first.
    """

    def test_spec_io_piezo_absent_table_leaves_the_field_unset(self):
        assert spec_from_dict(MINIMAL).piezo is None

    def test_spec_io_piezo_empty_table_decodes_to_an_inert_default_plan(self):
        from softae.core.deposition_recipe import PiezoPlan

        assert spec_from_dict({**MINIMAL, "piezo": {}}).piezo == PiezoPlan()

    @pytest.mark.parametrize("payload", [
        {}, {"piezo": {}}, {"piezo": {"enabled": False}}])
    def test_spec_io_piezo_three_inert_spellings_compile_no_piezo_steps(
        self, payload
    ):
        """Absent, empty and explicitly-off must be one experiment, not three."""
        from softae.core.deposition_recipe import (
            build_deposition_workflow,
            get_deposition_recipe,
        )

        spec = spec_from_dict({**MINIMAL, "channels": [21], **payload})
        workflow = build_deposition_workflow(
            get_deposition_recipe("single_drop"), [21], {21: [10.0, 30.0, 0.0]},
            settings=spec.deposition_settings(pcb=PIEZO_PCB),
            catalog=_piezo_catalog())

        assert not any(s.instrument == "piezo"
                       for s in workflow.resolve_steps())


class TestThePiezoPlanReachesTheCompiler:

    def test_spec_io_piezo_a_file_loaded_plan_compiles_actuation_steps(self):
        """THE POINT. A round trip through the codec proves nothing on its own:
        this proves the decoded plan is consumable by ``deposition_recipe``."""
        from softae.core.deposition_recipe import (
            build_deposition_workflow,
            get_deposition_recipe,
        )

        spec = spec_from_dict({**MINIMAL, "channels": [21],
                               "piezo": {"enabled": True}})
        workflow = build_deposition_workflow(
            get_deposition_recipe("single_drop"), [21], {21: [10.0, 30.0, 0.0]},
            settings=spec.deposition_settings(pcb=PIEZO_PCB),
            catalog=_piezo_catalog())

        names = [s.name for s in workflow.resolve_steps()]

        assert "piezo_on_ch21" in names
        assert names.index("piezo_on_ch21") < names.index("deposit_ch21")
        assert names.index("deposit_ch21") < names.index("piezo_off_ch21")
        assert names[-1] == "piezo_standby"
        assert workflow.metadata["piezo"] == "applied"

    def test_spec_io_piezo_a_file_loaded_all_elution_scope_reaches_the_branch(
        self
    ):
        """``elution_scope`` is the one field whose two values compile
        *differently*, so a codec that dropped it would look identical here."""
        from softae.core.deposition_recipe import (
            build_deposition_workflow,
            get_deposition_recipe,
        )

        spec = spec_from_dict({**MINIMAL, "channels": [21], "piezo": {
            "enabled": True, "elution_scope": "all_elution"}})
        workflow = build_deposition_workflow(
            get_deposition_recipe("single_drop"), [21], {21: [10.0, 30.0, 0.0]},
            settings=spec.deposition_settings(pcb=PIEZO_PCB),
            catalog=_piezo_catalog())

        names = [s.name for s in workflow.resolve_steps()]

        assert "piezo_on_startup_flush" in names      # the all-elution branch
        assert "piezo_on_ch21" not in names           # not the deposit-only one


# ──────────────────────── an axis's sampling `scale` ─────────────────────────
#
# ``CompositionAxis.scale`` ("linear" | "log") landed in ``composition_axes`` as a
# live Python field, so the GUI could ask for a log search and a *file* could not
# — the same one-surface gap the ``piezo`` codec above closed. The key is
# **permitted but not required** on read: every spec on disk predates it and each
# describes the linear search its absence already means, so requiring it would
# refuse a file for omitting a key it could not have known about. The encoder
# writes it regardless, because an omitted key silently taking a default is what
# the axis-key doctrine exists to prevent.

LOG_RATIO_AXIS = {**RATIO_AXIS, "scale": "log"}

#: The committed examples whose ``general_formulation`` declares axes at all —
#: named rather than counted, so the back-compatibility test below cannot pass by
#: finding no axes to check.
EXAMPLES_WITH_AXES = ("bench_instance", "bo_init_bench", "rung3a_fake_cast")


@pytest.fixture
def default_chemistry(monkeypatch):
    """Chemistry seam pointed at the shipped catalogs for the whole test."""
    use_default_chemistry(monkeypatch)


def _example_axes(example):
    gf = getattr(load_campaign_spec(example_path(example)),
                 "general_formulation", None)
    return tuple(getattr(gf, "axes", ()) or ())


class TestTheAxisScaleKey:

    def test_spec_io_axis_a_log_scale_axis_round_trips_through_the_file(
        self, stub_catalogs
    ):
        """THE ASK: written out, read back, and the same axis object."""
        from softae.core.campaign_spec_io import spec_to_dict

        spec = spec_from_dict(
            _payload([LOG_RATIO_AXIS], {"ratio_EO_Li": RATIO_PARAM}))
        axis = spec.general_formulation.axes[0]

        written = spec_to_dict(spec)["general_formulation"]["axes"][0]

        assert axis.scale == "log" and axis.is_log
        assert written["scale"] == "log"
        assert spec_from_dict(
            _payload([written], {"ratio_EO_Li": RATIO_PARAM})
        ).general_formulation.axes[0] == axis

    def test_spec_io_axis_without_a_scale_key_still_loads_as_linear(
        self, stub_catalogs
    ):
        """The regression this shape exists for: ``scale`` is optional on read,
        so every spec written before the key still describes its own search."""
        spec = spec_from_dict(
            _payload([RATIO_AXIS], {"ratio_EO_Li": RATIO_PARAM}))

        assert "scale" not in RATIO_AXIS
        assert spec.general_formulation.axes[0].scale == "linear"

    def test_spec_io_axis_a_linear_axis_is_written_with_its_scale_key(
        self, stub_catalogs
    ):
        """Optional on read, always written — the asymmetry, pinned. A file
        round-tripped through the GUI states the scale it searches."""
        from softae.core.campaign_spec_io import spec_to_dict

        spec = spec_from_dict(
            _payload([RATIO_AXIS], {"ratio_EO_Li": RATIO_PARAM}))

        written = spec_to_dict(spec)["general_formulation"]["axes"][0]

        assert written["scale"] == "linear"
        assert set(written) == {"kind", "a", "b", "low", "high", "basis", "scale"}

    def test_spec_io_axis_an_unknown_scale_is_refused_at_load(self, stub_catalogs):
        """Permitting the key must not mean accepting any value in it: an
        unrecognised scale would reach the sampler and quietly search linearly."""
        payload = _payload([{**RATIO_AXIS, "scale": "ln"}],
                           {"ratio_EO_Li": RATIO_PARAM})

        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict(payload, source="<bad-scale>")

        assert "scale" in str(exc.value)

    def test_spec_io_axis_an_unknown_axis_key_is_still_refused(self, stub_catalogs):
        """The control on widening the permitted set: only ``scale`` was added."""
        payload = _payload([{**RATIO_AXIS, "sclae": "log"}],
                           {"ratio_EO_Li": RATIO_PARAM})

        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict(payload, source="<typo>")

        assert "sclae" in str(exc.value)

    def test_spec_io_axis_a_log_axis_at_or_below_zero_is_refused_at_load(
        self, stub_catalogs
    ):
        """``CompositionAxis`` refuses ``low <= 0`` on a searched log axis; the
        file path must reach that refusal rather than construct around it."""
        payload = _payload([{**LOG_RATIO_AXIS, "low": 0.0}],
                           {"ratio_EO_Li": RATIO_PARAM})

        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict(payload, source="<log-zero>")

        assert "log" in str(exc.value)


@pytest.mark.parametrize("example", EXAMPLES)
def test_spec_io_axis_every_committed_example_still_loads_as_a_linear_search(
        example, default_chemistry):
    """No committed spec declares ``scale``, so each must load as the linear
    search it describes — the on-disk half of the optional-key regression."""
    assert all(a.scale == "linear" for a in _example_axes(example))


def test_spec_io_axis_the_examples_checked_above_do_carry_axes(default_chemistry):
    """The counter that stops the parametrized test passing on zero axes."""
    with_axes = {e for e in EXAMPLES if _example_axes(e)}

    assert with_axes == set(EXAMPLES_WITH_AXES)
