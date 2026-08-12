"""The shared deposition marshalling contract (P2.1).

One engine sat under the HT tab, the autonomous campaign, and the Process Studio
preview, but each marshalled its kwargs independently and the three had drifted.
These tests pin the contract that replaced them: identical inputs must produce
byte-identical workflows regardless of which surface launched the run.
"""

from __future__ import annotations

import pytest

from softae.config.loader import pico_for_channel
from softae.core.autonomous_wiring import CampaignSpec
from softae.core.deposition_recipe import (
    DepositionSettings,
    PiezoPlan,
    build_deposition_workflow,
    build_recipe_deposition_workflow,
    get_deposition_recipe,
)
from softae.core.task_catalog import Task, TaskCatalog
from softae.workflows.workflow_model import WorkflowStep

PCB = {"grid": [8, 4], "spacing_mm": [10, 10]}
CHANNELS = [21, 22]
FORMULATION = {21: [10.0, 30.0, 0.0], 22: [20.0, 20.0, 0.0]}


def _catalog() -> TaskCatalog:
    cat = TaskCatalog()
    for name, method in (
        ("startup_flush_full", "startup_flush_full"),
        ("precondition_flush", "precondition_flush"),
        ("single_drop_simul", "single_drop_simul"),
        ("final_flush", "final_flush"),
        ("alt_drop", "alt_drop"),
    ):
        cat.add(Task(name=name, instrument="liquid_handler", method=method,
                     params={"x": 0, "y": 0, "vols": [1, 1, 1], "disp_rate": 75}))
    cat.add(Task(name="piezo_channel_a_on", instrument="piezo",
                 method="set_channel", params={"channel": "A", "enabled": True}))
    cat.add(Task(name="piezo_channel_a_off", instrument="piezo",
                 method="set_channel", params={"channel": "A", "enabled": False}))
    cat.add(Task(name="piezo_standby", instrument="piezo", method="standby", params={}))
    return cat


def _eis_by_channel() -> dict[int, WorkflowStep]:
    return {
        ch: WorkflowStep(name=f"measure_eis_ch{ch}", instrument=pico_for_channel(ch),
                         method="sendscript_getdata", params={"chan": ch})
        for ch in CHANNELS
    }


def _fingerprint(wf) -> list[tuple]:
    """Structural identity of a workflow — step order, target, and params."""
    return [
        (s.name, s.instrument, s.method, repr(sorted(s.params.items())), repr(s.tags))
        for s in wf.resolve_steps()
    ]


# ── The wrapper must be a faithful passthrough ───────────────────────────────

def test_wrapper_matches_the_engine_called_directly():
    """The marshaller adds a contract; it must not change what the engine builds."""
    settings = DepositionSettings(
        pump_ids=(0, 1, 2), dispense_rate=100.0, flush_rate=500.0,
        flush_factor=3.0, settle_factor=2.0, settle_base_s=0.0,
        start_flush_uL=(80.0, 80.0, 80.0), pcb=PCB, origin_xy=(43.5, 50.0),
    )
    recipe = get_deposition_recipe("two_phase")
    eis = _eis_by_channel()

    via_wrapper = build_deposition_workflow(
        recipe, CHANNELS, FORMULATION, settings=settings,
        catalog=_catalog(), eis_step_by_channel=eis, name="wf")
    direct = build_recipe_deposition_workflow(
        recipe, CHANNELS, FORMULATION, catalog=_catalog(), pump_ids=[0, 1, 2],
        dispense_rate=100.0, flush_rate=500.0, flush_factor=3.0,
        settle_factor=2.0, settle_base_s=0.0, start_flush_uL=[80.0, 80.0, 80.0],
        eis_step_by_channel=eis, pcb=PCB, origin_xy=(43.5, 50.0), name="wf")

    assert _fingerprint(via_wrapper) == _fingerprint(direct)


def test_empty_start_flush_defers_to_the_engine_default():
    """`()` must mean "unset", not "flush zero" — the engine owns that default."""
    settings = DepositionSettings(pcb=PCB, origin_xy=(43.5, 50.0), start_flush_uL=())
    wf = build_deposition_workflow(
        get_deposition_recipe("single_drop"), CHANNELS, FORMULATION,
        settings=settings, catalog=_catalog())
    flush = next(s for s in wf.resolve_steps() if s.name == "startup_flush")
    assert any(v for k, v in flush.params.items() if "vol" in k.lower())


# ── HT and campaign must agree (the divergence this closed) ──────────────────

def test_ht_shaped_and_campaign_shaped_settings_agree():
    """Same physical intent from either surface → the same workflow."""
    spec = CampaignSpec(
        name="c", channels=(21, 22), pump_ids=(0, 1, 2), two_phase=True,
        disp_rate=100.0, line_flush_rate=500.0, flush_factor=3.0,
        settle_factor=2.0, settle_base_s=0.0,
        start_flush_uL=(80.0, 80.0, 80.0), time_scale=None,
    )
    from softae.core.liquid_handling import DeadVolumeCorrection

    campaign_settings = spec.deposition_settings(pcb=PCB, n_pumps=3)
    ht_settings = DepositionSettings(
        pump_ids=(0, 1, 2), dispense_rate=100.0, flush_rate=500.0,
        flush_factor=3.0, settle_factor=2.0, settle_base_s=0.0,
        start_flush_uL=(80.0, 80.0, 80.0), pcb=PCB,
        # Both surfaces build this from the same config (P2.2).
        correction=DeadVolumeCorrection.from_config((0, 1, 2)),
    )
    assert campaign_settings == ht_settings

    recipe = get_deposition_recipe("two_phase")
    a = build_deposition_workflow(recipe, CHANNELS, FORMULATION,
                                  settings=campaign_settings, catalog=_catalog(), name="w")
    b = build_deposition_workflow(recipe, CHANNELS, FORMULATION,
                                 settings=ht_settings, catalog=_catalog(), name="w")
    assert _fingerprint(a) == _fingerprint(b)


def test_campaign_can_now_express_deposit_method():
    """Regression: the campaign path silently dropped the deposit-method override."""
    spec = CampaignSpec(name="c", channels=(21,), deposit_method="alt_drop")
    assert spec.deposition_settings(pcb=PCB).deposit_method == "alt_drop"

    wf = build_deposition_workflow(
        get_deposition_recipe("single_drop"), [21], {21: [10.0, 30.0, 0.0]},
        settings=spec.deposition_settings(pcb=PCB), catalog=_catalog())
    assert next(s for s in wf.resolve_steps()
                if s.name == "deposit_ch21").method == "alt_drop"


def test_campaign_can_now_express_piezo():
    """Regression: campaigns never actuated the piezo, even with one configured."""
    spec = CampaignSpec(name="c", channels=(21,), piezo=PiezoPlan(enabled=True))
    wf = build_deposition_workflow(
        get_deposition_recipe("single_drop"), [21], {21: [10.0, 30.0, 0.0]},
        settings=spec.deposition_settings(pcb=PCB), catalog=_catalog())
    assert any(s.instrument == "piezo" for s in wf.resolve_steps())


def test_two_phase_flush_rate_selection_is_preserved():
    """Moving this branch onto the spec must not change which rate is broadcast."""
    common = dict(name="c", channels=(21,), line_flush_rate=500.0)
    assert CampaignSpec(**common, two_phase=True).deposition_settings(
        pcb=PCB).flush_rate == 500.0
    # Single-drop ignores the two-phase line rate and uses the prime-rate default.
    assert CampaignSpec(**common, two_phase=False).deposition_settings(
        pcb=PCB).flush_rate != 500.0


def test_pump_ids_trim_to_the_volume_vector_width():
    spec = CampaignSpec(name="c", channels=(21,), pump_ids=(0, 1, 2))
    assert spec.deposition_settings(pcb=PCB, n_pumps=2).pump_ids == (0, 1)
    assert spec.deposition_settings(pcb=PCB).pump_ids == (0, 1, 2)


# ── Config as the documented default source ──────────────────────────────────

def test_from_config_reads_dropcast_and_accepts_overrides():
    from softae.config.loader import dropcast_config

    dc = dropcast_config()
    s = DepositionSettings.from_config(PCB, pump_ids=[0, 1])
    assert s.dispense_rate == pytest.approx(float(dc["dispense_rate_uL_min"]))
    assert s.flush_rate == pytest.approx(float(dc["line_flush_rate_uL_min"]))
    assert s.pump_ids == (0, 1)
    assert s.pcb is PCB

    assert DepositionSettings.from_config(
        PCB, dispense_rate=1.5).dispense_rate == 1.5


def test_settings_are_immutable():
    """A run's settings must not be mutated midway by a later surface."""
    with pytest.raises(Exception):
        DepositionSettings(pcb=PCB).dispense_rate = 999.0


# ── P2.2: volume parity across surfaces, correction OFF and ON ───────────────

CORRECTION_CFG = {
    "enabled": True,
    "valves_in_series": 2,
    "beta": 0.30,
    "eta_ref_mpas": 1.0,
    "alpha_growth_per_run": 0.05,   # non-zero: dead volume grows down the run
    "pump_line": {"0": 0, "1": 1, "2": 2},
    "line": {
        str(i): {
            "cracking_kpa_per_valve": 8.0,
            "compliance_uL_per_kpa": 0.55,
            "alpha_base": 0.20,
            "viscosity_mpas": 1.0,
        }
        for i in range(3)
    },
}


def _deposit_volumes(wf) -> dict[str, list[float]]:
    """Per-channel volumes as they actually reach the hardware."""
    return {
        s.name: list(s.params["vols"])
        for s in wf.resolve_steps()
        if s.name.startswith("deposit_ch")
    }


def _workflow_with(correction) -> object:
    settings = DepositionSettings(
        pump_ids=(0, 1, 2), dispense_rate=100.0, flush_rate=500.0,
        pcb=PCB, origin_xy=(43.5, 50.0), correction=correction,
    )
    return build_deposition_workflow(
        get_deposition_recipe("single_drop"), CHANNELS, FORMULATION,
        settings=settings, catalog=_catalog(), name="w")


@pytest.mark.parametrize("enabled", [False, True])
def test_ht_and_campaign_dispense_identical_volumes(enabled):
    """The P2.2 acceptance criterion, exercised with correction off *and* on.

    Correction is off by default and has never run on the rig, so the paths
    agree today — but the dormant branch must not be allowed to rot, because
    switching it on is precisely when a silent divergence would start
    under-delivering on one surface.
    """
    from softae.core.liquid_handling import DeadVolumeCorrection

    correction = DeadVolumeCorrection.from_config(
        (0, 1, 2), CORRECTION_CFG, enabled=enabled)

    spec = CampaignSpec(
        name="c", channels=(21, 22), pump_ids=(0, 1, 2),
        disp_rate=100.0, line_flush_rate=500.0,
    )
    campaign_settings = spec.deposition_settings(pcb=PCB, n_pumps=3)
    # Same physical intent, expressed the way the HT tab expresses it.
    ht_settings = DepositionSettings(
        pump_ids=(0, 1, 2), dispense_rate=100.0,
        flush_rate=campaign_settings.flush_rate, pcb=PCB,
    )

    from dataclasses import replace
    a = build_deposition_workflow(
        get_deposition_recipe("single_drop"), CHANNELS, FORMULATION,
        settings=replace(campaign_settings, correction=correction),
        catalog=_catalog(), name="w")
    b = build_deposition_workflow(
        get_deposition_recipe("single_drop"), CHANNELS, FORMULATION,
        settings=replace(ht_settings, correction=correction),
        catalog=_catalog(), name="w")

    assert _deposit_volumes(a) == _deposit_volumes(b)


def test_disabled_correction_passes_volumes_through_untouched():
    """The live default: what the solver asked for is what the pump is told."""
    from softae.core.liquid_handling import DeadVolumeCorrection

    off = DeadVolumeCorrection.from_config((0, 1, 2), CORRECTION_CFG, enabled=False)
    vols = _deposit_volumes(_workflow_with(off))
    assert vols["deposit_ch21"] == pytest.approx(FORMULATION[21])
    assert vols["deposit_ch22"] == pytest.approx(FORMULATION[22])


def test_enabled_correction_adds_dead_volume_once():
    """Commanded > delivered, and by exactly one dead volume — not two.

    Guards the double-correction hazard: the HT path used to correct before the
    marshaller, so feeding it pre-corrected volumes would silently over-deliver.
    """
    from softae.core.liquid_handling import DeadVolumeCorrection

    on = DeadVolumeCorrection.from_config((0, 1, 2), CORRECTION_CFG, enabled=True)
    vols = _deposit_volumes(_workflow_with(on))

    expected_ch21 = on.commanded(FORMULATION[21], run_index=1)
    assert vols["deposit_ch21"] == pytest.approx(expected_ch21)
    # Strictly greater where a volume was actually requested.
    assert vols["deposit_ch21"][0] > FORMULATION[21][0]
    # A zero component stays zero — no dead volume on a pump that is not used.
    assert vols["deposit_ch21"][2] == pytest.approx(0.0)


def test_dead_volume_grows_with_position_in_the_run():
    """run_index is the channel's ordinal, matching the pre-move HT convention."""
    from softae.core.liquid_handling import DeadVolumeCorrection

    on = DeadVolumeCorrection.from_config((0, 1, 2), CORRECTION_CFG, enabled=True)
    first = on.commanded([10.0, 10.0, 10.0], run_index=1)
    later = on.commanded([10.0, 10.0, 10.0], run_index=5)
    assert later[0] < first[0]   # alpha grows → dead volume shrinks


def test_correction_is_not_applied_twice_by_the_marshaller():
    """Calling the marshaller on already-corrected volumes must be detectable."""
    from softae.core.liquid_handling import DeadVolumeCorrection

    on = DeadVolumeCorrection.from_config((0, 1, 2), CORRECTION_CFG, enabled=True)
    once = on.commanded(FORMULATION[21], run_index=1)
    twice = on.commanded(once, run_index=1)
    assert twice[0] > once[0]                       # double-correction is visible
    assert _deposit_volumes(_workflow_with(on))["deposit_ch21"] == pytest.approx(once)
