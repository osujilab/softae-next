"""A resume must not mix EIS estimators across an ``[eis] regime_aware`` edit (R6).

The campaign reads ``RegimeSettings`` once at launch, but the resume fingerprint
covers only the spec — so without this check a run could park, have the flag
edited, and resume, telling one search values from two estimators. Driven through
the real ``run_autonomous_campaign``: the refusal is only worth having if the
entry point actually applies it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import softae.analysis.eis.regime_route as rr
from softae.core.autonomous_wiring import (
    CampaignSpec,
    check_resume_regime,
    composition_target_objective,
    run_autonomous_campaign,
)
from softae.core.campaign_resume import ResumeMismatchError
from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}
OBJ = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="regime_resume", channels=(21, 22), pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE, vol_params=("vol_p0", "vol_p1"), pump_ids=(0, 1),
        time_scale=0.0, budget=2, seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


def _flag(monkeypatch, enabled: bool) -> None:
    monkeypatch.setattr(rr, "regime_settings",
                        lambda config=None: rr.RegimeSettings(enabled=enabled))


async def _park_a_run(connected, store: DataStore, monkeypatch, *, enabled: bool) -> None:
    """Run to budget with the clean-finish cleanup suppressed: a parked checkpoint."""
    _flag(monkeypatch, enabled)
    with monkeypatch.context() as m:
        m.setattr(store, "clear_campaign_checkpoint", lambda *_a, **_k: None)
        await run_autonomous_campaign(_spec(), manager=connected, data_store=store,
                                      objective_extractor=OBJ)
    assert store.campaign_checkpoint("regime_resume") is not None


async def _resume(connected, store: DataStore, events: list[dict]):
    return await run_autonomous_campaign(
        _spec(budget=3), manager=connected, data_store=store,
        objective_extractor=OBJ, on_event=events.append, resume=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("launched, now", [(True, False), (False, True)])
async def test_resume_regime_flag_changed_refuses_naming_both_values(
        connected, tmp_path: Path, monkeypatch, launched: bool, now: bool):
    store = DataStore(tmp_path / "proj")
    await _park_a_run(connected, store, monkeypatch, enabled=launched)
    recorded = json.loads(store.campaign_checkpoint("regime_resume")["spec_json"])
    assert recorded["regime"] == {"enabled": launched}

    _flag(monkeypatch, now)
    events: list[dict] = []
    expected = (f"regime_aware = {json.dumps(launched)} at launch, "
                f"{json.dumps(now)} now")
    with pytest.raises(ResumeMismatchError, match=expected) as err:
        await _resume(connected, store, events)
    assert "Restore the setting, or start a fresh run" in str(err.value)

    refused = [e for e in events if e["type"] == "resume_refused"]
    assert refused and expected in refused[0]["detail"]
    assert not any(e["type"] == "resumed" for e in events)
    # Refusal must not consume the checkpoint: restoring the flag still resumes.
    assert store.campaign_checkpoint("regime_resume") is not None
    store.close()


@pytest.mark.asyncio
async def test_resume_checkpoint_without_recorded_regime_warns_and_proceeds(
        connected, tmp_path: Path, monkeypatch):
    store = DataStore(tmp_path / "proj")
    await _park_a_run(connected, store, monkeypatch, enabled=True)
    # A checkpoint written before the settings were recorded: no "regime" key.
    cp = store.campaign_checkpoint("regime_resume")
    legacy = json.loads(cp["spec_json"])
    legacy.pop("regime")
    store.save_campaign_checkpoint(
        "regime_resume", iteration=cp["iteration"], run_id=cp["run_id"],
        loop_state=cp["loop_state"], board_id=cp["board_id"],
        spec_json=json.dumps(legacy, sort_keys=True),
        optimizer_json=cp["optimizer_json"])

    _flag(monkeypatch, False)
    events: list[dict] = []
    result = await _resume(connected, store, events)

    warned = [e for e in events if e["type"] == "resume_regime_unrecorded"]
    assert len(warned) == 1 and warned[0]["regime_aware"] is False
    assert any(e["type"] == "resumed" for e in events)
    assert not any(e["type"] == "resume_refused" for e in events)
    assert result.n_trials == 3
    store.close()


@pytest.mark.asyncio
async def test_resume_regime_unchanged_proceeds_silently(
        connected, tmp_path: Path, monkeypatch):
    store = DataStore(tmp_path / "proj")
    await _park_a_run(connected, store, monkeypatch, enabled=True)

    events: list[dict] = []
    result = await _resume(connected, store, events)

    assert any(e["type"] == "resumed" for e in events)
    assert not any(e["type"] in ("resume_refused", "resume_regime_unrecorded")
                   for e in events)
    assert result.n_trials == 3
    store.close()


# ── Union of recorded and current keys ([a414] §3) ──────────────────────────
# A field added to RegimeSettings after a run was checkpointed must still be
# guarded: these drive check_resume_regime directly with a widened settings type.


@dataclass(frozen=True)
class _WidenedSettings:
    enabled: bool = False
    regime_b: bool = False


def _checkpoint(regime: dict) -> str:
    return json.dumps({"regime": regime})


def test_check_resume_regime_new_field_at_default_proceeds():
    spec_json = _checkpoint({"enabled": True})
    assert check_resume_regime(spec_json, _WidenedSettings(enabled=True),
                               campaign="c") is None


def test_check_resume_regime_new_field_non_default_refuses_naming_it():
    spec_json = _checkpoint({"enabled": True})
    with pytest.raises(ResumeMismatchError,
                       match="regime_b = false at launch, true now"):
        check_resume_regime(spec_json, _WidenedSettings(enabled=True, regime_b=True),
                            campaign="c")


# None matters: a removed field read back as None must not equal a recorded null.
@pytest.mark.parametrize("value", [False, None])
def test_check_resume_regime_recorded_key_no_longer_a_field_refuses(value):
    spec_json = _checkpoint({"enabled": True, "retired_flag": value})
    with pytest.raises(ResumeMismatchError,
                       match=f"retired_flag = {json.dumps(value)} at launch, absent now"):
        check_resume_regime(spec_json, rr.RegimeSettings(enabled=True), campaign="c")
