"""``conditions.json`` — what a campaign publishes for a watcher that cannot look.

Stage 5, S5.F of ``docs/SubAgent docs/attach_stage5_gui_ownership.md``. An
attached GUI opens no instrument sessions, so it cannot read a temperature the
campaign owns; without this sidecar the Monitoring tab is blank for the whole of
an unattended run.

This is the only part of the attach arc that adds work *inside* a live campaign
process — one that runs for eight hours overnight with nobody watching — so the
cases below are weighted accordingly. Three properties carry the risk and each
is pinned here:

* **the cadence is a ceiling** — a beat arriving while a read is still in flight
  is counted and dropped, never queued behind it;
* **the read never runs on the event loop** — so a serial bus contended for tens
  of seconds cannot delay the ~1 s ``control.json`` poll, i.e. cannot delay an
  operator's Abort. That is the case that matters most and it is measured, not
  asserted structurally;
* **staleness is visible** — the file always carries the last *completed* read
  with its own stamps, and keeps being republished while a read hangs.

No rig: the reads are synchronous callables the test holds open exactly as a
driver retry holds ``_serial_lock``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from softae.core import campaign_events as ce
from softae.core.autonomous_wiring import CampaignSpec, run_autonomous_campaign
from softae.core.campaign_events import (
    CONDITIONS_FILENAME,
    ConditionsPublisher,
    conditions_path,
    open_control_watcher,
    write_control_request,
)
from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager

ENV_A = {
    "stage_temp_sp_C": 60.0,
    "chamber_air_C": 22.5,
    "stage_temp_pv_C": 59.4,
    "rh_sp_pct": 40.0,
    "rh_pv_pct": 41.2,
}


class _HeldRead:
    """A synchronous rig read the test can hold open, as a driver retry does.

    ``read_environment`` is straight-line blocking code behind a serial lock, so
    the fidelity that matters here is that it blocks a *thread* — the publisher
    must keep its loop free regardless.
    """

    def __init__(self, env: dict | None = None, *, hold: bool = True) -> None:
        self.env = dict(env if env is not None else ENV_A)
        self.gate = threading.Event()
        self.hold = hold
        self.raises = False
        self.started = 0
        self.finished = 0

    def release(self) -> None:
        self.hold = False
        self.gate.set()

    def __call__(self) -> dict:
        self.started += 1
        if self.hold:
            # Bounded: a test that hangs reports nothing.
            self.gate.wait(timeout=10.0)
        self.finished += 1
        if self.raises:
            raise RuntimeError("the bus said no")
        return dict(self.env)


async def _until(predicate, *, timeout: float = 5.0) -> bool:
    """Wait for *predicate*, bounded. Returns whether it came true."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.002)
    return predicate()


def _payload(run_dir: Path) -> dict:
    return json.loads(conditions_path(run_dir).read_text(encoding="utf-8"))


# ── The cadence is a ceiling ─────────────────────────────────────────────────

async def test_publisher_beat_while_a_read_is_in_flight_is_skipped_not_queued(
        tmp_path: Path):
    """One read outstanding, ever — the property that keeps this knob safe.

    A contended read can outlast six beats. If each of those beats started its
    own read, a monitoring cadence would take the serial lock away from the
    campaign that owns it, in a burst, at the worst possible moment. So a beat
    that finds a read in flight counts itself and returns.
    """
    read = _HeldRead()
    publisher = ConditionsPublisher(tmp_path / "run", read=read, poll_s=0.01)
    try:
        publisher.start()
        assert await _until(lambda: publisher.skipped_beats >= 3), (
            "the clock stopped while a read was in flight")
        assert read.started == 1, (
            f"{read.started} reads started while {publisher.skipped_beats} beats "
            "were skipped — the beats were queued, not dropped")
        assert read.finished == 0, "the test's own read did not stay held"
    finally:
        read.release()
        await publisher.aclose()


async def test_publisher_skipped_beat_republishes_the_last_completed_read(
        tmp_path: Path):
    """Never a wait, always a stamp.

    The contended read degrades to the *previous* value carrying its own older
    ``completed_at`` — never to a wait, and never to an old number wearing a
    fresh stamp. Republishing on the skipped beat is what distinguishes "the
    publisher died" from "the read is stuck", which have different answers.
    """
    run_dir = tmp_path / "run"
    read = _HeldRead(hold=False)
    publisher = ConditionsPublisher(run_dir, read=read, poll_s=0.01)
    try:
        publisher.start()
        assert await _until(
            lambda: conditions_path(run_dir).exists()
            and _payload(run_dir)["completed_at"] is not None)

        # Now hold the next read open and let the beats pile up behind it.
        read.gate.clear()
        read.hold = True
        assert await _until(lambda: publisher.skipped_beats >= 2)
        held = _payload(run_dir)
        assert await _until(
            lambda: _payload(run_dir)["skipped_beats"] > held["skipped_beats"])
        later = _payload(run_dir)
    finally:
        read.release()
        await publisher.aclose()

    assert held["env"] == ENV_A, "the held read wiped the value it replaced"
    assert later["completed_at"] == held["completed_at"], (
        "an in-flight read was stamped as though it had completed")
    assert later["env"] == held["env"]
    assert later["started_at"] > later["completed_at"], (
        "a read is in flight, and the payload does not say so")


async def test_publisher_slow_read_does_not_delay_the_control_poll(
        tmp_path: Path):
    """The one that matters: a monitoring knob must never delay an Abort.

    ``read_environment`` is synchronous and its driver calls hold
    ``_serial_lock`` for a deadline measured in tens of seconds. Awaited on the
    event loop, one bad read would stall the heartbeat *and* the ~1 s
    ``control.json`` poll — so the comfort knob could postpone the operator's
    stop. The read goes to a worker thread; this measures that it did.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    read = _HeldRead()
    seen: list[str] = []
    publisher = ConditionsPublisher(run_dir, read=read, poll_s=0.01)
    watcher = open_control_watcher(
        run_dir,
        handlers={"abort": lambda req: seen.append(req.action) or "applied"},
        poll_s=0.01,
    )
    assert watcher is not None
    try:
        publisher.start()
        watcher.start()
        assert await _until(lambda: read.started == 1 and publisher.skipped_beats >= 1), (
            "the read never began, so nothing was being contended for")

        write_control_request(run_dir, "abort", reason="operator E-Stop")

        assert await _until(lambda: seen == ["abort"], timeout=2.0), (
            "the abort did not reach its handler while a rig read was in "
            "flight — the publisher is blocking the event loop")
        assert read.finished == 0, (
            "the read completed before the abort landed, so this proved nothing")
    finally:
        read.release()
        await watcher.aclose()
        await publisher.aclose()


# ── One slot, replaced ───────────────────────────────────────────────────────

async def test_publisher_file_is_replaced_rather_than_appended(tmp_path: Path):
    """A growing conditions file would be a measurement log by another name.

    The events stream deliberately carries no measurements; an appended
    conditions history would be an unversioned second copy of the ``conditions``
    table, written by the process least able to say what it means. One slot,
    renamed onto — the same discipline ``write_control_request`` uses, so a
    reader sees a whole payload or the previous one, never a prefix.
    """
    run_dir = tmp_path / "run"
    read = _HeldRead(hold=False)
    publisher = ConditionsPublisher(run_dir, read=read, poll_s=0.005)
    try:
        publisher.start()
        assert await _until(lambda: read.finished >= 2)
        settled = conditions_path(run_dir).stat().st_size
        assert await _until(lambda: read.finished >= 8)
        grown = conditions_path(run_dir).stat().st_size
    finally:
        await publisher.aclose()

    # A tolerance rather than equality: the digit counts of ``read_ms`` and
    # ``skipped_beats`` move. Appending six payloads would add ~1.5 kB.
    assert grown < settled + 40, (
        f"the sidecar grew from {settled} to {grown} bytes across six beats — "
        "it is being appended to")
    # Exactly one JSON object, and nothing left behind by the rename.
    assert json.loads(conditions_path(run_dir).read_text(encoding="utf-8"))["env"] \
        == ENV_A
    assert sorted(p.name for p in run_dir.iterdir()) == [CONDITIONS_FILENAME], (
        "a temporary file survived the replace")


# ── A read that fails is still a payload ─────────────────────────────────────

async def test_publisher_read_that_raises_publishes_nulls_and_keeps_beating(
        tmp_path: Path):
    """A failing read must not take the publisher — or the campaign — with it.

    Nulls rather than the previous value, deliberately: an old number carrying a
    fresh stamp is the one lie this file must not tell.
    """
    run_dir = tmp_path / "run"
    read = _HeldRead(hold=False)
    read.raises = True
    publisher = ConditionsPublisher(run_dir, read=read, poll_s=0.005)
    try:
        publisher.start()
        assert await _until(
            lambda: conditions_path(run_dir).exists()
            and _payload(run_dir)["completed_at"] is not None)
        failed = _payload(run_dir)
        assert set(failed["env"]) == set(ENV_A), "the payload lost its shape"
        assert all(v is None for v in failed["env"].values())

        # And the task is still beating: a later good read lands.
        read.raises = False
        assert await _until(lambda: _payload(run_dir)["env"] == ENV_A), (
            "one failed read ended the publisher")
    finally:
        await publisher.aclose()


async def test_publisher_read_returning_nulls_still_publishes_five_keys(
        tmp_path: Path):
    """An unreadable instrument is a null field, not a missing one.

    A consumer that has to ask whether a key exists before asking what it says
    will get that wrong at 2 a.m.
    """
    run_dir = tmp_path / "run"
    read = _HeldRead({k: None for k in ENV_A}, hold=False)
    publisher = ConditionsPublisher(run_dir, read=read, poll_s=0.005)
    try:
        publisher.start()
        assert await _until(lambda: conditions_path(run_dir).exists())
        payload = _payload(run_dir)
    finally:
        await publisher.aclose()

    assert set(payload["env"]) == set(ENV_A)
    assert payload["completed_at"] is not None
    assert isinstance(payload["read_ms"], int)


async def test_publisher_unwritable_run_directory_does_not_raise(tmp_path: Path):
    """Best-effort, like every other sidecar: it degrades, it does not raise."""
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x", encoding="utf-8")
    read = _HeldRead(hold=False)
    publisher = ConditionsPublisher(blocker / "run", read=read, poll_s=0.005)
    try:
        publisher.start()
        assert await _until(lambda: read.finished >= 2), (
            "the publisher stopped reading because it could not write")
    finally:
        await publisher.aclose()


async def test_publisher_poll_zero_publishes_nothing(tmp_path: Path):
    """``0`` is off, and off means no file and no rig traffic at all.

    Against a headless run this sidecar is *net-new* Modbus traffic on the same
    serial lock the anneal hold polls, so switching it off must genuinely stop
    it rather than merely stop the writing.
    """
    run_dir = tmp_path / "run"
    read = _HeldRead(hold=False)
    publisher = ConditionsPublisher(run_dir, read=read, poll_s=0)
    publisher.start()
    await asyncio.sleep(0.05)
    await publisher.aclose()

    assert read.started == 0, "a disabled publisher still read the rig"
    assert not conditions_path(run_dir).exists()


# ── Wired into a campaign ────────────────────────────────────────────────────

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="published_campaign",
        channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,
        budget=2,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


async def test_campaign_publishes_conditions_beside_the_event_stream(
        connected, tmp_path: Path):
    """The wiring, not the mechanism: a headless run leaves the sidecar behind.

    Beside ``events.jsonl`` because that is the discovery contract — a watcher
    finds the run directory from the rig lock's ``log_path`` and names both
    files without globbing.
    """
    store = DataStore(tmp_path / "proj")
    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store, conditions_poll_s=0.02)
    finally:
        store.close()

    run_dir = store.run_dir(result.run_id)
    payload = _payload(run_dir)
    assert (run_dir / ce.EVENTS_FILENAME).exists()
    assert set(payload["env"]) == set(ENV_A)
    assert payload["started_at"] is not None
    assert payload["skipped_beats"] >= 0


async def test_campaign_conditions_poll_zero_writes_no_sidecar(
        connected, tmp_path: Path):
    store = DataStore(tmp_path / "proj")
    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store, conditions_poll_s=0)
    finally:
        store.close()

    assert not conditions_path(store.run_dir(result.run_id)).exists()


async def test_campaign_publisher_failure_never_reaches_the_run(
        connected, tmp_path: Path, monkeypatch):
    """A campaign runs whether or not it can be watched.

    The whole sidecar is a courtesy to an operator; a courtesy that can end an
    eight-hour unattended run is a defect, not a feature.
    """
    def _boom(_manager):
        raise RuntimeError("serial bus gone")

    monkeypatch.setattr(
        "softae.core.conditions_capture.read_environment", _boom)

    store = DataStore(tmp_path / "proj")
    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store, conditions_poll_s=0.02)
    finally:
        store.close()

    assert result.n_trials >= 1, "a failing conditions read stopped the campaign"
    payload = _payload(store.run_dir(result.run_id))
    assert all(v is None for v in payload["env"].values())


async def test_campaign_cadences_come_from_the_config_when_unset(
        connected, tmp_path: Path, monkeypatch):
    """``[campaign]`` reaches both sidecars — and ``0`` disables each.

    The heartbeat cadence has been a constructor kwarg reachable from config
    nowhere; this is the pass-through that makes the documented ``0`` convention
    real for both knobs at once.
    """
    from softae.config import loader

    monkeypatch.setattr(loader, "campaign_heartbeat_s", lambda: 0.0)
    monkeypatch.setattr(loader, "campaign_conditions_poll_s", lambda: 0.0)

    store = DataStore(tmp_path / "proj")
    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store)
    finally:
        store.close()

    run_dir = store.run_dir(result.run_id)
    assert not conditions_path(run_dir).exists()
    stream = (run_dir / ce.EVENTS_FILENAME).read_text(encoding="utf-8")
    assert '"heartbeat"' not in stream, "heartbeat_s = 0 did not disable the beat"
