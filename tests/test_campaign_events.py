"""``events.jsonl`` — the campaign's narration stream and its heartbeat.

Stage 3 of ``docs/SubAgent docs/campaign_attach_architecture.md``. The stream is
the durable channel a future attached GUI reads; these tests pin the five
properties that make it worth attaching to, and the one that makes it safe to
leave switched on.

**Worth attaching to:** records land beside the run, a reader tailing mid-run
sees them, a crash leaves what was already written intact, and the heartbeat
advances *inside* a single long step — the case a step-boundary heartbeat cannot
serve and the only case a watcher really cares about.

**Safe to leave on:** a narration failure never fails a trial.

No rig — ``create_mock_manager`` and ``tmp_path`` stores throughout. The long-step
case is driven on an injected clock rather than by waiting, so an eight-hour
anneal costs no wall-clock time here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from softae.core import campaign_events as ce
from softae.core.autonomous_wiring import CampaignSpec, run_autonomous_campaign
from softae.core.campaign_events import EVENTS_FILENAME, CampaignNarrator
from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="narrated_campaign",
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


def _read(path: Path) -> list[dict]:
    """Parse the stream, tolerating a torn final line as a reader must."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return records


# ── Where the records land ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_stream_lands_in_the_run_directory_beside_the_other_sidecars(
        connected, tmp_path: Path):
    """The location is the discovery contract.

    A watcher finds the run directory from the rig lock's ``log_path`` and opens
    the stream by name. A timestamped filename, or a directory of this module's
    own choosing, would make that a glob-and-guess.
    """
    store = DataStore(tmp_path / "proj")
    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store)
    finally:
        store.close()

    path = store.run_dir(result.run_id) / EVENTS_FILENAME
    assert path.exists(), "the campaign left no transcript"
    assert path.parent == store.run_dir(result.run_id)


@pytest.mark.asyncio
async def test_the_stream_persists_the_emit_vocabulary_rather_than_a_new_one(
        connected, tmp_path: Path):
    """A replayed line must be feedable to the same handler as a live event.

    The stream is ``emit()``'s existing vocabulary made durable, keyed on
    ``type`` exactly as the in-memory dispatch is. If this ever diverges, every
    consumer needs a translation layer that has to be kept in step with an
    emitter it cannot see.
    """
    store = DataStore(tmp_path / "proj")
    live: list[dict] = []
    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store,
            on_event=lambda e: live.append(dict(e)))
    finally:
        store.close()

    written = _read(store.run_dir(result.run_id) / EVENTS_FILENAME)
    narration = [r for r in written if r["type"] != "heartbeat"]

    assert [r["type"] for r in narration] == [e["type"] for e in live], (
        "the durable stream and the live dispatch disagree about what happened")
    assert narration[0]["type"] == "run_started", (
        "a reader replaying from byte 0 must learn the run's identity first")
    assert narration[0]["run_id"] == result.run_id
    # Monotone, gapless, and assigned by the writer — the ordering a reader
    # relies on when it interleaves this stream with the DataStore.
    assert [r["seq"] for r in written] == list(range(len(written)))


@pytest.mark.asyncio
async def test_the_stream_does_not_restate_the_scientific_record(
        connected, tmp_path: Path):
    """Narration and liveness only.

    Measurements, spectra and settle verdicts are durable already — in
    ``measurements``, under ``runs/<run_id>/data/``, and in ``settle.json``.
    Restating them here would make the sidecar a second, unversioned copy of the
    record, which is how two answers to one question start.
    """
    store = DataStore(tmp_path / "proj")
    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store)
    finally:
        store.close()

    text = (store.run_dir(result.run_id) / EVENTS_FILENAME).read_text(
        encoding="utf-8")
    for forbidden in ("frequencies", "z_real", "z_imag", "spectrum"):
        assert forbidden not in text, (
            f"the narration stream is carrying {forbidden!r} — that is record, "
            "not narration")


# ── A reader tailing mid-run ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_reader_tailing_mid_run_sees_records_before_the_campaign_ends(
        connected, tmp_path: Path):
    """Flushed per record, so the file is useful while it is being written.

    This is the whole point of the flush discipline: a stream only readable
    after the process exits answers none of the questions an unattended run
    raises.
    """
    store = DataStore(tmp_path / "proj")
    seen_mid_run: list[list[dict]] = []

    def tail(event: dict) -> None:
        if event.get("type") != "suggestion" or seen_mid_run:
            return
        run_dir = store.run_dir(str(event.get("run_id") or _run_id[0]))
        seen_mid_run.append(_read(run_dir / EVENTS_FILENAME))

    _run_id: list[str] = []

    def watch(event: dict) -> None:
        if event.get("type") == "run_started":
            _run_id.append(str(event["run_id"]))
        tail(event)

    try:
        await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store, on_event=watch)
    finally:
        store.close()

    assert seen_mid_run, "no suggestion was emitted"
    types = [r["type"] for r in seen_mid_run[0]]
    assert "run_started" in types, "nothing was readable while the run was live"
    assert "run_finished" not in types, "the read happened after the run ended"


@pytest.mark.asyncio
async def test_repeated_attach_and_detach_does_not_disturb_the_campaign(
        connected, tmp_path: Path):
    """A reader opening and closing the file is a non-event for the writer.

    This is "the GUI attached, detached, and attached again", and the campaign
    must not be able to tell.
    """
    store = DataStore(tmp_path / "proj")
    baseline = DataStore(tmp_path / "baseline")
    run_ids: list[str] = []

    def churn(event: dict) -> None:
        if event.get("type") == "run_started":
            run_ids.append(str(event["run_id"]))
        if not run_ids:
            return
        path = store.run_dir(run_ids[0]) / EVENTS_FILENAME
        for _ in range(3):
            if path.exists():
                with open(path, encoding="utf-8") as handle:
                    handle.read()

    try:
        watched = await run_autonomous_campaign(
            _spec(seed=11), manager=connected, data_store=store, on_event=churn)
        alone = await run_autonomous_campaign(
            _spec(seed=11), manager=connected, data_store=baseline)
    finally:
        store.close()
        baseline.close()

    assert watched.n_trials == alone.n_trials == 2
    assert watched.final_state == alone.final_state
    assert watched.park_reason is alone.park_reason is None


# ── A crash keeps what was already written ───────────────────────────────────

@pytest.mark.asyncio
async def test_a_campaign_that_dies_leaves_its_transcript_intact(
        connected, tmp_path: Path):
    """Append-only is what buys this.

    The rewrite-the-whole-file pattern ``settle.json`` uses is right for a
    handful of verdicts and wrong here: a crash mid-rewrite loses the history,
    and the history is the only account of a run nobody watched.
    """
    store = DataStore(tmp_path / "proj")
    run_ids: list[str] = []

    def explode(event: dict) -> None:
        if event.get("type") == "run_started":
            run_ids.append(str(event["run_id"]))
        if event.get("type") == "suggestion":
            raise RuntimeError("trial blew up")

    try:
        with pytest.raises(RuntimeError):
            await run_autonomous_campaign(
                _spec(), manager=connected, data_store=store, on_event=explode)
    finally:
        store.close()

    records = _read(store.run_dir(run_ids[0]) / EVENTS_FILENAME)
    types = [r["type"] for r in records]
    assert "run_started" in types
    # The suggestion is persisted *before* `on_event` is called, so the record of
    # what the campaign was doing when it died survives the thing that killed it.
    assert "suggestion" in types, (
        "the event that raised was lost — narration must be written before "
        "dispatch, not after")
    assert "park" in types, "the crash park never reached the transcript"


# ── Narration never fails a trial ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_campaign_whose_transcript_cannot_be_written_still_runs(
        connected, tmp_path: Path, monkeypatch):
    """The contract ``settle.json`` and ``conditions_capture`` already keep.

    A full disk must cost the operator their log, not their night. Every append
    raises here, from the first record to the last.
    """
    store = DataStore(tmp_path / "proj")
    baseline = DataStore(tmp_path / "baseline")

    def boom(self, record):
        raise OSError("no space left on device")

    try:
        alone = await run_autonomous_campaign(
            _spec(seed=3), manager=connected, data_store=baseline)
        monkeypatch.setattr(
            "softae.workflows.experiment_logger.ExperimentLogger.log_record", boom)
        broken = await run_autonomous_campaign(
            _spec(seed=3), manager=connected, data_store=store)
    finally:
        store.close()
        baseline.close()

    assert broken.n_trials == alone.n_trials == 2
    assert broken.final_state == alone.final_state
    assert broken.park_reason is None, "narration turned into a park"


@pytest.mark.asyncio
async def test_a_campaign_whose_transcript_cannot_be_opened_still_runs(
        connected, tmp_path: Path, monkeypatch):
    """The failure that happens *before* there is anything to write.

    Construction is the one place a best-effort writer can still take a campaign
    down, so ``open_narrator`` answers ``None`` — "run unnarrated", never "do not
    run".
    """
    store = DataStore(tmp_path / "proj")
    monkeypatch.setattr(
        ce, "CampaignNarrator",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")))

    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store)
    finally:
        store.close()

    assert result.n_trials == 2
    assert not (store.run_dir(result.run_id) / EVENTS_FILENAME).exists()


def test_a_narrator_on_an_unwritable_path_swallows_every_call(tmp_path: Path):
    """Degraded, not dead: the object stays usable and answers nothing."""
    narrator = CampaignNarrator(tmp_path / "run")
    narrator.close()                 # as an open failure would have left it

    narrator.record("suggestion", {"iteration": 1})
    narrator.beat()
    narrator.close()                 # idempotent


# ── The heartbeat ────────────────────────────────────────────────────────────

class _FakeClock:
    """A wall clock and an ``asyncio.sleep`` that advance it without waiting.

    ``sleep`` yields to the event loop so the heartbeat task and the "long step"
    interleave exactly as they would in a real run — but eight hours of hold
    costs the suite nothing.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += float(seconds)
        import asyncio

        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_the_heartbeat_advances_during_a_single_eight_hour_step(
        tmp_path: Path):
    """The case a step-boundary heartbeat cannot serve, and the reason this exists.

    ``campaign_checkpoints.updated_at`` moves only when an *iteration* completes,
    so inside a long anneal a wedged process and a healthy slow one look
    identical. The beat runs on its own clock and keeps ticking with no
    narration event in sight.
    """
    import asyncio

    clock = _FakeClock()
    narrator = CampaignNarrator(
        tmp_path / "run", heartbeat_s=30.0, now=clock.now, sleep=clock.sleep)
    narrator.record("suggestion", {"iteration": 4})
    narrator.start_heartbeat()

    # One step, eight hours of fake time, and nothing else emitted throughout.
    guard = 0
    while clock.t < 8 * 3600:
        await asyncio.sleep(0)
        guard += 1
        assert guard < 100_000, "the heartbeat task never ran"
    await narrator.aclose()

    records = _read(narrator.path)
    beats = [r for r in records if r["type"] == "heartbeat"]

    assert len(beats) >= 900, (
        f"only {len(beats)} beat(s) inside an 8-hour step at a 30 s cadence — "
        "a watcher cannot tell alive from wedged")
    assert all(b["iteration"] == 4 for b in beats), (
        "the beat lost track of which iteration is running")
    assert all(b["phase"] == "suggestion" for b in beats), (
        "the beat cannot say what the campaign is doing")
    # Staleness is computable, and phase age is what distinguishes a long hold
    # from a stuck one.
    assert beats[-1]["phase_age_s"] > beats[0]["phase_age_s"] > 0
    assert beats[-1]["uptime_s"] >= 8 * 3600 - 120

    # And no narration was invented to carry it.
    assert [r["type"] for r in records if r["type"] != "heartbeat"] \
        == ["suggestion"]


@pytest.mark.asyncio
async def test_the_heartbeat_stops_when_the_campaign_does(tmp_path: Path):
    """A beat that outlived the run would report a torn-down campaign as alive."""
    import asyncio

    clock = _FakeClock()
    narrator = CampaignNarrator(
        tmp_path / "run", heartbeat_s=30.0, now=clock.now, sleep=clock.sleep)
    narrator.start_heartbeat()
    for _ in range(10):
        await asyncio.sleep(0)
    await narrator.aclose()

    settled = len(_read(narrator.path))
    for _ in range(50):
        await asyncio.sleep(0)

    assert len(_read(narrator.path)) == settled, "the heartbeat outlived the run"


@pytest.mark.asyncio
async def test_a_real_campaign_beats_without_being_asked(
        connected, tmp_path: Path, monkeypatch):
    """The wiring, not the mechanism: the loop's beat is actually started.

    Cadence dropped to near zero so a two-trial mock campaign is long enough to
    contain one; the shipped default is 30 s, argued in the module.
    """
    monkeypatch.setattr(ce, "DEFAULT_HEARTBEAT_S", 0.02)
    store = DataStore(tmp_path / "proj")
    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store)
    finally:
        store.close()

    records = _read(store.run_dir(result.run_id) / EVENTS_FILENAME)
    assert any(r["type"] == "heartbeat" for r in records), (
        "no heartbeat reached the stream — a watcher has no liveness signal")


# ── The file is bounded ──────────────────────────────────────────────────────

def test_the_stream_rotates_once_it_reaches_its_cap(tmp_path: Path):
    """Bounded at twice the cap, permanently, with one generation of history kept.

    A backstop rather than a routine event — at the shipped cadence 32 MB is
    roughly three months of running — but an unattended process writing to the
    disk that also holds the DataStore does not get to be unbounded.
    """
    narrator = CampaignNarrator(tmp_path / "run", heartbeat_s=0, max_bytes=4096)
    for i in range(400):
        narrator.record("step_recovered", {"iteration": i, "pad": "x" * 60})
    narrator.close()

    previous = tmp_path / "run" / ce.PREVIOUS_FILENAME
    assert previous.exists(), "the cap was never enforced"
    assert narrator.path.stat().st_size < 4096 * 2
    assert previous.stat().st_size < 4096 * 2

    # The rotation announces itself, so a reader knows where the earlier history
    # went rather than silently believing the run began mid-stream.
    kinds = [r["type"] for r in _read(narrator.path)]
    assert "stream_rotated" in kinds
    rotated = next(r for r in _read(narrator.path) if r["type"] == "stream_rotated")
    assert rotated["previous"] == ce.PREVIOUS_FILENAME
    # Sequence numbers survive rotation, so a reader stitching the two
    # generations can tell whether it lost anything.
    assert _read(previous)[-1]["seq"] < rotated["seq"]


def test_a_rotation_that_cannot_happen_does_not_stop_the_stream(
        tmp_path: Path, monkeypatch):
    """On Windows a reader tailing the file can block the rename.

    ``os.replace`` onto a file another process holds open without
    ``FILE_SHARE_DELETE`` raises. The stream must keep flowing and try again
    later — a watcher holding the file open is a watcher that will let go.
    """
    narrator = CampaignNarrator(tmp_path / "run", heartbeat_s=0, max_bytes=2048)

    def refuse(src, dst):
        raise PermissionError("the file is in use by another process")

    monkeypatch.setattr(ce.os, "replace", refuse)
    for i in range(200):
        narrator.record("step_recovered", {"iteration": i, "pad": "x" * 60})
    narrator.close()

    assert not (tmp_path / "run" / ce.PREVIOUS_FILENAME).exists()
    records = _read(narrator.path)
    assert len(records) == 200, "the stream stopped when rotation was refused"


# ── The writer is the one this codebase already had ──────────────────────────

def test_the_narrator_writes_through_the_existing_experiment_logger(
        tmp_path: Path, monkeypatch):
    """One JSONL writer in the process, one flush discipline.

    A second writer beside ``ExperimentLogger`` — same format, same directory,
    its own idea of when to flush — is duplication that only shows up as a bug
    the first time the two disagree about durability.
    """
    from softae.workflows.experiment_logger import ExperimentLogger

    written: list[dict] = []
    original = ExperimentLogger.log_record

    def spy(self, record):
        written.append(record)
        return original(self, record)

    monkeypatch.setattr(ExperimentLogger, "log_record", spy)
    narrator = CampaignNarrator(tmp_path / "run", heartbeat_s=0)
    narrator.record("run_started", {"run_id": "r1"})
    narrator.close()

    assert [r["type"] for r in written] == ["run_started"]


def test_the_experiment_logger_still_names_its_own_files_by_default(
        tmp_path: Path):
    """The ``filename`` override must not disturb the existing callers.

    ``workflows/__main__.py`` and the Experiment tab both rely on the
    timestamped default to keep successive runs from overwriting each other.
    """
    from softae.workflows.experiment_logger import ExperimentLogger

    logger = ExperimentLogger(tmp_path, "some_workflow")
    try:
        assert logger.log_path.name.startswith("some_workflow_")
        assert logger.log_path.name.endswith(".jsonl")
    finally:
        logger.close()


def test_each_record_is_flushed_rather_than_buffered(tmp_path: Path):
    """A hard kill must not lose what was already narrated.

    The crash case the shutdown work exists for: nothing here gets a chance to
    close the file, so anything still in a buffer is anything the operator never
    learns.
    """
    narrator = CampaignNarrator(tmp_path / "run", heartbeat_s=0)
    narrator.record("run_started", {"run_id": "r1"})
    narrator.record("suggestion", {"iteration": 1})

    # Read without closing — exactly what a `kill -9` would leave behind.
    records = _read(narrator.path)
    assert [r["type"] for r in records] == ["run_started", "suggestion"]
    narrator.close()
