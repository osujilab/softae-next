"""Start a campaign the way a terminal would, then let go of the rig.

The Live BO tab used to run its campaign **inside the GUI process**, on a daemon
thread, over the window's own :class:`~softae.server.manager.InstrumentManager`.
Two things follow from that, and both are defects rather than trade-offs:

*The run had no identity of its own.* :func:`softae.core.run_lock.acquire_run_lock`
is re-entrant for the process that already holds the lock, and since the GUI
claims the rig for its instrument session
(:func:`softae.core.rig_session.claim_rig_session`), the campaign's own acquire
handed back the window's ``gui:desktop`` claim unchanged. The lock never said
``campaign:<name>:<run_id>`` and never carried a run directory, so a second window
— or ``softae-campaign control`` — was told there was nothing to attach to while a
campaign was mid-anneal.

*Two processes read the same bus.* An in-process campaign runs its conditions
publisher (5 s) beside the window's own ``InstrumentPoller`` (2 s), both taking
``AsyncTempController._serial_lock``. The sidecar exists precisely so that the
process holding the sessions is the only one reading them.

So the tab does what :class:`~softae.gui.widgets.calibration_launcher.CalibrationLauncherDialog`
already does for bench sequences: write the spec down, hand the instruments back,
and spawn a **detached** child that owns the rig for the run's whole length. The
GUI then attaches to that child through the ordinary discovery path
(:func:`softae.core.campaign_discovery.find_running_campaign`) — the same path it
would use for a campaign someone else started from a terminal. There is
deliberately no shortcut for the campaign this window happened to start: a
channel that exists only for the GUI-started case is a second safety posture, and
the drift between two postures is what this arc exists to end.

**What this module may not do is decide that a file is a faithful copy of the
spec.** :func:`softae.core.campaign_spec_io.spec_toml_completeness` decides that,
and the caller must ask before writing anything: a composition campaign written
to TOML loses ``general_formulation``, reloads with no ``vol_params`` either, and
searches its composition axes as raw µL volumes without raising. Spawning that is
worse than refusing to spawn at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Where a launched campaign's spec file is kept, under the project directory.
#: Mirrors :data:`softae.core.rejected_launch.REJECTED_DIRNAME` — one directory
#: for the launches that happened, one for the launches that were refused.
LAUNCHED_DIRNAME = "launched"

#: Where a launched child's stdout/stderr lands.
LOG_DIRNAME = "logs"


@dataclass(frozen=True)
class DetachedCampaign:
    """A campaign running in a child process this window started and let go of.

    **There is no ``abort()`` here, and its absence is the design.**
    :meth:`softae.gui.tabs._bo_base.BOTabBase._abort_run_impl` looks for one on
    whatever it finds in ``_runner``, and
    :meth:`softae.gui.daemon_runner.DaemonRunnerMixin.cleanup` calls that from the
    window's ``closeEvent``. A handle that could stop the run would therefore stop
    it every time the operator closed the window — killing the overnight campaign
    that the whole detachment exists to protect. The stop that *does* reach this
    child is :class:`~softae.gui.widgets.campaign_control.CampaignControlBar`,
    which writes a request into the run directory and waits to be acknowledged.
    """

    pid: int
    name: str
    spec_path: Path
    log_path: Path
    project_dir: Path
    started_at: str

    def describe(self) -> str:
        """The operator-facing paragraph shown once, at launch."""
        return (
            f"'{self.name}' is running as PID {self.pid}.\n\n"
            f"It keeps running if you close this tab or the whole application — "
            f"the rig belongs to that process now, not to this window.\n\n"
            f"Spec:\n    {self.spec_path}\n"
            f"Log:\n    {self.log_path}"
        )


def _slug(text: str) -> str:
    """A filename stem that cannot escape the directory it is written into."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("._")
    return cleaned[:60] or "campaign"


def _stamp(now: datetime | None = None) -> str:
    return (now or datetime.now(tz=timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def write_launch_spec(spec: Any, project_dir: "str | Path", *,
                      now: datetime | None = None) -> Path:
    """Write *spec* where the child will read it from. Returns the path.

    Kept beside the project rather than in a temp directory: the file is the only
    durable statement of what was launched, and an operator asking "what did I
    start last night?" should find it next to the data it produced.

    Deliberately **not** guarded here. The caller must have proved the file
    carries the whole spec (see this module's docstring); a guard in the writer
    would let a caller skip the proof and still feel checked.
    """
    from softae.core.campaign_spec_io import write_campaign_spec_toml

    base = Path(project_dir).expanduser() / LAUNCHED_DIRNAME
    path = base / f"{_slug(getattr(spec, 'name', ''))}_{_stamp(now)}.toml"
    return write_campaign_spec_toml(spec, path)


def launch_log_path(project_dir: "str | Path", name: str, *,
                    now: datetime | None = None) -> Path:
    """Where the detached child's console output goes."""
    base = Path(project_dir).expanduser() / LOG_DIRNAME
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_stamp(now)}_{_slug(name)}.log"


def campaign_run_argv(
    spec_path: "str | Path",
    project_dir: "str | Path",
    *,
    mock: bool = False,
    head_up: bool | None = None,
) -> list[str]:
    """The child's argv, in the form ``softae-campaign`` actually parses.

    ``-m softae.tools.campaign`` rather than the console script, for the reason
    :func:`~softae.gui.widgets.calibration_launcher.spawn_detached` runs
    ``sys.executable``: the entry point on ``PATH`` may belong to a different
    virtualenv than the one this window is running out of, and a campaign that
    starts against the wrong interpreter fails hours later with an import error.

    ``--yes`` is not optional. The child's stdin is ``DEVNULL``, so every prompt
    ``softae-campaign run`` would otherwise raise — the projection approval, the
    head question — resolves to a refusal, and the campaign would exit before it
    connected. The prompts the flag skips are the ones this window has already
    asked (:meth:`~softae.gui.tabs._autonomous_run.AutonomousRunMixin._preflight_projection_ok`,
    :meth:`~softae.gui.tabs._autonomous_run.AutonomousRunMixin._verify_head_position`).

    *head_up* is passed through for the same reason and must be genuinely known:
    the loop drives the head with **conditional** commands, so a guess costs one
    wrong flip. ``None`` produces no flag, which the child refuses to start on —
    which is the correct outcome, loudly, rather than a guess.
    """
    argv = ["-m", "softae.tools.campaign", "run", str(spec_path),
            "--project", str(project_dir), "--yes"]
    if mock:
        argv.append("--mock")
    if head_up is not None:
        argv.append("--head-up" if head_up else "--head-down")
    return argv


def connected_instruments(manager: Any) -> list[str]:
    """Names of the instruments this process still has open.

    **An enumeration that fails answers "something is open".** The single use of
    this is deciding whether it is safe to hand the ports to another process, and
    "I could not tell" must not be spelled the same way as "nothing is held".
    """
    try:
        return [str(s.get("name")) for s in manager.list_instruments()
                if s.get("connected")]
    except Exception:
        logger.warning("instrument_enumeration_failed", exc_info=True)
        return ["(could not enumerate the instruments)"]


def campaign_runs_on_mocks(manager: Any) -> bool:
    """Whether the child should be started with ``--mock``.

    :func:`softae.core.rig_session.session_is_simulated` rather than
    :func:`softae.core.run_lock.rig_is_simulated`, because the question is about
    **ports**, not motion: a manager with a real potentiostat and a mock stage
    reads as simulated to a motion-shaped question, and starting that child with
    ``--mock`` would record fabricated spectra as real data.
    """
    from softae.core.rig_session import session_is_simulated

    return session_is_simulated(manager)


def spawn_campaign(argv: list[str], *, log_file: "str | Path") -> int:
    """Start the child and return its PID.

    Delegates to :func:`~softae.gui.widgets.calibration_launcher.spawn_detached`
    rather than reimplementing it — that is the shipped statement of what "the
    child outlives this window" means (``DETACHED_PROCESS |
    CREATE_NEW_PROCESS_GROUP`` on Windows, ``start_new_session`` elsewhere, output
    to a file rather than a pipe nobody will read), and a second copy of it is a
    second thing that can quietly stop being true.
    """
    from softae.gui.widgets.calibration_launcher import spawn_detached

    return spawn_detached(argv, log_file=Path(log_file))
