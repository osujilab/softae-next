"""The ``softae-commission`` CLI surface.

Written after two defects reached the operator's hands rather than a test: the
console-script entry existed in ``pyproject.toml`` but no shim had been generated, and
``--project`` was *required* — a question the operator has no reason to be able to
answer while standing at the rig with a jumper in one hand.

Both were found by running the thing manually. The argument surface is where a CLI's
behaviour actually lives, it is cheap to test, and nothing else in the suite touches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.tools.commission import _open_store, _parse_channels, build_parser


class _Args:
    """Just enough of an argparse namespace for the store opener."""

    def __init__(self, project=None, mock=False):
        self.project = project
        self.mock = mock


class TestChannelParsing:
    def test_a_single_channel(self):
        assert _parse_channels("32") == [32]

    def test_ranges_and_lists_mix(self):
        assert _parse_channels("1-4, 7") == [1, 2, 3, 4, 7]

    def test_duplicates_collapse_in_order(self):
        assert _parse_channels("3, 1-3, 3") == [3, 1, 2]

    def test_whitespace_and_empty_entries_are_tolerated(self):
        assert _parse_channels(" 1 , , 2 ") == [1, 2]


class TestArgumentSurface:
    def test_a_run_needs_no_project_because_the_answer_is_never_a_choice(self):
        # This was `required=True` and produced an argparse error at the bench. The
        # store is "the same one as everything else"; the operator should not have to
        # know its path.
        args = build_parser().parse_args(
            ["run", "blank_short", "--channels", "32", "--fixture", "mux16"])
        assert args.project is None
        assert args.role == "blank_short"

    def test_derive_and_history_are_equally_undemanding(self):
        p = build_parser()
        assert p.parse_args(["derive", "--fixture", "mux16"]).project is None
        assert p.parse_args(["history", "--fixture", "mux16"]).project is None

    def test_an_unknown_role_is_rejected_by_the_parser_itself(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "blank_magic"])

    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_every_commissioning_role_is_offered(self):
        from softae.analysis.eis.calibration import COMMISSIONING_ROLES

        p = build_parser()
        for role in COMMISSIONING_ROLES:
            assert p.parse_args(["run", role]).role == role


class TestStoreSelection:
    def test_an_explicit_project_is_honoured(self, tmp_path):
        store, project = _open_store(_Args(project=str(tmp_path / "proj")))
        assert Path(project) == tmp_path / "proj"
        store.close()

    def test_the_default_is_the_store_everything_else_uses(self, monkeypatch, tmp_path):
        # Sharing the store is what lets `derive` combine artifacts acquired weeks
        # apart without anyone remembering which directory the last session used.
        from softae.config import loader

        monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
        store, project = _open_store(_Args())
        assert Path(project) == tmp_path / "real"
        store.close()

    def test_a_mock_run_is_isolated_from_the_production_store(self, monkeypatch, tmp_path):
        # A --mock sweep writes synthetic spectra tagged exactly like real ones, so
        # landing them in the production store would let a later `derive` build a
        # calibration from simulated data with nothing in the record to say so. That
        # happened once in development and had to be deleted by hand.
        from softae.config import loader

        monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
        store, project = _open_store(_Args(mock=True))
        assert Path(project) == tmp_path / "real" / "mock"
        store.close()

    def test_an_explicit_project_still_wins_under_mock(self, tmp_path):
        # Someone deliberately pointing a dry run at a directory means it.
        store, project = _open_store(_Args(project=str(tmp_path / "chosen"), mock=True))
        assert Path(project) == tmp_path / "chosen"
        store.close()


class TestManagerConstruction:
    """``InstrumentManager.from_config()`` never existed.

    Both CLIs called it, so every non-mock invocation raised ``AttributeError`` before
    touching an instrument — ``softae-campaign run`` has only ever worked with
    ``--mock``, and the commissioning command failed the first time it was pointed at
    real hardware. The factory the GUI uses is ``drivers.factory.create_manager``.
    """

    def test_the_factory_the_cli_needs_actually_exists(self):
        from softae.drivers import factory

        assert hasattr(factory, "create_manager")

    def test_neither_cli_still_calls_the_method_that_never_existed(self):
        # A grep-style guard: the attribute does not exist, so a call to it is an
        # AttributeError at the bench rather than a test failure here.
        from softae.server.manager import InstrumentManager

        assert not hasattr(InstrumentManager, "from_config")
        for module in ("softae.tools.commission", "softae.tools.campaign"):
            source = Path(__import__(module, fromlist=["x"]).__file__).read_text(
                encoding="utf-8")
            calls = [ln for ln in source.splitlines()
                     if "InstrumentManager.from_config()" in ln
                     and not ln.strip().startswith("#")]
            assert calls == [], f"{module} still calls it: {calls}"

    def test_a_real_run_never_silently_falls_back_to_mock_drivers(self, monkeypatch,
                                                                  tmp_path):
        # The auto mode (mock=None) falls back to mocks when hardware is absent. For
        # commissioning that would write synthetic spectra tagged as a real blank, so
        # the CLI must force mock=False and let the failure surface.
        import softae.drivers.factory as factory
        from softae.config import loader
        from softae.tools.commission import _cmd_run

        seen: dict = {}

        def _spy(*, mock=None, config=None):
            seen["mock"] = mock
            raise RuntimeError("no hardware here")

        monkeypatch.setattr(factory, "create_manager", _spy)
        monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
        args = build_parser().parse_args(
            ["run", "blank_short", "--channels", "1", "--yes"])
        _cmd_run(args)
        assert seen["mock"] is False, "a real commissioning run must force real drivers"

    def test_the_mock_flag_forces_mocks_rather_than_auto_detecting(self, monkeypatch,
                                                                  tmp_path):
        import softae.drivers.factory as factory
        from softae.config import loader
        from softae.tools.commission import _cmd_run

        seen: dict = {}

        def _spy(*, mock=None, config=None):
            seen["mock"] = mock
            raise RuntimeError("stop here")

        monkeypatch.setattr(factory, "create_manager", _spy)
        monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
        args = build_parser().parse_args(
            ["run", "blank_short", "--channels", "1", "--mock", "--yes"])
        _cmd_run(args)
        assert seen["mock"] is True


class TestHardwareInterlock:
    def test_an_unarmed_rig_declines_before_a_run_row_is_created(self, monkeypatch,
                                                                tmp_path):
        # The executor asserts the interlock too — that is the real choke point — but
        # reaching it first left an empty run row behind on every declined attempt.
        import softae.drivers.factory as factory
        from softae.config import loader
        from softae.core.data_store import DataStore
        from softae.core.hardware_safety import HardwareNotArmedError
        from softae.tools.commission import EXIT_DECLINED, _cmd_run

        project = tmp_path / "real"
        monkeypatch.setattr(loader, "data_project_dir", lambda: str(project))
        monkeypatch.setattr(factory, "create_manager",
                            lambda **kw: object())          # never connected

        def _refuse(manager, *, action="move hardware"):
            raise HardwareNotArmedError(f"SAFETY INTERLOCK: refusing to {action}.")

        import softae.core.hardware_safety as safety

        monkeypatch.setattr(safety, "assert_hardware_armed", _refuse)

        args = build_parser().parse_args(
            ["run", "blank_short", "--channels", "1", "--yes"])
        assert _cmd_run(args) == EXIT_DECLINED

        # No store, no run row — nothing to clean up after a declined attempt.
        if (project / "db").exists():
            store = DataStore(project)
            rows = list(store._conn.execute(
                "SELECT run_id FROM experiments WHERE run_id LIKE '%commission%'"))
            store.close()
            assert rows == []


class TestNominalIsRequiredWhereItMatters:
    """A part's marked value is not optional for the artifacts that have one.

    Overhaul 3.7: the capacitor marked "102" (1 nF) measured ~150 nF with tan d = 0.18,
    unusable as a phase reference. The marking and the measurement *disagreeing* is what
    revealed it — with only one of the two numbers, nothing would have flagged it.
    """

    def test_the_parts_with_markings_are_the_ones_that_demand_one(self):
        from softae.analysis.eis.calibration import ARTIFACT_NOMINAL_UNITS

        assert set(ARTIFACT_NOMINAL_UNITS) == {
            "blank_load", "reference_r", "reference_cap"}

    def test_a_short_blank_needs_no_nominal_because_it_has_no_marking(self):
        from softae.analysis.eis.calibration import ARTIFACT_NOMINAL_UNITS

        assert "blank_short" not in ARTIFACT_NOMINAL_UNITS
        assert "blank_open" not in ARTIFACT_NOMINAL_UNITS

    def test_running_a_reference_part_without_its_value_is_refused(self, monkeypatch,
                                                                   tmp_path):
        from softae.config import loader
        from softae.tools.commission import EXIT_FAILED, _cmd_run

        monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
        args = build_parser().parse_args(
            ["run", "reference_cap", "--channels", "1", "--mock", "--yes"])
        assert _cmd_run(args) == EXIT_FAILED     # refused before touching hardware

    def test_the_units_are_named_so_the_operator_knows_what_to_type(self):
        from softae.analysis.eis.calibration import ARTIFACT_NOMINAL_UNITS

        assert ARTIFACT_NOMINAL_UNITS["reference_cap"] == "farads"
        assert ARTIFACT_NOMINAL_UNITS["blank_load"] == "ohms"


# ── Every run row is closed, on every exit path ──────────────────────────────

class _Manager:
    """A manager that opens and closes and drives nothing."""

    async def connect_all(self):
        return None

    async def disconnect_all(self):
        return None


def _arrange_run(monkeypatch, outcome: BaseException | None = None):
    """Wire ``_cmd_run`` to an executor that ends with *outcome* (None = success).

    The interlock is stubbed *open* here rather than closed: these tests are
    about what happens to a run row once one exists, and a declined run never
    creates one (see ``TestHardwareInterlock``).
    """
    import softae.core.hardware_safety as safety
    import softae.drivers.factory as factory
    import softae.workflows.workflow_executor as wfx

    monkeypatch.setattr(factory, "create_manager", lambda **_kw: _Manager())
    monkeypatch.setattr(safety, "assert_hardware_armed", lambda *_a, **_kw: None)

    class _Executor:
        def __init__(self, *_a, **_kw):
            pass

        async def run(self, _wf):
            if outcome is not None:
                raise outcome
            return None

    monkeypatch.setattr(wfx, "WorkflowExecutor", _Executor)


def _the_only_outcome(project: Path) -> dict:
    """``run_outcome`` for the single run the command wrote."""
    from softae.core.data_store import DataStore

    with DataStore(project) as ds:
        run_ids = [r[0] for r in ds._conn.execute(
            "SELECT run_id FROM experiments ORDER BY started_at")]
        assert len(run_ids) == 1, run_ids
        return ds.run_outcome(run_ids[0])


class TestRunRowFinalization:
    """``start_run`` had no matching ``finish_run`` anywhere in this tool.

    Neither the CLI nor ``WorkflowExecutor`` closed the row, so **every**
    successful commissioning sweep left ``finished_at`` NULL — which is
    byte-for-byte what a killed process leaves behind. The next GUI launch read
    it as an unclean shutdown and offered to park the rig over a run that had
    finished perfectly, which is how a real crash report gets trained out of an
    operator.
    """

    def _run(self, monkeypatch, project: Path, outcome=None):
        from softae.tools.commission import _cmd_run

        _arrange_run(monkeypatch, outcome)
        args = build_parser().parse_args(
            ["run", "blank_short", "--channels", "1", "--yes",
             "--project", str(project)])
        return _cmd_run(args)

    def test_a_completed_sweep_closes_its_row_done(self, monkeypatch, tmp_path):
        from softae.tools.commission import EXIT_OK

        project = tmp_path / "proj"
        assert self._run(monkeypatch, project) == EXIT_OK
        assert _the_only_outcome(project) == {"status": "done", "finished": True}

    def test_a_completed_sweep_is_not_reported_as_an_unclean_shutdown(
            self, monkeypatch, tmp_path):
        """The defect as the operator met it, pinned at its own surface."""
        from softae.core.data_store import DataStore

        project = tmp_path / "proj"
        self._run(monkeypatch, project)
        with DataStore(project) as ds:
            assert ds.unfinished_runs() == []

    def test_a_ctrl_c_closes_its_row_interrupted(self, monkeypatch, tmp_path):
        from softae.tools.commission import EXIT_FAILED

        project = tmp_path / "proj"
        assert self._run(monkeypatch, project, KeyboardInterrupt()) == EXIT_FAILED
        assert _the_only_outcome(project)["status"] == "interrupted"

    def test_an_unarmed_rig_at_the_executor_closes_its_row_aborted(
            self, monkeypatch, tmp_path):
        from softae.core.hardware_safety import HardwareNotArmedError
        from softae.tools.commission import EXIT_DECLINED

        project = tmp_path / "proj"
        outcome = HardwareNotArmedError("SAFETY INTERLOCK")
        assert self._run(monkeypatch, project, outcome) == EXIT_DECLINED
        assert _the_only_outcome(project)["status"] == "aborted"

    def test_an_unnamed_failure_still_closes_its_row_error(
            self, monkeypatch, tmp_path):
        """The ``finally`` catch-all: no ``except`` names a bare RuntimeError."""
        project = tmp_path / "proj"
        with pytest.raises(RuntimeError):
            self._run(monkeypatch, project, RuntimeError("the mux stopped replying"))
        assert _the_only_outcome(project)["status"] == "error"

    def test_a_finalization_failure_does_not_fail_the_run(
            self, monkeypatch, tmp_path):
        """Recording *how* a run ended must not decide *whether* it succeeded."""
        from softae.core.data_store import DataStore
        from softae.tools.commission import EXIT_OK

        def _boom(self, *_a, **_kw):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(DataStore, "finish_run", _boom)
        assert self._run(monkeypatch, tmp_path / "proj") == EXIT_OK

    def test_the_row_is_closed_before_the_store_is(self, monkeypatch, tmp_path):
        """The finalizer and ``store.close()`` share one ``finally``, in order.

        A closed connection can record nothing, so the ordering inside that
        block is the whole of the fix on the failure paths.
        """
        from softae.core.data_store import DataStore

        events: list[str] = []
        real_finish, real_close = DataStore.finish_run, DataStore.close
        monkeypatch.setattr(
            DataStore, "finish_run",
            lambda self, *a, **k: events.append("finish") or real_finish(self, *a, **k))
        monkeypatch.setattr(
            DataStore, "close",
            lambda self, *a, **k: events.append("close") or real_close(self, *a, **k))

        with pytest.raises(RuntimeError):
            self._run(monkeypatch, tmp_path / "proj", RuntimeError("boom"))
        assert events[:2] == ["finish", "close"]


class TestImportRowFinalization:
    """The tool's *second* ``start_run`` site.

    The two are alternatives, not nested: ``import`` opens its own store and
    runs no workflow, so it needs its own finalizer rather than sharing
    ``run``'s.
    """

    def _a_spectrum(self, path: Path) -> Path:
        import numpy as np

        from softae.analysis.eis_data import EISResult

        eis = EISResult(
            channel=1,
            frequency=np.array([1e5, 1e4, 1e3]),
            z_magnitude=np.array([100.0, 110.0, 120.0]),
            phase=np.array([-1.0, -2.0, -3.0]),
            z_real=np.array([100.0, 110.0, 120.0]),
            z_imag_neg=np.array([1.0, 2.0, 3.0]),
        )
        eis.save(path)
        return path

    def _import(self, tmp_path: Path, project: Path):
        from softae.tools.commission import _cmd_import

        src = self._a_spectrum(tmp_path / "spectrum.csv")
        args = build_parser().parse_args(
            ["import", "blank_short", "--file", str(src), "--channel", "1",
             "--electrode-mode", "two", "--project", str(project)])
        return _cmd_import(args)

    def test_a_completed_import_closes_its_row_done(self, tmp_path):
        from softae.tools.commission import EXIT_OK

        project = tmp_path / "proj"
        assert self._import(tmp_path, project) == EXIT_OK
        assert _the_only_outcome(project) == {"status": "done", "finished": True}

    def test_a_failed_import_closes_its_row_error(self, monkeypatch, tmp_path):
        from softae.core.data_store import DataStore

        def _boom(self, *_a, **_kw):
            raise RuntimeError("the measurements table is locked")

        monkeypatch.setattr(DataStore, "record_measurement", _boom)
        project = tmp_path / "proj"
        with pytest.raises(RuntimeError):
            self._import(tmp_path, project)
        assert _the_only_outcome(project)["status"] == "error"
