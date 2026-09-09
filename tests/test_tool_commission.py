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
            ["run", "blank_short", "--channels", "1", "--yes",
             "--electrode-mode", "two"])
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
            ["run", "blank_short", "--channels", "1", "--mock", "--yes",
             "--electrode-mode", "two"])
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

        # --electrode-mode is explicit here so the command reaches the interlock at
        # all: --yes no longer invents it, and a run refused for a MISSING mode would
        # return the same "nothing happened" shape while testing nothing.
        args = build_parser().parse_args(
            ["run", "blank_short", "--channels", "1", "--yes",
             "--electrode-mode", "two"])
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
                                                                   tmp_path, capsys):
        from softae.config import loader
        from softae.tools.commission import EXIT_FAILED, _cmd_run

        monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
        args = build_parser().parse_args(
            ["run", "reference_cap", "--channels", "1", "--mock", "--yes"])
        assert _cmd_run(args) == EXIT_FAILED     # refused before touching hardware
        # ...and refused for the NOMINAL, not for the missing --electrode-mode, which
        # this same invocation now also lacks. Two refusals sharing one exit code is
        # exactly how a check starts passing for the wrong reason.
        assert "--nominal" in capsys.readouterr().err

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
             "--electrode-mode", "two", "--project", str(project)])
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


# -- The declaration names its rows -------------------------------------------

def _capacitive_sweep(C: float, tand: float, n: int = 41):
    """A capacitor of *C* farads with a **constant** loss tangent.

    Constant tan d is the shape of a lossy dielectric and of an instrument residual
    alike -- which is the point: the two are told apart by the *size* of the angle,
    never by its frequency dependence. |Z| still falls at -1 per decade, so
    ``phase_table_gate`` keeps every point and the plausibility judgement is made on
    the values that would actually be tabulated rather than on gate leftovers.
    """
    import numpy as np

    f = np.logspace(np.log10(200_000.0), np.log10(4.0), n)
    Xc = -1.0 / (2 * np.pi * f * C)
    return f, (abs(Xc) * tand) + 1j * Xc


def _cap_acq(channel: int, C: float, tand: float, *, nominal: float,
             measurement_id: int | None = None):
    from softae.workflows.commissioning import AcquiredSpectrum

    f, Z = _capacitive_sweep(C, tand)
    return AcquiredSpectrum(channel, f, Z, nominal=nominal, electrode_mode="two",
                            measurement_id=measurement_id)


class _CommissioningStore:
    """A real store holding commissioning rows, so the UPDATE is exercised for real.

    A fake connection would let a scoped ``UPDATE`` "work" while its ``WHERE`` clause
    was nonsense -- the check would then be about the fake, not about SQLite
    (SUBAGENT_RULES.md 3.1d).
    """

    def __init__(self, project: Path):
        import numpy as np

        from softae.analysis.eis_data import EISResult
        from softae.core.data_store import DataStore

        self.project = project
        self.store = DataStore(project)
        self.run_id = self.store.start_run("commission_fixture", mode="commissioning")
        self._np = np
        self._EISResult = EISResult

    def add(self, role: str, channel: int, *, mode: str = "unknown",
            nominal: float | None = None) -> int:
        np = self._np
        f = np.logspace(5, 1, 9)
        eis = self._EISResult(
            channel=channel, frequency=f,
            z_magnitude=np.full(f.size, 10.0), phase=np.full(f.size, -1.0),
            z_real=np.full(f.size, 10.0), z_imag_neg=np.full(f.size, 0.5),
        )
        dest = self.project / "eis" / f"{role}_ch{channel}_{id(eis) % 9973}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        eis.save(dest)
        eis.raw_file_path = str(dest)
        # Recorded through the real API at 'two', then rewound to the historical
        # mode. See `_rewind_to_a_pre_check_mode`.
        mid = self.store.record_measurement(
            self.run_id, eis, role=role, fixture_id="mux16",
            nominal_value=nominal, electrode_mode="two")
        if mode != "two":
            self._rewind_to_a_pre_check_mode(mid, mode)
        return mid

    def _rewind_to_a_pre_check_mode(self, measurement_id: int, mode: str) -> None:
        """Put back the mode a row of this age actually carries, behind the API.

        DELIBERATE: ``record_measurement`` now REFUSES any ``TWO_TERMINAL_ROLES`` row
        whose mode is not ``'two'`` (R24/F17), so this state is unreachable through
        the public write path -- by design, and that refusal must not be weakened.

        It is nonetheless the state under test. The rows ``--declare-ids`` exists for
        -- mux16's 1931/1932 (ch32 short and load) and 1933 (ch31 reference cap) --
        carry an unknown mode *because they predate the check*. They were written when
        the boundary did not exist and can never be written again. A fixture confined
        to what today's API permits could only build rows that need no declaration at
        all, and the feature's whole subject would go untested: green tests asserting
        nothing (SUBAGENT_RULES.md 3.2 -- a sound check no data ever reaches).

        So the fixture reproduces history the way history produced it: the same write
        path, minus the guard that did not yet exist. Hence record-then-UPDATE rather
        than a hand-written INSERT -- the row is exactly what ``record_measurement``
        builds (column set, path relativisation, run linkage), with only the one
        column the new boundary governs rewound. An INSERT here would be a second,
        drifting copy of that column list, and these tests would then be about the
        fixture's SQL instead of about the store's.
        """
        self.store._conn.execute(
            "UPDATE measurements SET electrode_mode = ? WHERE measurement_id = ?",
            (mode, int(measurement_id)))
        self.store._conn.commit()

    def modes(self) -> dict[int, str]:
        return {int(mid): str(mode) for mid, mode in self.store._conn.execute(
            "SELECT measurement_id, electrode_mode FROM measurements "
            "WHERE role != 'sample'")}

    def artifacts(self) -> dict:
        from softae.tools.commission import _load_role_spectra

        return _load_role_spectra(self.store, "mux16")[0]

    def close(self):
        self.store.close()


@pytest.fixture()
def commissioning_store(tmp_path):
    """The mux16 shape the scoped declaration exists for.

    Three unknown-mode rows, and the operator can vouch for two of them: the ch32
    short and load blanks were sensed with the jumper fitted, the ch31 reference
    capacitor was not necessarily. Declaring all three is what the unscoped flag does.
    """
    store = _CommissioningStore(tmp_path / "proj")
    ids = {
        "short": store.add("blank_short", 32),
        "load": store.add("blank_load", 32),
        "cap": store.add("reference_cap", 31, nominal=1e-10),
    }
    try:
        yield store, ids
    finally:
        store.close()


class TestScopedElectrodeModeDeclaration:
    """``--declare-electrode-mode`` used to have no scope but the role.

    The UPDATE was keyed on ``role`` alone, so declaring the two ch32 blanks
    necessarily declared the ch31 reference capacitor too -- a different assertion
    about a different part, made on the operator's behalf and recorded as their
    provenance. The role is a *kind of artifact*, never a scope.
    """

    def test_naming_two_ids_leaves_the_third_row_unknown(self, commissioning_store):
        from softae.tools.commission import _declare_electrode_mode

        store, ids = commissioning_store
        _declare_electrode_mode(
            store.store, "mux16", "two", store.artifacts(),
            measurement_ids=[ids["short"], ids["load"]])

        modes = store.modes()
        assert modes[ids["short"]] == "two"
        assert modes[ids["load"]] == "two"
        assert modes[ids["cap"]] == "unknown", (
            "the row the operator did not vouch for must not be declared")

    def test_the_unscoped_sweep_still_declares_every_unknown_row(
            self, commissioning_store):
        # The pre-existing behaviour is reachable and unchanged: it is the right shape
        # when a whole pre-jumper session is being declared at once.
        from softae.tools.commission import _declare_electrode_mode

        store, ids = commissioning_store
        _declare_electrode_mode(store.store, "mux16", "two", store.artifacts())
        assert set(store.modes().values()) == {"two"}

    def test_a_channel_filter_would_not_have_been_enough(self, commissioning_store):
        # ch32 separates the two rows to declare from the one to leave alone TODAY.
        # A second ch32 unknown row -- a later import, a re-run -- is swept in by a
        # channel filter and not by an id list, which is the whole reason the scope is
        # by id.
        from softae.tools.commission import _declare_electrode_mode

        store, ids = commissioning_store
        latecomer = store.add("blank_short", 32)

        _declare_electrode_mode(
            store.store, "mux16", "two", store.artifacts(),
            measurement_ids=[ids["short"], ids["load"]])

        modes = store.modes()
        assert modes[latecomer] == "unknown"
        assert modes[ids["short"]] == "two"

    def test_an_id_that_names_no_unknown_row_is_reported(self, commissioning_store,
                                                         capsys):
        # Silently declaring nothing for a mistyped id is a refusal wearing a pass's
        # clothes: the operator walks away believing the row was declared.
        from softae.tools.commission import _declare_electrode_mode

        store, ids = commissioning_store
        _declare_electrode_mode(store.store, "mux16", "two", store.artifacts(),
                                measurement_ids=[ids["short"], 999999])

        out = capsys.readouterr().out
        assert "999999" in out
        assert store.modes()[ids["short"]] == "two"

    def test_an_already_declared_id_is_reported_rather_than_overridden(
            self, commissioning_store, capsys):
        # An explicitly recorded mode is information, not an absence -- F17 keeps it.
        from softae.tools.commission import _declare_electrode_mode

        store, ids = commissioning_store
        explicit = store.add("blank_short", 30, mode="three")
        _declare_electrode_mode(store.store, "mux16", "two", store.artifacts(),
                                measurement_ids=[explicit])

        assert store.modes()[explicit] == "three"
        # The id must appear in the *unmatched* report, not merely somewhere in the
        # output: measurement ids in this fixture are single digits, so a bare
        # `str(explicit) in out` is satisfied by any row count that happens to share
        # the digit -- a pass bought by coincidence (SUBAGENT_RULES.md 3.1).
        assert f"id(s) {explicit}" in capsys.readouterr().out

    def test_the_log_records_what_was_scoped(self, commissioning_store, monkeypatch):
        # A declaration naming two rows and one that swept three are different
        # assertions. The provenance log is the only place that difference survives.
        import softae.tools.commission as mod
        from softae.tools.commission import _declare_electrode_mode

        events: list[tuple] = []
        monkeypatch.setattr(mod.logger, "warning",
                            lambda ev, **kw: events.append((ev, kw)))

        store, ids = commissioning_store
        _declare_electrode_mode(store.store, "mux16", "two", store.artifacts(),
                                measurement_ids=[ids["short"], ids["load"]])
        scoped = [kw for ev, kw in events
                  if ev == "commissioning_electrode_mode_declared"]
        assert len(scoped) == 1
        assert scoped[0]["scope"] == "measurement_ids"
        assert scoped[0]["declared_ids"] == sorted([ids["short"], ids["load"]])
        assert "operator assertion" in scoped[0]["msg"]

        events.clear()
        _declare_electrode_mode(store.store, "mux16", "two", store.artifacts())
        swept = [kw for ev, kw in events
                 if ev == "commissioning_electrode_mode_declared"]
        assert swept and swept[0]["scope"] == "all_unknown"
        assert swept[0]["declared_ids"] is None


class TestDeclareIdsArgumentSurface:
    def test_the_flag_parses_the_same_list_syntax_as_channels(self):
        args = build_parser().parse_args(
            ["derive", "--declare-electrode-mode", "two", "--declare-ids", "1931-1932"])
        assert _parse_channels(args.declare_ids) == [1931, 1932]

    def test_a_scope_without_a_mode_is_refused_rather_than_ignored(self, capsys):
        # Dropping the scope would declare EVERY unknown row of every role -- the
        # opposite of what was typed, and invisible until the next derive.
        from softae.tools.commission import EXIT_FAILED, _cmd_derive

        args = build_parser().parse_args(["derive", "--declare-ids", "1931,1932"])
        assert _cmd_derive(args) == EXIT_FAILED
        assert "--declare-ids" in capsys.readouterr().err

    def test_derive_without_the_flag_is_unchanged(self):
        assert build_parser().parse_args(["derive"]).declare_ids is None


class TestYesCannotInventTheElectrodeMode:
    """``--yes`` skips a *prompt*; it cannot answer a question never asked.

    The jumper question lived entirely inside the interactive branch, so ``--yes`` fell
    through to ``electrode_mode = "two"``: a positive claim about how the bench was
    wired, recorded as the operator's provenance, that nobody made. It is worse than an
    honest ``'unknown'`` -- an unknown is refused downstream and can still be rescued by
    a declaration, whereas a fabricated ``'two'`` passes every check there is and is
    indistinguishable from an operator who really did fit the jumper.

    Recording ``'unknown'`` instead is no longer on the table: ``record_measurement``
    refuses any two-terminal role whose mode is not ``'two'``. So the only two correct
    outcomes are a refusal or an EXPLICIT assertion, and ``--electrode-mode two`` is
    already that assertion -- ``import`` demands it (``required=True``) for the same
    fact, so no second flag is invented for it.
    """

    def _args(self, *extra):
        return build_parser().parse_args(
            ["run", "blank_short", "--channels", "1", *extra])

    def test_yes_without_an_explicit_mode_is_refused_before_the_instruments(
            self, monkeypatch, capsys):
        import softae.drivers.factory as factory
        from softae.tools.commission import EXIT_FAILED, _cmd_run

        opened: list = []
        monkeypatch.setattr(factory, "create_manager",
                            lambda **kw: opened.append(kw))

        assert _cmd_run(self._args("--yes")) == EXIT_FAILED
        err = capsys.readouterr().err
        assert "--electrode-mode two" in err
        # The refusal is worthless if it lands after the rig is open: this command's
        # next act is a real sweep.
        assert opened == [], "refused only after opening the instruments"

    def test_an_explicit_two_still_runs_under_yes(self, monkeypatch, tmp_path):
        # NEGATIVE CONTROL. Without it, a guard that refused *every* --yes run would
        # pass the test above and quietly remove the headless path altogether.
        import softae.drivers.factory as factory
        from softae.config import loader
        from softae.tools.commission import _cmd_run

        seen: dict = {}

        def _spy(*, mock=None, config=None):
            seen["mock"] = mock
            raise RuntimeError("far enough")

        monkeypatch.setattr(factory, "create_manager", _spy)
        monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
        _cmd_run(self._args("--yes", "--electrode-mode", "two"))
        assert seen["mock"] is False, "an explicit assertion must reach the hardware"

    def test_three_is_refused_for_a_two_terminal_role_before_any_prompt(
            self, monkeypatch, capsys):
        # Not merely bad practice: the store REFUSES the row, so the sweep would drive
        # the rig and then discard the spectrum. No --yes here -- the operator must not
        # be asked to confirm hardware for a run that cannot be recorded either way.
        import softae.drivers.factory as factory
        from softae.tools.commission import EXIT_FAILED, _cmd_run

        monkeypatch.setattr("builtins.input",
                            lambda *_a: pytest.fail("prompted for a doomed run"))
        monkeypatch.setattr(factory, "create_manager",
                            lambda **_kw: pytest.fail("opened the instruments"))

        assert _cmd_run(self._args("--electrode-mode", "three")) == EXIT_FAILED
        assert "uncalibratable" in capsys.readouterr().err

    def test_every_role_run_accepts_is_one_the_guard_governs(self):
        # The guard is only worth anything if real invocations reach it (SUBAGENT_RULES
        # 3.2). Every role `run` accepts is two-terminal, so `needs_two` is true for all
        # of them -- and the CLI refusal is the same predicate the write boundary
        # enforces, not a house rule free to drift away from it.
        from softae.analysis.eis.calibration import (
            COMMISSIONING_ROLES,
            electrode_mode_ok,
        )

        for role in COMMISSIONING_ROLES:
            assert not electrode_mode_ok(role, "unknown")[0], role
            assert not electrode_mode_ok(role, "three")[0], role
            assert electrode_mode_ok(role, "two")[0], role


# -- A lossy part is not the instrument's phase floor --------------------------

class TestReferenceCapPhaseGate:
    """``derive_calibration`` fed the reference capacitor's phase table in ungated.

    mux16 id 1933 is marked 100 pF, measures 26.22 nF, and contributes eps of
    58.88/31.05/29.27 deg -- tan d between 0.56 and 1.65. Admitting it replaces the
    system-wide phase floor (headline 6.120 deg) with that part's loss tangent.

    **The gate is on phase, not on the marking**, and these pin both halves of that.
    """

    def _derive(self, artifacts):
        from softae.workflows.commissioning import derive_calibration

        return derive_calibration(artifacts, fixture_id="mux16")

    def _id_3493(self, channel: int = 25):
        """Mismarked and perfectly usable: 1 nF marked, 119.4 pF measured, eps 2.65."""
        return _cap_acq(channel, 119.4e-12, 0.0463, nominal=1e-9, measurement_id=3493)

    def _id_1933(self, channel: int = 31):
        """Implausible as a phase reference: eps ~31 deg, tan d = 0.60."""
        return _cap_acq(channel, 26.22e-9, 0.602, nominal=1e-10, measurement_id=1933)

    def test_a_mismarked_but_low_loss_part_still_contributes(self):
        # THE REGRESSION PIN. 3493 is 0.119x its marking and fires the existing
        # mismatch warning; its eps values are in family with the correctly-marked
        # parts and are already in the live table without harm. A gate keyed on the
        # marking ratio would drop it and change a good phase table.
        cal = self._derive({"reference_cap": [self._id_3493()]})
        assert not cal.phase_acc.is_empty
        assert cal.phase_acc.load == "capacitive"
        assert max(cal.phase_acc.eps_deg) < 15.0

    def test_the_mismarked_part_really_does_fire_the_marking_warning(self, monkeypatch):
        # Without this the test above proves only that *some* capacitor contributes,
        # not that a MISMARKED one does -- and the marking-ratio gate it exists to
        # forbid would still pass it.
        import softae.analysis.eis.calibration_derive as cd

        fired: list = []
        real = cd.logger.warning

        def spy(ev, **kw):
            fired.append(ev)
            return real(ev, **kw)

        monkeypatch.setattr(cd.logger, "warning", spy)
        self._derive({"reference_cap": [self._id_3493()]})
        assert "eis_reference_cap_mismatch" in fired

    def test_a_lossy_part_contributes_nothing_to_the_phase_table(self):
        cal = self._derive({"reference_cap": [self._id_1933()]})
        assert cal.phase_acc.is_empty, (
            "a lossy dielectric's tan d must not become the instrument phase floor")
        assert cal.z_min_ohm != cal.z_min_ohm      # NaN: z_points is the only source
        assert not cal.capabilities().phase_floor_measured

    def test_the_ladder_still_reports_the_phase_floor_as_unmeasured(self):
        # The artifact-level surface of the refusal: the set does not claim a floor it
        # does not have, and the capability ladder still asks for the part.
        cal = self._derive({"reference_cap": [self._id_1933()]})
        caps = cal.capabilities()
        assert not caps.phase_floor_measured
        assert "reference capacitor" in caps.blocked["qualified upper bounds"]

    def test_one_bad_part_does_not_take_the_good_one_with_it(self):
        cal = self._derive({"reference_cap": [self._id_3493(), self._id_1933()]})
        assert not cal.phase_acc.is_empty
        assert max(cal.phase_acc.eps_deg) < 15.0

    def test_the_exclusion_is_logged_with_what_it_refused(self, monkeypatch):
        # Visible, not silent: a dropped artifact the operator never hears about is
        # indistinguishable from one that was never run.
        import softae.workflows.commissioning as mod

        events: list[tuple] = []
        monkeypatch.setattr(mod.logger, "warning",
                            lambda ev, **kw: events.append((ev, kw)))
        self._derive({"reference_cap": [self._id_1933()]})

        by_event = dict(events)
        assert "commissioning_reference_cap_phase_implausible" in by_event
        detail = by_event["commissioning_reference_cap_phase_implausible"]
        assert detail["channel"] == 31
        assert detail["measurement_id"] == 1933
        assert detail["limit_deg"] == 15.0
        assert min(detail["eps_deg"]) > 15.0
        assert "commissioning_no_usable_phase_reference" in by_event

    def test_a_pass_with_a_survivor_does_not_claim_the_floor_is_unmeasured(
            self, monkeypatch):
        import softae.workflows.commissioning as mod

        events: list[str] = []
        monkeypatch.setattr(mod.logger, "warning",
                            lambda ev, **kw: events.append(ev))
        self._derive({"reference_cap": [self._id_3493(), self._id_1933()]})
        assert "commissioning_no_usable_phase_reference" not in events


class TestPhaseReferencePlausibility:
    """The judgement itself, on the four real per-decade eps families."""

    #: The values the live mux16 record actually produced, per measurement id.
    REAL = {
        3491: [3.85, 2.20, 1.03, 1.12, 1.34],
        3492: [3.85, 2.21, 0.91, 1.71, 4.54],
        3493: [4.35, 1.96, 2.67, 2.63, 1.42, 3.63],
        1933: [58.88, 31.05, 29.27],
    }

    def test_the_three_in_family_parts_are_admitted(self):
        from softae.workflows.commissioning import phase_reference_is_plausible

        for mid in (3491, 3492, 3493):
            ok, why = phase_reference_is_plausible(self.REAL[mid])
            assert ok, f"id {mid} would have been dropped: {why}"

    def test_the_lossy_part_is_refused_with_its_reason(self):
        from softae.workflows.commissioning import phase_reference_is_plausible

        ok, why = phase_reference_is_plausible(self.REAL[1933])
        assert not ok
        assert "31.05" in why and "lossy dielectric" in why

    def test_the_threshold_sits_between_the_two_families_not_beside_one(self):
        # 3.3x above the worst admitted per-decade point and 2x below the best
        # refused decade. A threshold hugging either family is one noisy sweep from
        # changing its own answer.
        from softae.workflows.commissioning import PHASE_REFERENCE_MAX_EPS_DEG

        worst_admitted = max(max(self.REAL[m]) for m in (3491, 3492, 3493))
        best_refused = min(self.REAL[1933])
        assert worst_admitted * 3 < PHASE_REFERENCE_MAX_EPS_DEG < best_refused / 1.9

    def test_nothing_to_judge_is_not_a_refusal(self):
        # An empty table means the point gate dropped everything, which is already
        # reported there. Refusing again would file one absence as two.
        from softae.workflows.commissioning import phase_reference_is_plausible

        assert phase_reference_is_plausible([])[0]
        assert phase_reference_is_plausible([float("nan")])[0]

    def test_the_median_not_the_max_decides(self):
        from softae.workflows.commissioning import phase_reference_is_plausible

        # One noisy decade must not discard a part whose sweep is otherwise clean...
        assert phase_reference_is_plausible([1.0, 2.0, 3.0, 60.0])[0]
        # ...and one lucky decade must not admit a part that is lossy everywhere.
        assert not phase_reference_is_plausible([0.5, 40.0, 42.0, 44.0])[0]
