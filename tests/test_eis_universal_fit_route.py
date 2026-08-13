"""P.20 — one configurable analysis route for every FIT site, and the end of ``z_to_sigma``.

P.16 migrated six σ sites and shipped an AST guard as proof of completeness. Both were
incomplete: the guard matched the literal name ``z_to_sigma`` and scanned one directory,
so it saw neither ``tab_analysis.py``'s ``fr.sigma(L, t, w)`` nor any of the seven places
that called ``fit_circuit`` directly and therefore ignored ``[eis] engine`` entirely.

Three things are pinned here.

1. **Every fit site routes through** :func:`~softae.analysis.eis.engine.analyze_spectrum`
   **with ``engine`` unset**, so ``[eis] engine`` is the whole cutover. Hardcoding
   ``"legacy"`` at a call site would pass a naive "did you migrate?" check while defeating
   the point of migrating.
2. **A replacement completeness guard** that scans all of ``src/softae/`` plus ``scripts/``
   and matches four shapes rather than one name. Its allowlist is small and every entry
   carries a reason, because an allowlist that grows to silence a noisy test is exactly how
   the first guard died.
3. **``z_to_sigma`` and ``FitResult.sigma`` warn on call and have no production callers** —
   but are *not* deleted and *not* reimplemented in terms of ``CellConstant``. They are the
   independent parity oracle; collapsing them into the survivor would turn
   ``cell.sigma(R) == z_to_sigma(L, t, w, R)`` into ``x == x``.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from softae.analysis.circuit_fitting import FitResult
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis.report import SigmaReport, SpectrumReport
from softae.analysis.eis.settings import eis_settings
from softae.analysis.eis_data import EISResult

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "softae"
SCRIPTS = REPO / "scripts"


# ── The sanctioned set ───────────────────────────────────────────────────────
#
# Modules permitted to spell σ arithmetic or call ``fit_circuit`` directly. Each entry
# states *why*; ``test_the_sanctioned_allowlist_cannot_grow_without_a_documented_reason``
# asserts the reasons exist and that the set is exactly this one.

SANCTIONED: dict[str, str] = {
    "analysis/circuit_fitting.py":
        "defines fit_circuit, z_to_sigma and FitResult.sigma — the legacy fitter the "
        "engine calls verbatim, and the deprecated parity oracle kept beside it",
    "analysis/eis/geometry.py":
        "defines CellConstant.sigma, the one σ implementation intended to survive",
    "analysis/eis/engine.py":
        "analyze_spectrum IS the route: _legacy_report calls fit_circuit and cell.sigma "
        "deliberately, and the gated path calls fit_circuit as its fallback fitter",
    "gui/eis_sigma.py":
        "the GUI's single CellConstant builder; cell_sigma is the shared arithmetic every "
        "surface that holds a bare R1 divides through",
}

#: Sites Stage A did not migrate, because afl-session HELD them under mail [a18].
#: **Empty, and it must stay empty.** They were released in [a22] and Stage B landed
#: both: ``analysis/eis/router.py`` (site 9, the workflow auto-fit) now calls
#: ``analyze_spectrum`` with ``engine`` unset, and ``core/data_store.py`` (site 10,
#: ``record_fit``'s inline ``L/(R1·t·w)``) now takes σ from
#: ``CellConstant.from_legacy(...).sigma(R1)``. The entries asserted the two files
#: *still offended* precisely so that landing Stage B would fail this file rather than
#: leave a stale exemption standing over migrated code — which it did.
#:
#: Both are now pinned positively instead, by
#: ``test_every_fit_site_routes_through_analyze_spectrum_with_the_engine_left_unset``
#: and ``test_the_persistence_layers_sigma_is_the_cell_constants_not_a_third_spelling``.
STAGE_B_HELD: dict[str, str] = {}

#: Modules permitted to name an engine at a call site. **Empty, and it must stay empty.**
#: The user ruled in mail [a23]: "The GUI and objective should report the same
#: conductivity, as nothing changes about the casting nor measurement between them."
#: ``core/autonomous_wiring.py`` was the last holdout with its ``engine="gated"``
#: hardcode; afl-session retired it under T2.6b on 2026-08-09. Anything added back here
#: is a second place that decides which physics runs.
ENGINE_DECISION_EXEMPT: dict[str, str] = {}

#: Every migrated fit site — Stage A's six, plus Stage B's ``router.py`` (site 9, the
#: workflow auto-fit, the origin of nearly every stored ``fit_results`` row). All must
#: name ``analyze_spectrum``.
MIGRATED_FIT_SITES = (
    "analysis/eis/router.py",
    "gui/tabs/tab_analysis.py",
    "gui/tabs/tab_experiment.py",
    "gui/tabs/tab_manual.py",
    "gui/widgets/eis_visualizer_widget.py",
    "web/data_adapter.py",
    "workflows/temp_eis_sweep.py",
)


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.py")) + sorted(SCRIPTS.rglob("*.py"))


def _rel(path: Path) -> str:
    try:
        return path.relative_to(SRC).as_posix()
    except ValueError:
        return "scripts/" + path.relative_to(SCRIPTS).as_posix()


def _trees() -> list[tuple[str, ast.Module]]:
    # utf-8-sig: at least one module is BOM-prefixed, which ast.parse refuses.
    return [(_rel(p), ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p)))
            for p in _sources()]


def _tree(rel: str) -> ast.Module:
    root = SCRIPTS.parent if rel.startswith("scripts/") else SRC
    path = root / rel
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _func(tree: ast.Module, name: str) -> ast.AST:
    return next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == name)


# ── The four off-route shapes ────────────────────────────────────────────────


def _leaf(node: ast.AST) -> str | None:
    """The identifier at the end of a name or attribute chain, else ``None``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _normalise(name: str) -> str:
    """``L_cm`` → ``l``, ``R1`` → ``r1``, ``t_cm`` → ``t``. Unit suffixes only."""
    lowered = name.lower()
    for suffix in ("_cm", "_ohm"):
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)]
    return lowered


def _mult_leaves(node: ast.AST) -> list[str] | None:
    """Flatten ``a * b * c`` to its leaf identifiers; ``None`` if it is anything else.

    Deliberately narrow. A call, a subscript or a literal anywhere in the chain aborts
    the match, because the shape being hunted is the hand-spelled ``R·t·w`` product and
    nothing else. Narrowing beats extending the allowlist: a false positive silenced by
    an exemption is how the P.16 guard stopped meaning anything.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = _mult_leaves(node.left), _mult_leaves(node.right)
        return None if left is None or right is None else left + right
    name = _leaf(node)
    return [name] if name is not None else None


def _is_raw_sigma_arithmetic(node: ast.AST) -> bool:
    """``L / (R · t · w)`` spelled out by hand, in any of its unit-suffixed forms."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return False
    numerator = _leaf(node.left)
    denominator = _mult_leaves(node.right)
    if numerator is None or denominator is None or len(denominator) != 3:
        return False
    return (_normalise(numerator) == "l"
            and {_normalise(d) for d in denominator} in ({"r", "t", "w"},
                                                         {"r1", "t", "w"}))


def _is_cell_constant_construction(node: ast.AST) -> bool:
    """True for ``CellConstant(...)`` / ``CellConstant.from_legacy(...)`` and nothing else.

    Deliberately structural rather than by variable name. ``cell.sigma(R)`` reads as
    obviously sound and is exactly the shape a stale ``FitResult`` named ``cell`` would
    also have, so a name-based exemption would admit the very thing site 1 was: σ
    computed off a bare R by a method that merely *sounds* like the route. A receiver
    that is the constructor call itself cannot be anything else.
    """
    if not isinstance(node, ast.Call):
        return False
    return _leaf(node.func) in ("CellConstant", "from_legacy")


def _offences(rel: str, tree: ast.Module) -> list[str]:
    """Every off-route σ or fit this module *spells*, with a shape label."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in ("z_to_sigma", "fit_circuit"):
                    found.append(f"{rel}:{node.lineno} imports {alias.name}")
        elif isinstance(node, ast.Name) and node.id in ("z_to_sigma", "fit_circuit"):
            found.append(f"{rel}:{node.lineno} references {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in ("z_to_sigma",
                                                               "fit_circuit"):
            found.append(f"{rel}:{node.lineno} references .{node.attr}")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "sigma"
              and not _is_cell_constant_construction(node.func.value)):
            found.append(f"{rel}:{node.lineno} calls .sigma(...) off-route")
        elif _is_raw_sigma_arithmetic(node):
            found.append(f"{rel}:{node.lineno} spells L/(R·t·w) by hand")
    return found


# ── The completeness guard ───────────────────────────────────────────────────


def test_no_module_outside_the_sanctioned_set_computes_sigma_or_fits_off_route():
    """Four shapes, all of ``src/softae`` and ``scripts/``, one small allowlist.

    The predecessor matched ``ast.Name(id="z_to_sigma")`` and scanned ``src/softae/gui``
    only, so a method call named ``.sigma(`` and every ``fit_circuit`` outside the GUI
    were both invisible to it.

    **What this cannot prove**, so that nobody over-reads it: it is source inspection.
    A σ built in a dynamically-composed expression, reached through ``getattr``, or
    spread across two statements is beyond it. It proves no module *spells* an off-route
    σ — the behavioural claim is carried by the ``analyze_spectrum`` spy tests below.
    """
    offenders = [
        offence
        for rel, tree in _trees()
        if rel not in SANCTIONED and rel not in STAGE_B_HELD
        for offence in _offences(rel, tree)
    ]
    assert offenders == []


def test_the_sanctioned_allowlist_cannot_grow_without_a_documented_reason():
    """Exactly four sanctioned modules, each with a real reason, and nothing pending."""
    assert set(SANCTIONED) == {
        "analysis/circuit_fitting.py",
        "analysis/eis/geometry.py",
        "analysis/eis/engine.py",
        "gui/eis_sigma.py",
    }

    for rel, reason in {**SANCTIONED, **STAGE_B_HELD, **ENGINE_DECISION_EXEMPT}.items():
        assert (SRC / rel).exists(), f"{rel} is exempted but does not exist"
        assert len(reason.split()) >= 8, f"{rel}'s reason is not a reason: {reason!r}"

    # Stage B landed both held files, so the guard now scans the whole tree with no
    # exemptions at all. A file re-entered here would be a migrated site being excused
    # a second time, which is how the P.16 guard stopped meaning anything.
    assert STAGE_B_HELD == {}

    # [a23] is a user ruling, not a preference: no surface may name its own engine.
    assert ENGINE_DECISION_EXEMPT == {}


# ── Every fit site is config-governed ────────────────────────────────────────


def test_every_fit_site_routes_through_analyze_spectrum_with_the_engine_left_unset():
    """Migrating a site and then naming its engine would defeat the whole exercise.

    ``[a23]`` (user ruling, 2026-08-09) makes this binding rather than stylistic: "the
    GUI and objective should report the same conductivity, as nothing changes about the
    casting nor measurement between them." One resolver, no per-surface opinions.
    """
    for rel in MIGRATED_FIT_SITES:
        tree = _tree(rel)
        assert any(isinstance(n, ast.Call)
                   and getattr(n.func, "id", getattr(n.func, "attr", None))
                   == "analyze_spectrum"
                   for n in ast.walk(tree)), f"{rel} does not call analyze_spectrum"

    hardcoded = [
        f"{rel}:{node.lineno}"
        for rel, tree in _trees()
        if rel not in ENGINE_DECISION_EXEMPT
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        == "analyze_spectrum"
        and any(kw.arg == "engine" for kw in node.keywords)
    ]
    assert hardcoded == []


def test_analyze_spectrum_is_the_single_engine_resolution_point():
    """The one place ``[eis] engine`` is read. [a23]: one σ everywhere, one resolver.

    Every Stage A site leaves ``engine`` unset precisely so this is the only function
    that decides. Anything that resolves the engine somewhere else — a second default,
    a per-surface config read — reintroduces the divergence the ruling closed.
    """
    import inspect

    from softae.analysis.eis.engine import analyze_spectrum

    assert inspect.signature(analyze_spectrum).parameters["engine"].default is None

    resolvers = [rel for rel, tree in _trees()
                 if rel != "analysis/eis/settings.py"
                 for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == "engine"
                 and isinstance(n.value, ast.Name) and n.value.id in ("cfg", "settings")]
    assert resolvers == ["analysis/eis/engine.py"]


def test_the_legacy_report_still_passes_the_model_name_positionally_so_existing_patches_hold():
    """``_legacy_report`` must keep ``fit_circuit(eis, model_name)`` positional.

    ``tests/test_tab_experiment.py`` asserts ``fake.call_args.args[1]``. Switching to a
    keyword breaks six tests at once with an ``IndexError`` and no useful message.
    """
    call = next(
        n for n in ast.walk(_func(_tree("analysis/eis/engine.py"), "_legacy_report"))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "fit_circuit"
    )
    assert len(call.args) == 2 and not call.keywords

    # …and the import stays inside the function body. Hoisting it to module scope would
    # convert late binding to early binding and silently kill ~8 module-scope patches.
    body = _func(_tree("analysis/eis/engine.py"), "_legacy_report")
    assert any(isinstance(n, ast.ImportFrom)
               and n.module == "softae.analysis.circuit_fitting"
               and any(a.name == "fit_circuit" for a in n.names)
               for n in ast.walk(body))


# ── Behaviour: the migrated sites really do go through the engine ────────────


def _eis(r0: float = 100.0, r1: float = 5.0e4) -> EISResult:
    f = np.logspace(1, 5, 30)
    tau = 2 * np.pi * f * 1e-6
    return EISResult.from_arrays(
        channel=1, f=f, z_real=r0 + r1 / (1 + tau**2),
        z_imag_neg=np.abs(r1 * tau / (1 + tau**2)),
    )


def _fit(R1: float = 5.0e4, R0: float = 100.0, success: bool = True) -> FitResult:
    return FitResult(model_name="simpleSalt", parameters=np.array([R0, R1]),
                     R0=R0, R1=R1, R0_guess=R0, R1_guess=R1, z_indices=[0, 1],
                     success=success)


@pytest.fixture()
def legacy_settings():
    """Pin the engine in the test rather than inheriting it from the config file.

    These tests mean "the legacy path"; a flip of ``[eis] engine`` would otherwise send
    them down the fitter/gates path, where a patched ``fit_circuit`` is called with
    different surroundings.
    """
    return eis_settings({"engine": "legacy"})


def test_a_migrated_fit_site_still_honours_a_module_scope_patch_of_fit_circuit(
        monkeypatch, legacy_settings):
    """Routing behind one entry point usually converts late binding to early binding.

    It does not here, and the reason is load-bearing: ``_legacy_report`` imports
    ``fit_circuit`` *inside the function body*, so the module attribute is resolved per
    call and the ~8 existing patches keep landing.
    """
    calls = []

    def fake(eis_result, model_name="simpleSalt", **kw):
        calls.append(model_name)
        return _fit()

    monkeypatch.setattr("softae.analysis.circuit_fitting.fit_circuit", fake)
    monkeypatch.setattr("softae.analysis.eis.engine.eis_settings",
                        lambda *a, **k: legacy_settings)

    from softae.analysis.eis.engine import analyze_spectrum

    report = analyze_spectrum(_eis(), cell=CellConstant.from_legacy(0.2, 0.015, 0.2),
                              model_name="simpleSaltMembrane")
    assert calls == ["simpleSaltMembrane"]
    assert report.fit.R1 == 5.0e4


def test_the_temperature_sweep_sigma_is_bit_comparable_to_the_legacy_value_within_one_ulp(
        monkeypatch, legacy_settings):
    """Sites 5–7 feed *stored* Arrhenius σ, so the association-order shift reaches disk.

    ``z_to_sigma`` evaluates ``L/((R·t)·w)``; ``CellConstant.sigma`` evaluates ``K/R``
    with ``K = L/(t·w)`` precomputed. Same arithmetic, different association order, so
    the last bit can differ — and unlike the GUI's display σ, this one is persisted.
    One ULP is orders below the fit tolerance and below the Arrhenius fitter's
    sensitivity; the point of the test is that it is one ULP and not more.
    """
    from softae.analysis.circuit_fitting import z_to_sigma
    from softae.workflows.temp_eis_sweep import ArrheniusSweep

    monkeypatch.setattr("softae.analysis.eis.engine.eis_settings",
                        lambda *a, **k: legacy_settings)

    geom = {"L_cm": 0.2, "t_cm": 0.175, "w_cm": 0.2}
    sweep = object.__new__(ArrheniusSweep)
    sweep.config = SimpleNamespace(electrode_geometry=geom, eis_model="simpleSalt")

    # Every value stays above the `simpleSalt` R₁ lower bound of 100 Ω. Below it a
    # fit is *railed* and the engine now withholds σ entirely, which is a different
    # test's subject: this one is about association order, and needs σ to exist.
    for R1 in (150.0, 333.0, 1234.5, 5.0e4, 9.87e5):
        monkeypatch.setattr("softae.analysis.circuit_fitting.fit_circuit",
                            lambda eis, model_name, _R=R1: _fit(R1=_R))
        with pytest.warns(DeprecationWarning):
            legacy = float(z_to_sigma(geom["L_cm"], geom["t_cm"], geom["w_cm"], R1))

        routed = sweep._sigma_from_eis(_eis())
        assert abs(routed - legacy) <= math.ulp(legacy)

        # The live-plot callback takes the same route, so the trace on screen and the
        # number that reaches the Arrhenius fit cannot disagree.
        r0, r1, live_sigma = sweep._live_point(_eis())
        assert (r0, r1) == (100.0, R1)
        assert live_sigma == routed


def test_the_temperature_sweep_withholds_sigma_when_the_geometry_is_absent(
        monkeypatch, legacy_settings):
    """No geometry is no conductivity — never one built on the 0.175 cm placeholder."""
    from softae.workflows.temp_eis_sweep import ArrheniusSweep

    monkeypatch.setattr("softae.analysis.eis.engine.eis_settings",
                        lambda *a, **k: legacy_settings)
    monkeypatch.setattr("softae.analysis.circuit_fitting.fit_circuit",
                        lambda eis, model_name: _fit())

    sweep = object.__new__(ArrheniusSweep)
    sweep.config = SimpleNamespace(electrode_geometry=None, eis_model="simpleSalt")
    assert math.isnan(sweep._sigma_from_eis(_eis()))
    assert sweep._cell() is None

    sweep.config = SimpleNamespace(electrode_geometry={"L_cm": 0.2, "t_cm": 0.0,
                                                       "w_cm": 0.2},
                                   eis_model="simpleSalt")
    assert sweep._cell() is None
    assert math.isnan(sweep._sigma_from_eis(_eis()))


def test_the_web_adapter_reports_no_sigma_rather_than_a_nominal_when_geometry_is_absent(
        tmp_path, monkeypatch, legacy_settings):
    """``FileAdapter`` fits through the engine, and withholds σ when it cannot claim one."""
    from softae.web.data_adapter import FileAdapter, _cell_from_geometry

    monkeypatch.setattr("softae.analysis.eis.engine.eis_settings",
                        lambda *a, **k: legacy_settings)
    monkeypatch.setattr("softae.analysis.circuit_fitting.fit_circuit",
                        lambda eis, model_name: _fit(R1=5.0e4))

    path = tmp_path / "ch01_eis.txt"
    _eis().save(path)

    # No geometry at all, an incomplete one, and a degenerate one all yield no cell…
    assert _cell_from_geometry(None) is None
    assert _cell_from_geometry({"L_cm": 0.2, "w_cm": 0.2}) is None
    assert _cell_from_geometry({"L_cm": 0.2, "t_cm": 0.0, "w_cm": 0.2}) is None

    assert FileAdapter([path]).get_entries()[0].sigma is None
    assert FileAdapter([path], electrode_geometry={"L_cm": 0.2, "w_cm": 0.2}
                       ).get_entries()[0].sigma is None

    # …and a complete one yields the cell constant's σ, not a hand-rolled quotient.
    entry = FileAdapter([path], electrode_geometry={"L_cm": 0.2, "t_cm": 0.015,
                                                    "w_cm": 0.2}).get_entries()[0]
    assert entry.sigma == CellConstant.from_legacy(0.2, 0.015, 0.2).sigma(5.0e4)


# ── The Arrhenius params panel (site 1) ──────────────────────────────────────


@pytest.fixture()
def analysis_tab(qapp):
    from softae.drivers.mock_factory import create_mock_manager
    from softae.gui.tabs.tab_analysis import AnalysisTab

    tab = AnalysisTab(create_mock_manager(config={}))
    yield tab
    tab.close()


def _sigma_cell_text(tab) -> str:
    return tab._arrh_params_table.item(0, 1).text()


def test_the_arrhenius_params_panel_shows_the_sigma_the_fit_used_not_a_live_recomputation(
        analysis_tab):
    """The panel reads the worker's cached σ, not ``R1`` divided by the live spins.

    ``_ArrhFitWorker`` builds one cell at launch and the Arrhenius curve is fitted
    against the σ that cell produced. Recomputing on row-click from spin boxes that may
    have moved since put a second number on screen that no fit had ever consumed, with
    nothing indicating which of the two was stale.
    """
    tab = analysis_tab
    tab._arrh_loaded = [_eis()]
    tab._arrh_fit_by_index = {0: _fit(R1=5.0e4)}
    tab._arrh_sigma_by_index = {0: 1.2345e-05}

    tab._arrh_on_row_selected(0)
    assert _sigma_cell_text(tab) == "1.2345e-05"


def test_moving_a_geometry_spin_after_fitting_does_not_change_the_displayed_row_sigma(
        analysis_tab):
    """The one number P.20 deliberately moves — and only when the spins have moved.

    Before this change the panel divided the *stale* ``R1`` by the *live* geometry. An
    operator who nudged a spin box after fitting saw a σ that disagreed with the one the
    Arrhenius fit consumed. After it, the displayed value is fixed at fit time.
    """
    tab = analysis_tab
    tab._arrh_loaded = [_eis()]
    tab._arrh_fit_by_index = {0: _fit(R1=5.0e4)}
    tab._arrh_sigma_by_index = {0: CellConstant.from_legacy(0.2, 0.015, 0.2).sigma(5.0e4)}

    tab._arrh_spin_L.setValue(0.2)
    tab._arrh_spin_t.setValue(0.015)
    tab._arrh_spin_w.setValue(0.2)
    tab._arrh_on_row_selected(0)
    before = _sigma_cell_text(tab)

    tab._arrh_spin_t.setValue(0.03)          # halve the film after the fit
    tab._arrh_on_row_selected(0)
    assert _sigma_cell_text(tab) == before


def test_the_arrhenius_worker_caches_the_sigma_it_fitted_against(monkeypatch, qapp):
    """The cached σ is the report's, so the panel and the curve share one number."""
    from softae.gui.tabs import tab_analysis

    def spy(eis_result, **kwargs):
        assert "engine" not in kwargs                 # config-governed, still
        return SpectrumReport(engine="legacy", fit=_fit(R1=5.0e4),
                              sigma=SigmaReport(mode="value",
                                                value=kwargs["cell"].sigma(5.0e4)))

    monkeypatch.setattr(tab_analysis, "analyze_spectrum", spy)

    captured: dict = {}
    worker = tab_analysis._ArrhFitWorker(
        [(0, _eis(), 25.0, None), (1, _eis(), 45.0, None)],
        "simpleSalt", 0.2, 0.015, 0.2)
    worker.finished.connect(
        lambda fits, sigmas, keyed, rh: captured.update(fits=fits, sigmas=sigmas))
    worker.run()

    expected = CellConstant.from_legacy(0.2, 0.015, 0.2).sigma(5.0e4)
    assert captured["sigmas"] == {0: expected, 1: expected}
    assert set(captured["fits"]) == {0, 1}


# ── The manual tab ───────────────────────────────────────────────────────────


def _manual_stub(*, geom: bool, K: float = 0.0,
                 L: float = 0.2, t: float = 0.015, w: float = 0.2):
    """Just enough of ``ManualControlTab`` for ``_conductivity_from_fit``."""
    return SimpleNamespace(
        _rb_geom=SimpleNamespace(isChecked=lambda: geom),
        _spin_eis_K=SimpleNamespace(value=lambda: K),
        _spin_eis_L=SimpleNamespace(value=lambda: L),
        _spin_eis_t=SimpleNamespace(value=lambda: t),
        _spin_eis_w=SimpleNamespace(value=lambda: w),
    )


def test_the_manual_tabs_series_sigma_comes_from_a_fit_produced_by_analyze_spectrum(
        monkeypatch, qapp):
    """The payload's ``fit_result`` is the engine's, so the series σ inherits its R.

    ``_conductivity_from_fit`` is injected as ``sigma_fn`` into ``EisSeriesPlotWidget``,
    so an R of unsound provenance would spread across the whole series plot. Its
    signature is unchanged: the fix is upstream, at the fit.
    """
    from softae.gui.tabs.tab_manual import ManualControlTab, _ManualEisWorker

    routed = _fit(R1=5.0e4)
    seen: list[dict] = []

    def spy(eis_result, **kwargs):
        seen.append(kwargs)
        return SpectrumReport(engine="legacy", fit=routed,
                              sigma=SigmaReport(mode="unavailable"))

    monkeypatch.setattr("softae.analysis.eis.engine.analyze_spectrum", spy)
    monkeypatch.setattr("softae.drivers.mscr_library.eis_run_mscrbuild",
                        lambda *a, **k: None)
    monkeypatch.setattr(EISResult, "from_raw",
                        classmethod(lambda cls, raw, **kw: _eis()))

    pico = SimpleNamespace(sendscript_getdata=lambda *a: object(), _output_dir=".")
    worker = _ManualEisWorker(
        SimpleNamespace(get=lambda name: pico), None,
        channels=[1], eis_params={}, auto_fit=True,
        fit_model="simpleSalt", auto_save=False,
    )

    payload = worker._measure_one(1, "pico1", None)

    assert payload["fit_result"] is routed
    assert seen and "engine" not in seen[0]          # [eis] engine still governs
    assert seen[0]["model_name"] == "simpleSalt"

    # …and the σ the series plot renders is that fit's R1 through the shared cell.
    sigma = ManualControlTab._conductivity_from_fit(_manual_stub(geom=True),
                                                    payload["fit_result"])
    assert sigma == CellConstant.from_legacy(0.2, 0.015, 0.2).sigma(5.0e4)


def test_the_manual_tabs_empirical_K_branch_is_still_not_a_cell_constant_route():
    """``K / R1`` is an operator-typed cell constant. There is no geometry to migrate.

    Pinned at source as well as behaviourally: the branch must build no ``CellConstant``
    and call no shared σ helper, or a future "unification" would fold a typed-in number
    into a geometry route it never came from.
    """
    from softae.gui.tabs.tab_manual import ManualControlTab

    assert ManualControlTab._conductivity_from_fit(
        _manual_stub(geom=False, K=12.5), _fit(R1=500.0)) == 12.5 / 500.0
    assert ManualControlTab._conductivity_from_fit(
        _manual_stub(geom=False, K=0.0), _fit(R1=500.0)) is None

    fn = _func(_tree("gui/tabs/tab_manual.py"), "_conductivity_from_fit")
    branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If))
    names = {getattr(n.func, "id", getattr(n.func, "attr", None))
             for n in ast.walk(ast.Module(body=branch.orelse, type_ignores=[]))
             if isinstance(n, ast.Call)}
    assert not names & {"gui_cell", "cell_sigma", "CellConstant", "from_legacy"}


# ── The deprecation ──────────────────────────────────────────────────────────


def test_z_to_sigma_warns_on_call_and_has_no_remaining_production_caller():
    """Deprecated, kept, and unused — the three conditions together.

    Kept because it is the *independent* oracle: ``tests/test_eis_geometry.py`` and
    ``tests/test_gui_eis_sigma_source.py`` assert ``cell.sigma(R) == z_to_sigma(...)``,
    and that proof only means something while the two implementations are unrelated.
    """
    from softae.analysis.circuit_fitting import z_to_sigma

    with pytest.warns(DeprecationWarning, match="z_to_sigma is deprecated"):
        assert z_to_sigma(0.2, 0.015, 0.2, 5.0e4) > 0

    callers = [rel for rel, tree in _trees()
               if rel != "analysis/circuit_fitting.py"
               for n in ast.walk(tree)
               if (isinstance(n, ast.Name) and n.id == "z_to_sigma")
               or (isinstance(n, ast.Attribute) and n.attr == "z_to_sigma")]
    assert callers == []


def test_fit_result_sigma_warns_on_call_and_sigma_report_does_not():
    """``sigma_report(cell)`` is the sanctioned bare-``FitResult`` route and stays quiet."""
    import warnings

    fit = _fit(R1=5.0e4)
    with pytest.warns(DeprecationWarning, match="FitResult.sigma is deprecated"):
        assert fit.sigma(0.2, 0.015, 0.2) > 0

    cell = CellConstant.from_legacy(0.2, 0.015, 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        report = fit.sigma_report(cell)
    assert report.mode == "value"
    assert report.value == cell.sigma(5.0e4)


def test_cell_constant_sigma_does_not_call_z_to_sigma_so_the_parity_oracle_stays_independent():
    """Unifying the two would turn ``cell.sigma(R) == z_to_sigma(...)`` into ``x == x``.

    Green forever, cited as evidence, proving nothing — at exactly the moment the gated
    engine starts moving real numbers. The 1-ULP gap is the *measurement* of the
    duplication and is kept deliberately.
    """
    import warnings

    from softae.analysis.circuit_fitting import z_to_sigma

    cell = CellConstant.from_legacy(0.2, 0.015, 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert cell.sigma(5.0e4) > 0                 # would raise if it delegated

    fn = _func(_tree("analysis/eis/geometry.py"), "sigma")
    assert not any(isinstance(n, ast.Name) and n.id == "z_to_sigma"
                   for n in ast.walk(fn))

    # …and they still agree to within one unit in the last place.
    with pytest.warns(DeprecationWarning):
        for L, t, w in ((0.2, 0.175, 0.2), (0.2, 0.015, 0.2), (0.15, 0.002, 0.17)):
            for R in (1.0, 333.0, 5.0e4, 9.87e5):
                legacy = float(z_to_sigma(L, t, w, R))
                assert abs(CellConstant.from_legacy(L, t, w).sigma(R) - legacy) \
                    <= math.ulp(legacy)


def test_the_user_guide_conductivity_example_does_not_teach_the_deprecated_route():
    """A worked example teaches faster than a deprecation retires.

    ``USER_GUIDE.md`` presented ``fit_circuit`` + ``z_to_sigma`` (and ``fit.sigma(...)``)
    as *the* way to obtain conductivity. Left alone it would re-seed the pattern into
    every reader who follows it.
    """
    guide = (REPO / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    block = guide.split("### Circuit Fitting", 1)[1].split("### Available Models", 1)[0]
    code = block.split("```python", 1)[1].split("```", 1)[0]

    assert "z_to_sigma" not in code
    assert "fit.sigma(" not in code
    assert "fit_circuit(" not in code
    assert "analyze_spectrum(" in code
    assert "report.sigma.mode" in code                # the unavailable case is taught
    assert "engine" in code                           # …and so is who chooses it


# ── Stage B: the two files afl-session held ──────────────────────────────────


@pytest.fixture()
def store(tmp_path):
    from softae.core.data_store import DataStore

    ds = DataStore(tmp_path / "project")
    yield ds
    ds.close()


def test_the_persistence_layers_sigma_is_the_cell_constants_not_a_third_spelling(store):
    """``record_fit`` used to divide ``L_cm / (R1 · t_cm · w_cm)`` itself.

    Not ``z_to_sigma`` and not ``CellConstant.sigma`` — a *third* implementation of the
    formula, in the persistence layer, so a correction to the physics could reach every
    display and still miss the column downstream analysis actually queries.
    """
    run_id = store.start_run("stage_b_sigma")
    mid = store.record_measurement(run_id, _eis())
    store.record_fit(mid, _fit(R1=5.0e4), L_cm=0.2, t_cm=0.015, w_cm=0.2)

    stored = store.query_fits(measurement_id=mid)[0]["sigma_S_per_cm"]
    assert stored == CellConstant.from_legacy(0.2, 0.015, 0.2).sigma(5.0e4)

    # …and within one ULP of what the column held before Stage B. The stored σ is the
    # one number here that persists, so the association-order shift reaches disk.
    legacy = 0.2 / (5.0e4 * 0.015 * 0.2)
    assert abs(stored - legacy) <= math.ulp(legacy)


@pytest.mark.parametrize("geometry,R1", [
    ({"L_cm": 0.2, "w_cm": 0.2}, 5.0e4),                    # a term is absent
    ({"L_cm": 0.2, "t_cm": 0.015, "w_cm": 0.2}, 0.0),       # R1 is unusable
    ({}, 5.0e4),                                            # nothing declared
])
def test_record_fit_still_withholds_sigma_when_a_term_or_the_resistance_is_missing(
        store, geometry, R1):
    """Every guard the inline arithmetic carried survives the move onto the cell.

    A σ built on two of three dimensions, or on a zero resistance, is not a weaker
    measurement — it is not one at all.
    """
    run_id = store.start_run("stage_b_guards")
    mid = store.record_measurement(run_id, _eis())
    store.record_fit(mid, _fit(R1=R1), **geometry)
    assert store.query_fits(measurement_id=mid)[0]["sigma_S_per_cm"] is None


def test_the_persistence_layer_spells_no_sigma_arithmetic_of_its_own():
    """Pinned at source as well as behaviourally: the quotient must not come back."""
    fn = _func(_tree("core/data_store.py"), "record_fit")
    assert not any(_is_raw_sigma_arithmetic(n) for n in ast.walk(fn))
    assert any(isinstance(n, ast.Call) and _leaf(n.func) == "from_legacy"
               for n in ast.walk(fn))


@pytest.mark.asyncio
async def test_the_workflow_auto_fit_routes_through_analyze_spectrum_with_engine_unset(
        store, monkeypatch):
    """Site 9 — the origin of nearly every stored ``fit_results`` row.

    It called ``fit_circuit`` directly, so it was the one surface that ignored
    ``[eis] engine`` entirely while writing the rows everything downstream reads.
    """
    from softae.analysis.eis import router as eis_router
    from softae.workflows.workflow_model import WorkflowStep

    seen: list[dict] = []

    def spy(eis_result, **kwargs):
        seen.append(kwargs)
        return SpectrumReport(engine="legacy", fit=_fit(R1=5.0e4),
                              sigma=SigmaReport(mode="unavailable"))

    monkeypatch.setattr(eis_router, "analyze_spectrum", spy)
    monkeypatch.setattr(EISResult, "from_raw",
                        classmethod(lambda cls, raw, **kw: _eis()))

    run_id = store.start_run("stage_b_autofit")
    step = WorkflowStep(
        name="measure", instrument="pico1", method="sendscript_getdata",
        params={"chan": 1, "circuit_model": "simpleSalt",
                "electrode_L_cm": 0.5, "electrode_t_cm": 0.001,
                "electrode_w_cm": 0.1},
    )
    await eis_router.EISResultRouter().handle(
        step, object(), eis_router.RouterContext(data_store=store, run_id=run_id))

    assert seen, "the auto-fit never reached analyze_spectrum"
    assert "engine" not in seen[0]                        # [eis] engine still governs
    assert seen[0]["model_name"] == "simpleSalt"
    # The cell is the step's own declaration — the same L/t/w stored beside the row,
    # so the geometry analysed against and the geometry recorded cannot disagree.
    assert seen[0]["cell"].as_legacy_triple() == (0.5, 0.001, 0.1)

    row = store.query_fits(run_id=run_id)[0]
    assert row["R1"] == 5.0e4                            # the engine's fit, persisted
    assert row["sigma_S_per_cm"] == CellConstant.from_legacy(
        0.5, 0.001, 0.1).sigma(5.0e4)
    # P.18, not this stage: no report is passed, so the gate columns keep their defaults.
    assert row["gate_verdict"] is None


@pytest.mark.asyncio
async def test_a_step_that_declares_no_geometry_is_analysed_without_a_cell(
        store, monkeypatch):
    """No geometry is no conductivity — never one built on a nominal thickness."""
    from softae.analysis.eis import router as eis_router
    from softae.workflows.workflow_model import WorkflowStep

    seen: list[dict] = []
    monkeypatch.setattr(eis_router, "analyze_spectrum",
                        lambda eis_result, **kw: (seen.append(kw), SpectrumReport(
                            engine="legacy", fit=_fit(), sigma=SigmaReport()))[1])
    monkeypatch.setattr(EISResult, "from_raw",
                        classmethod(lambda cls, raw, **kw: _eis()))

    run_id = store.start_run("stage_b_no_geometry")
    for params in ({"chan": 1, "circuit_model": "simpleSalt"},
                   {"chan": 1, "circuit_model": "simpleSalt",
                    "electrode_L_cm": 0.5, "electrode_w_cm": 0.1},
                   {"chan": 1, "circuit_model": "simpleSalt",
                    "electrode_L_cm": 0.5, "electrode_t_cm": 0.0,
                    "electrode_w_cm": 0.1}):
        await eis_router.EISResultRouter().handle(
            WorkflowStep(name="m", instrument="pico1",
                         method="sendscript_getdata", params=params),
            object(),
            eis_router.RouterContext(data_store=store, run_id=run_id))

    assert len(seen) == 3
    assert [kw["cell"] for kw in seen] == [None, None, None]


def test_narrowing_the_sigma_matcher_did_not_reopen_the_hole_it_was_written_for():
    """Stage B made the guard flag its own sanctioned route; the fix was the matcher.

    ``CellConstant.from_legacy(...).sigma(R1)`` had to stop being an offence. Adding
    ``core/data_store.py`` to the allowlist instead would have excused every future
    off-route σ in the persistence layer along with it — which is precisely how the
    P.16 guard stopped meaning anything.
    """
    caught = _offences("x.py", ast.parse("s = fr.sigma(L, t, w)"))
    assert caught and "off-route" in caught[0]

    assert _offences("x.py",
                     ast.parse("s = CellConstant.from_legacy(L, t, w).sigma(R1)")) == []

    # A receiver that merely reads like a cell is not one: the exemption is structural.
    assert _offences("x.py", ast.parse("s = cell.sigma(R1)"))
