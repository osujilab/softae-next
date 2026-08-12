"""P.16 Stage 1 — one EIS analysis source across the GUI.

Every GUI surface that reports conductivity now builds the same
:class:`~softae.analysis.eis.geometry.CellConstant` and reads σ off a
:class:`~softae.analysis.eis.report.SpectrumReport`, instead of hand-rolling
``z_to_sigma(L, t, w, R1)``. Stage 1 changes the *route* only — no displayed
number moves, and ``[eis] engine`` (not a literal at a call site) still decides
which engine runs.

The two tests that carry the weight are the first two. Existing coverage proves
``cell.sigma == z_to_sigma`` for a *hand-built* cell (``test_eis_geometry.py``) and
that ``_legacy_report`` composes its parts correctly (``test_eis_engine.py``);
nothing pinned that the cell the **GUI** builds is the right one, and a mistake
there — an electrode-configuration factor of 2, say — moves every σ on screen while
every self-consistent test still passes.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from softae.analysis.circuit_fitting import FitResult, z_to_sigma
from softae.analysis.eis.geometry import CONFIG_FACTORS, CellConstant
from softae.analysis.eis.report import SigmaReport, SpectrumReport
from softae.analysis.eis_data import EISResult
from softae.gui.eis_sigma import cell_sigma, gui_cell, report_sigma

GUI_ROOT = Path(__file__).resolve().parents[1] / "src" / "softae" / "gui"


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])

#: The geometries the GUI actually offers: the analysis tab's spin defaults, the
#: browser's ``DEFAULT_GEOMETRY``, a realistic drop-cast film, and a board whose
#: L and w differ.
GEOMETRIES = [
    (0.2, 0.175, 0.2),      # the historical placeholder thickness
    (0.2, 0.015, 0.2),      # the datum test_eis_geometry.py pins
    (0.3, 0.1, 0.25),       # test_eis_browser_fit.py's geometry
    (0.15, 0.002, 0.17),    # L != w, a 20 µm film
]
RESISTANCES = [1.0, 333.0, 1234.5, 5.0e4, 9.87e5, 7.0e7]


def _legacy_sigma(L: float, t: float, w: float, R: float) -> float:
    """The deprecated oracle, called deliberately — the warning is expected here.

    ``z_to_sigma`` emits a ``DeprecationWarning`` since P.20 and has zero production
    callers. It survives only as the *independent* proof that the new route computes the
    same physics, so these tests state the expectation rather than suppressing it: the
    oracle keeps working, and a production re-adoption is caught by the completeness
    guard in ``test_eis_universal_fit_route.py``.
    """
    with pytest.warns(DeprecationWarning, match="z_to_sigma is deprecated"):
        return float(z_to_sigma(L, t, w, R))


def _gui_sources() -> list[Path]:
    return sorted(p for p in GUI_ROOT.rglob("*.py"))


def _trees() -> list[tuple[Path, ast.Module]]:
    # utf-8-sig: at least one GUI module is BOM-prefixed, which ast.parse refuses.
    return [(p, ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p)))
            for p in _gui_sources()]


# ── The cell the GUI builds ──────────────────────────────────────────────────


def test_the_cell_the_gui_builds_reproduces_the_tabs_previous_l_t_w_sigma_exactly():
    """``gui_cell(L, t, w).sigma(R)`` is the σ the tabs used to print.

    Exact at the datum ``test_eis_geometry.py::TestLegacyParity`` pins, and within
    one unit in the last place everywhere else. It is *not* bit-identical in
    general, and the spec's claim that it is overstates what that test proves:
    ``CellConstant.sigma`` evaluates ``K/R`` with ``K = L/(t·w)`` precomputed, while
    ``z_to_sigma`` evaluates ``L/((R·t)·w)``. Same arithmetic, different association
    order, so the final rounding can differ by one ULP (~2e-16 relative). Every
    precision the GUI prints — 3 or 4 significant figures — is identical, which is
    the property that matters: no displayed number moves.
    """
    L, t, w, R = 0.2, 0.015, 0.2, 5.0e4
    assert gui_cell(L, t, w).sigma(R) == _legacy_sigma(L, t, w, R)

    for L, t, w in GEOMETRIES:
        cell = gui_cell(L, t, w)
        assert (cell.L_gap_cm, cell.thickness_cm, cell.L_stripe_cm) == (L, t, w)
        assert cell.dead_height_cm == 0.0
        for R in RESISTANCES:
            mine = cell.sigma(R)
            legacy = _legacy_sigma(L, t, w, R)
            assert abs(mine - legacy) <= math.ulp(legacy)
            for fmt in (".4e", ".3e", ".4g", ".3g", ".6e"):
                assert format(mine, fmt) == format(legacy, fmt)


def test_the_gui_cell_never_activates_the_electrode_config_factor():
    """``k_config_factor`` stays 1.0, so ``K_per_cm`` is the bare geometric K.

    ``CONFIG_FACTORS["3-electrode"]`` is 2.0. Threading an electrode configuration
    into the cell the GUI builds would halve every K and so halve every σ, and no
    test asserting "σ from R" would notice, because the arithmetic stays internally
    consistent. This is the guard.
    """
    assert CONFIG_FACTORS["3-electrode"] == 2.0        # the landmine is still armed
    for L, t, w in GEOMETRIES:
        cell = gui_cell(L, t, w)
        assert cell.k_config_factor == 1.0
        assert cell.electrode_config == "unverified"
        assert cell.k_config_verified is False
        assert cell.re_contact_verified is False
        assert cell.config_factor_verified is False
        assert cell.K_per_cm == cell.K_geometric_per_cm


# ── One source, config-governed ──────────────────────────────────────────────


def test_every_gui_sigma_path_goes_through_analyze_spectrum():
    """No ``z_to_sigma`` import or reference survives anywhere under ``src/softae/gui``.

    AST rather than a text scan so prose in a docstring cannot satisfy or break it.
    """
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if any(a.name == "z_to_sigma" for a in node.names):
                    offenders.append(f"{path.name}:{node.lineno} imports z_to_sigma")
            elif isinstance(node, ast.Name) and node.id == "z_to_sigma":
                offenders.append(f"{path.name}:{node.lineno} references z_to_sigma")
            elif isinstance(node, ast.Attribute) and node.attr == "z_to_sigma":
                offenders.append(f"{path.name}:{node.lineno} references .z_to_sigma")
    assert offenders == []

    routed = {p.name for p, tree in _trees()
              for n in ast.walk(tree)
              if isinstance(n, ast.Name) and n.id == "analyze_spectrum"}
    assert {"tab_analysis.py", "eis_visualizer_widget.py"} <= routed


def test_the_engine_is_config_governed_not_hardcoded_at_any_call_site(monkeypatch, qapp):
    """No GUI call site names an engine, so ``[eis] engine`` is the whole cutover."""
    hardcoded = [
        f"{path.name}:{node.lineno}"
        for path, tree in _trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        == "analyze_spectrum"
        and any(kw.arg == "engine" for kw in node.keywords)
    ]
    assert hardcoded == []

    # …and behaviourally: the workers hand the engine no opinion of its own.
    seen: list[dict] = []

    def spy(eis_result, **kwargs):
        seen.append(kwargs)
        return SpectrumReport(engine="legacy", fit=None, sigma=SigmaReport())

    from softae.gui.tabs import tab_analysis

    monkeypatch.setattr(tab_analysis, "analyze_spectrum", spy)
    worker = tab_analysis._FitAllWorker(
        [SimpleNamespace(channel=1, raw_file_path="")], "simpleSalt", 0.2, 0.015, 0.2)
    worker.run()
    assert seen and all("engine" not in kw for kw in seen)
    assert seen[0]["cell"] == CellConstant.from_legacy(0.2, 0.015, 0.2)


# ── A missing thickness is still no number ───────────────────────────────────


@pytest.mark.parametrize("t", [None, 0.0, -0.01, float("nan"), "", "abc"])
def test_a_missing_thickness_still_yields_no_sigma_rather_than_a_nominal(t):
    """No cell, therefore ``sigma.mode == "unavailable"``, therefore a blank cell.

    Never the 0.175 cm placeholder, and never a nominal of any other kind.
    """
    assert gui_cell(0.2, t, 0.2) is None
    assert cell_sigma(gui_cell(0.2, t, 0.2), 1000.0) is None

    from softae.analysis.eis.engine import analyze_spectrum

    f = np.logspace(1, 5, 30)
    tau = 2 * np.pi * f * 1e-6
    eis = EISResult.from_arrays(
        channel=1, f=f, z_real=100 + 50 / (1 + tau**2),
        z_imag_neg=np.abs(50 * tau / (1 + tau**2)),
    )
    report = analyze_spectrum(eis, cell=gui_cell(0.2, t, 0.2))
    assert report.sigma.mode == "unavailable"
    assert math.isnan(report_sigma(report))


# ── The manual tab's second, non-geometric route ─────────────────────────────


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


def _fit(R1: float) -> FitResult:
    return FitResult(model_name="simpleSalt", parameters=np.array([0.0, R1]),
                     R0=0.0, R1=R1, R0_guess=0.0, R1_guess=R1, z_indices=[0, 1])


def test_the_manual_tabs_empirical_K_branch_is_preserved():
    """``K / R1`` from the spin box is not a cell-constant route and did not move."""
    from softae.gui.tabs.tab_manual import ManualControlTab

    sigma = ManualControlTab._conductivity_from_fit(
        _manual_stub(geom=False, K=12.5), _fit(500.0))
    assert sigma == 12.5 / 500.0

    # A zero or unset K still yields nothing rather than an infinity.
    assert ManualControlTab._conductivity_from_fit(
        _manual_stub(geom=False, K=0.0), _fit(500.0)) is None

    # …and the geometry branch beside it now runs off the shared cell constant.
    geom_sigma = ManualControlTab._conductivity_from_fit(
        _manual_stub(geom=True), _fit(5.0e4))
    assert geom_sigma == pytest.approx(_legacy_sigma(0.2, 0.015, 0.2, 5.0e4),
                                       rel=1e-15)

    # A degenerate geometry withholds σ instead of reporting one from a nominal.
    assert ManualControlTab._conductivity_from_fit(
        _manual_stub(geom=True, t=0.0), _fit(5.0e4)) is None


# ── The browser's single choke point ─────────────────────────────────────────


def test_the_browser_fit_entry_migration_covers_both_embedded_and_popout_use(monkeypatch):
    """``fit_entry`` is the only fit path in the browser, so migrating it is enough.

    ``_InspectionPane`` (embedded in ``EISVisualizerWidget``) and
    ``EISVisualizerWindow`` (the pop-out, which *wraps* that same widget) both fit
    through ``_SpectrumFitWorker``, which calls ``fit_entry`` — so there is exactly
    one place the engine is chosen for either.
    """
    from softae.gui.widgets import eis_visualizer_widget as vw

    src = Path(vw.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    worker = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and n.name == "_SpectrumFitWorker")
    assert any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "fit_entry"
               for n in ast.walk(worker))
    # The pop-out window owns an EISVisualizerWidget rather than its own fit path.
    window = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and n.name == "EISVisualizerWindow")
    assert any(isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "EISVisualizerWidget"
               for n in ast.walk(window))
    assert not any(isinstance(n, ast.Call)
                   and getattr(n.func, "id", None) in ("fit_circuit", "fit_entry")
                   for n in ast.walk(window))

    # And the one choke point routes through the engine, cell and all.
    captured: dict = {}

    def spy(eis_result, **kwargs):
        captured.update(kwargs)
        return SpectrumReport(
            engine="legacy", fit=_fit(500.0),
            sigma=SigmaReport(mode="value", value=kwargs["cell"].sigma(500.0)),
        )

    monkeypatch.setattr("softae.analysis.eis.engine.analyze_spectrum", spy)

    f = np.logspace(1, 5, 30)
    tau = 2 * np.pi * f * 1e-6
    entry = vw.EISEntry(
        label="Ch01", fit=None, sigma=None,
        eis=EISResult.from_arrays(channel=1, f=f, z_real=100 + 50 / (1 + tau**2),
                                  z_imag_neg=np.abs(50 * tau / (1 + tau**2))),
    )
    out = vw.fit_entry(entry, "simpleSalt", 0.3, 0.1, 0.25)
    assert out is entry
    assert entry.geometry == (0.3, 0.1, 0.25)
    assert "engine" not in captured
    assert captured["cell"] == CellConstant.from_legacy(0.3, 0.1, 0.25)
    assert entry.sigma == pytest.approx(_legacy_sigma(0.3, 0.1, 0.25, 500.0),
                                        rel=1e-15)
