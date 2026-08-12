"""GUI tests for VFT/Arrhenius model selection in the post-hoc Analysis tab."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from softae.analysis.arrhenius import ArrheniusResult
from softae.analysis.vft import VftResult
from softae.gui.tabs.tab_analysis import _ArrhFitWorker


@pytest.fixture()
def analysis_tab(qapp):
    from softae.drivers.mock_factory import create_mock_manager
    from softae.gui.tabs.tab_analysis import AnalysisTab

    return AnalysisTab(create_mock_manager(config={}))


_KB_EV = 8.617333e-5


def _vft_result(ch=1):
    return VftResult(
        channel=ch, run_id="t",
        temperatures_C=[25.0, 45.0, 65.0, 85.0],
        conductivities=[1e-5, 2e-5, 4e-5, 8e-5],
        A=1.0, ln_A=0.0, Ea_eV=600.0 * _KB_EV, Ea_kJ_per_mol=600.0 * _KB_EV * 96.485,
        B=600.0, T0_K=180.0, T0_C=180.0 - 273.15,
        R_squared=0.999, T_min_C=25.0, T_max_C=85.0, n_points=4, fit_success=True,
    )


def _arrhenius_result(ch=1):
    return ArrheniusResult(
        channel=ch, run_id="t",
        temperatures_C=[25.0, 45.0, 65.0, 85.0],
        conductivities=[1e-5, 2e-5, 4e-5, 8e-5],
        Ea_eV=0.3, Ea_kJ_per_mol=28.9, ln_A=0.0, R_squared=0.998,
        T_min_C=25.0, T_max_C=85.0, n_points=4, fit_success=True,
    )


def test_posthoc_has_thermal_selector(analysis_tab):
    items = [
        analysis_tab._arrh_combo_thermal.itemText(i)
        for i in range(analysis_tab._arrh_combo_thermal.count())
    ]
    assert items == ["arrhenius", "vft"]


def test_worker_uses_selected_thermal_model():
    w = _ArrhFitWorker([], "simpleSalt", 0.2, 0.175, 0.2, thermal_model="vft")
    assert w._thermal_model == "vft"


def _sim_eis(Tc, ch=1):
    """Synthetic simpleSalt EIS whose R1 follows a VFT σ(T) trend."""
    import math

    import numpy as np
    from impedance.models.circuits import CustomCircuit

    from softae.analysis.eis_data import EISResult

    freq = np.logspace(5, -1, 41)
    A, B, T0, L, t, w = 1.0, 600.0, 180.0, 0.2, 0.175, 0.2
    sigma = A * math.exp(-B / (Tc + 273.15 - T0)) * 1e-3
    R1 = L / (sigma * t * w)
    cc = CustomCircuit("R0-CPE0-p(R1,C0)", initial_guess=[50.0, 1e-7, 0.8, R1, 1e-10])
    cc.parameters_ = np.array([50.0, 1e-7, 0.8, R1, 1e-10])
    Z = cc.predict(freq, use_initial=True)
    return EISResult.from_arrays(channel=ch, f=freq, z_real=Z.real, z_imag_neg=-Z.imag)


@pytest.mark.parametrize("thermal_model", ["arrhenius", "vft"])
def test_posthoc_no_rh_groups_into_single_fit(qapp, thermal_model):
    """Regression: no-RH measurements must form ONE group, not one per file.

    A fresh ``float('nan')`` rh-key per measurement (nan != nan) previously
    scattered every no-RH file into its own single-point group, so all fits
    failed (Arrhenius needs ≥2, VFT ≥3 points).
    """
    pytest.importorskip("impedance")
    pairs = [(i, _sim_eis(Tc), Tc, None) for i, Tc in enumerate([25.0, 45.0, 65.0, 85.0])]
    captured: dict = {}
    worker = _ArrhFitWorker(pairs, "simpleSalt", 0.2, 0.175, 0.2,
                            thermal_model=thermal_model)
    # (fit_by_index, sigma_by_index, arrh_keyed, rh_results) — σ is cached alongside
    # the fit so the params panel can show the number the fit consumed (P.20).
    worker.finished.connect(lambda a, s, k, r: captured.__setitem__("keyed", k))
    worker.run()  # synchronous

    keyed = captured["keyed"]
    assert len(keyed) == 1                       # all 4 temps in ONE group
    _ch, _rh, res = keyed[0]
    assert res.model == thermal_model
    assert res.n_points == 4
    assert res.fit_success


def test_ea_table_headers_switch_for_vft(analysis_tab):
    tab = analysis_tab
    tab._arrh_model = "vft"
    tab._arrh_results_keyed = [(1, float("nan"), _vft_result(1))]
    tab._arrh_update_ea_table()
    headers = [
        tab._arrh_ea_table.horizontalHeaderItem(c).text()
        for c in range(tab._arrh_ea_table.columnCount())
    ]
    assert headers[2] == "Eₐ (eV)"
    assert headers[3] == "T₀ (°C)"
    # VFT activation energy Eₐ = B·k_B (B=600 K → ~0.0517 eV).
    assert tab._arrh_ea_table.item(0, 2).text() == f"{600.0 * _KB_EV:.4f}"


def test_ea_table_headers_arrhenius(analysis_tab):
    tab = analysis_tab
    tab._arrh_model = "arrhenius"
    tab._arrh_results_keyed = [(1, float("nan"), _arrhenius_result(1))]
    tab._arrh_update_ea_table()
    headers = [
        tab._arrh_ea_table.horizontalHeaderItem(c).text()
        for c in range(tab._arrh_ea_table.columnCount())
    ]
    assert headers[2] == "Eₐ (eV)"
    assert tab._arrh_ea_table.item(0, 2).text() == "0.3000"


@pytest.mark.skipif(
    __import__("softae.gui.tabs.tab_analysis", fromlist=["_HAS_MPL"])._HAS_MPL is False,
    reason="matplotlib not available",
)
def test_plot_titles_are_model_aware(analysis_tab):
    tab = analysis_tab
    # VFT
    tab._arrh_model = "vft"
    tab._arrh_results_keyed = [(1, float("nan"), _vft_result(1))]
    tab._arrh_update_plot()
    assert tab._arrh_fig.axes[0].get_title() == "VFT Plot"
    # Arrhenius
    tab._arrh_model = "arrhenius"
    tab._arrh_results_keyed = [(1, float("nan"), _arrhenius_result(1))]
    tab._arrh_update_plot()
    assert tab._arrh_fig.axes[0].get_title() == "Arrhenius Plot"


def test_vft_fit_curve_is_curved(analysis_tab):
    # The VFT model curve must not be a straight line in 1000/T (unlike Arrhenius).
    import numpy as np
    if not __import__("softae.gui.tabs.tab_analysis", fromlist=["_HAS_MPL"])._HAS_MPL:
        pytest.skip("matplotlib not available")
    tab = analysis_tab
    tab._arrh_model = "vft"
    tab._arrh_results_keyed = [(1, float("nan"), _vft_result(1))]
    tab._arrh_update_plot()
    # find the fitted line (100 points) and check curvature (non-constant slope)
    curved = False
    for line in tab._arrh_fig.axes[0].get_lines():
        x, y = line.get_xdata(), line.get_ydata()
        if len(x) >= 50:
            d = np.diff(y) / np.diff(x)
            if np.ptp(d) > 1e-6:   # slope varies → curved
                curved = True
    assert curved
