"""Composition-targets mode in the Deposition twin.

Drives the mode toggle, the targets table, and the solve_formulation →
elution_from_stock_volumes → simulate path added to DepositionPanel.
"""

from __future__ import annotations

import pytest

from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
)
from softae.gui.widgets.deposition_panel import (
    _SUM_ERROR_STYLE,
    _SUM_OK_STYLE,
    DepositionPanel,
)


def _catalogs():
    chem = ChemicalCatalog()
    chem.add(Chemical("PolyA", density_g_per_mL=1.2, molar_mass_g_per_mol=100.0))
    chem.add(Chemical("PolyB", density_g_per_mL=1.5, molar_mass_g_per_mol=120.0))
    chem.add(Chemical("Water", density_g_per_mL=1.0, molar_mass_g_per_mol=18.0))
    sol = SolutionCatalog()
    sol.add(Solution("A", [SolutionComponent("PolyA", "dep", 2.0, "mL"),
                           SolutionComponent("Water", "carrier", 8.0, "mL")]))  # dep_frac 0.2
    sol.add(Solution("B", [SolutionComponent("PolyB", "dep", 3.0, "mL"),
                           SolutionComponent("Water", "carrier", 7.0, "mL")]))  # dep_frac 0.3
    return chem, sol


@pytest.fixture
def panel(qtbot):
    chem, sol = _catalogs()
    p = DepositionPanel(chem, sol)
    qtbot.addWidget(p)
    return p


def _add_dried_fraction(panel, component: str, value: float):
    panel._targets_editor.add_target("Dried fraction", a=component, value=str(value))


# ── Mode toggle ──────────────────────────────────────────────────────────────


def test_manual_is_default_and_targets_hidden(panel):
    assert panel._form_mode() == "manual"
    assert panel._targets_editor.isHidden()


def test_switch_to_targets_shows_editor_and_disables_fraction_spins(panel):
    panel._combo_form_mode.setCurrentIndex(1)
    assert panel._form_mode() == "targets"
    assert not panel._targets_editor.isHidden()
    assert all(not s.isEnabled() for s in panel._spin_fractions)  # fractions are solved
    assert not panel._btn_auto_balance.isEnabled()


# ── Solve + render ───────────────────────────────────────────────────────────


def test_targets_mode_solves_and_renders(panel):
    panel._combo_form_mode.setCurrentIndex(1)
    panel._spin_target.setValue(3.0)          # small → within well capacity
    _add_dried_fraction(panel, "PolyB", 0.4)
    panel._recompute()

    assert panel._last_plan is not None
    assert panel._last_plan.feasible is True
    assert panel._last_summary is not None    # the twin produced a film prediction
    assert panel._last_plan.achieved["dried_frac[PolyB]"] == pytest.approx(0.4, rel=1e-6)
    # Feasibility surfaced on the Σ label, achieved on the elution label.
    assert panel._lbl_fraction_sum.styleSheet() == _SUM_OK_STYLE
    assert "dried_frac[PolyB]" in panel._lbl_elution.text()


def test_over_capacity_flagged_infeasible(panel):
    panel._combo_form_mode.setCurrentIndex(1)
    panel._spin_target.setValue(20.0)         # large → cast exceeds the ~39 µL well
    _add_dried_fraction(panel, "PolyB", 0.4)
    panel._recompute()

    assert panel._last_plan is not None
    assert panel._last_plan.feasible is False
    assert panel._lbl_fraction_sum.styleSheet() == _SUM_ERROR_STYLE
    assert any("budget" in n for n in panel._last_plan.notes)


def test_switch_back_to_manual_restores_fraction_path(panel):
    panel._combo_form_mode.setCurrentIndex(1)
    _add_dried_fraction(panel, "PolyB", 0.4)
    panel._recompute()
    assert panel._last_plan is not None

    panel._combo_form_mode.setCurrentIndex(0)  # back to manual
    assert panel._form_mode() == "manual"
    assert panel._targets_editor.isHidden()
    assert panel._last_plan is None            # manual path clears the solve plan
