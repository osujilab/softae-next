"""The shared TargetsEditor: dropdown population, read-out, and validation."""

from __future__ import annotations

import pytest

from softae.core.formulation import (
    DriedFractionTarget,
    MolarRatioTarget,
)
from softae.gui.widgets.targets_editor import TargetsEditor


@pytest.fixture
def editor(qtbot):
    e = TargetsEditor()
    qtbot.addWidget(e)
    e.set_available(["PEO", "LiCl"], ["PEO", "SiO2"], n_stocks=3)
    return e


def test_targets_read_from_combos(editor):
    editor.add_target("Molar ratio", a="PEO", b="LiCl", value="20")
    ts = editor.targets()
    assert len(ts) == 1
    assert isinstance(ts[0], MolarRatioTarget)
    assert (ts[0].numerator, ts[0].denominator, ts[0].value) == ("PEO", "LiCl", 20.0)


def test_dried_fraction_disables_B_and_uses_components(editor):
    row = editor.add_target("Dried fraction", a="SiO2", value="0.1")
    assert not editor._table.cellWidget(row, 2).isEnabled()  # B unused for dried fraction
    ts = editor.targets()
    assert isinstance(ts[0], DriedFractionTarget) and ts[0].component == "SiO2"


def test_underdetermined_flagged(editor):
    # 3 stocks need 3 targets (incl. the µL box); one ratio + µL = 2 → short by 1.
    editor.add_target("Molar ratio", a="PEO", b="LiCl", value="20")
    assert any("Under-determined" in m for m in editor.issues())


def test_over_determined_flagged(editor):
    editor.set_available(["PEO", "LiCl"], ["SiO2"], n_stocks=2)
    editor.add_target("Molar ratio", a="PEO", b="LiCl", value="20")
    editor.add_target("Dried fraction", a="SiO2", value="0.1")  # 2 targets + µL = 3 > 2
    assert any("Over-determined" in m for m in editor.issues())


def test_unknown_species_flagged(editor):
    editor.set_available(["PEO", "LiCl"], ["SiO2"], n_stocks=2)
    editor.add_target("Molar ratio", a="PEO_stock", b="LiCl", value="20")  # stock name, not species
    assert any("PEO_stock" in m and "species" in m for m in editor.issues())


def test_unknown_component_flagged(editor):
    editor.set_available(["PEO", "LiCl"], ["SiO2"], n_stocks=2)
    editor.add_target("Dried fraction", a="Alumina", value="0.1")  # not a deposited component
    assert any("Alumina" in m and "component" in m for m in editor.issues())


def test_clean_target_set_has_no_issues(editor):
    editor.set_available(["PEO", "LiCl"], ["SiO2"], n_stocks=3)
    editor.add_target("Molar ratio", a="PEO", b="LiCl", value="20")
    editor.add_target("Dried fraction", a="SiO2", value="0.1")  # 2 + µL = 3 == 3 stocks
    assert editor.issues() == []
