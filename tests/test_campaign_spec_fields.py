"""How a declared scale crosses the file boundary (W1, D2 / D2b).

``docs/SubAgent docs/thickness_target_all_paths.md``. ``target_deposition_uL``
and ``thickness_um`` both fix the scale of a cast, so a spec carrying both
states it twice and one would be silently preferred — the failure this module's
whole docstring is about. The decoder takes exactly one, and refuses both and
neither.
"""

from __future__ import annotations

import pytest

from softae.core.campaign_spec_fields import (
    UNREPRESENTABLE,
    decode_general_formulation,
    encode_general_formulation,
)
from softae.core.deposit_target import DepositTargetSpec

AXIS = {"kind": "molar_ratio", "a": "EO", "b": "Li",
        "low": 5.0, "high": 40.0, "basis": "volume"}


@pytest.fixture
def stub_catalogs(monkeypatch):
    """Stock names resolve without a data root — the seam's stated purpose."""
    from softae.core import campaign_spec_fields as fields
    from softae.core.formulation import ChemicalCatalog, Solution, SolutionCatalog

    sol = SolutionCatalog()
    sol.add(Solution(name="PEO stock"))
    chem = ChemicalCatalog()
    monkeypatch.setattr(fields, "catalogs", lambda: (chem, sol))
    return chem, sol


def _table(**scale):
    return {"stocks": ["PEO stock"], "pump_assignment": {"PEO stock": 0},
            "axes": [AXIS], **scale}


def _supports_deposit_target() -> bool:
    """Whether the campaign context has grown its field yet (wave W3)."""
    from dataclasses import fields as dataclass_fields

    from softae.core.autonomous_wiring import GeneralFormulation

    return any(f.name == "deposit_target"
               for f in dataclass_fields(GeneralFormulation))


class TestScaleKeyExclusivity:
    def test_spec_declaring_both_thickness_and_deposition_volume_is_refused(
        self, stub_catalogs
    ):
        """D2. Accepting both would let one be silently ignored."""
        with pytest.raises(ValueError, match="both"):
            decode_general_formulation(
                _table(target_deposition_uL=4.5, thickness_um=60.0))

    def test_spec_declaring_neither_scale_key_is_refused(self, stub_catalogs):
        """D2's other half: a solve with no scale row is under-determined."""
        with pytest.raises(ValueError, match="neither"):
            decode_general_formulation(_table())

    def test_a_thickness_basis_without_a_thickness_is_refused(self, stub_catalogs):
        with pytest.raises(ValueError, match="thickness_basis"):
            decode_general_formulation(
                _table(target_deposition_uL=4.5, thickness_basis="wet"))

    def test_a_volume_spec_still_decodes_exactly_as_before(self, stub_catalogs):
        """Every spec that loaded before this change must still load."""
        gf = decode_general_formulation(_table(target_deposition_uL=4.5))

        assert gf.target_deposition_uL == pytest.approx(4.5)


class TestThicknessDecoding:
    def test_a_thickness_spec_builds_a_thickness_deposit_target(self, stub_catalogs):
        if not _supports_deposit_target():
            pytest.skip("campaign context grows 'deposit_target' in wave W3")

        gf = decode_general_formulation(
            _table(thickness_um=60.0, thickness_basis="wet"))

        assert gf.deposit_target == DepositTargetSpec("thickness", 60.0, "wet")
        # D2b: no volume exists until an area is resolved, and a placeholder
        # would be a scale nobody declared travelling into a solve.
        assert gf.target_deposition_uL is None

    def test_a_thickness_is_refused_rather_than_dropped_before_its_field_exists(
        self, stub_catalogs
    ):
        """A spec that decoded without its scale would run a different
        experiment from the one the file describes."""
        if _supports_deposit_target():
            pytest.skip("the field exists; the thickness is carried, not refused")

        with pytest.raises(ValueError, match="thickness_um"):
            decode_general_formulation(_table(thickness_um=60.0))

    def test_a_non_numeric_thickness_is_refused(self, stub_catalogs):
        with pytest.raises(ValueError, match="thickness_um"):
            decode_general_formulation(_table(thickness_um="thick"))


class TestEncoding:
    def test_a_volume_context_writes_the_volume_key(self, stub_catalogs):
        gf = decode_general_formulation(_table(target_deposition_uL=4.5))

        out = encode_general_formulation(gf)

        assert out is not UNREPRESENTABLE
        assert out["target_deposition_uL"] == pytest.approx(4.5)
        assert "thickness_um" not in out

    def test_a_thickness_context_round_trips_through_the_file(self, stub_catalogs):
        if not _supports_deposit_target():
            pytest.skip("campaign context grows 'deposit_target' in wave W3")

        gf = decode_general_formulation(
            _table(thickness_um=60.0, thickness_basis="dry"))

        out = encode_general_formulation(gf)

        assert out["thickness_um"] == pytest.approx(60.0)
        # Written even at its default, for the reason the axis keys are: an
        # omitted key would silently take a default and cast a different film.
        assert out["thickness_basis"] == "dry"
        assert "target_deposition_uL" not in out
        assert decode_general_formulation(
            {**_table(), **{k: out[k]
                            for k in ("thickness_um", "thickness_basis")}}
        ).deposit_target == gf.deposit_target
