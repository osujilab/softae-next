"""The decode-time pinning check in :mod:`softae.core.campaign_spec_io`.

The rest of that module's coverage lives in ``tests/test_campaign_cli.py``,
which was written when the loader and the CLI arrived together; this file holds
only the refusal added with :mod:`softae.core.pinned_recipe`, so the defect and
its controls sit in one place rather than being appended to an 800-line file
about something else.

**What the refusal is for.** ``composition_axes.build_targets_from_axes``
substitutes an axis's ``low`` when a suggestion does not carry it, deliberately:
a solver on a stale bound is recoverable, a campaign dying mid-round on a
``KeyError`` is not. That is right at run time and silent at load time — a spec
declaring a searched axis it never listed in ``parameter_space`` casts every
trial at the corner of its declared box and reports a search. The GUI's axes
editor is refused that shape by ``validate_axes``; a TOML-loaded campaign
reached no check at all until this one.
"""

from __future__ import annotations

import pytest
import structlog

from softae.core.campaign_spec_io import SpecLoadError, spec_from_dict

#: A legacy volume-mode spec — no composition context, nothing to pin.
MINIMAL = {
    "name": "c",
    "parameter_space": {"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
}

RATIO_AXIS = {"kind": "molar_ratio", "a": "EO", "b": "Li",
              "low": 5.0, "high": 40.0, "basis": "volume"}
PINNED_RATIO_AXIS = {**RATIO_AXIS, "low": 20.0, "high": 20.0}
PINNED_SILICA_AXIS = {"kind": "dried_fraction", "a": "SiO2", "b": "",
                      "low": 0.1, "high": 0.1, "basis": "volume"}

RATIO_PARAM = {"type": "float", "low": 5.0, "high": 40.0}
REPLICATE_PARAM = {"type": "int", "low": 1, "high": 4}


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


def _payload(axes, space):
    return {
        "name": "comp",
        "budget": 4,
        "parameter_space": space,
        "general_formulation": {
            "stocks": ["PEO stock"],
            "pump_assignment": {"PEO stock": 0},
            "target_deposition_uL": 4.5,
            "axes": axes,
        },
    }


class TestPinningRefusal:

    def test_a_searched_axis_absent_from_parameter_space_is_refused(
        self, stub_catalogs
    ):
        """THE DEFECT. Before the decode hook this loaded and ran, silently."""
        payload = _payload([RATIO_AXIS], {"replicate": REPLICATE_PARAM})

        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict(payload, source="<defect>")

        assert "ratio_EO_Li" in str(exc.value)
        assert "lower bound" in str(exc.value)

    def test_a_searched_axis_listed_in_parameter_space_still_decodes(
        self, stub_catalogs
    ):
        """The control: the check must not refuse an ordinary search spec."""
        payload = _payload([RATIO_AXIS], {"ratio_EO_Li": RATIO_PARAM})

        spec = spec_from_dict(payload)

        assert set(spec.parameter_space) == {"ratio_EO_Li"}
        assert spec.general_formulation.axes[0].high == 40.0

    def test_a_fully_pinned_recipe_decodes_with_only_a_replicate_parameter(
        self, stub_catalogs
    ):
        """The shape the check exists to permit, not to refuse."""
        payload = _payload([PINNED_RATIO_AXIS, PINNED_SILICA_AXIS],
                           {"replicate": REPLICATE_PARAM})

        spec = spec_from_dict(payload)

        assert all(a.is_fixed for a in spec.general_formulation.axes)
        assert set(spec.parameter_space) == {"replicate"}

    def test_a_pinned_axis_beside_a_searched_one_is_not_itself_required(
        self, stub_catalogs
    ):
        """Only *searched* axes need a parameter; a pinned one must not."""
        payload = _payload([RATIO_AXIS, PINNED_SILICA_AXIS],
                           {"ratio_EO_Li": RATIO_PARAM})

        spec = spec_from_dict(payload)

        assert len(spec.general_formulation.axes) == 2

    def test_a_volume_mode_spec_is_unaffected_by_the_pinning_check(self):
        """Every spec that loaded before this hook must still load."""
        assert spec_from_dict(MINIMAL).name == "c"


class TestLoadEvent:

    def _event(self, payload):
        with structlog.testing.capture_logs() as logs:
            spec_from_dict(payload)
        return next(e for e in logs if e["event"] == "campaign_spec_loaded")

    def test_a_pinned_recipe_is_recorded_as_pinned(self, stub_catalogs):
        event = self._event(_payload([PINNED_RATIO_AXIS, PINNED_SILICA_AXIS],
                                     {"replicate": REPLICATE_PARAM}))

        assert event["recipe_pinned"] is True
        assert event["n_searched_axes"] == 0

    def test_a_search_spec_is_not_recorded_as_pinned(self, stub_catalogs):
        event = self._event(_payload([RATIO_AXIS, PINNED_SILICA_AXIS],
                                     {"ratio_EO_Li": RATIO_PARAM}))

        assert event["recipe_pinned"] is False
        assert event["n_searched_axes"] == 1

    def test_a_volume_mode_spec_is_not_recorded_as_pinned(self):
        """"Nothing to pin" and "pinned deliberately" must not share a token."""
        event = self._event(MINIMAL)

        assert event["recipe_pinned"] is False
        assert event["n_searched_axes"] == 0
