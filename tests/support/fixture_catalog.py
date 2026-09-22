"""The shipped placeholder catalogs, offered to tests through the public seams.

A test that compiles a campaign needs a task catalog and a chemistry catalog.
Taking them from the instance's ``data/`` would tie the result to one machine,
so tests take them from :mod:`softae.catalog_defaults`, which ships with the
package and is the same on every checkout.

Two seams carry them, and no private is patched: ``build_trial_workflow`` takes
a ``catalog=`` argument, and ``campaign_spec_fields.catalogs`` is a documented
hook for supplying chemistry without a data root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from softae.catalog_defaults import DefaultCatalogs, load_defaults

#: The committed campaign specs under ``examples/``. The ``.py`` files beside
#: them are demo scripts with no spec to compile.
EXAMPLES: tuple[str, ...] = (
    "bench_instance", "bo_init_bench", "rung3a_fake_cast", "shadow_campaign",
)


def example_path(example: str) -> Path:
    """The committed spec file for *example*."""
    return Path("examples") / f"{example}.toml"


def default_catalogs() -> DefaultCatalogs:
    """The three shipped catalogs, loaded and proved non-empty."""
    return load_defaults()


def default_task_catalog() -> Any:
    """The shipped task catalog ``build_trial_workflow`` resolves names against."""
    return load_defaults().tasks


@pytest.fixture
def task_catalog():
    """The task catalog the package ships, loaded once per test.

    Imported by name into the test modules that use it — pytest finds a fixture
    in the module's namespace however it got there — so the two files that
    compile workflows share one definition instead of repeating it.
    """
    return default_task_catalog()


def use_default_chemistry(monkeypatch: Any) -> None:
    """Point the chemistry seam at the shipped catalogs for one test.

    Applied before a spec is *read*, not before it is compiled: an unknown stock
    name is refused during decoding, well ahead of any workflow being built.
    """
    from softae.core import campaign_spec_fields

    defaults = load_defaults()
    monkeypatch.setattr(
        campaign_spec_fields, "catalogs",
        lambda: (defaults.chemicals, defaults.solutions),
    )


def load_example(example: str, monkeypatch: Any):
    """The ``CampaignSpec`` for one committed example, decoded against defaults."""
    from softae.core.campaign_spec_io import load_campaign_spec

    use_default_chemistry(monkeypatch)
    return load_campaign_spec(example_path(example))


def midpoint(spec: Any) -> dict[str, Any]:
    """The parameter-space midpoint, as ``preflight.project_campaign`` derives it.

    One reproducible suggestion is what a structural assertion can honestly be
    taken at; a sampled point would differ per run. It mirrors preflight
    deliberately, so the two describe the same representative trial.
    """
    out: dict[str, Any] = {}
    for name, axis in (getattr(spec, "parameter_space", {}) or {}).items():
        if axis.get("type") in ("float", "int"):
            out[name] = (float(axis["low"]) + float(axis["high"])) / 2.0
        else:
            out[name] = (axis.get("choices") or [None])[0]
    return out


def compile_example(example: str, catalog: Any, monkeypatch: Any):
    """Compile one committed example into a ``Workflow`` against the defaults."""
    from softae.core.autonomous_wiring import build_trial_workflow

    spec = load_example(example, monkeypatch)
    return build_trial_workflow(spec, midpoint(spec), catalog=catalog)
