"""Which stock solution is loaded on which pump (P8 catalog linkage).

The missing binding, and it is narrower than "link the catalog to the rig".
:attr:`~softae.core.formulation.Chemical.is_particulate` already exists and is
fully wired — persisted, editable, browsable, round-tripped. What did *not*
exist is any record of **which solution is currently loaded on which pump**, so
nothing downstream could be derived from the chemistry:

* ``[purge] particulate_pumps`` had to be hand-maintained in TOML, where it was
  found **wrong** (declared pump 0; the particulate line is pump 1);
* the stock ledger tracks volumes against a bare pump index with no idea what
  is in the syringe.

One declaration fixes both.

**Direction matters: this is position → solution, never the reverse.** Pump IDs
are physically meaningful bench positions — pump 1 is a particular syringe in a
particular place — so the operator's mental model is "what is loaded in position
1", not "where did my silica solution end up". Every accessor, message, and
warning here leads with the pump index for that reason.

**An undeclared pump is treated as particulate**, consistent with "undeclared =
unknown, never empty" elsewhere in the consumables work: spending a few extra µL
purging a line that turned out not to need it is recoverable, a clogged check
valve on an unattended overnight run is not. But a *completely* empty loadout
falls back to configuration instead — see :func:`derive_particulate_pumps`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Key holding the JSON pump→solution map in ``rig_state``.
_LOADOUT_KEY = "pump_loadout"


@dataclass
class PumpLoadout:
    """What is loaded on each pump, keyed by physical pump index."""

    #: pump id -> solution name. A pump absent from this map is *undeclared*,
    #: which is a distinct state from "declared as empty" and is treated
    #: conservatively everywhere.
    by_pump: dict[int, str] = field(default_factory=dict)

    def solution_for(self, pump_id: int) -> str | None:
        return self.by_pump.get(int(pump_id))

    def declared_pumps(self) -> tuple[int, ...]:
        return tuple(sorted(self.by_pump))

    def is_empty(self) -> bool:
        return not self.by_pump

    def assign(self, pump_id: int, solution_name: str | None) -> None:
        """Bind a solution to a pump, or clear it with ``None``/empty."""
        pid = int(pump_id)
        if solution_name:
            self.by_pump[pid] = str(solution_name)
        else:
            self.by_pump.pop(pid, None)

    def describe(self) -> str:
        """Operator-facing summary, pump index first."""
        if self.is_empty():
            return "No stocks declared."
        return "; ".join(
            f"pump {pid}: {self.by_pump[pid]}" for pid in self.declared_pumps()
        )


def solution_is_particulate(solution: Any, chem_catalog: Any) -> bool:
    """Whether *solution* contains any chemical flagged as particulate."""
    if solution is None or chem_catalog is None:
        return False
    for component in getattr(solution, "components", []) or []:
        name = getattr(component, "chemical_name", None)
        if not name:
            continue
        try:
            chem = chem_catalog.get(name)
        except Exception:
            chem = None
        if chem is not None and bool(getattr(chem, "is_particulate", False)):
            return True
    return False


def derive_particulate_pumps(
    loadout: PumpLoadout,
    *,
    chem_catalog: Any,
    sol_catalog: Any,
    pumps: "tuple[int, ...]",
    fallback: "tuple[int, ...]",
) -> tuple[int, ...]:
    """Which pumps carry particulate stock, derived from declared chemistry.

    Precedence, and the reasoning for each rung:

    1. **Nothing declared at all** → *fallback* (the configured value). The
       operator has not opted into this mechanism, so behaviour must not change
       under them — and treating every line as particulate would silently raise
       consumption by half again.
    2. **Declared and resolvable** → derived from ``is_particulate``.
    3. **Partially declared** → an undeclared pump counts as **particulate**.
       Partial declaration is the genuinely dangerous state: the operator has
       engaged with the mechanism, so silence about one line is much more likely
       to be an oversight than an assertion that it is clean.
    4. **Declared but the solution is unknown to the catalog** → particulate,
       for the same reason, and logged loudly since it means the two records
       have drifted.
    """
    if loadout.is_empty():
        logger.info("particulate_pumps_from_config", pumps=list(fallback))
        return tuple(fallback)

    particulate: list[int] = []
    for pump_id in pumps:
        name = loadout.solution_for(pump_id)
        if not name:
            particulate.append(pump_id)
            logger.info(
                "pump_stock_undeclared", pump_id=pump_id,
                msg="treating as particulate — declare its stock to refine this",
            )
            continue
        try:
            solution = sol_catalog.get(name)
        except Exception:
            solution = None
        if solution is None:
            particulate.append(pump_id)
            logger.warning(
                "pump_stock_not_in_catalog", pump_id=pump_id, solution=name,
                msg="treating as particulate — the loadout and catalog disagree",
            )
            continue
        if solution_is_particulate(solution, chem_catalog):
            particulate.append(pump_id)

    logger.info("particulate_pumps_derived", pumps=particulate,
                loadout=loadout.describe())
    return tuple(particulate)


# ── Persistence ──────────────────────────────────────────────────────────────

def save_loadout(data_store: Any, loadout: PumpLoadout) -> None:
    """Persist the pump→solution map. Durable, not per-session."""
    if data_store is None:
        return
    try:
        payload = json.dumps({str(k): v for k, v in loadout.by_pump.items()})
        data_store._kv_set_text(_LOADOUT_KEY, payload)
    except Exception:
        logger.warning("pump_loadout_persist_failed", exc_info=True)
        return
    logger.info("pump_loadout_saved", loadout=loadout.describe())


def load_loadout(data_store: Any = None) -> PumpLoadout:
    """Read the pump→solution map; an unreadable record reads as undeclared."""
    if data_store is None:
        return PumpLoadout()
    try:
        raw = data_store._kv_get_text(_LOADOUT_KEY)
    except Exception:
        logger.warning("pump_loadout_read_failed", exc_info=True)
        return PumpLoadout()
    if not raw:
        return PumpLoadout()
    try:
        parsed = json.loads(raw)
        return PumpLoadout({int(k): str(v) for k, v in parsed.items() if v})
    except Exception:
        logger.warning("pump_loadout_parse_failed", raw=raw[:120], exc_info=True)
        return PumpLoadout()


def catalogs_from_data_root():
    """``(ChemicalCatalog, SolutionCatalog)`` from the data root.

    Degrades to empty catalogs on any failure. That is safe for the particulate
    resolver — an unresolvable solution counts as particulate, so a missing
    catalog purges more rather than less — but **not** for a thickness
    prediction, where an unresolvable stock would silently under-count the film.
    Callers in that position must check that every stock they cast resolved (see
    ``ExperimentBuilderTab._predicted_cast``) rather than trusting the return.

    Public because the HT tab needs the *chemical* catalog to elute against, and
    :func:`softae.core.composition_axes.stocks_from_loadout` returns only
    solutions.
    """
    from softae.config import loader
    from softae.core.formulation import ChemicalCatalog, SolutionCatalog

    try:
        root = loader.data_root()
        return (
            ChemicalCatalog.load_csv(root / "chemicals.csv"),
            SolutionCatalog.load_csv(root / "solutions.csv"),
        )
    except Exception:
        logger.warning("stock_catalogs_unreadable", exc_info=True)
        return ChemicalCatalog(), SolutionCatalog()


def resolve_particulate_pumps(
    settings: Any,
    *,
    data_store: Any = None,
    chem_catalog: Any = None,
    sol_catalog: Any = None,
) -> tuple[int, ...]:
    """Best available particulate-pump set for *settings*.

    One entry point so the GUI, the headless CLI, and the projection all agree.
    Any failure falls back to the configured value rather than guessing, because
    a wrong *derived* answer is worse than a stale declared one — the operator
    can at least see the config.
    """
    try:
        if chem_catalog is None or sol_catalog is None:
            loaded_chem, loaded_sol = catalogs_from_data_root()
            chem_catalog = chem_catalog or loaded_chem
            sol_catalog = sol_catalog or loaded_sol

        return derive_particulate_pumps(
            load_loadout(data_store),
            chem_catalog=chem_catalog,
            sol_catalog=sol_catalog,
            pumps=tuple(settings.pumps),
            fallback=tuple(settings.particulate_pumps),
        )
    except Exception:
        logger.warning("particulate_pumps_resolve_failed", exc_info=True)
        return tuple(settings.particulate_pumps)
