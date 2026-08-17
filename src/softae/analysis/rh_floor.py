"""The enclosure's attainable RH floor, read out of data the rig already has.

A setpoint below what the chamber can deliver hot saturates the humidity PID: the
PV sits above the command indefinitely with nothing broken, because the flush
basin holds water inside the heated enclosure. ``drivers.contracts`` grades that
state live (:data:`~softae.drivers.contracts.RH_FLOOR_LIMITED`) and refuses to
park a run on it. That is the safety net. **This module is the durable fix's
first half:** the floor as a function of chamber temperature is already latent in
``conditions``, which stores ``rh_sp_pct`` and ``rh_pv_pct`` beside the resolved
temperature at every EIS measurement. It needs a query, not an experiment.

Two properties, deliberate:

**Read-only by construction.** ``mode=ro`` makes SQLite itself refuse every write
on this handle, and :class:`~softae.core.data_store.DataStore` is never
constructed — its ``__init__`` runs DDL and eight migrations, and idempotent is
not read-only.

**A ``MIN(rh_pv_pct)`` is not automatically a floor.** It is a floor only where a
setpoint at or below it was actually *commanded* and the controller still failed
to reach it. A bin whose lowest commanded setpoint was 40 %RH says nothing about
how dry the enclosure could get — it says the operator never asked. That
distinction is :attr:`TemperatureBin.saturated`, and without it the naive query
reads room humidity as a hard physical limit.

The temperature binned on is ``conditions.temperature_C``, the answer
:func:`~softae.analysis.conditions.resolve_temperature_C` recorded at write time.
That precedence is **not** re-implemented here (see that module's warning); the
source label it was recorded with is carried through per bin instead, and a bin
fed by more than one thermometer is labelled
:data:`~softae.analysis.conditions.TEMPERATURE_MIXED` rather than inheriting
whichever row came first.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import structlog

from softae.analysis.conditions import TEMPERATURE_MIXED, TEMPERATURE_UNAVAILABLE

logger = structlog.get_logger(__name__)

#: Bin width in °C. Wide, because the floor is a property of a warm enclosure
#: rather than of a particular setpoint, and the sweep temperatures the rig
#: actually visits (27.5 / 45 / 65 / 85) separate cleanly at this width.
DEFAULT_BIN_WIDTH_C = 10.0


@dataclass(frozen=True)
class TemperatureBin:
    """The lowest %RH observed in one temperature bin, and whether it means anything."""

    temperature_C: float
    #: ``MIN(rh_pv_pct)`` in the bin.
    rh_floor_pct: float
    #: The lowest %RH anyone actually asked for in the bin.
    rh_setpoint_min_pct: float
    n_rows: int
    temperature_source: str = TEMPERATURE_UNAVAILABLE

    @property
    def saturated(self) -> bool:
        """True when the controller was asked for less than it delivered.

        Only then is :attr:`rh_floor_pct` a measurement of the enclosure's limit.
        False means the setpoint was met and the real floor is somewhere at or
        below this number, unprobed.
        """
        return self.rh_floor_pct > self.rh_setpoint_min_pct

    def describe(self) -> str:
        if self.saturated:
            return (f"{self.temperature_C:g} C: floor {self.rh_floor_pct:.1f} %RH "
                    f"(asked for {self.rh_setpoint_min_pct:.1f}, n={self.n_rows}, "
                    f"{self.temperature_source}) -- do not command below this")
        return (f"{self.temperature_C:g} C: reached {self.rh_floor_pct:.1f} %RH, the "
                f"lowest commanded (n={self.n_rows}, {self.temperature_source}) -- "
                f"floor not probed")


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """The one place a connection is made, and SQLite itself refuses the write."""
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def resolve_db_path(path: Path | str) -> Path:
    """Accept either the ``softae.db`` file or the project directory holding it."""
    p = Path(path)
    return p if p.suffix == ".db" else p / "db" / "softae.db"


def rh_floor_by_temperature(
    path: Path | str,
    *,
    bin_width_C: float = DEFAULT_BIN_WIDTH_C,
    run_id: str | None = None,
) -> list[TemperatureBin]:
    """Lowest %RH observed per temperature bin, coldest first.

    A bin with no rows is **absent**, never zero: an unvisited temperature has no
    floor, and a zero there would read as a perfectly dry enclosure.
    """
    width = float(bin_width_C)
    if width <= 0:
        raise ValueError(f"bin_width_C must be positive; got {bin_width_C!r}")

    sql = (
        "SELECT ROUND(temperature_C / ?) * ? AS t_bin, "
        "       MIN(rh_pv_pct), MIN(rh_sp_pct), COUNT(*), "
        "       MIN(temperature_source), MAX(temperature_source) "
        "FROM conditions "
        "WHERE rh_sp_pct IS NOT NULL AND rh_pv_pct IS NOT NULL "
        "  AND temperature_C IS NOT NULL"
    )
    params: list[object] = [width, width]
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    sql += " GROUP BY t_bin ORDER BY t_bin"

    conn = _connect_ro(resolve_db_path(path))
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    bins = [
        TemperatureBin(
            temperature_C=float(t_bin),
            rh_floor_pct=float(rh_min),
            rh_setpoint_min_pct=float(sp_min),
            n_rows=int(n),
            temperature_source=(str(src_lo) if src_lo == src_hi
                                else TEMPERATURE_MIXED),
        )
        for t_bin, rh_min, sp_min, n, src_lo, src_hi in rows
    ]
    logger.info("rh_floor_binned", n_bins=len(bins), bin_width_C=width, run_id=run_id)
    return bins


def describe_floor(bins: list[TemperatureBin]) -> str:
    """One line per bin, plus the sentence an operator actually needs."""
    if not bins:
        return ("No conditions rows carry both an RH setpoint and a PV: the floor "
                "has never been observed on this project.")
    lines = [b.describe() for b in bins]
    probed = [b for b in bins if b.saturated]
    if probed:
        worst = max(probed, key=lambda b: b.rh_floor_pct)
        lines.append(
            f"Worst observed floor: {worst.rh_floor_pct:.1f} %RH at "
            f"{worst.temperature_C:g} C. Commanding below that there buys an unmet "
            f"setpoint, not a drier chamber.")
    else:
        lines.append("No bin was ever asked for less than it delivered, so no floor "
                     "has been probed yet.")
    return "\n".join(lines)
