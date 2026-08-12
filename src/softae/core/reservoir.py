"""Stock-volume ledger — a **safety interlock**, not consumable bookkeeping.

Running a syringe dry does not dispense air: the plunger drives into a mechanical
dead-end hard stop, which is a hazard to the pump and the syringe. That reclassifies
volume tracking from housekeeping to rig protection, and is why this lands early
rather than with the rest of the consumables work.

Nothing in the system previously knew how much stock was actually left (see the
``res_vol`` warning below). This ledger makes that state persistent and enforces
two thresholds:

* **soft warn** — raise an alert, keep going (time to plan a refill);
* **hard stop** — refuse the dispense with :class:`SafetyError` before the plunger
  can reach its dead stop.

``SafetyError`` is deliberate: :meth:`AutonomousLoop._is_hard_fault` already treats
it as a park-immediately fault class, so a depleted reservoir stops the campaign
and drives the rig safe **without consuming retries** — retrying a mechanical
dead-end is exactly the wrong response.

**Accounting is conservative: stock is debited when the dispense is *commanded*,
not when it succeeds.** A command that fails midway may still have moved fluid, so
erring toward "less remaining than you think" is the safe direction for an
interlock whose failure mode is mechanical.

.. warning::

   **Never seed, initialise, or cross-check this ledger from ``single_pump``'s
   ``res_vol`` argument.**  Despite the name, ``res_vol`` is *not* a measure of
   stock on hand: it is written straight to the pump firmware as that pump's
   **declared syringe volume** (``{ID} svolume {res_vol} ml``), and by long-
   standing operator convention it is simply padded to comfortably exceed the
   elution volume of the command so the pump's own limit logic never trips a
   hardware stop mid-dispense.  Real volume checks were always visual.  Deriving
   the ledger from it would import a padded fiction into a safety interlock and
   silently defeat the hard stop — the failure would be invisible until a plunger
   hit its dead-end.  The ledger's only inputs are :meth:`refill` (an operator
   asserting a measured quantity) and the debits it makes itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import structlog

from softae.errors import SafetyError

if TYPE_CHECKING:
    from softae.core.data_store import DataStore

logger = structlog.get_logger(__name__)

#: Warn (but keep running) once a stock falls below this.
DEFAULT_SOFT_WARN_UL = 1000.0
#: Refuse to dispense below this — the margin that keeps the plunger off its stop.
DEFAULT_HARD_STOP_UL = 250.0


class LevelStore(Protocol):
    """The narrow persistence surface the ledger needs.

    Keeps the driver layer from depending on the whole experiment database — any
    object with these two methods works, which also makes the ledger trivial to
    test without SQLite.
    """

    def reservoir_level_uL(self, pump_id: int) -> float | None: ...
    def set_reservoir_level(self, pump_id: int, remaining_uL: float) -> None: ...


@dataclass(frozen=True)
class ReservoirStatus:
    """Result of debiting a dispense against a stock."""

    pump_id: int
    remaining_uL: float
    warned: bool = False        # crossed the soft threshold on this dispense


class ReservoirLedger:
    """Persistent per-pump stock levels with soft-warn / hard-stop enforcement.

    A pump with no recorded level is treated as **unknown, not empty**: unknown
    stocks are passed through untouched so that adding the ledger cannot break a
    rig whose reservoirs have never been declared. Call :meth:`refill` to bring a
    pump under management.
    """

    def __init__(
        self,
        store: "LevelStore | None" = None,
        *,
        soft_warn_uL: float = DEFAULT_SOFT_WARN_UL,
        hard_stop_uL: float = DEFAULT_HARD_STOP_UL,
        on_warn=None,
    ) -> None:
        if hard_stop_uL < 0:
            raise ValueError("hard_stop_uL must be >= 0")
        if soft_warn_uL < hard_stop_uL:
            raise ValueError("soft_warn_uL must be >= hard_stop_uL")
        self._store = store
        self.soft_warn_uL = float(soft_warn_uL)
        self.hard_stop_uL = float(hard_stop_uL)
        self._on_warn = on_warn
        self._levels: dict[int, float] = {}

    # ── State ───────────────────────────────────────────────────────────

    def remaining_uL(self, pump_id: int) -> float | None:
        """Stock left on *pump_id*, or ``None`` when it is unmanaged."""
        pid = int(pump_id)
        if pid in self._levels:
            return self._levels[pid]
        if self._store is not None:
            try:
                level = self._store.reservoir_level_uL(pid)
            except Exception:
                logger.warning("reservoir_read_failed", pump_id=pid, exc_info=True)
                return None
            if level is not None:
                self._levels[pid] = float(level)
                return self._levels[pid]
        return None

    def refill(self, pump_id: int, volume_uL: float) -> None:
        """Declare *pump_id* loaded with ``volume_uL`` (also brings it under management)."""
        pid, vol = int(pump_id), float(volume_uL)
        self._levels[pid] = vol
        self._persist(pid, vol)
        logger.info("reservoir_refilled", pump_id=pid, volume_uL=vol)

    def _persist(self, pump_id: int, remaining_uL: float) -> None:
        if self._store is None:
            return
        try:
            self._store.set_reservoir_level(pump_id, remaining_uL)
        except Exception:
            logger.warning("reservoir_persist_failed", pump_id=pump_id, exc_info=True)

    # ── Enforcement ─────────────────────────────────────────────────────

    def check(self, pump_id: int, dispense_uL: float, *, instrument: str = "syringe") -> None:
        """Raise :class:`SafetyError` if this dispense would breach the hard stop.

        Unmanaged pumps pass through untouched.
        """
        remaining = self.remaining_uL(pump_id)
        if remaining is None:
            return
        after = remaining - float(dispense_uL)
        if after < self.hard_stop_uL:
            raise SafetyError(
                f"Pump {pump_id}: dispensing {float(dispense_uL):.1f} µL would leave "
                f"{after:.1f} µL, below the {self.hard_stop_uL:.0f} µL hard stop "
                f"(running dry drives the plunger into its mechanical stop). "
                f"Refill before continuing.",
                instrument=instrument,
                requested=float(dispense_uL),
                limit=max(0.0, remaining - self.hard_stop_uL),
            )

    def debit(self, pump_id: int, dispense_uL: float) -> ReservoirStatus | None:
        """Deduct a commanded dispense. ``None`` for an unmanaged pump.

        Call :meth:`check` first — this does not enforce, it records.
        """
        remaining = self.remaining_uL(pump_id)
        if remaining is None:
            return None
        pid = int(pump_id)
        after = max(0.0, remaining - float(dispense_uL))
        was_above = remaining > self.soft_warn_uL
        self._levels[pid] = after
        self._persist(pid, after)

        warned = was_above and after <= self.soft_warn_uL
        if warned:
            logger.warning(
                "reservoir_low", pump_id=pid, remaining_uL=after,
                soft_warn_uL=self.soft_warn_uL, hard_stop_uL=self.hard_stop_uL,
            )
            if self._on_warn is not None:
                try:
                    self._on_warn(pid, after)
                except Exception:
                    logger.warning("reservoir_warn_hook_failed", exc_info=True)
        return ReservoirStatus(pump_id=pid, remaining_uL=after, warned=warned)

    def check_and_debit(
        self, pump_id: int, dispense_uL: float, *, instrument: str = "syringe"
    ) -> ReservoirStatus | None:
        """Enforce the hard stop, then record the draw. The normal entry point."""
        self.check(pump_id, dispense_uL, instrument=instrument)
        return self.debit(pump_id, dispense_uL)


# ── Wiring ──────────────────────────────────────────────────────────────

def attach_reservoir_ledger(
    manager,
    data_store: "DataStore | None" = None,
    *,
    config: dict | None = None,
    instrument: str = "syringe",
) -> "ReservoirLedger | None":
    """Build a ledger, wire its soft-warn to the alert seam, and attach it.

    One entry point for every host — GUI, headless CLI, tests — so the interlock
    cannot be live on one surface and inert on another. Returns the attached
    ledger, or ``None`` if there is no syringe to attach it to (a rig configured
    without one is not an error).

    Thresholds come from ``[safety]``: ``reservoir_soft_warn_uL`` and
    ``reservoir_hard_stop_uL``. **Levels do not** — they are read from
    *data_store*, and are only ever written by :meth:`ReservoirLedger.refill`
    (an operator asserting a measured quantity). In particular they are never
    derived from ``single_pump``'s ``res_vol``; see this module's docstring.
    """
    try:
        syringe = manager.get(instrument)
    except Exception:
        logger.info("reservoir_ledger_no_syringe", instrument=instrument)
        return None
    if syringe is None:
        return None

    if config is None:
        try:
            from softae.config import loader

            config = loader.safety()
        except Exception:
            config = {}

    def _threshold(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return default

    soft = _threshold("reservoir_soft_warn_uL", DEFAULT_SOFT_WARN_UL)
    hard = _threshold("reservoir_hard_stop_uL", DEFAULT_HARD_STOP_UL)
    if soft < hard:
        logger.warning(
            "reservoir_thresholds_inverted", soft_warn_uL=soft, hard_stop_uL=hard,
            msg="falling back to defaults",
        )
        soft, hard = DEFAULT_SOFT_WARN_UL, DEFAULT_HARD_STOP_UL

    def _on_warn(pump_id: int, remaining_uL: float) -> None:
        # Durable, because an unattended run may cross this at 3 a.m. and the
        # in-process event stream dies with its host.
        from softae.core.alerts import WARNING, Alert, raise_alert

        raise_alert(
            Alert(
                kind="reservoir",
                severity=WARNING,
                message=(
                    f"Pump {pump_id} stock low: {remaining_uL:.0f} µL left "
                    f"(hard stop at {hard:.0f} µL). Refill soon."
                ),
                details={"pump_id": pump_id, "remaining_uL": remaining_uL,
                         "hard_stop_uL": hard},
            ),
            data_store=data_store,
        )

    ledger = ReservoirLedger(
        data_store, soft_warn_uL=soft, hard_stop_uL=hard, on_warn=_on_warn
    )
    syringe.reservoir_ledger = ledger
    logger.info(
        "reservoir_ledger_attached", instrument=instrument,
        soft_warn_uL=soft, hard_stop_uL=hard, persistent=data_store is not None,
    )
    return ledger
