"""Anti-clog purge scheduling (P8).

A syringe line carrying a **particulate** stock has a check valve that clogs when
the stock sits stagnant — observable on a ~10-minute timescale. Two things follow
that are easy to get wrong:

* **Being mid-run is not protection.** A campaign can run many trials without
  drawing meaningfully from the particulate stock, so the line clogs during an
  otherwise healthy run.
* **All lines purge together.** Purging only the particulate line would make one
  of the others the new stagnation site. Volumes differ (the particulate line
  gets more); the *schedule* does not.

This module decides **when** and **how much**. It deliberately does not actuate:
the caller owns the hardware, which keeps the policy testable against a virtual
clock and keeps a background timer from ever holding a pump directly.

Scheduling is *due-based, not interrupting*. :meth:`PurgeScheduler.due` reports
that a purge is owed; the caller performs it at the next safe boundary — an
anneal or measurement interim, or alongside a precondition flush. The interval is
a **floor**, not a deadline, so a purge is never a reason to cut into a cast.

Every dispense the rig makes counts, so :meth:`note_dispense` resets a pump's
timer: a pump the campaign just used does not need purging, which is what keeps
the consumption bill down during an active run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Operator-set defaults (2026-07-31). Interval is the observed stagnation
#: onset with margin; volumes are what the bench found sufficient.
DEFAULT_INTERVAL_S = 900.0
DEFAULT_PARTICULATE_UL = 20.0
DEFAULT_OTHER_UL = 10.0


@dataclass
class PurgeSettings:
    """Everything reconfigurable about purging, in one place.

    Kept as a single value object so the operator surface, the config file, and
    the consumption projection all read the *same* numbers — the alternative is
    three places that drift, which is the failure P2 spent its effort removing.
    """

    #: Purging is *planned* — the schedule is live and its consumption counts
    #: toward the stock/waste runway.
    enabled: bool = True
    #: Purging actually *moves fluid*. Deliberately separate from
    #: :attr:`enabled` and shipped **off**: this is the only mechanism in the
    #: system that actuates hardware with nobody asking it to, so it stays
    #: inert until an operator turns it on at the bench. With ``enabled`` on and
    #: this off, the schedule runs and logs what it *would* purge — and the
    #: projection still bills the consumption, so the runway is not flattered by
    #: the fact that nothing is dispensing yet.
    actuate: bool = False
    interval_s: float = DEFAULT_INTERVAL_S
    particulate_uL: float = DEFAULT_PARTICULATE_UL
    other_uL: float = DEFAULT_OTHER_UL
    #: Operator-confirmed 2026-08-02. Pump 1 is deliberately *not* one of
    #: ``precondition_flush``'s ``plug_ids`` (0 and 2), which push a fixed plug
    #: every channel — so this line only moves when its own component is
    #: non-zero, and a trial that zeroes it correctly leaves it needing a purge.
    particulate_pumps: tuple[int, ...] = (1,)
    pumps: tuple[int, ...] = (0, 1, 2)

    def volume_for(self, pump_id: int) -> float:
        """Purge volume for one pump."""
        return (
            self.particulate_uL
            if int(pump_id) in self.particulate_pumps
            else self.other_uL
        )

    def per_purge_uL(self) -> dict[int, float]:
        return {int(p): self.volume_for(p) for p in self.pumps}

    def uL_per_day(self) -> dict[int, float]:
        """Consumption rate per pump — what the preflight projection needs.

        Expressed per *day* because purging accrues with elapsed time, not with
        iterations; a multi-day campaign's purge bill can exceed its trial draw.
        """
        if not self.enabled or self.interval_s <= 0:
            return {}
        per_day = 86400.0 / float(self.interval_s)
        return {p: v * per_day for p, v in self.per_purge_uL().items()}

    def total_uL_per_day(self) -> float:
        return sum(self.uL_per_day().values())

    def describe(self) -> str:
        """One line an operator can sanity-check the settings against."""
        if not self.enabled:
            return "Anti-clog purging is off."
        prefix = "" if self.actuate else "[not actuating] "
        mins = self.interval_s / 60.0
        parts = ", ".join(
            f"pump {p} {self.volume_for(p):.0f} µL"
            f"{' (particulate)' if p in self.particulate_pumps else ''}"
            for p in self.pumps
        )
        return (
            f"{prefix}Purge every {mins:.0f} min: {parts}. "
            f"About {self.total_uL_per_day() / 1000.0:.1f} mL/day total."
        )

    def validated(self) -> "PurgeSettings":
        """Return self, or raise if the settings cannot work."""
        if self.enabled:
            if self.interval_s <= 0:
                raise ValueError("purge interval must be > 0 s")
            if self.particulate_uL < 0 or self.other_uL < 0:
                raise ValueError("purge volumes must be >= 0 µL")
            if not self.pumps:
                raise ValueError("purge needs at least one pump")
        return self


def purge_settings(config: dict[str, Any] | None = None) -> PurgeSettings:
    """Read ``[purge]`` — the single parse point for these numbers."""
    if config is None:
        try:
            from softae.config import loader

            config = loader.load().get("purge", {}) or {}
        except Exception:
            config = {}

    def _f(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _ids(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
        raw = config.get(key, default)
        try:
            return tuple(int(v) for v in raw)
        except (TypeError, ValueError):
            return default

    return PurgeSettings(
        enabled=bool(config.get("enabled", True)),
        actuate=bool(config.get("actuate", False)),
        interval_s=_f("interval_s", DEFAULT_INTERVAL_S),
        particulate_uL=_f("particulate_uL", DEFAULT_PARTICULATE_UL),
        other_uL=_f("other_uL", DEFAULT_OTHER_UL),
        particulate_pumps=_ids("particulate_pumps", (1,)),
        pumps=_ids("pumps", (0, 1, 2)),
    )


#: Keys used for the operator's durable overrides in ``rig_state``.
_OVERRIDE_KEYS = ("interval_s", "particulate_uL", "other_uL")


def save_purge_settings(data_store: Any, settings: PurgeSettings) -> None:
    """Persist the operator's purge numbers.

    Precedence is deliberately simple and one-directional: **config supplies the
    defaults, an operator override wins.** The bench is where the clogging
    behaviour is actually observed, so someone at the rig must be able to retune
    the interval without editing a TOML file — but the override is durable and
    visible rather than a per-session tweak that silently reverts.
    """
    settings.validated()
    try:
        data_store._kv_set("purge_interval_s", settings.interval_s)
        data_store._kv_set("purge_particulate_uL", settings.particulate_uL)
        data_store._kv_set("purge_other_uL", settings.other_uL)
        data_store._kv_set("purge_enabled", 1.0 if settings.enabled else 0.0)
    except Exception:
        logger.warning("purge_settings_persist_failed", exc_info=True)
        return
    logger.info("purge_settings_saved", **{k: getattr(settings, k)
                                           for k in _OVERRIDE_KEYS})


def load_purge_settings(data_store: Any = None) -> PurgeSettings:
    """Config defaults, with any durable operator override applied on top."""
    settings = purge_settings()
    if data_store is None:
        return settings

    try:
        interval = data_store._kv_get("purge_interval_s")
        particulate = data_store._kv_get("purge_particulate_uL")
        other = data_store._kv_get("purge_other_uL")
        enabled = data_store._kv_get("purge_enabled")
    except Exception:
        logger.warning("purge_settings_read_failed", exc_info=True)
        return settings

    if interval is not None:
        settings.interval_s = float(interval)
    if particulate is not None:
        settings.particulate_uL = float(particulate)
    if other is not None:
        settings.other_uL = float(other)
    if enabled is not None:
        settings.enabled = bool(enabled)
    return settings


@dataclass
class PurgeDue:
    """A purge that is owed, and what it should dispense."""

    volumes_uL: dict[int, float]
    overdue_s: float

    @property
    def total_uL(self) -> float:
        return sum(self.volumes_uL.values())


class PurgeScheduler:
    """Tracks when each line last moved and reports when a purge is owed.

    Per-pump timers rather than one global timer: a pump the campaign has just
    used needs no purge, and skipping those is what keeps an active run from
    paying the full idle consumption rate.
    """

    def __init__(
        self,
        settings: PurgeSettings | None = None,
        *,
        now: Any = None,
    ) -> None:
        self.settings = (settings or purge_settings()).validated()
        self._now = now or time.monotonic
        self._last: dict[int, float] = {}
        start = self._now()
        for pump_id in self.settings.pumps:
            self._last[int(pump_id)] = start

    def note_dispense(self, pump_id: int) -> None:
        """Any dispense counts — the line just moved, so its timer resets."""
        self._last[int(pump_id)] = self._now()

    def note_purged(self, pump_ids=None) -> None:
        """Record that a purge was performed."""
        now = self._now()
        for pump_id in (pump_ids if pump_ids is not None else self.settings.pumps):
            self._last[int(pump_id)] = now

    def seconds_since(self, pump_id: int) -> float:
        return self._now() - self._last.get(int(pump_id), self._now())

    def due(self) -> PurgeDue | None:
        """A purge is owed, or ``None``.

        Purges **all** configured pumps whenever *any* line is due: leaving the
        others idle through a purge cycle would make one of them the next
        stagnation site.
        """
        if not self.settings.enabled:
            return None

        overdue = [
            self.seconds_since(p) - self.settings.interval_s
            for p in self.settings.pumps
        ]
        worst = max(overdue, default=-1.0)
        if worst < 0:
            return None
        return PurgeDue(self.settings.per_purge_uL(), overdue_s=worst)

    def next_due_in_s(self) -> float | None:
        """Seconds until the next purge is owed (0 if already), or ``None``."""
        if not self.settings.enabled:
            return None
        remaining = [
            self.settings.interval_s - self.seconds_since(p)
            for p in self.settings.pumps
        ]
        return max(0.0, min(remaining, default=0.0))


def attach_purge_scheduler(
    manager: Any,
    settings: "PurgeSettings | None" = None,
    *,
    data_store: Any = None,
    instrument: str = "syringe",
) -> "PurgeScheduler | None":
    """Attach a scheduler that observes every dispense the rig makes.

    One entry point for every host — GUI, headless CLI, tests — mirroring
    :func:`softae.core.reservoir.attach_reservoir_ledger`, and sited at the same
    choke point (:meth:`PumpSafetyMixin._validate_single_pump`) so no dispense
    path can be forgotten.

    Without this the per-pump timers never see the campaign's own dispensing, and
    the harness would purge at the full idle rate *during* an active run — paying
    the whole consumption bill for lines that had just moved anyway.

    Returns the attached scheduler, or ``None`` if there is no syringe (a rig
    configured without one is not an error).
    """
    try:
        syringe = manager.get(instrument)
    except Exception:
        logger.info("purge_scheduler_no_syringe", instrument=instrument)
        return None
    if syringe is None:
        return None

    if settings is None:
        settings = load_purge_settings(data_store)

    # Derive which lines carry particulates from the declared stock rather than
    # from the hand-maintained config list — which was found wrong (it named
    # pump 0; the particulate line is pump 1). Falls back to the config when
    # nothing is declared, so behaviour does not change under an operator who
    # has not opted in. See `core.stock_assignment`.
    from softae.core.stock_assignment import resolve_particulate_pumps

    derived = resolve_particulate_pumps(settings, data_store=data_store)
    if tuple(derived) != tuple(settings.particulate_pumps):
        logger.info(
            "particulate_pumps_overridden_by_loadout",
            configured=list(settings.particulate_pumps), derived=list(derived),
        )
        settings.particulate_pumps = tuple(derived)

    scheduler = PurgeScheduler(settings)
    syringe.purge_scheduler = scheduler
    logger.info(
        "purge_scheduler_attached", instrument=instrument,
        enabled=settings.enabled, actuate=settings.actuate,
        interval_s=settings.interval_s, uL_per_day=settings.total_uL_per_day(),
    )
    return scheduler
