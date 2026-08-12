"""Shared mock/real driver contract helpers.

The real drivers' public signatures and semantics are canonical; the mock
drivers must mirror them exactly.  The safety-validation and config-parsing
logic that both sides of each mock/real pair share lives here so the two
implementations cannot drift.

Contents
--------
- :class:`ParallelSyringeMixin` — parallel-syringe config parsing, volume
  splitting, and pump rate/volume safety checks (MockSyringe / AsyncSyringe).
- :func:`check_stage_bounds` — stage travel-limit check (MockStage / AsyncStage).
- :func:`validate_temp_setpoint` — temperature setpoint limit check
  (MockTempController / AsyncTempController).
- :func:`validate_rh_setpoint` — relative-humidity setpoint cap
  (MockRHController / AsyncRHController).
- :func:`apply_piezo_profile` — combined frequency + sweep profile application
  (MockPiezoController / AsyncPiezoController).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

from softae.errors import SafetyError

logger = structlog.get_logger(__name__)

#: Number of addressable pumps on the Harvard Apparatus chain.
N_PUMPS = 3

#: A commanded dispense volume at or below this (µL) means "leave this pump
#: alone": no rate/volume validation and no hardware command are issued.  This
#: lets a formulation zero out a component (e.g. elute only two of three stocks)
#: without a 0-volume pump — whose proportional rate is also 0 — tripping the
#: ``min_rate`` safety limit.  Enforced at the :meth:`single_pump` choke point so
#: it covers every pump actuation (deposit, flush, precondition, manual).
PUMP_NOOP_VOLUME_UL = 0.0


# ── Syringe ──────────────────────────────────────────────────────────────────

class ParallelSyringeMixin:
    """Parallel-syringe bookkeeping shared by MockSyringe and AsyncSyringe.

    Host classes must call :meth:`_init_parallel_syringes` in ``__init__``
    and define ``self.name``, ``self._max_rate`` and ``self._min_rate``
    before using :meth:`_validate_single_pump`.
    """

    _parallel_syringes: int
    _parallel_syringes_by_pump: dict[int, int]

    def _init_parallel_syringes(self, config: dict[str, Any]) -> None:
        """Parse ``parallel_syringes`` / ``parallel_syringes_pump<N>`` config."""
        self._parallel_syringes = max(1, int(config.get("parallel_syringes", 1)))
        self._parallel_syringes_by_pump = {}
        for pump_id in range(N_PUMPS):
            key = f"parallel_syringes_pump{pump_id}"
            try:
                self._parallel_syringes_by_pump[pump_id] = max(
                    1, int(config.get(key, self._parallel_syringes))
                )
            except (TypeError, ValueError):
                self._parallel_syringes_by_pump[pump_id] = self._parallel_syringes

    def set_parallel_syringes(self, n: int, pump_id: int | None = None) -> None:
        """Set active parallel syringe count used for volume splitting.

        When *pump_id* is ``None``, the count is applied to all pumps.
        """
        value = int(n)
        if value < 1:
            raise ValueError("parallel syringe count must be >= 1")
        if pump_id is None:
            self._parallel_syringes = value
            for pid in range(N_PUMPS):
                self._parallel_syringes_by_pump[pid] = value
        else:
            self._parallel_syringes_by_pump[int(pump_id)] = value

    def effective_per_syringe_volume(self, commanded_uL: float, pump_id: int | None = None) -> float:
        """Return per-syringe hardware volume for a total commanded volume."""
        if pump_id is None:
            count = self._parallel_syringes
        else:
            count = self._parallel_syringes_by_pump.get(int(pump_id), self._parallel_syringes)
        return float(commanded_uL) / float(count)

    def _is_noop_pump_command(self, dispense_vol: float) -> bool:
        """True if *dispense_vol* means "leave this pump alone" (≤ the no-op floor).

        A no-op command is skipped entirely by :meth:`single_pump` — no safety
        validation and no hardware write — so a zeroed formulation component does
        not trip the ``min_rate`` limit.  Shared by MockSyringe and AsyncSyringe
        so both sides skip on exactly the same condition.
        """
        try:
            return float(dispense_vol) <= PUMP_NOOP_VOLUME_UL
        except (TypeError, ValueError):
            return False

    def _validate_single_pump(
        self,
        res_vol: float,
        rate: float,
        dispense_vol: float,
        pump_id: int | None = None,
    ) -> None:
        """Raise :class:`SafetyError` for out-of-limit rate or volume requests.

        Mirrors the three-branch validation of ``AsyncSyringe.single_pump``, and
        additionally enforces the stateful stock interlock when a
        ``reservoir_ledger`` is attached (``pump_id`` identifies the stock).

        ``res_vol`` is the **declared syringe volume written to the pump
        firmware** (``{ID} svolume {res_vol} ml``), *not* a measure of stock on
        hand.  By operator convention it is padded to comfortably exceed the
        command's elution volume purely so the pump's own limit logic does not
        trip a hardware stop mid-dispense; real volume checks were always
        visual.  The ``dispense_vol > res_vol`` branch below is therefore a
        pump-firmware sanity check and carries no consumables meaning — the
        ledger is the sole authority on remaining stock, and the two are
        deliberately independent.  See :mod:`softae.core.reservoir`.
        """
        if rate > self._max_rate:
            raise SafetyError(
                f"Pump rate {rate} µL/min exceeds max {self._max_rate}",
                instrument=self.name,
                requested=rate,
                limit=self._max_rate,
            )
        if rate < self._min_rate:
            raise SafetyError(
                f"Pump rate {rate} µL/min below min {self._min_rate}",
                instrument=self.name,
                requested=rate,
                limit=self._min_rate,
            )
        # Pump-firmware sanity check only: res_vol is the *declared syringe
        # volume* sent to the pump, not the stock actually loaded. Remaining
        # stock is the ledger's business, below.
        if dispense_vol > res_vol * 1000:
            raise SafetyError(
                f"Dispense volume {dispense_vol} µL exceeds declared syringe "
                f"volume {res_vol} mL ({res_vol * 1000} µL)",
                instrument=self.name,
                requested=dispense_vol,
                limit=res_vol * 1000,
            )

        # Stateful stock interlock. The check above only validates one call
        # against a caller-supplied number; the ledger knows what is *actually*
        # left and refuses before the plunger can reach its mechanical stop.
        # Sited here because every dispense — HT, campaign, manual, CLI — passes
        # through this one validator. Debiting on *command* (not on success) is
        # deliberate: a partially-completed dispense may still have moved fluid.
        ledger = getattr(self, "reservoir_ledger", None)
        if ledger is not None:
            ledger.check_and_debit(pump_id, dispense_vol, instrument=self.name)

        # Anti-clog observation (P8). A line that just moved is not stagnating,
        # so any dispense resets its purge timer — which is what stops an active
        # campaign from paying the full idle purge rate for lines it was already
        # using. Deliberately *after* the ledger: if the stock interlock refuses,
        # no fluid moves and the timer must keep running.
        #
        # Noted on command rather than on success, matching the ledger. The two
        # are wrong in opposite directions when a dispense fails mid-way, and
        # both errors are bounded and benign: the ledger over-counts stock spent
        # (conservative), while this may defer one purge by at most one interval.
        scheduler = getattr(self, "purge_scheduler", None)
        if scheduler is not None and pump_id is not None:
            try:
                scheduler.note_dispense(pump_id)
            except Exception:      # observation must never fail a dispense
                pass


# ── Stage ────────────────────────────────────────────────────────────────────

def check_stage_bounds(
    x: float,
    y: float,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    instrument: str,
) -> None:
    """Raise :class:`SafetyError` if ``(x, y)`` is outside stage bounds."""
    if x < x_min or x > x_max:
        raise SafetyError(
            f"X={x} mm outside stage bounds [{x_min}, {x_max}]",
            instrument=instrument,
            requested=x,
            limit=x_max if x > x_max else x_min,
        )
    if y < y_min or y > y_max:
        raise SafetyError(
            f"Y={y} mm outside stage bounds [{y_min}, {y_max}]",
            instrument=instrument,
            requested=y,
            limit=y_max if y > y_max else y_min,
        )


def check_head_clear_to_move(syringe: Any, *, instrument: str) -> None:
    """Raise :class:`SafetyError` if the dispenser head is down.

    A lowered head is immersed — in a well, the wick pad, or the flush basin.
    Translating the stage under it drags the tip across whatever lies between.

    This was previously safe only *by convention*: every hand-written sequence
    happened to call ``head_retract`` before moving, and head-down was always a
    brief interval inside a known sequence. **That convention breaks the moment
    head-down becomes the rig's resting state** (P8 idle rest, and the anneal
    hold in the flush basin), because a path that forgets to retract now finds
    the head down by default rather than by accident.

    Sited beside :func:`check_stage_bounds` so mock and real stages cannot drift.

    Callers that genuinely intend to move while lowered — the in-drop mixing
    pattern is the only one today — must say so explicitly via
    ``move_to(..., head_may_be_down=True)`` rather than being silently exempt.
    """
    if syringe is None:
        return                      # no syringe configured: nothing to protect
    is_up = getattr(syringe, "is_head_up", None)
    if not callable(is_up):
        return                      # driver does not track it; do not invent a belief
    try:
        lowered = not is_up()
    except Exception:
        return                      # unreadable state must not block a move
    if lowered:
        raise SafetyError(
            "Stage move refused: the dispenser head is lowered. Retract it "
            "first, or pass head_may_be_down=True if the move is deliberate "
            "(e.g. in-drop mixing).",
            instrument=instrument,
        )


# ── Temperature controller ───────────────────────────────────────────────────

def temp_setpoint_limits(
    config: dict[str, Any] | None = None,
    safety_config: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """``(min_C, max_C)`` for the temperature setpoint — the single parse point.

    Two sources declare these: ``[safety] temp_min_C/temp_max_C`` and the
    instrument's own ``min_temp``/``max_temp``. **The narrower window wins** — the
    highest floor and the lowest ceiling — rather than one source taking precedence.

    That is deliberate and is *not* the precedence pattern
    :mod:`softae.core.dropcast` uses for pump rates. A pump rate is a capability; a
    thermal limit is an interlock, and an interlock must only ever be tightenable.
    Under plain precedence, a global ``[safety] temp_max_C = 200`` would silently
    *raise* the ceiling of a controller that declared a stricter 150 °C — adding a
    second limit would have loosened the first, which is the one thing a safety
    limit must never do.

    **Why both exist.** Until 2026-08-05 the ``[safety]`` pair was read by nothing,
    and because it held values identical to the instrument's, the disconnection had
    no visible symptom. An operator lowering ``[safety] temp_max_C`` to 80 would have
    read that as capping the heater and it would have changed nothing.
    """
    if safety_config is None:
        try:
            from softae.config.loader import safety

            safety_config = safety()
        except Exception:
            safety_config = {}
    inst = config or {}

    def _values(primary: str, fallback: str) -> list[float]:
        out: list[float] = []
        for source, key in ((safety_config, primary), (inst, fallback)):
            if key in (source or {}):
                try:
                    out.append(float(source[key]))
                except (TypeError, ValueError):
                    continue
        return out

    mins = _values("temp_min_C", "min_temp")
    maxes = _values("temp_max_C", "max_temp")
    return (max(mins) if mins else 5.0, min(maxes) if maxes else 200.0)


def validate_temp_setpoint(T_SP: float, config: dict[str, Any], instrument: str) -> None:
    """Enforce the min/max temperature-setpoint limits.

    Mirrors ``AsyncTempController.write_sp`` (limits are re-read on every call).
    See :func:`temp_setpoint_limits` for where they come from and why.
    """
    min_temp, max_temp = temp_setpoint_limits(config)
    if T_SP > max_temp:
        raise SafetyError(
            f"Setpoint {T_SP} °C exceeds max allowed {max_temp} °C",
            instrument=instrument,
            requested=T_SP,
            limit=max_temp,
        )
    if T_SP < min_temp:
        raise SafetyError(
            f"Setpoint {T_SP} °C below min allowed {min_temp} °C",
            instrument=instrument,
            requested=T_SP,
            limit=min_temp,
        )


# ── Anneal hold watchdog ─────────────────────────────────────────────────────

#: Deviation from the anneal target that raises a warning but keeps holding (°C).
DEFAULT_ANNEAL_WARN_C = 3.0
#: Sustained deviation that aborts the hold (°C).
DEFAULT_ANNEAL_FAULT_C = 10.0
#: How long a deviation (or an unreadable PV) must persist before it is a fault.
#: Short excursions are normal — a lid opened, a transient — and must not kill a
#: multi-hour anneal; a sustained one means the sample is no longer being made.
DEFAULT_ANNEAL_GRACE_S = 120.0
#: Interval between PV samples during a hold (s).
DEFAULT_ANNEAL_POLL_S = 30.0


@dataclass
class HoldReport:
    """Outcome of a monitored anneal hold."""

    held_s: float
    n_samples: int
    pv_min: float | None = None
    pv_max: float | None = None
    n_warn: int = 0
    aborted: bool = False

    @property
    def excursion_C(self) -> float | None:
        """Peak-to-peak PV spread observed during the hold."""
        if self.pv_min is None or self.pv_max is None:
            return None
        return self.pv_max - self.pv_min


def monitored_hold(
    hold_time_s: float,
    *,
    read_pv: Any,
    target_C: float,
    instrument: str,
    warn_C: float = DEFAULT_ANNEAL_WARN_C,
    fault_C: float = DEFAULT_ANNEAL_FAULT_C,
    grace_s: float = DEFAULT_ANNEAL_GRACE_S,
    poll_interval_s: float = DEFAULT_ANNEAL_POLL_S,
    on_warn: Any = None,
    should_abort: Any = None,
    sleep: Any = None,
    now: Any = None,
) -> HoldReport:
    """Hold at temperature while watching the process value.

    Replaces a bare ``time.sleep(hold_time_s)``. Soft-material anneals run for
    *hours*, and an unwatched sleep has two failure modes that both look like
    success: a heater that drifts, sticks, or loses its setpoint produces a
    wrongly-annealed sample that is then measured and recorded as valid data;
    and an abort cannot interrupt the sleep, so stopping a campaign mid-anneal
    waits out the full hold.

    Policy is graded, because the response to a 2 °C wobble and to a dead heater
    are not the same:

    * within ``warn_C`` — normal;
    * beyond ``warn_C`` — warn once per excursion via *on_warn*, keep holding;
    * beyond ``fault_C`` **continuously for** ``grace_s`` — raise
      :class:`SafetyError`, which the autonomous loop treats as a park-immediately
      fault class (retrying a thermal fault cannot help);
    * PV unreadable for ``grace_s`` — also a fault. "We cannot verify the hold"
      is not a safe state for an unattended run: the alternative is asserting a
      sample was annealed correctly with no evidence.

    A momentary excursion resets the grace timer, so only sustained faults abort.
    *sleep* and *now* are injectable for testing.
    """
    _sleep = sleep or time.sleep
    _now = now or time.monotonic

    hold_time_s = max(0.0, float(hold_time_s))
    poll = max(0.1, float(poll_interval_s))

    start = _now()
    deadline = start + hold_time_s
    bad_since: float | None = None
    was_warning = False
    n_warn = 0
    n_samples = 0
    pv_min: float | None = None
    pv_max: float | None = None

    while True:
        remaining = deadline - _now()
        if remaining <= 0:
            break
        if should_abort is not None and should_abort():
            return HoldReport(
                held_s=_now() - start, n_samples=n_samples, pv_min=pv_min,
                pv_max=pv_max, n_warn=n_warn, aborted=True,
            )

        _sleep(min(poll, remaining))

        try:
            pv = float(read_pv())
            readable = True
        except Exception:
            pv = float("nan")
            readable = False

        t = _now()
        if readable:
            n_samples += 1
            pv_min = pv if pv_min is None else min(pv_min, pv)
            pv_max = pv if pv_max is None else max(pv_max, pv)
            deviation = abs(pv - float(target_C))
            bad = deviation > float(fault_C)

            if deviation > float(warn_C):
                if not was_warning:
                    n_warn += 1
                    was_warning = True
                    logger_msg = (
                        f"anneal PV {pv:.1f} °C is {deviation:.1f} °C from the "
                        f"{float(target_C):.1f} °C target"
                    )
                    if on_warn is not None:
                        try:
                            on_warn(pv, deviation, logger_msg)
                        except Exception:
                            pass
            else:
                was_warning = False
        else:
            bad = True   # cannot verify the hold

        if bad:
            if bad_since is None:
                bad_since = t
            elif t - bad_since >= float(grace_s):
                held = t - start
                detail = (
                    f"PV unreadable" if not readable
                    else f"PV {pv:.1f} °C vs target {float(target_C):.1f} °C"
                )
                raise SafetyError(
                    f"Anneal hold aborted after {held:.0f}s of {hold_time_s:.0f}s: "
                    f"{detail} for more than {float(grace_s):.0f}s. The sample is "
                    f"no longer being annealed as specified.",
                    instrument=instrument,
                    requested=float(target_C),
                    limit=float(fault_C),
                )
        else:
            bad_since = None   # a momentary excursion must not accumulate

    return HoldReport(
        held_s=_now() - start, n_samples=n_samples, pv_min=pv_min,
        pv_max=pv_max, n_warn=n_warn,
    )


def run_anneal_hold(controller, hold_time_s: float, target_C: float) -> HoldReport:
    """Watched hold for a temperature controller — the one call both drivers make.

    Pulls the PV reader and abort signal off *controller* so the mock and real
    implementations cannot drift into different watchdog behaviour, which is the
    whole reason this module exists.

    Anti-clog purging is **not** wired here. It is executor-driven for every kind
    of dead time (see ``PURGE_WINDOW_TAG``), so this watchdog stays purely
    thermal and no driver needs to know purging exists.
    """
    cfg = anneal_watchdog_config()
    stop = getattr(controller, "_stop_wait", None)

    def _on_warn(pv: float, deviation: float, message: str) -> None:
        logger.warning(
            "anneal_pv_excursion", instrument=getattr(controller, "name", "temp"),
            pv=pv, target_C=target_C, deviation_C=round(deviation, 2),
            detail=message,
        )

    return monitored_hold(
        hold_time_s,
        read_pv=controller.get_pv,
        target_C=target_C,
        instrument=getattr(controller, "name", "temp_controller"),
        on_warn=_on_warn,
        should_abort=(stop.is_set if stop is not None else None),
        **cfg,
    )


def anneal_watchdog_config(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolve the watchdog thresholds from ``[safety]`` (single parse point)."""
    if config is None:
        try:
            from softae.config.loader import safety

            config = safety()
        except Exception:
            config = {}

    def _f(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "warn_C": _f("anneal_deviation_warn_C", DEFAULT_ANNEAL_WARN_C),
        "fault_C": _f("anneal_deviation_fault_C", DEFAULT_ANNEAL_FAULT_C),
        "grace_s": _f("anneal_deviation_grace_s", DEFAULT_ANNEAL_GRACE_S),
        "poll_interval_s": _f("anneal_poll_interval_s", DEFAULT_ANNEAL_POLL_S),
    }


# ── RH controller ────────────────────────────────────────────────────────────

def validate_rh_setpoint(val: float, max_rh: float, instrument: str) -> None:
    """Raise :class:`SafetyError` if *val* exceeds the configured RH cap.

    Mirrors ``AsyncRHController.set_setpoint`` (``max_rh`` config key,
    default 95.0 %).
    """
    if val > max_rh:
        raise SafetyError(
            f"RH setpoint {val}% exceeds limit of {max_rh}%",
            instrument=instrument,
            requested=val,
            limit=max_rh,
        )


# ── Piezo ────────────────────────────────────────────────────────────────────

def apply_piezo_profile(
    controller,
    frequency_hz: int,
    on_s: float,
    rest_s: float,
    *,
    allow_legacy_noop: bool = True,
) -> str:
    """Apply frequency then sweep settings; collapse legacy no-ops.

    *controller* must expose ``set_frequency`` and ``set_sweep`` (both the
    mock and real piezo controllers do).  Returns ``"LEGACY_NOOP"`` if either
    setting was skipped on legacy firmware, otherwise the sweep response.
    """
    first = controller.set_frequency(frequency_hz, allow_legacy_noop=allow_legacy_noop)
    second = controller.set_sweep(on_s, rest_s, allow_legacy_noop=allow_legacy_noop)
    if first == "LEGACY_NOOP" or second == "LEGACY_NOOP":
        return "LEGACY_NOOP"
    return second
