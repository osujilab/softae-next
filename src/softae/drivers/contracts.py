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
- :func:`sustained_above` / :func:`sustained_below` — the one grace-windowed,
  one-sided excursion test, shared by the equilibration overshoot guard and the
  RH hold watchdog.
- :func:`classify_rh_hold` — three-state verdict on a held %RH series.
- :func:`apply_piezo_profile` — combined frequency + sweep profile application
  (MockPiezoController / AsyncPiezoController).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

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


# ── Excursion tests: one implementation, both signs, every axis ──────────────

def sustained_above(
    series: Sequence[tuple[float, float]], target: float, band: float, grace_s: float
) -> bool:
    """True when the trailing run of samples above ``target + band`` spans *grace_s*.

    Two properties, both load-bearing, which is why there is exactly one of these.

    **One-sided.** :func:`monitored_hold` grades ``abs(pv - target)``, so a stage
    that cannot *reach* 85 °C and a heater that *runs past* it produce the same
    :class:`SafetyError`. The first is data; the second is a hazard. The sign is
    checked here, from the caller's own recorded series.

    **Grace-windowed on the trailing run only.** A sample inside the band — or an
    unreadable one — breaks the run, so a legitimate ramp *through* the band is
    not read as a fault. That was written for a temperature approach and is the
    same hazard during an RH approach, which is why this is shared rather than
    reimplemented: two grace-windowed excursion tests would drift apart.

    Fewer than two consecutive samples cannot span an interval, so a young run is
    never an excursion.
    """
    above: list[tuple[float, float]] = []
    for t, v in reversed(list(series)):
        if isinstance(v, float) and math.isfinite(v) and v > target + band:
            above.append((t, v))
        else:
            break
    if len(above) < 2:
        return False
    return (above[0][0] - above[-1][0]) >= float(grace_s)


def sustained_below(
    series: Sequence[tuple[float, float]], target: float, band: float, grace_s: float
) -> bool:
    """The mirror of :func:`sustained_above` — reflected, not reimplemented."""
    reflected = [(t, -v if isinstance(v, float) else v) for t, v in series]
    return sustained_above(reflected, -float(target), band, grace_s)


# ── Anneal hold watchdog ─────────────────────────────────────────────────────

# DEFECT, NOTED NOT FIXED: the two values below disagree with `[safety]`
# (`anneal_deviation_warn_C = 2.0` / `_fault_C = 5.0`). The toml comment records
# 3/10 as the explicitly *rejected* engineering guess — "a heater 9 C off target
# would never have faulted" — superseded by the measured 2/5. Because
# `anneal_watchdog_config` swallows a config-load exception and proceeds with
# `{}`, a config-load failure silently restores the rejected bands. Recommended
# resolution: align these to 2.0/5.0. Left alone deliberately: changing a thermal
# fault band is a decision that needs its own review, and the RH demotion that
# found this does not own the temperature axis. (The RH side below is clean and
# has a test pinning it that way — see `test_the_rh_defaults_match_the_shipped_config`.)

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
    #: The humidity verdict, when the caller supplied an ``rh_reader``; ``None``
    #: otherwise. :func:`monitored_hold` never sets it — it knows nothing about
    #: humidity and must not start to. :func:`run_anneal_hold` attaches it on the
    #: return path, so a film annealed under an RH fault is a weaker provenance
    #: claim carried as a *value* rather than an error.
    rh: "RHHoldVerdict | None" = None

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
                    "PV unreadable" if not readable
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


def run_anneal_hold(
    controller,
    hold_time_s: float,
    target_C: float,
    *,
    rh_reader: Any = None,
    rh_setpoint_pct: float | None = None,
    data_store: Any = None,
    run_id: str | None = None,
    sleep: Any = None,
    now: Any = None,
) -> HoldReport:
    """Watched hold for a temperature controller — the one call both drivers make.

    Pulls the PV reader and abort signal off *controller* so the mock and real
    implementations cannot drift into different watchdog behaviour, which is the
    whole reason this module exists.

    **Humidity is watched too, when the caller supplies a reader.** Pass
    *rh_reader* (``get_TH`` returning ``(chamber_T_C, %RH)``, or any callable
    returning a bare %RH) together with the *rh_setpoint_pct* that was commanded,
    and the hold samples humidity on the temperature poll's own cadence —
    throttled to ``[safety] rh_poll_interval_s`` — and classifies it per
    :func:`classify_rh_hold`.

    **No humidity verdict stops the hold.** A polymer cure at >100 °C is
    dominated by its thermal history; humidity modulates the result rather than
    defining it, so killing an 8 h cure — and with it the board, the stock and
    the overnight slot — because a *secondary* variable left its band trades a
    certain loss for an uncertain one. Part of the RH trajectory through a long
    cure is a consumable draining rather than a process going wrong, which is a
    supply curve and not a fault. So humidity is **announced and recorded**: the
    alert row is the durable evidence, ``report.rh`` is the in-process value for
    whoever records the sample. Temperature stays blocking exactly as it was,
    including its unreadable-PV fault.

    With no *rh_reader* this is exactly the thermal-only hold it has always been.

    Anti-clog purging is **not** wired here. It is executor-driven for every kind
    of dead time (see ``PURGE_WINDOW_TAG``), so this watchdog stays purely
    thermal and no driver needs to know purging exists.
    """
    cfg = anneal_watchdog_config()
    stop = getattr(controller, "_stop_wait", None)
    instrument = getattr(controller, "name", "temp_controller")
    read_pv = controller.get_pv
    should_abort = stop.is_set if stop is not None else None

    watch: RHHoldWatch | None = None
    if rh_reader is not None and rh_setpoint_pct is not None:
        watch = RHHoldWatch(
            rh_reader, float(rh_setpoint_pct), fallback_temperature_C=float(target_C),
            data_store=data_store, run_id=run_id, now=now or time.monotonic,
        )
        read_pv = watch.wrap_reader(read_pv)

    def _on_warn(pv: float, deviation: float, message: str) -> None:
        logger.warning(
            "anneal_pv_excursion", instrument=instrument,
            pv=pv, target_C=target_C, deviation_C=round(deviation, 2),
            detail=message,
        )

    clock = {k: v for k, v in (("sleep", sleep), ("now", now)) if v is not None}
    report = monitored_hold(
        hold_time_s,
        read_pv=read_pv,
        target_C=target_C,
        instrument=instrument,
        on_warn=_on_warn,
        should_abort=should_abort,
        **cfg,
        **clock,
    )
    if watch is not None:
        report.rh = watch.verdict
    return report


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


# ── RH hold watchdog ─────────────────────────────────────────────────────────
#
# `validate_rh_setpoint` caps the commanded value at write time and never looks
# again, so through a multi-hour hot hold nothing watches what the humidity did.
# The interesting failure is not drift — the loop is PID-controlled and effective
# — it is a PV that sits away from the command for hours. Measured 2026-08-11:
# 15 %RH commanded returned 16.9–20.4 at 65 °C and 19.5–23.2 at 85 °C.
#
# **Why the states below name an observation and not a cause.** At least two
# explanations fit that record — an attainable floor that rises with chamber
# temperature, and a flush basin still evaporating its water out — and the
# distinguishing measurement was never taken (basin fill is uninstrumented; no
# column in ``conditions`` records it). Both produce the same reading and both
# want the same response, so the classifier tests what it can see: how far the PV
# sat from the command, for how long, and on which side.

#: Sustained |PV − SP| worth saying out loud (%RH).
DEFAULT_RH_WARN_PCT = 3.0
#: Sustained |PV − SP| worth waking someone for (%RH). Symmetric with the warn
#: band and strictly outside it. An operator judgement about how far off command
#: is loud — **not** a magnitude fitted to any run: a deviation band fitted to the
#: 2026-08-11 data would encode that morning's basin fill, which the planned
#: plumbing change designs away.
DEFAULT_RH_FAULT_PCT = 5.0
#: Long **on purpose**: a humidity loop settling over minutes is normal, and a
#: short grace would produce alerts an operator learns to ignore — which is the
#: failure this whole mechanism exists to prevent.
DEFAULT_RH_GRACE_S = 600.0
#: Floor on the interval between RH samples (s).
DEFAULT_RH_POLL_S = 60.0

#: PV outside the band, but the hold is young or still trending in. An ordinary
#: approach: say nothing.
RH_CONVERGING = "converging"
#: PV sustained off SP beyond ``warn_pct`` but inside ``fault_pct``, **in either
#: direction**. Alert at ``WARNING``, record — and keep holding.
RH_OFF_SETPOINT_SUSTAINED = "off_setpoint_sustained"
#: Sustained beyond ``fault_pct`` in either direction, or no readable PV at all.
#: Inside an anneal this stops nothing — see :func:`run_anneal_hold`.
RH_FAULT = "fault"

#: ``Alert.kind`` for the sustained-off-setpoint finding. Spelled apart from
#: :data:`RH_OFF_SETPOINT_SUSTAINED` for two reasons, and the second is now the
#: stronger. Alert kinds are one flat namespace shared with ``park`` and
#: ``reservoir``, where a bare "off_setpoint_sustained" says nothing about which
#: axis went off setpoint. And this value is **persisted** (``alerts.kind``,
#: written by ``DataStore.record_alert``, read back by ``query_alerts``), so it is
#: **frozen** at its original spelling: changing it would silently orphan every
#: historical row from any query written afterwards. A reviewer who wants the
#: value to match the identifier must own that migration deliberately.
ALERT_RH_OFF_SETPOINT = "rh_floor_limited"
#: ``Alert.kind`` for a humidity fault that did **not** stop the hold.
ALERT_RH_FAULT = "rh_fault"

#: The states worth announcing. Named once so :func:`alert_rh_verdict` and
#: :class:`RHHoldWatch`'s once-per-state throttle cannot disagree about which
#: findings get announced — a disagreement would silently drop one.
_RH_ANNOUNCED_STATES = frozenset({RH_OFF_SETPOINT_SUSTAINED, RH_FAULT})


@dataclass(frozen=True)
class RHHoldVerdict:
    """What the humidity did during a watched hold, in three states rather than two.

    The third state earns its place on **operator attention**, not on plumbing: a
    sustained excursion that has not reached the fault band is still worth
    announcing without stopping anything. Ten minutes at 3 %RH off command is a
    fact an operator should have; it is not a fact that should carry ``CRITICAL``
    or mark a sample's provenance. Collapsing to two states forces every excursion
    into either silence or the loud severity, and both are wrong for the middle
    band. That argument is indifferent to *why* the humidity is off command, which
    is the property the original floor/basin warrant lacked.

    Both bands are **symmetric**: over- and undershoot are graded identically and
    nested, ``warn_pct < fault_pct``. The earlier asymmetry — undershoot faulting
    at the warn band because "the basin can only push humidity up" — rested on a
    mechanism the record does not settle, and was removed with it.
    """

    state: str
    setpoint_pct: float
    pv_pct: float = float("nan")
    temperature_C: float = float("nan")
    deviation_pct: float = float("nan")
    n_samples: int = 0
    reason: str = ""

    @property
    def is_fault(self) -> bool:
        """Classification only. What a fault *costs* is the caller's decision:
        the anneal records it and holds on; the settle path may choose otherwise."""
        return self.state == RH_FAULT

    def describe(self) -> str:
        """The operator's sentence. SP, PV and chamber temperature **together** —
        any one of the three alone is not actionable."""
        at = ("" if math.isnan(self.temperature_C)
              else f" at {self.temperature_C:.1f} C")
        seen = (f"{self.pv_pct:.1f} %RH" if math.isfinite(self.pv_pct)
                else "no readable PV")
        head = (f"RH commanded {self.setpoint_pct:.1f} %RH, measured {seen}{at} "
                f"over {self.n_samples} sample(s)")
        if self.state == RH_OFF_SETPOINT_SUSTAINED:
            # Above and below the command are different findings to an operator
            # even when they are one state to the classifier.
            side = "below" if self.deviation_pct < 0 else "above"
            return (f"{head}: the humidity stayed sustained {side} the command "
                    f"for the whole grace window. Recorded; the hold continues.")
        if self.state == RH_FAULT:
            return f"{head}: {self.reason or 'humidity fault'}."
        return f"{head}: within band or still approaching."


def rh_watchdog_config(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolve the RH watchdog thresholds from ``[safety]`` (single parse point).

    Mirrors :func:`anneal_watchdog_config` exactly — same shape, same discipline.
    """
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
        "warn_pct": _f("rh_deviation_warn_pct", DEFAULT_RH_WARN_PCT),
        "fault_pct": _f("rh_deviation_fault_pct", DEFAULT_RH_FAULT_PCT),
        "grace_s": _f("rh_deviation_grace_s", DEFAULT_RH_GRACE_S),
        "poll_interval_s": _f("rh_poll_interval_s", DEFAULT_RH_POLL_S),
    }


def classify_rh_hold(
    series: Sequence[tuple[float, float]],
    setpoint_pct: float,
    *,
    warn_pct: float = DEFAULT_RH_WARN_PCT,
    fault_pct: float = DEFAULT_RH_FAULT_PCT,
    grace_s: float = DEFAULT_RH_GRACE_S,
    temperature_C: float = float("nan"),
    poll_interval_s: float | None = None,   # accepted so **rh_watchdog_config() fits
) -> RHHoldVerdict:
    """Grade a ``(t, %RH)`` series into :data:`RH_CONVERGING` /
    :data:`RH_OFF_SETPOINT_SUSTAINED` / :data:`RH_FAULT`.

    Every excursion question is asked of :func:`sustained_above` — the same
    grace-windowed, one-sided test the equilibration overshoot guard uses — so a
    transient ramp through the band cannot trip any of them.

    Four branches, three states, and **the order is load-bearing**: a series
    beyond ``fault_pct`` is also beyond ``warn_pct``, so each sign's fault test
    must precede its warn test. The bands are symmetric and nested
    (``warn_pct < fault_pct``); the two signs are graded identically.
    """
    del poll_interval_s
    finite = [(t, v) for t, v in series
              if isinstance(v, float) and math.isfinite(v)]
    sp = float(setpoint_pct)
    if not finite:
        return RHHoldVerdict(
            RH_FAULT, sp, temperature_C=float(temperature_C),
            n_samples=0, reason="no readable %RH for the whole window",
        )
    pv = finite[-1][1]
    common: dict[str, Any] = dict(
        setpoint_pct=sp, pv_pct=pv, temperature_C=float(temperature_C),
        deviation_pct=pv - sp, n_samples=len(finite),
    )
    if sustained_above(series, sp, float(fault_pct), grace_s):
        return RHHoldVerdict(
            RH_FAULT, reason=f"sustained more than {float(fault_pct):g} %RH ABOVE "
                             f"the setpoint", **common)
    if sustained_below(series, sp, float(fault_pct), grace_s):
        return RHHoldVerdict(
            RH_FAULT, reason=f"sustained more than {float(fault_pct):g} %RH BELOW "
                             f"the setpoint", **common)
    if sustained_above(series, sp, float(warn_pct), grace_s):
        return RHHoldVerdict(
            RH_OFF_SETPOINT_SUSTAINED,
            reason=f"sustained more than {float(warn_pct):g} %RH above the setpoint",
            **common)
    if sustained_below(series, sp, float(warn_pct), grace_s):
        return RHHoldVerdict(
            RH_OFF_SETPOINT_SUSTAINED,
            reason=f"sustained more than {float(warn_pct):g} %RH below the setpoint",
            **common)
    return RHHoldVerdict(RH_CONVERGING, reason="converging", **common)


def alert_rh_verdict(
    verdict: RHHoldVerdict,
    *,
    data_store: Any = None,
    run_id: str | None = None,
    instrument: str = "rh_controller",
) -> None:
    """Announce a humidity finding through the existing alert path. Never raises.

    **Both** non-converging states are announced here, and the fault is the louder
    of the two. Before the demotion a fault travelled as a :class:`SafetyError`
    and was announced by the machinery that parked on it — sound reasoning only
    while the fault parked. Nothing parks on it now, so without this an 8 h cure
    at a badly wrong humidity would run to completion and report clean, and the
    operator's first evidence would be the σ.

    The fault takes ``CRITICAL`` deliberately. A park is self-announcing: the
    operator finds a stopped rig and goes looking for the reason. A demoted fault
    is the opposite — the run completes, the samples get measured, and the data
    enters the record looking ordinary — so it needs the loud severity *more*
    than the park does, not less. ``raise_alert`` never raises, so promoting a
    fault into it cannot reintroduce a stop by the back door.
    """
    if verdict.state not in _RH_ANNOUNCED_STATES:
        return
    from softae.core.alerts import CRITICAL, WARNING, Alert, raise_alert

    kind, severity = (
        (ALERT_RH_FAULT, CRITICAL) if verdict.is_fault
        else (ALERT_RH_OFF_SETPOINT, WARNING)
    )

    raise_alert(
        Alert(
            kind=kind,
            message=verdict.describe(),
            severity=severity,
            run_id=run_id,
            details={
                "instrument": instrument,
                "rh_setpoint_pct": verdict.setpoint_pct,
                "rh_pv_pct": verdict.pv_pct,
                "temperature_C": verdict.temperature_C,
                "deviation_pct": verdict.deviation_pct,
            },
        ),
        data_store=data_store,
    )


class RHHoldWatch:
    """Samples %RH alongside a temperature hold and keeps the verdict as a value.

    Rides the temperature poll rather than opening a second loop (``get_TH``
    returns both in one transaction), throttled to its own
    ``rh_poll_interval_s``.

    The alert throttle is **per state**, not per hold: a verdict that is true for
    eight hours is one finding, not four hundred — but two different findings are
    two findings. A hold that goes off-setpoint at hour 1 and degrades to a fault
    at hour 4 must announce both; a single per-hold flag would swallow the second,
    which is exactly the silent 8 h cure the demotion exists to prevent. Worst
    case is two alerts per hold.

    **It keeps the verdict; it does not act on it.** There is deliberately no
    abort composition and no raise here — see :func:`run_anneal_hold`.
    """

    def __init__(
        self,
        reader: Any,
        setpoint_pct: float,
        *,
        fallback_temperature_C: float = float("nan"),
        thresholds: dict[str, float] | None = None,
        data_store: Any = None,
        run_id: str | None = None,
        instrument: str = "rh_controller",
        now: Any = None,
    ) -> None:
        self._reader = reader
        self._setpoint_pct = float(setpoint_pct)
        self._fallback_C = float(fallback_temperature_C)
        self._cfg = dict(thresholds if thresholds is not None else rh_watchdog_config())
        self._data_store = data_store
        self._run_id = run_id
        self._instrument = instrument
        self._now = now or time.monotonic
        self.series: list[tuple[float, float]] = []
        self.temperature_C = float(fallback_temperature_C)
        self.verdict = RHHoldVerdict(
            RH_CONVERGING, self._setpoint_pct,
            temperature_C=self._fallback_C, reason="not yet sampled")
        self._alerted: set[str] = set()
        self._last_sample_t: float | None = None

    # ── sampling ─────────────────────────────────────────────────────────────

    def wrap_reader(self, read_pv: Any) -> Any:
        """Return *read_pv* with an RH sample taken on the same call.

        Failure to read humidity must never look like a dead thermocouple, so
        every exception from the RH side is swallowed here and recorded as a
        non-finite sample — which breaks the trailing run and therefore cannot
        manufacture a verdict either.
        """

        def _read() -> float:
            pv = read_pv()
            try:
                self.sample()
            except Exception:
                logger.warning("rh_hold_sample_failed", exc_info=True)
            return pv

        return _read

    def sample(self) -> None:
        """Take one throttled %RH sample and re-classify."""
        t = float(self._now())
        interval = float(self._cfg.get("poll_interval_s", DEFAULT_RH_POLL_S))
        if self._last_sample_t is not None and t - self._last_sample_t < interval:
            return
        self._last_sample_t = t
        rh, temp = _read_rh(self._reader)
        if math.isfinite(temp):
            self.temperature_C = temp
        self.series.append((t, rh))
        self.verdict = classify_rh_hold(
            self.series, self._setpoint_pct, temperature_C=self.temperature_C,
            **{k: v for k, v in self._cfg.items() if k != "poll_interval_s"},
        )
        state = self.verdict.state
        if state in _RH_ANNOUNCED_STATES and state not in self._alerted:
            self._alerted.add(state)
            alert_rh_verdict(self.verdict, data_store=self._data_store,
                             run_id=self._run_id, instrument=self._instrument)


def _read_rh(reader: Any) -> tuple[float, float]:
    """``(%RH, temperature_C)`` from a ``get_TH``-shaped or bare-%RH reader.

    ``get_TH`` returns ``(chamber_air_C, %RH)``, and the chamber air is the right
    thermometer for a claim about the enclosure's humidity floor — the basin sits
    in that air, not on the stage. A bare reader yields NaN and the caller's
    fallback stands.
    """
    value = reader()
    if isinstance(value, (tuple, list)) and len(value) == 2:
        temp, rh = value
        return float(rh), float(temp)
    return float(value), float("nan")


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
