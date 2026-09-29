"""Linear setpoint-ramp schedule and runner shared by the real and mock temperature drivers.

One spelling of the step math (:func:`ramp_schedule`), one blocking runner
(:func:`run_ramp`) and the progress/report types a caller sees. The real
driver delegates its ``ramp_linear`` here; the mock uses the schedule only for
its step count, since its ramp is instant by design.

**The schedule is derived from an update interval, not a step count.** The
setpoint is re-written at least once per :data:`MAX_UPDATE_INTERVAL_S`, and no
more often than the controller's :data:`SP_RESOLUTION_C` register resolution
makes meaningful — but never faster than :data:`MIN_UPDATE_INTERVAL_S`, since
each write is two serial transactions (``write_sp`` reads then writes).

**Rates here are °C/min** (operator ruling 2026-09-29), matching the Manual
tab's "Rate (°C/min)". ``anneal`` once took ``ramp_rate`` in °C/s; that key is
refused rather than reinterpreted (:func:`legacy_ramp_rate_problem`), because a
silently reinterpreted value would ramp 60x slower.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Mapping

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: Hard ceiling on the gap between setpoint writes (operator ruling 2026-09-29).
MAX_UPDATE_INTERVAL_S = 60.0
#: Floor on the gap: each write is two serial transactions (SP read + SP write).
MIN_UPDATE_INTERVAL_S = 1.0
#: The N1040 setpoint register holds tenths of a degree.
SP_RESOLUTION_C = 0.1
#: Granularity of the wait between writes, and so the worst-case cancel latency.
POLL_S = 0.5

#: The anneal ramp-rate parameter, in °C/min.
ANNEAL_RATE_PARAM = "ramp_rate_C_per_min"
#: The retired °C/s spelling, refused wherever it appears.
LEGACY_ANNEAL_RATE_PARAM = "ramp_rate"


@dataclass(frozen=True)
class RampStep:
    """One setpoint write, as reported to an ``on_step`` callback."""

    index: int          # 1-based
    total: int
    sp_C: float         # the value just written
    T_start: float
    T_end: float
    elapsed_s: float
    t_span: float
    next_in_s: float    # seconds until the next write; 0.0 on the last step


@dataclass(frozen=True)
class RampReport:
    """Outcome of a ramp. ``last_sp is None`` means nothing was written."""

    total: int
    steps_done: int
    last_sp: float | None
    cancelled: bool
    zero_span: bool
    elapsed_s: float


def update_interval_s(T_start: float, T_end: float, t_span: float,
                      up_int: float | None = None) -> float:
    """Seconds between setpoint writes.

    ``up_int=None`` applies the rule ``clamp(SP_RESOLUTION_C / |rate|,
    MIN_UPDATE_INTERVAL_S, MAX_UPDATE_INTERVAL_S)``; an explicit ``up_int`` is
    honoured but capped at :data:`MAX_UPDATE_INTERVAL_S`.
    """
    if up_int is not None:
        if not up_int > 0:
            raise ValueError(f"ramp update interval must be > 0 s, got {up_int!r}")
        return min(float(up_int), MAX_UPDATE_INTERVAL_S)
    if t_span <= 0:
        return MIN_UPDATE_INTERVAL_S
    rate_C_per_s = abs(T_end - T_start) / t_span
    if rate_C_per_s == 0:
        return MAX_UPDATE_INTERVAL_S
    return min(max(SP_RESOLUTION_C / rate_C_per_s, MIN_UPDATE_INTERVAL_S),
               MAX_UPDATE_INTERVAL_S)


def ramp_schedule(T_start: float, T_end: float, t_span: float,
                  up_int: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(times_s, setpoints_C)`` for a linear ramp; empty arrays for a zero span.

    ``n = ceil(t_span / interval)`` uniform intervals of ``t_span / n`` seconds
    (never longer than the interval, so never longer than
    :data:`MAX_UPDATE_INTERVAL_S`), giving ``n + 1`` writes: the first
    re-asserts ``T_start`` at t = 0, the last is ``T_end`` at exactly ``t_span``.
    Endpoints that differ always give at least two writes — a one-point schedule
    would write ``T_start`` and never reach the target. Setpoints are rounded to
    :data:`SP_RESOLUTION_C`.
    """
    if t_span < 0:
        raise ValueError(f"ramp t_span must be >= 0 s, got {t_span!r}")
    if round(float(T_start), 1) == round(float(T_end), 1):
        return np.array([]), np.array([])
    interval = update_interval_s(T_start, T_end, t_span, up_int)
    # round() absorbs float noise such as 4200 / 6.000000000000001 -> 699.9999.
    n_intervals = max(math.ceil(round(t_span / interval, 9)), 1)
    times = np.linspace(0.0, float(t_span), n_intervals + 1)
    temps = np.round(np.linspace(float(T_start), float(T_end), n_intervals + 1), 1)
    return times, temps


def ramp_span_s(T_start: float, T_end: float, rate_C_per_min: float) -> float:
    """Duration (s) of a ramp from *T_start* to *T_end* at *rate_C_per_min*."""
    return abs(T_end - T_start) / rate_C_per_min * 60.0


def legacy_ramp_rate_problem(params: Mapping[str, Any]) -> str | None:
    """Why *params* must be refused for ``anneal``, or ``None`` if they are sound.

    The retired ``ramp_rate`` key was °C/s. It is refused, never reinterpreted:
    read as °C/min it would ramp 60x slower than whoever wrote it intended.
    """
    if LEGACY_ANNEAL_RATE_PARAM not in params:
        return None
    value = params[LEGACY_ANNEAL_RATE_PARAM]
    try:
        hint = f" (here {value!r} °C/s -> {ANNEAL_RATE_PARAM} = {float(value) * 60:g})"
    except (TypeError, ValueError):
        hint = ""
    return (f"anneal param '{LEGACY_ANNEAL_RATE_PARAM}' (°C/s) is retired; use "
            f"'{ANNEAL_RATE_PARAM}' in °C/min, i.e. multiply by 60{hint}")


def refuse_unknown_anneal_kwargs(extra: Mapping[str, Any]) -> None:
    """Raise for any keyword ``anneal`` does not take — the retired rate key loudly.

    Called first thing in ``anneal``, before the setpoint is read or written.
    """
    if not extra:
        return
    from softae.errors import ValidationError_

    problem = legacy_ramp_rate_problem(extra)
    if problem is not None:
        raise ValidationError_(problem)
    raise TypeError(f"anneal() got unexpected keyword argument(s): {sorted(extra)}")


class ProgressNotifier:
    """Calls ``on_step``; a failing callback is logged once and never stops the ramp.

    Progress is observability: a broken label must not strand the heater at a
    mid-ramp setpoint.
    """

    def __init__(self, on_step: Callable[[RampStep], None] | None):
        self._on_step = on_step
        self._warned = False

    def __call__(self, step: RampStep) -> None:
        if self._on_step is None:
            return
        try:
            self._on_step(step)
        except Exception:
            if not self._warned:
                self._warned = True
                logger.warning("ramp_progress_callback_failed", exc_info=True)


def _is_set(cancel: Any) -> bool:
    return cancel is not None and cancel.is_set()


def zero_span_report(T_start: float, T_end: float) -> RampReport:
    """Log and report a ramp whose rounded endpoints are equal (nothing written)."""
    logger.info("ramp_zero_span", T_start=T_start, T_end=T_end)
    return RampReport(total=0, steps_done=0, last_sp=None, cancelled=False,
                      zero_span=True, elapsed_s=0.0)


def run_ramp(
    write_sp: Callable[[float], None],
    T_start: float,
    T_end: float,
    t_span: float,
    up_int: float | None = None,
    *,
    on_step: Callable[[RampStep], None] | None = None,
    cancel: Any = None,
    read_pv: Callable[[], float] | None = None,
    write_lock: ContextManager[Any] | None = None,
) -> RampReport:
    """Execute a blocking linear ramp by writing the schedule through *write_sp*.

    *cancel* (anything with ``is_set()``) is checked on every :data:`POLL_S`
    wait iteration, and once more **inside** *write_lock* immediately before
    each write, so a cancelled ramp writes nothing further and the last-written
    SP holds. A write already in flight when *cancel* is set completes and is
    reported as ``last_sp``. *read_pv*, when given, adds an INFO ``ramp_step``
    line with PV per write (the legacy ``print_flag=1`` behaviour). Write
    errors propagate.

    *write_lock* makes check-and-write atomic against every other setpoint
    writer that takes the same lock. A park that sets *cancel* and then writes
    its own SP therefore always lands last: either the ramp holds the lock and
    its write finishes before the park's, or the park's cancel is already set
    when the ramp takes the lock and the ramp writes nothing. Without it the
    ramp could pass its check, lose the CPU, and write over the park.
    """
    lock = write_lock if write_lock is not None else contextlib.nullcontext()
    times, temps = ramp_schedule(T_start, T_end, t_span, up_int)
    total = len(times)
    if total == 0:
        return zero_span_report(T_start, T_end)
    step_s = float(times[1] - times[0])
    logger.info("ramp_started", T_start=T_start, T_end=T_end, t_span=t_span,
                n=total, step_s=round(step_s, 3))
    notify = ProgressNotifier(on_step)
    t0 = time.monotonic()
    last_sp: float | None = None
    for i, (t_pt, sp) in enumerate(zip(times, temps)):
        while True:
            if _is_set(cancel):
                return _ended("ramp_cancelled", total, i, last_sp, t0, cancelled=True)
            remaining = t_pt - (time.monotonic() - t0)
            if remaining <= 0:
                break
            time.sleep(min(POLL_S, remaining))
        sp = float(sp)
        with lock:
            cancelled_at_write = _is_set(cancel)
            if not cancelled_at_write:
                write_sp(sp)
        if cancelled_at_write:
            return _ended("ramp_cancelled", total, i, last_sp, t0, cancelled=True)
        last_sp = sp
        elapsed = time.monotonic() - t0
        if read_pv is not None:
            logger.info("ramp_step", pv=read_pv(), sp=sp, step=i + 1, total=total)
        else:
            logger.debug("ramp_step", sp=sp, step=i + 1, total=total)
        next_in = max(float(times[i + 1]) - elapsed, 0.0) if i + 1 < total else 0.0
        notify(RampStep(index=i + 1, total=total, sp_C=sp, T_start=T_start, T_end=T_end,
                        elapsed_s=elapsed, t_span=t_span, next_in_s=next_in))
    return _ended("ramp_finished", total, total, last_sp, t0, cancelled=False)


def _ended(event: str, total: int, steps_done: int, last_sp: float | None,
           t0: float, *, cancelled: bool) -> RampReport:
    elapsed = time.monotonic() - t0
    logger.info(event, steps_done=steps_done, total=total, last_sp=last_sp,
                elapsed_s=round(elapsed, 1))
    return RampReport(total=total, steps_done=steps_done, last_sp=last_sp,
                      cancelled=cancelled, zero_span=False, elapsed_s=elapsed)
