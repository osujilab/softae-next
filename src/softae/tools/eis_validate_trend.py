"""Direction, not just magnitude: each channel's sigma against its own past.

The ``[settle]`` line answers *"how far is the worst channel from its own recent
mean?"* and it answers it as a **magnitude** -- ``max|sigma - mean| / |mean|``
over the trailing window, with the sign thrown away by the absolute value. That
is the right quantity for the gate, which is asking whether the window is flat.
It is the wrong one for the operator standing in front of it, who is asking a
different question: is this film still *going somewhere*, or is it merely
jittering? Those two call for opposite decisions -- wait longer, versus the
condition is unreachable and the setpoint has to move -- and a magnitude cannot
tell them apart.

So the same rounds are rendered a second way, as a **signed** comparison against
a baseline built from the rounds *before* the current one.

**Console only.** Nothing here is persisted and nothing here is added to the
run's narration payload: the durable event stream carries the gate's state, and
an earlier decision deliberately kept per-channel sigma out of it. A table whose
whole purpose is to be read while a hold is running does not need to survive it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from softae.analysis.equilibration import (
    DEFAULT_SETTLE_N_ROUNDS,
    EXCLUDED_ABSENT,
)

#: Smoothing factor, **derived rather than chosen**.
#:
#: The standard ``alpha = 2 / (N + 1)`` is the value at which an EMA and an
#: N-point boxcar have the same centre of mass, so an EMA built with it
#: "remembers" as far back as the window the gate is actually judging. Sourcing
#: ``N`` from :data:`~softae.analysis.equilibration.DEFAULT_SETTLE_N_ROUNDS`
#: rather than restating 3 is the point: if the gate's window is ever widened,
#: the two views move together instead of silently disagreeing about what
#: *recent* means -- which would be the worst possible failure for a table whose
#: only job is to be read alongside the gate's own line.
TREND_ALPHA = 2.0 / (float(DEFAULT_SETTLE_N_ROUNDS) + 1.0)


@dataclass(frozen=True)
class ChannelTrend:
    """One channel's row: where it is, where it was, and which way it moved."""

    channel: int
    #: This round's sigma, or ``None`` when the round carried no usable number.
    sigma: float | None
    #: EMA of the **previous** rounds only. ``None`` before any exists.
    ema: float | None
    #: How many previous rounds were folded into :attr:`ema`. ``0`` means there
    #: is no baseline; ``1`` means the baseline is a single reading rather than
    #: an average, and the column exists so that can never be mistaken.
    n_prior: int
    #: Signed ``(sigma - ema) / |ema|``. ``None`` whenever it is not defined.
    departure_rel: float | None
    #: Provisional band, read off the pre-equilibration apex sweep.
    band: str
    #: Why the settle gate is not counting this channel, or ``""``.
    note: str


def prior_ema(
    sigmas: Sequence[float | None], alpha: float = TREND_ALPHA,
) -> tuple[float | None, int]:
    """EMA over *sigmas*, skipping the rounds that carried no number.

    The caller passes the previous rounds and **not** the current one. That
    exclusion is the whole point of this module: an EMA that has already
    absorbed the sample it is being compared against has baked the sample into
    its own baseline, and the departure it then reports is a fraction of the
    real one -- smallest exactly when the reading is most anomalous.

    A missing round is skipped rather than treated as a zero or as a repeat of
    the last value: a sweep that failed says nothing about where sigma went.
    """
    ema: float | None = None
    folded = 0
    for value in sigmas:
        if value is None:
            continue
        ema = value if ema is None else alpha * value + (1.0 - alpha) * ema
        folded += 1
    return ema, folded


def trend_rows(
    rounds: Sequence[Sequence[Any]],
    channels: Sequence[int],
    *,
    bands: Mapping[int, str] | None = None,
    excluded: Mapping[int, str] | None = None,
    participating: Sequence[int] | None = None,
    alpha: float = TREND_ALPHA,
) -> list[ChannelTrend]:
    """One row per channel under study, in the plan's own channel order.

    *rounds* is the **full** history including the current round as its last
    entry; the EMA is taken over everything before that. *participating* and
    *excluded* come from the gate's own
    :class:`~softae.analysis.equilibration.SettleCheck`, and ``participating is
    None`` means no window has been judged yet -- which is not the same as "no
    channel participates", so nothing is marked in that case.
    """
    history = [list(fits) for fits in rounds]
    wanted = [int(channel) for channel in channels]
    current = _sigmas_by_channel(history[-1]) if history else {}
    prior: dict[int, list[float | None]] = {channel: [] for channel in wanted}
    for fits in history[:-1]:
        seen = _sigmas_by_channel(fits)
        for channel in wanted:
            prior[channel].append(seen.get(channel))

    rows: list[ChannelTrend] = []
    for channel in wanted:
        ema, folded = prior_ema(prior[channel], alpha)
        sigma = current.get(channel)
        rows.append(ChannelTrend(
            channel=channel, sigma=sigma, ema=ema, n_prior=folded,
            departure_rel=_departure(sigma, ema),
            band=str((bands or {}).get(channel, "")),
            note=_note(channel, excluded, participating),
        ))
    return rows


def render_trend_table(rows: Sequence[ChannelTrend]) -> str:
    """The block printed under the ``[settle]`` line, once per round.

    **The column header is reprinted every round**, deliberately. It costs one
    line against fifteen, and the alternative -- a header printed only on round
    one -- leaves an operator forty lines down a scrolling console reading five
    unlabelled numeric columns. Every other reduction available here (dropping
    the legend after round one, one line per channel, no blank separators)
    buys far more than that one line costs.
    """
    lines = [_PREFIX + _HEADER]
    lines += [_INDENT + _row_text(row) for row in rows]
    return "\n".join(lines)


def render_trend_legend(
    alpha: float = TREND_ALPHA, n_rounds: int = DEFAULT_SETTLE_N_ROUNDS,
) -> str:
    """Printed once, on the first round. ASCII only, like every block here.

    Every claim the table could be *mis*read as making is denied here in
    writing: sigma is not a calibrated conductivity, the EMA is not a
    comparison until it has something to average, and the band is not a
    measurement of the equilibrated film.
    """
    return "\n".join([
        _PREFIX + "per-channel sigma against its own trailing EMA -- the "
                  "DIRECTION that the",
        _INDENT + "[settle] line's magnitude cannot show. Console only; not "
                  "persisted.",
        _INDENT + "sigma = 1/Re(Z) at the LOWEST measured frequency with the "
                  "cell constant set",
        _INDENT + "to 1. A relative tracking number, NOT a calibrated "
                  "conductivity, and NOT",
        _INDENT + "comparable between channels -- only against its own past.",
        _INDENT + "EMA folds the PREVIOUS rounds ONLY; the current reading is "
                  "excluded, so",
        _INDENT + "`vs EMA` is a departure from a baseline that has not "
                  "absorbed it. alpha",
        _INDENT + f"{alpha:.2f} = 2/(N+1), N = {n_rounds} = the settle window, "
                  f"so the EMA remembers as",
        _INDENT + "far back as the gate judges.",
        _INDENT + "`n` = rounds folded in: 0 = no baseline yet, 1 = one "
                  "previous reading and",
        _INDENT + "not an average. `!` = the gate is NOT counting this "
                  "channel; the note says",
        _INDENT + "why. `band?` is PROVISIONAL -- read off pre-equilibration "
                  "apexes.",
    ])


# ── Formatting ───────────────────────────────────────────────────────────────

_PREFIX = "[trend ] "
_INDENT = " " * len(_PREFIX)
_NONE = "n/a"
_HEADER = (f"{'':<2}{'ch':>3}  {'sigma':>9}  {'EMA':>9}  {'n':>2}  "
           f"{'vs EMA':>8}  {'band?':<12}note")


def _sigmas_by_channel(fits: Sequence[Any]) -> dict[int, float | None]:
    return {int(fit.channel): _finite(fit.sigma) for fit in fits}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _departure(sigma: float | None, ema: float | None) -> float | None:
    if sigma is None or ema is None or ema == 0.0:
        return None
    return (sigma - ema) / abs(ema)


def _note(
    channel: int,
    excluded: Mapping[int, str] | None,
    participating: Sequence[int] | None,
) -> str:
    """Why the gate is not counting this channel, in the gate's own vocabulary.

    A channel the gate never saw at all is not in ``excluded`` either -- it is
    simply missing from the window -- so it is reported as ``absent``, which is
    the reason ``settle_check`` would have given had it seen a placeholder.
    """
    if excluded and channel in excluded:
        return str(excluded[channel])
    if participating is not None and channel not in participating:
        return EXCLUDED_ABSENT
    return ""


def _fmt_sigma(value: float | None) -> str:
    """Fixed-width scientific, so ~1e-4 magnitudes stay aligned down the column."""
    return _NONE if value is None else f"{value:.2e}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return _NONE
    pct = value * 100.0
    # A departure past 1000 % is a broken channel, not a drift; it is allowed to
    # widen its own row rather than be clipped into looking like a small number.
    return f"{pct:+.0f}%" if abs(pct) >= 1000.0 else f"{pct:+.1f}%"


def _row_text(row: ChannelTrend) -> str:
    flag = "!" if row.note else ""
    return (f"{flag:<2}{row.channel:>3}  {_fmt_sigma(row.sigma):>9}  "
            f"{_fmt_sigma(row.ema):>9}  {row.n_prior:>2}  "
            f"{_fmt_pct(row.departure_rel):>8}  "
            f"{(row.band or '-'):<12}{row.note}").rstrip()


__all__ = [
    "TREND_ALPHA", "ChannelTrend", "prior_ema", "render_trend_legend",
    "render_trend_table", "trend_rows",
]
