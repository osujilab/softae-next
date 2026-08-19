"""Did the impedance semicircle close inside the swept window?

R₁ is a *measurement* only once −Z″ has risen, peaked and fallen again before the
sweep runs out of frequency. While the lowest swept point is still on the rising
limb, R₁ is reached by extrapolating off the high-frequency side — the same number,
a weaker claim, and nothing downstream currently says so.

Measured over run ``20260811T023757Z_equilibration_characterization`` (1440 spectra,
``Quick`` preset, 200 kHz → 20 Hz): a third of the sweep peaks sit on the lowest
measured point. The rate is conductivity-driven, not a fault — 8 % at the hot, wet
end of the up leg against 73 % at the cold end of the down leg, which is precisely
where an Arrhenius slope has its leverage.

**Nothing here demotes a fit.** A railed parameter reports a property of
``CIRCUIT_MODELS``; an extrapolated R₁ still reports the film. So this annotates and
:func:`~softae.analysis.eis.engine._demote_if_railed` does not, and the difference is
deliberate: refusing the open third would throw away most of the cold end of every
temperature sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: The arc peaked inside the window; R₁ is bracketed by measured points.
CLOSED = "closed"
#: The peak sits on the lowest measured point; R₁ is extrapolated.
OPEN = "open"
#: Not enough sweep to judge either way — distinct from ``CLOSED``, as
#: ``resolve_thickness_cm`` keeps "absent" distinct from "zero".
UNKNOWN = "unknown"

#: Lowest-frequency points the shape rule inspects. Three is the smallest number
#: that can distinguish a trend from a single excursion.
TRAILING_POINTS = 3

#: How large a rise-with-frequency counts as structure rather than noise, relative
#: to the local ``−Z″`` level. Without it the shape rule rescued two genuinely open
#: spectra in the characterisation run on wobbles of 0.5 % and 1.8 % — scatter, not
#: a peak.
NOISE_TOL_REL = 0.05


def _finite_or_none(value: float) -> float | None:
    """JSON has no NaN, and a NaN read back as a number is worse than a null."""
    return float(value) if value == value and np.isfinite(value) else None


def _peak_prominence(y: np.ndarray, k: int) -> float:
    """Topographic prominence of ``y[k]`` — how far it stands above its saddles.

    The standard definition, and the one :func:`scipy.signal.peak_prominences`
    computes: walk out either side until a value higher than the peak is met or the
    array ends, take the lowest point reached on each side, and measure the peak
    above the *higher* of those two bases. Written out here in nine lines because
    this module imports only numpy and sits on the router's per-fit path; a scipy
    import for one array walk is not a trade worth making.
    """
    left = y[k]
    for i in range(k - 1, -1, -1):
        if y[i] > y[k]:
            break
        left = min(left, y[i])
    right = y[k]
    for i in range(k + 1, y.size):
        if y[i] > y[k]:
            break
        right = min(right, y[i])
    return float(y[k] - max(left, right))


def _interior_apex(y_s: np.ndarray) -> int:
    """Index of the tallest **strict-interior** local maximum, or ``-1``.

    Endpoints are never candidates. That is not a stylistic choice: this tree has
    already paid twice for an extremum search that admitted a window edge (F15, the
    ``|Z|`` minimum read as the interior ``−Z″`` valley, wrong by 10× and published),
    and :func:`~softae.analysis.eis.gates.gate_valley_feature` was written to fix it
    on the sibling feature. The same policy applies here on the maximum.
    """
    best = -1
    for k in range(1, y_s.size - 1):
        if y_s[k] > y_s[k - 1] and y_s[k] > y_s[k + 1]:
            if best < 0 or y_s[k] > y_s[best]:
                best = k
    return best


@dataclass(frozen=True)
class ArcClosure:
    """Where the ``−Z″`` peak sat, and how far the sweep was from closing.

    ``phase_low_deg`` is the severity, and it is the most interpretable one
    available: near 0° the response at the sweep floor is already resistive and a
    modestly lower floor would close the arc; near −90° it is still essentially
    capacitive and no realistic extension of the preset will rescue it. It is NaN
    when the caller supplied no phase — the *state* never depended on it, so its
    absence costs interpretability, not the verdict.
    """

    state: str
    f_peak_hz: float = float("nan")
    f_low_hz: float = float("nan")
    phase_low_deg: float = float("nan")
    #: Why the state is ``UNKNOWN``, or why a peak at the sweep floor was still
    #: read as ``CLOSED``. Empty on a plain closed arc.
    reason: str = ""

    # ── Additive, and never consulted by ``state`` or ``f_peak_hz`` ──────────
    #
    # Appended *after* ``reason`` because the constructions below are positional;
    # inserting them earlier would silently rebind ``reason``. They answer the
    # acquisition question — *where should the next sweep put its points?* — which
    # is not the question ``state`` answers, and they carry no threshold: the cut
    # lives in :mod:`softae.analysis.eis.scout`, because for a guard the cheap
    # error is a missed warning while for a planner it is the opposite.
    #
    # On a tail-dominated spectrum the global maximum of ``−Z″`` *is* the tail
    # (measured: 8.1e5 Ω of tail against a 5.3e4 Ω arc apex), so ``f_peak_hz`` is
    # honestly the sweep floor and the arc worth measuring is 20 Hz up-sweep.

    #: Tallest strict-interior local maximum of ``−Z″``. NaN when there is none.
    f_apex_interior_hz: float = float("nan")
    #: Its topographic prominence divided by **its own height** — never by the
    #: spectrum's global maximum, which a blocking tail owns. Measured on a 15-well
    #: humid series: 1/15 arcs detected with global scaling, 12/15 with relative.
    apex_prominence_rel: float = float("nan")
    #: ``log10(f_apex_interior_hz / f_low_hz)`` — how much band was measured below
    #: the arc. Kept separate from the prominence on purpose: a prominent apex
    #: 0.1 decades off the floor and a weak shoulder 2 decades up demand opposite
    #: actions, and one score cannot tell them apart.
    band_below_apex_decades: float = float("nan")

    @property
    def closed(self) -> bool:
        return self.state == CLOSED

    @property
    def detail(self) -> str:
        if self.state == UNKNOWN:
            return f"arc closure undetermined: {self.reason}"
        where = f"−Z″ peak at {self.f_peak_hz:.4g} Hz, sweep floor {self.f_low_hz:.4g} Hz"
        if self.phase_low_deg == self.phase_low_deg:
            where += f", phase there {self.phase_low_deg:.1f}°"
        if self.state == OPEN:
            return f"arc did not close in band ({where}) — R1 extrapolated"
        return f"arc closed in band ({where})"

    def as_record(self) -> dict[str, Any]:
        """The persisted form — ``run_gates`` log shape plus the four numbers.

        Log shape because the column this travels in is ``fit_results.gate_log_json``,
        whose readers sum ``n_dropped`` over every entry; an entry missing that key
        would be a foreign object in a list with a contract.
        """
        return {
            "gate": "arc_closure",
            "severity": "annotate",
            "passed": self.state == CLOSED,
            "n_dropped": 0,
            "detail": self.detail,
            "state": self.state,
            "f_peak_hz": _finite_or_none(self.f_peak_hz),
            "f_low_hz": _finite_or_none(self.f_low_hz),
            "phase_low_deg": _finite_or_none(self.phase_low_deg),
        }


def arc_closure(
    freq: Any,
    z_imag_neg: Any,
    phase: Any = None,
    *,
    min_points: int = 5,
) -> ArcClosure:
    """Read the closure state off one spectrum. Pure: no fit, no rig, no database.

    *z_imag_neg* is ``−Z″`` in the file convention (positive for a capacitive
    response), matching :attr:`~softae.analysis.eis_data.EISResult.z_imag_neg`.
    Sweep order is irrelevant — the arrays are sorted here, because the instrument
    sweeps high→low and every caller would otherwise have to remember that.

    **The robustness rule.** ``argmax`` alone lets one noisy point at the sweep
    floor declare an open arc, and the floor is exactly where a spectrum is
    noisiest. So a peak at the lowest frequency is read as OPEN only when the three
    lowest-frequency points are *rising toward low frequency* — a shape test rather
    than a threshold on the peak's height, which would have to be tuned against a
    noise level spanning three decades across this board. A genuine unclosed arc
    rises monotonically there; a lone excursion leaves its neighbours still falling
    and is rejected. Scatter of up to :data:`NOISE_TOL_REL` of the local ``−Z″``
    level does not count as a fall: at that size it is not evidence of a peak.

    The rule is deliberately biased: a spectrum it gets wrong is called CLOSED, so
    the cost of the guard is an occasional missed warning rather than an occasional
    false one. Over the 1440-spectrum characterisation run it reclassifies two
    spectra out of the 478 that bare ``argmax`` calls open.

    The three interior-apex fields are computed alongside and reported alongside;
    they never enter the rule above, so ``state`` and ``f_peak_hz`` are what they
    have always been on every spectrum this rig has stored.
    """
    f = np.asarray(freq, dtype=float).ravel()
    y = np.asarray(z_imag_neg, dtype=float).ravel()
    if f.size != y.size:
        return ArcClosure(UNKNOWN, reason=f"{f.size} frequencies vs {y.size} points")
    if f.size < min_points:
        return ArcClosure(UNKNOWN, reason=f"{f.size} points, need {min_points}")
    if not (np.isfinite(f).all() and np.isfinite(y).all()):
        return ArcClosure(UNKNOWN, reason="non-finite point in the sweep")

    order = np.argsort(f)
    f_s, y_s = f[order], y[order]
    if f_s[0] == f_s[-1] or y_s.max() == y_s.min():
        return ArcClosure(UNKNOWN, reason="degenerate sweep")

    ph = float("nan")
    if phase is not None:
        p = np.asarray(phase, dtype=float).ravel()
        if p.size == f.size:
            ph = float(p[order][0])

    # Parallel result, computed off the same sorted arrays so it inherits the
    # order-invariance above for free. It feeds only the three new fields — the
    # verdict below is the same code, in the same order, that it always was.
    interior: dict[str, float] = {}
    apex = _interior_apex(y_s)
    if apex >= 0 and y_s[apex] > 0.0:
        interior = {
            "f_apex_interior_hz": float(f_s[apex]),
            "apex_prominence_rel": _peak_prominence(y_s, apex) / float(y_s[apex]),
            "band_below_apex_decades": float(np.log10(f_s[apex] / f_s[0])),
        }

    peak = int(np.argmax(y_s))
    if peak != 0:
        return ArcClosure(CLOSED, float(f_s[peak]), float(f_s[0]), ph, **interior)

    trailing = y_s[:TRAILING_POINTS]
    # Each rise is judged against the point it climbs *to*, never against the floor
    # point: a spike there would otherwise inflate its own tolerance and hide the
    # very fall that exposes it.
    if np.any(np.diff(trailing) > NOISE_TOL_REL * np.abs(trailing[1:])):
        # The floor point is above its neighbours but they are still climbing with
        # frequency: the arc's real peak is up-sweep, so report that one.
        runner_up = 1 + int(np.argmax(y_s[1:]))
        return ArcClosure(CLOSED, float(f_s[runner_up]), float(f_s[0]), ph,
                          reason="lone excursion at the sweep floor", **interior)
    return ArcClosure(OPEN, float(f_s[0]), float(f_s[0]), ph, **interior)


def annotate_arc_closure(fit: Any, eis_result: Any) -> ArcClosure:
    """Attach the closure state to *fit*, and attach only that.

    ``success``, ``R1`` and ``parameters`` are untouched, unlike
    :func:`~softae.analysis.eis.engine._demote_if_railed`. A railed fit reports a
    property of ``CIRCUIT_MODELS``; an extrapolated R₁ still reports the film, and
    demoting a third of every run would cost far more evidence than it saved.
    """
    arc = arc_closure(
        getattr(eis_result, "frequency", ()),
        getattr(eis_result, "z_imag_neg", ()),
        getattr(eis_result, "phase", None),
    )
    fit.arc_closure = arc
    return arc
