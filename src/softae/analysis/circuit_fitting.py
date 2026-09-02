"""Equivalent-circuit fitting and conductivity analysis for EIS data.

Ports and streamlines the circuit-model definitions and fitting logic
from the legacy ``eis_analyzer.py``, using the structured
:class:`~softae.analysis.eis_data.EISResult` container.

Requires the ``impedance`` package (``pip install impedance``).

Circuit Models
--------------
Each entry in :data:`CIRCUIT_MODELS` contains the ``impedance``-compatible
circuit string, initial parameter guesses, optional bounds and constants,
and the indices into the fitted parameter vector for R0 (series resistance)
and R1 (bulk / charge-transfer resistance).

Example::

    from softae.analysis.eis.engine import analyze_spectrum
    from softae.analysis.eis.geometry import CellConstant
    from softae.analysis.eis_data import EISResult

    result = EISResult.load("data/sample_E1_eisdata.txt")
    report = analyze_spectrum(              # ``engine`` unset: ``[eis] engine`` decides
        result, cell=CellConstant.from_legacy(0.2, 0.175, 0.2), model_name="simpleSalt")
    if report.sigma.mode == "value":
        sigma = report.sigma.value          # S/cm

:func:`fit_circuit` remains the legacy engine's fitter and is called by it verbatim.
:func:`z_to_sigma` and :meth:`FitResult.sigma` are **deprecated**: they are kept only
as the independent parity oracle the test suite checks the new route against, and
emit :class:`DeprecationWarning` on call.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: Values of :attr:`FitResult.failure_kind`.  Three distinguishable refusals, because
#: they were indistinguishable in the GUI and got read as one thing.
FAILURE_TOO_FEW_POINTS = "too_few_points"
FAILURE_BUDGET_EXHAUSTED = "budget_exhausted"
FAILURE_FIT_ERROR = "fit_error"

# ---------------------------------------------------------------------------
# Circuit model registry
# ---------------------------------------------------------------------------

CIRCUIT_MODELS: dict[str, dict[str, Any]] = {
    "simpleSalt": {
        "circuit": "R0-CPE0-p(R1,C0)",
        "initial_guess": [1e2, 1e-7, 0.7, 1e3, 1e-10],
        "bounds": ([0, 1e-8, 0.4, 1e2, 1e-11], [np.inf, 9e-6, 0.9, np.inf, 5e-9]),
        "constants": None,
        "z_indices": [0, 3],
        "description": "Simple ionic conductor: R0 + CPE || (R1 + C)",
    },
    "flexSalt": {
        "circuit": "R0-CPE0-p(R1,C0)",
        "initial_guess": [None, 1e-7, 0.83, None, None],
        "bounds": None,
        "constants": {"C0": 2e-10},
        "z_indices": [0, 3],
        "description": "Flexible salt model with fixed stray capacitance",
    },
    "simpleSaltMembrane": {
        "circuit": "R0-CPE0-p(R1-Wo1,C0)",
        "initial_guess": [None, 1e-7, 0.83, None, 1e-2, 1e2, None],
        "bounds": None,
        "constants": None,
        "z_indices": [0, 3],
        "description": "Salt + membrane diffusion (Warburg open)",
    },
}


# ---------------------------------------------------------------------------
# Fit result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Structured output from :func:`fit_circuit`."""

    model_name: str
    parameters: np.ndarray
    R0: float
    R1: float
    R0_guess: float
    R1_guess: float
    z_indices: list[int]
    success: bool = True
    error_msg: str = ""
    # Fitted impedance evaluated at the measured frequencies (complex).  Stored
    # at fit time so callers can overlay it without rebuilding the circuit model.
    z_fit: np.ndarray | None = field(default=None)
    #: Goodness-of-fit metrics (P4.1): ``chi2``, ``chi2_reduced``, ``r_squared``,
    #: ``residual_rms_pct``, ``residual_max_pct``.  Computed at fit time, because
    #: a non-converged fit still reports an ``R1`` and the conductivity derived
    #: from it is indistinguishable from a good one without these.  Empty when
    #: the fit failed or produced no ``z_fit`` to compare against.
    quality: dict[str, float] = field(default_factory=dict)
    #: Parameter covariance from the gated engine's fitter (E1), as a
    #: ``softae.analysis.eis.fitter.FitCovariance``.  ``None`` on this legacy path,
    #: which fits through ``impedance.py``'s ``CustomCircuit`` — that discards
    #: ``pcov`` and returns only ``sqrt(diag(pcov))``, so the off-diagonal
    #: ``ρ(R_series, R_bulk)`` that decides whether the split may be reported at all
    #: is simply not available here.
    covariance: Any | None = field(default=None)
    #: Admission-gate log for the spectrum this fit came from (E0), in
    #: ``run_gates`` entry form.  Empty on the legacy path and on pre-E0 records.
    gate_log: list[dict[str, Any]] = field(default_factory=list)
    #: Which resistance the evidence licensed reporting: ``"split"`` | ``"sum"`` |
    #: ``"bound"`` | ``"bound_unqualified"``.  Legacy always reports ``"split"``,
    #: which is what it has always done.
    report_mode: str = "split"
    #: How many of the spectrum's points the optimiser actually saw, and how many
    #: :func:`usable_points` withheld from it.  These describe *this fitter's* mask
    #: only — the gated engine reports its own drops through ``gate_log`` — and they
    #: exist because a fit on 37 of 53 points and a fit on all 53 are different
    #: measurements that were previously indistinguishable in the result.
    n_points_used: int = 0
    n_points_dropped: int = 0
    #: *Why* the fit failed, as a token rather than as prose: ``""`` when it did not,
    #: else :data:`FAILURE_TOO_FEW_POINTS`, :data:`FAILURE_BUDGET_EXHAUSTED` or
    #: :data:`FAILURE_FIT_ERROR`.  It exists because the first two both surface in the
    #: GUI's Error column as a refusal and were read there as "the EIS gates are
    #: enabled" — neither is a gate, and they are not each other.  ``error_msg`` says
    #: which in words; this says which in a value a caller can branch on, and no
    #: existing field can (``success`` is one bit, ``n_points_dropped`` is nonzero for
    #: a *successful* masked fit too).
    failure_kind: str = ""

    def sigma(self, L: float, t: float, w: float) -> float:
        """Ionic conductivity σ = L / (R1 · t · w)  [S/cm].

        .. deprecated:: P.20
           Use :meth:`sigma_report` with a
           :class:`~softae.analysis.eis.geometry.CellConstant`, or take σ off the
           :class:`~softae.analysis.eis.report.SpectrumReport` that
           :func:`~softae.analysis.eis.engine.analyze_spectrum` returns.

        This existed because the GUI "expect[s] a bare float" — an expectation P.16
        retired. It survives as the independent parity oracle: the tests assert
        ``cell.sigma(R) == z_to_sigma(L, t, w, R)``, and that proof is only worth
        anything while the two implementations stay unrelated.
        """
        warnings.warn(
            "FitResult.sigma is deprecated; use FitResult.sigma_report(cell) or "
            "analyze_spectrum(...).sigma. It is retained only as the parity oracle.",
            DeprecationWarning,
            stacklevel=2,
        )
        return z_to_sigma(L, t, w, self.R1)

    def sigma_report(self, cell, *, envelope=None):
        """R2/R6-compliant conductivity for a per-sample :class:`CellConstant`.

        A sibling of :meth:`sigma`, never a replacement: that method keeps its
        signature *and* its formula, because the GUI, the Arrhenius sweep and the web
        layer all call it and all expect a bare float.

        Returns a :class:`softae.analysis.eis.report.SigmaReport`, which can say
        "upper bound" or "unavailable" — things a float cannot.
        """
        from softae.analysis.eis.report import SigmaReport

        if cell is None:
            return SigmaReport(mode="unavailable", R_reported_ohm=float(self.R1))
        return SigmaReport(
            mode="value",
            value=cell.sigma(self.R1),
            R_reported_ohm=float(self.R1),
            R_basis="split_bulk",
            K_per_cm=cell.K_per_cm,
            K_route=cell.K_route,
            thickness_method=cell.thickness_method,
        )


# ---------------------------------------------------------------------------
# Feature extraction (ported from eis_analyzer.py)
# ---------------------------------------------------------------------------

def _local_minima(arr: np.ndarray, window: int = 5) -> list[int]:
    """Return indices of local minima within a sliding window."""
    n = len(arr)
    minima: list[int] = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbours = np.concatenate([arr[lo:i], arr[i + 1:hi]])
        if len(neighbours) > 0 and np.all(arr[i] < neighbours):
            minima.append(i)
    return minima


def extract_features(freq: np.ndarray, z_real: np.ndarray,
                     z_imag_neg: np.ndarray) -> dict[str, Any]:
    """Extract characteristic impedance features for initial-guess estimation.

    This replaces the legacy ``parse_input`` feature extraction with a
    cleaner dict-based return.

    Returns
    -------
    dict with keys:
        z_real_min, z_real_max, z_imag_min, z_imag_max,
        z_real_local_min_idx, z_imag_local_min_idx,
        r0_guess, r1_guess, min_pairs
    """
    zr_min_idx = int(np.argmin(z_real))
    zr_max_idx = int(np.argmax(z_real))
    zi_min_idx = int(np.argmin(z_imag_neg))
    zi_max_idx = int(np.argmax(z_imag_neg))

    zi_local_min = _local_minima(z_imag_neg)

    # Initial guesses for R0 and CT resistance
    r0_guess = float(z_real[zr_min_idx])
    if r0_guess < 0:
        r0_guess = 0.0

    # Use the Z'/Z'' pair at the lowest-frequency local Z'' minimum
    if zi_local_min:
        pair_idx = zi_local_min[-1]  # lowest-frequency local min
        ct_guess = float(z_real[pair_idx]) - r0_guess
        min_pairs = [float(z_real[pair_idx]), float(z_imag_neg[pair_idx])]
    else:
        ct_guess = float(z_real[zr_max_idx]) - r0_guess
        min_pairs = [float(z_real[zr_max_idx]), float(z_imag_neg[zr_max_idx])]

    if ct_guess <= 0:
        ct_guess = r0_guess
        r0_guess = 0.0
    if ct_guess < r0_guess:
        r0_guess, ct_guess = ct_guess, r0_guess

    return {
        "z_real_min": float(z_real[zr_min_idx]),
        "z_real_max": float(z_real[zr_max_idx]),
        "z_imag_min": float(z_imag_neg[zi_min_idx]),
        "z_imag_max": float(z_imag_neg[zi_max_idx]),
        "z_real_local_min_idx": _local_minima(z_real),
        "z_imag_local_min_idx": zi_local_min,
        "r0_guess": r0_guess,
        "r1_guess": ct_guess,
        "min_pairs": min_pairs,
    }


# ---------------------------------------------------------------------------
# Admissible points
# ---------------------------------------------------------------------------

#: Fallback for ``[quality] min_points`` when the config system cannot be reached at
#: all.  Equal to :data:`softae.analysis.quality.DEFAULT_MIN_POINTS`, which is also the
#: shipped value, so this changes nothing when it fires — it exists so that a fitter
#: cannot be silently relaxed by a broken loader.
FALLBACK_MIN_FIT_POINTS = 8


def usable_points(freq, z_real, z_imag_neg) -> np.ndarray:
    """Which points may reach the optimiser: finite ``Z``, finite ``f``, ``f > 0``.

    Deliberately the **same predicate** as
    :func:`softae.analysis.eis.gates.gate_finiteness` and
    :func:`softae.analysis.eis.fitter._fit_with_covariance` — ``isfinite(f) &
    isfinite(Z) & (f > 0)`` — because two definitions of "usable point" in one codebase
    is the defect one layer up from the one this fixes.

    Three things that predicate settles, none of them obvious:

    * **Real and imaginary parts are judged jointly, not separately.** ``np.isfinite``
      on a complex array is already the conjunction, and it has to be: ``curve_fit``
      is handed ``hstack([Z.real, Z.imag])``, so a point whose ``Z''`` is NaN cannot
      contribute its ``Z'`` either. Half a point is not a point.
    * **Frequency is masked as well as impedance.** ``curve_fit`` only rejects a
      non-finite *ordinate*, so a NaN frequency would pass its check and then poison
      the circuit evaluation instead of failing loudly. ``f ≤ 0`` goes for the same
      reason it does in ``gate_finiteness``: the model is evaluated at ``jω``, and a
      non-positive frequency is not a measurement.
    * **Duplicate frequencies are *not* dropped here**, though ``gate_finiteness``
      drops them. That check protects the Kramers–Kronig basis and the topology
      triad's ``polyfit(log10(f), …)``, neither of which runs on this path, and
      removing points from spectra that fit correctly today would be a behaviour
      change unrelated to finiteness. ``fitter.py`` — the closer precedent, being a
      fitter rather than a gate — omits it for the same reason.

    Returns a boolean mask over the first ``min(len(freq), len(z_real),
    len(z_imag_neg))`` points.
    """
    f = np.asarray(freq, dtype=float)
    zr = np.asarray(z_real, dtype=float)
    zi = np.asarray(z_imag_neg, dtype=float)
    n = int(min(f.size, zr.size, zi.size))
    return (
        np.isfinite(f[:n]) & np.isfinite(zr[:n]) & np.isfinite(zi[:n]) & (f[:n] > 0)
    )


def _min_fit_points(override: int | None = None) -> int:
    """How many usable points a fit must be supported by.

    Sourced from ``[quality] min_points``, which is not a borrowed number: its existing
    consumer :func:`softae.analysis.quality.validate_eis_trace` already spells it
    ``if n_finite < min_points`` and calls the result "usable points". That is the same
    question asked here, of the same spectrum, one stage earlier — so a second knob
    would only let the trace validator and the fitter disagree about whether the same
    file is fittable.

    ``[eis.gates] min_fit_pts`` is the other candidate and is deliberately not used:
    it belongs to the gated engine's settings object, which this legacy path does not
    build and must not start depending on. Both ship at 8.
    """
    if override is not None:
        return int(override)
    try:
        from softae.analysis.quality import quality_config

        return int(quality_config()["min_points"])
    except Exception:  # config system unreachable — do not relax the requirement
        logger.warning("eis_fit_min_points_unresolved", exc_info=True)
        return FALLBACK_MIN_FIT_POINTS


# ---------------------------------------------------------------------------
# Optimiser budget
# ---------------------------------------------------------------------------

#: Shipped default for ``[eis] legacy_max_nfev``, and the fallback when the config
#: system cannot be reached.  Deliberately the same 2000 as ``[eis.pregate] max_nfev``
#: so the codebase carries one number for "how long may a fit run" rather than two.
#:
#: The gap it sits in is three orders wide and empty.  Over the 54-spectrum
#: ``20260825T154521Z_arrhenius_sweep`` corpus the worst *converging* fit costs 65
#: function evaluations on ``simpleSalt`` and 87 on ``flexSalt`` (median 12, p95 31),
#: while the one pathological spectrum — measurement 3840, ch20, 60.1 C — spends
#: ~100,000 over 348 s at 100 % CPU and then raises anyway.  2000 is ~23x headroom
#: over the worst success and 2 % of the cost of the one failure.
DEFAULT_LEGACY_MAX_NFEV = 2000

#: ``curve_fit``'s wording when it runs out of evaluations.  Matched as a substring
#: because it arrives wrapped: impedance.py lets ``scipy`` raise, and the text is
#: ``"Optimal parameters not found: The maximum number of function evaluations is
#: exceeded."``  Lower-cased before comparison so the leading "The" cannot matter.
_BUDGET_EXHAUSTED_SIGNATURE = "maximum number of function evaluations is exceeded"


def _legacy_max_nfev(override: int | None = None) -> int | None:
    """The optimiser's evaluation budget for this legacy path, or ``None`` for none.

    Read from ``[eis] legacy_max_nfev`` through :mod:`softae.config.loader` — the same
    single parse point :func:`softae.analysis.eis.settings.eis_settings` reads ``[eis]
    engine`` from, so a test that patches the loader moves both and there is no second
    mechanism to keep in step.  It is read *here* rather than added to ``EISSettings``
    because ``EISSettings`` is the **gated** engine's settings object, which this path
    does not build and must not start depending on — exactly the reasoning
    :func:`_min_fit_points` gives for preferring ``[quality] min_points`` over
    ``[eis.gates] min_fit_pts``.

    **Zero or negative means uncapped**, and returns ``None`` so the caller omits the
    kwarg entirely and inherits impedance.py's own ``maxfev = 1e5``.  That is the
    pre-cap behaviour restored exactly, not approximated by a large number.

    A caller-supplied *override* wins over configuration, matching
    :func:`_min_fit_points`'s precedence.
    """
    if override is not None:
        raw: Any = override
    else:
        try:
            from softae.config import loader

            raw = (loader.load().get("eis", {}) or {}).get(
                "legacy_max_nfev", DEFAULT_LEGACY_MAX_NFEV
            )
        except Exception:  # config system unreachable — cap anyway, do not hang
            logger.warning("eis_fit_max_nfev_unresolved", exc_info=True)
            raw = DEFAULT_LEGACY_MAX_NFEV

    try:
        budget = int(raw)
    except (TypeError, ValueError):
        logger.warning("eis_fit_max_nfev_uninterpretable", value=repr(raw))
        budget = DEFAULT_LEGACY_MAX_NFEV
    return budget if budget > 0 else None


def _budget_exhausted_message(
    budget: int | None, model_name: str, n_dropped: int = 0
) -> str:
    """The refusal an operator reads when the optimiser ran out of evaluations.

    Written to be unmistakable against the *other* refusal this module produces — the
    ``usable_points`` remnant message — because in the GUI's Error column both are just
    red text, and one was read as "the EIS gates are enabled". Neither is a gate, so
    this says so in words rather than leaving it to be inferred.

    *n_dropped* is taken rather than assumed: a spectrum can lose points to
    :func:`usable_points` **and then** exhaust the budget on what survived, and a
    message that flatly claimed "no points were withheld" would be false exactly there —
    which is the same "unknown spelled as clean" mistake this message exists to undo.

    ASCII only, deliberately: this string reaches a Qt table, a console and a log file,
    and the em dashes elsewhere in this module do not survive all three.
    """
    if budget is None:
        limit = "impedance.py's own ceiling (the [eis] legacy_max_nfev cap is disabled)"
        knob = "Set [eis] legacy_max_nfev to a positive number to fail faster instead."
    else:
        limit = f"its budget of {budget} function evaluations"
        knob = (
            "Raise [eis] legacy_max_nfev to allow a longer fit, or set it to 0 or a "
            "negative number for no cap."
        )
    points = (
        "no points were withheld for being non-finite"
        if n_dropped == 0
        else f"separately, {n_dropped} non-finite point(s) had been dropped first"
    )
    return (
        f"optimiser evaluation budget exhausted: the '{model_name}' fit stopped at "
        f"{limit} without converging, so this spectrum has no fitted R1. "
        f"This is the fitter's own iteration limit, NOT a gate - no gate rejected "
        f"this spectrum and {points}. {knob}"
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_circuit(eis_result, model_name: str = "simpleSalt", *,
                fit_plots: bool = False,
                fixed_params: dict[str, float] | None = None,
                min_points: int | None = None,
                max_nfev: int | None = None) -> FitResult:
    """Fit an equivalent-circuit model to EIS data.

    Non-finite points are **masked, not fatal.** ``curve_fit`` calls
    ``asarray_chkfinite`` on its ordinate, so before this every spectrum carrying a
    single NaN failed whole: ``ch22_003`` failed with 52 good points of 53, exactly as
    its sibling with 37 of 53 did, and the pair were read as a hardware fault for two
    days because a spectrum that cannot be fitted looks like a spectrum that is wrong.
    See :func:`usable_points` for what "usable" means and why it is that and not
    something else.

    The mask is applied to the initial-guess extraction as well as to the optimiser,
    which is not optional: ``extract_features`` takes ``np.argmin``, and ``argmin``
    returns the index of the NaN, so a masked fit given unmasked guesses starts from
    ``r0_guess = nan``.

    **Nothing is dropped silently.** Every drop is logged and counted into
    ``FitResult.n_points_dropped``, and a spectrum reduced below *min_points* usable
    points is refused with a message saying so rather than fitted on the remnant.

    **The optimiser is given a budget, and running out of it is safe.** Without one it
    inherits impedance.py's ``maxfev = 1e5`` at ``ftol = 1e-13``, which on measurement
    3840 of ``20260825T154521Z_arrhenius_sweep`` burned **348 s at 100 % CPU** and then
    raised regardless — the GUI's Arrhenius fit looked hung because, for that one
    spectrum in 54, it effectively was.  A cap cannot degrade a result silently:
    ``curve_fit`` **raises** when the budget is exhausted rather than returning a
    best-so-far, so a cap either does not fire (bit-identical fit) or fires and the fit
    fails as it was already going to.  Measured across caps from 20 to 5000 on that
    corpus, ``max |ΔR1|/R1 = 0.000``.  See :func:`_legacy_max_nfev`.

    Parameters
    ----------
    eis_result : EISResult
        Structured impedance data.
    model_name : str
        Key in :data:`CIRCUIT_MODELS`.
    fit_plots : bool
        If True, display Nyquist / Bode fit overlays (requires matplotlib).
    fixed_params : dict, optional
        Parameter values to hold fixed during fitting.
    min_points : int, optional
        Override the ``[quality] min_points`` floor. For tests that need to pin the
        threshold rather than inherit the operator's configuration.
    max_nfev : int, optional
        Override the ``[eis] legacy_max_nfev`` optimiser budget. ``0`` or negative
        means no cap — impedance.py's ``maxfev = 1e5``, the pre-cap behaviour.

    Returns
    -------
    FitResult
    """
    from softae.analysis.eis_data import EISResult  # deferred to avoid circular

    if model_name not in CIRCUIT_MODELS:
        raise ValueError(
            f"Unknown circuit model '{model_name}'. "
            f"Available: {list(CIRCUIT_MODELS)}"
        )

    config = CIRCUIT_MODELS[model_name]
    z_idx = config["z_indices"]

    mask = usable_points(
        eis_result.frequency, eis_result.z_real, eis_result.z_imag_neg
    )
    n_total = int(mask.size)
    n_kept = int(mask.sum())
    n_dropped = n_total - n_kept

    fit_freq = np.asarray(eis_result.frequency, dtype=float)[:n_total]
    fit_z_real = np.asarray(eis_result.z_real, dtype=float)[:n_total]
    fit_z_imag_neg = np.asarray(eis_result.z_imag_neg, dtype=float)[:n_total]

    if n_dropped:
        # The floor is checked only when this mask actually removed something. A short
        # all-finite sweep is an operator's deliberate choice of preset and has always
        # been fitted; revoking that would be a capability change smuggled in under a
        # finiteness fix. What is new here is the *remnant*, and a remnant is what this
        # guard exists to refuse.
        need = _min_fit_points(min_points)
        if n_kept < need:
            reason = (
                f"only {n_kept} of {n_total} points are usable (need {need}): "
                f"{n_dropped} point(s) dropped as non-finite or f <= 0"
            )
            logger.warning(
                "eis_fit_refused_too_few_points", model=model_name,
                n_total=n_total, n_used=n_kept, n_dropped=n_dropped, need=need,
                detail=reason,
            )
            return FitResult(
                model_name=model_name,
                parameters=np.full(len(config["initial_guess"]), np.nan),
                R0=np.nan, R1=np.nan, R0_guess=np.nan, R1_guess=np.nan,
                z_indices=z_idx,
                success=False,
                error_msg=reason,
                failure_kind=FAILURE_TOO_FEW_POINTS,
                n_points_used=n_kept,
                n_points_dropped=n_dropped,
            )

        fit_freq = fit_freq[mask]
        fit_z_real = fit_z_real[mask]
        fit_z_imag_neg = fit_z_imag_neg[mask]
        # The surviving band is the useful half of this message: on ch22_001 the
        # dropped points are a contiguous run at the *top* (6.45 kHz-200 kHz), which
        # reads as a high-frequency resolution limit rather than scattered noise.
        logger.info(
            "eis_fit_points_dropped", model=model_name,
            n_total=n_total, n_used=n_kept, n_dropped=n_dropped,
            detail=(
                f"{n_dropped} of {n_total} points non-finite or f <= 0; fitting the "
                f"remaining {n_kept} over "
                f"{fit_freq.min():.4g}-{fit_freq.max():.4g} Hz"
            ),
        )

    # Extract features for initial guesses — on the masked arrays, because argmin
    # over a NaN returns the NaN.
    features = extract_features(fit_freq, fit_z_real, fit_z_imag_neg)
    r0_guess = features["r0_guess"]
    r1_guess = features["r1_guess"]

    # Build initial guess and constants based on model
    initial_guess = list(config["initial_guess"])
    bounds = config["bounds"]
    constants = dict(config["constants"]) if config["constants"] else None

    if model_name == "simpleSalt":
        initial_guess = [r0_guess, 1e-7, 0.7, r1_guess, 1e-10]
    elif model_name == "flexSalt":
        if fixed_params and "R0" in fixed_params and "R1" in fixed_params:
            initial_guess = [None, 8e-5, 0.83, None, None]
            constants = {
                "R0": fixed_params["R0"],
                "R1": fixed_params["R1"],
                "C0": 2e-10,
            }
        else:
            initial_guess = [r0_guess, 1e-7, 0.83, r1_guess, None]
            constants = {"C0": 2e-10}
    elif model_name == "simpleSaltMembrane":
        initial_guess = [r0_guess, 1e-7, 0.83, r1_guess, 1e-2, 1e2, None]
        constants = {"R0": r0_guess, "R1": r1_guess, "C0": 2e-10}

    # Build complex impedance: Z = Z' + jZ'' (note Z'' stored as -Z'')
    #
    # ``Z``/``freq`` are the *masked* arrays — they are what the optimiser sees.  The
    # full grid is kept separately because ``z_fit`` must stay aligned with
    # ``eis_result``: ``compute_fit_quality`` indexes the two together and applies its
    # own finiteness mask, and ``tab_analysis`` overlays ``z_fit`` on the measured
    # trace. A ``z_fit`` one element shorter than the spectrum it annotates would
    # silently shift every point after the drop.
    Z = fit_z_real + 1j * (-fit_z_imag_neg)
    freq = fit_freq
    full_freq = np.asarray(eis_result.frequency, dtype=float)

    budget = _legacy_max_nfev(max_nfev)
    # ``maxfev``, not ``max_nfev``, and the distinction is not cosmetic on either call
    # site.  ``circuit_fit`` fills ``bounds`` with per-element defaults when the caller
    # passes none, so **both** branches below reach ``curve_fit`` bounded and therefore
    # run ``trf``, whose own kwarg is ``max_nfev``.  ``curve_fit`` bridges that itself —
    # ``if 'max_nfev' not in kwargs: kwargs['max_nfev'] = kwargs.pop('maxfev', None)`` —
    # so passing ``max_nfev`` would leave impedance.py's unconditional ``maxfev = 1e5``
    # unpopped beside it and raise ``TypeError: least_squares() got an unexpected
    # keyword argument 'maxfev'``.  Verified both ways against impedance 1.7.1 /
    # scipy 1.17.1 before this comment was written.
    fit_kwargs: dict[str, Any] = {} if budget is None else {"maxfev": budget}

    try:
        from impedance.models.circuits import CustomCircuit  # type: ignore

        model = CustomCircuit(
            config["circuit"],
            initial_guess=initial_guess,
            constants=constants or {},
        )
        if bounds:
            model.fit(freq, Z, bounds=bounds, **fit_kwargs)
        else:
            model.fit(freq, Z, **fit_kwargs)

        params = model.parameters_

        # Capture fitted impedance for later overlay without re-fitting.  Predicted on
        # the full grid: the circuit is defined at every frequency, whether or not the
        # instrument returned a number there.
        try:
            _z_fit = model.predict(full_freq)
        except Exception:
            _z_fit = None

        if fit_plots:
            import matplotlib.pyplot as plt
            model.plot(f_data=freq, Z_data=Z, kind="nyquist")
            plt.tight_layout()
            plt.show()
            model.plot(f_data=freq, Z_data=Z, kind="bode")
            plt.tight_layout()
            plt.show()

        # Grade the fit against the data it was fitted to, while both are in
        # hand — a caller inspecting the result later has no way to tell a
        # converged-but-wrong fit from a good one.
        try:
            from softae.analysis.quality import compute_fit_quality

            _quality = compute_fit_quality(eis_result, _z_fit, n_params=len(params))
        except Exception:
            _quality = {}

        return FitResult(
            model_name=model_name,
            parameters=params,
            R0=float(params[z_idx[0]]),
            R1=float(params[z_idx[1]]),
            R0_guess=r0_guess,
            R1_guess=r1_guess,
            z_indices=z_idx,
            z_fit=_z_fit,
            quality=_quality,
            n_points_used=n_kept,
            n_points_dropped=n_dropped,
        )

    except Exception as exc:
        n_params = len(initial_guess)
        if _BUDGET_EXHAUSTED_SIGNATURE in str(exc).lower():
            kind = FAILURE_BUDGET_EXHAUSTED
            reason = _budget_exhausted_message(budget, model_name, n_dropped)
            logger.warning(
                "eis_fit_budget_exhausted", model=model_name, max_nfev=budget,
                n_used=n_kept, n_dropped=n_dropped, detail=reason,
            )
        else:
            kind = FAILURE_FIT_ERROR
            reason = str(exc)
        return FitResult(
            model_name=model_name,
            parameters=np.full(n_params, np.nan),
            R0=np.nan,
            R1=np.nan,
            R0_guess=r0_guess,
            R1_guess=r1_guess,
            z_indices=z_idx,
            success=False,
            error_msg=reason,
            failure_kind=kind,
            n_points_used=n_kept,
            n_points_dropped=n_dropped,
        )


# ---------------------------------------------------------------------------
# Fit-curve reconstruction
# ---------------------------------------------------------------------------

def predict_fit_curve(
    fit_result: "FitResult",
    freq: np.ndarray,
    model_name: str | None = None,
) -> np.ndarray | None:
    """Rebuild the circuit from a :class:`FitResult` and return predicted Z.

    Reconstructs the ``impedance`` ``CustomCircuit`` for the fit's model using
    ``CIRCUIT_MODELS`` plus the stored parameter vector, then evaluates it at
    *freq*, returning the complex impedance array.

    This is the single shared implementation of the "rebuild circuit → predict"
    logic that overlay/thumbnail plotting code across the GUI and web layers
    used to reproduce inline.

    Parameters
    ----------
    fit_result : FitResult
        A completed fit; ``fit_result.parameters`` supplies the circuit's
        initial guess and ``fit_result.model_name`` selects the topology
        (unless *model_name* overrides it).
    freq : np.ndarray
        Frequencies (Hz) at which to evaluate the model.
    model_name : str, optional
        Circuit-model key in :data:`CIRCUIT_MODELS`. Defaults to
        ``fit_result.model_name``.

    Returns
    -------
    np.ndarray or None
        Complex impedance evaluated at *freq*, or ``None`` if the ``impedance``
        backend is unavailable, the model is unknown, or prediction fails.
    """
    name = model_name if model_name is not None else fit_result.model_name
    cfg = CIRCUIT_MODELS.get(name)
    if cfg is None:
        return None
    try:
        from impedance.models.circuits import CustomCircuit  # type: ignore

        model = CustomCircuit(
            cfg["circuit"],
            initial_guess=fit_result.parameters.tolist(),
        )
        return model.predict(np.asarray(freq))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Conductivity
# ---------------------------------------------------------------------------

def z_to_sigma(L: float, t: float, w: float, R1: float | np.ndarray) -> float | np.ndarray:
    """Convert impedance to ionic conductivity.

    σ = L / (R1 · t · w)

    .. deprecated:: P.20
       Build a :class:`~softae.analysis.eis.geometry.CellConstant` and use
       :meth:`CellConstant.sigma`, or read σ off the
       :class:`~softae.analysis.eis.report.SpectrumReport` returned by
       :func:`~softae.analysis.eis.engine.analyze_spectrum` — which is the only
       surface that honours ``[eis] engine``. Production callers: zero.

    **Kept, not deleted, and deliberately not reimplemented in terms of
    ``CellConstant``.** This function is the independent oracle the parity tests
    check the new route against (``tests/test_eis_geometry.py``,
    ``tests/test_gui_eis_sigma_source.py``). Wiring it into the survivor would turn
    ``cell.sigma(R) == z_to_sigma(L, t, w, R)`` into ``x == x``: green forever, and
    proving nothing at exactly the moment the gated engine starts moving numbers.

    Parameters
    ----------
    L : float
        Electrode separation (cm).
    t : float
        Sample thickness (cm).
    w : float
        Electrode width (cm).
    R1 : float or array
        Bulk impedance (Ω).

    Returns
    -------
    float or ndarray
        Conductivity in S/cm.
    """
    warnings.warn(
        "z_to_sigma is deprecated; use CellConstant.sigma or "
        "analyze_spectrum(...).sigma. It is retained only as the parity oracle.",
        DeprecationWarning,
        stacklevel=2,
    )
    return L / (np.asarray(R1) * t * w)


# ---------------------------------------------------------------------------
# Styled EIS fit diagnostic plot
# ---------------------------------------------------------------------------

def compute_fit_residuals(eis_result, z_fit_complex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return residuals (%) for Z' and -Z'' channels.

    Residual definition for each channel:

    residual = (measured - fitted) / abs(measured) * 100
    """
    eps = 1e-12
    z_real_fit = np.real(z_fit_complex)
    z_imag_neg_fit = -np.imag(z_fit_complex)

    resid_real = (eis_result.z_real - z_real_fit) / np.maximum(np.abs(eis_result.z_real), eps) * 100.0
    resid_imag = (eis_result.z_imag_neg - z_imag_neg_fit) / np.maximum(np.abs(eis_result.z_imag_neg), eps) * 100.0
    return resid_real, resid_imag


def plot_eis_fit(
    eis_result, fit_result: FitResult | None = None, *, show: bool = True, fig=None
) -> "plt.Figure":
    """Plot Nyquist, Bode, and split fit residuals with a consistent style.

    Layout (2 rows × 2 columns): Nyquist, Bode, residual Z', residual -Z''.

    **``fit_result`` is optional**, and passing ``None`` is what an unfitted
    measurement should use. The data panes are identical either way — same palette,
    same markers, same axes, same Nyquist aspect ratio — so turning auto-fit off
    changes *what is known about* the spectrum, not how it is drawn. Only the
    residual row is dropped, because residuals of no model do not exist; the figure
    becomes a single row of Nyquist + Bode at full height rather than two half-empty
    panes.

    That is a distinct state from a fit that was *attempted and failed*
    (``fit_result.success is False``), which keeps the 2×2 layout and says so in the
    residual panes — asking for a fit and not getting one is worth seeing.

    Pass ``fig`` to draw into an existing :class:`matplotlib.figure.Figure` (e.g.
    an embedded Qt canvas) instead of creating a pyplot-managed one; ``plt.show``
    is then never called (embedding owns the display).

    Style convention
    ----------------
    * Measured data  → blue filled circles (``"o"``, ``color="#1f77b4"``)
    * Model fit line → golden-yellow solid line (``color="#FFB300"``)
    * Residuals      → grey stems with red zero-line (separate real/imag)

    Parameters
    ----------
    eis_result : EISResult
        Structured impedance measurement.
    fit_result : FitResult
        Output of :func:`fit_circuit`.
    show : bool
        If True, call ``plt.show()`` (blocks in interactive sessions).
        Set False for embedded / testing use.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from softae.analysis.eis_data import EISResult  # deferred, avoid circular

    # Wong (2011) colour-blind-safe palette — also distinguishable in greyscale
    Z_REAL_COLOR  = "#000000"   # black      — Z′  data
    Z_IMAG_COLOR  = "#D55E00"   # vermillion — −Z″ data
    PHASE_COLOR   = "#0072B2"   # blue       — phase data
    FIT_COLOR     = "#E69F00"   # orange     — all fit lines
    RESID_COLOR   = "#555555"   # dark grey  — residual stems

    freq   = eis_result.frequency
    z_real = eis_result.z_real
    z_imag = eis_result.z_imag_neg   # stored as -Z'', positive upward
    z_mag  = eis_result.z_magnitude

    # No fit object at all is a *different* state from a fit that failed, and the
    # two are reported differently below.
    no_fit_requested = fit_result is None

    # Build model-predicted impedance if the fit succeeded
    z_fit_complex: np.ndarray | None = None
    if not no_fit_requested and fit_result.success:
        z_fit_complex = predict_fit_curve(fit_result, freq)

    created_fig = fig is None
    if created_fig:
        fig = plt.figure(figsize=(12, 5) if no_fit_requested else (12, 9))
    if no_fit_requested:
        subtitle = "  (raw — auto-fit off)"
    elif fit_result.success:
        subtitle = f"  [{fit_result.model_name}]"
    else:
        subtitle = f"  [{fit_result.model_name}]  ⚠ fit failed"
    fig.suptitle(
        f"EIS Diagnostic — CH{eis_result.channel}{subtitle}",
        fontsize=13,
        fontweight="bold",
    )

    # Residuals of no model do not exist, so an unfitted spectrum gets one full-height
    # row rather than two panes explaining their own emptiness.
    if no_fit_requested:
        gs = fig.add_gridspec(1, 2, wspace=0.35)
    else:
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35, height_ratios=[2, 1])
    # Nyquist and Bode first in both modes, so ``fig.axes[0]`` and ``[1]`` are the
    # data panes regardless of whether residuals exist — anything indexing the
    # figure then means the same thing either way.
    ax_nyq = fig.add_subplot(gs[0, 0])
    ax_bode = fig.add_subplot(gs[0, 1])
    if no_fit_requested:
        ax_res_real = ax_res_imag = None
    else:
        ax_res_real = fig.add_subplot(gs[1, 0])
        ax_res_imag = fig.add_subplot(gs[1, 1])

    # ── Nyquist ──────────────────────────────────────────────────────────────
    ax_nyq.scatter(z_real, z_imag, color=Z_REAL_COLOR, s=30, zorder=2,
                   marker="o", label="Measured")
    if z_fit_complex is not None:
        ax_nyq.plot(z_fit_complex.real, -z_fit_complex.imag,
                    color=FIT_COLOR, linewidth=2, zorder=4, label="Model fit")
    ax_nyq.set_xlabel("Z′ (Ω)")
    ax_nyq.set_ylabel("−Z″ (Ω)")
    ax_nyq.set_title("Nyquist")
    ax_nyq.set_aspect("equal", adjustable="datalim")
    ax_nyq.grid(True, linestyle="--", alpha=0.5)
    ax_nyq.legend(fontsize=9)

    # ── Bode ─────────────────────────────────────────────────────────────────
    ax_bode_phase = ax_bode.twinx()

    # Left axis: Z′ and −Z″ (log–log)
    ax_bode.scatter(freq, z_real,    color=Z_REAL_COLOR, s=20, marker="o",
                    zorder=2, label="Z′ meas.")
    ax_bode.scatter(freq, z_imag,    color=Z_IMAG_COLOR, s=20, marker="s",
                    zorder=2, label="−Z″ meas.")
    ax_bode.set_xscale("log")
    ax_bode.set_yscale("log")
    ax_bode.set_xlabel("Frequency (Hz)")
    ax_bode.set_ylabel("Z′, −Z″ (Ω)")

    # Right axis: phase
    phase_measured = -eis_result.phase   # convention: positive peak downward
    ax_bode_phase.scatter(freq, phase_measured, color=PHASE_COLOR, s=15,
                          marker="^", zorder=2, label="Phase meas.")
    ax_bode_phase.set_ylabel("−Phase (°)", color=PHASE_COLOR)
    ax_bode_phase.tick_params(axis="y", labelcolor=PHASE_COLOR)

    # Fit overlays
    if z_fit_complex is not None:
        z_fit_real     =  np.real(z_fit_complex)
        z_fit_imag_neg = -np.imag(z_fit_complex)
        z_fit_phase    = -np.angle(z_fit_complex, deg=True)

        ax_bode.plot(freq, z_fit_real,     color=FIT_COLOR, linewidth=2,
                     linestyle="-",  zorder=4, label="Z′ fit")
        ax_bode.plot(freq, z_fit_imag_neg, color=FIT_COLOR, linewidth=2,
                     linestyle="--", zorder=4, label="−Z″ fit")
        ax_bode_phase.plot(freq, z_fit_phase, color=FIT_COLOR, linewidth=2,
                           linestyle=":",  zorder=4, label="Phase fit")

    ax_bode.set_title("Bode")
    ax_bode.grid(True, which="both", linestyle="--", alpha=0.4)

    # Combined legend for both y-axes
    lines1, labels1 = ax_bode.get_legend_handles_labels()
    lines2, labels2 = ax_bode_phase.get_legend_handles_labels()
    ax_bode.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # ── Residuals (split by channel) ────────────────────────────────────────
    if z_fit_complex is not None:
        resid_real, resid_imag = compute_fit_residuals(eis_result, z_fit_complex)

        ax_res_real.axhline(0, color="red", linewidth=1.0, zorder=1)
        markerline_r, stemlines_r, _ = ax_res_real.stem(
            freq, resid_real, linefmt=RESID_COLOR, markerfmt="o", basefmt=" "
        )
        plt.setp(stemlines_r, linewidth=0.8)
        plt.setp(markerline_r, color=RESID_COLOR, markersize=4)
        ax_res_real.set_xscale("log")
        ax_res_real.set_xlabel("Frequency (Hz)")
        ax_res_real.set_ylabel("Residual (%)")
        ax_res_real.set_title("Z' Residuals")
        ax_res_real.grid(True, linestyle="--", alpha=0.4)

        ax_res_imag.axhline(0, color="red", linewidth=1.0, zorder=1)
        markerline_i, stemlines_i, _ = ax_res_imag.stem(
            freq, resid_imag, linefmt=RESID_COLOR, markerfmt="o", basefmt=" "
        )
        plt.setp(stemlines_i, linewidth=0.8)
        plt.setp(markerline_i, color=RESID_COLOR, markersize=4)
        ax_res_imag.set_xscale("log")
        ax_res_imag.set_xlabel("Frequency (Hz)")
        ax_res_imag.set_ylabel("Residual (%)")
        ax_res_imag.set_title("-Z'' Residuals")
        ax_res_imag.grid(True, linestyle="--", alpha=0.4)

        # Keep residuals on the result object so save() can optionally persist
        # them as additional columns in the EIS text output.
        eis_result.residual_real_pct = resid_real
        eis_result.residual_imag_pct = resid_imag
    elif ax_res_real is not None and ax_res_imag is not None:
        # A fit was attempted and failed — say so where the residuals would be. (When
        # no fit was requested these axes do not exist, so there is nothing to label.)
        for ax, title in ((ax_res_real, "Z' Residuals"), (ax_res_imag, "-Z'' Residuals")):
            ax.text(
                0.5,
                0.5,
                "No fit available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#888888",
                fontsize=10,
            )
            ax.set_title(title)
            ax.set_axis_off()

    if show and created_fig:
        plt.show()

    return fig
