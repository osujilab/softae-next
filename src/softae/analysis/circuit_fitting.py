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
# Fitting
# ---------------------------------------------------------------------------

def fit_circuit(eis_result, model_name: str = "simpleSalt", *,
                fit_plots: bool = False,
                fixed_params: dict[str, float] | None = None) -> FitResult:
    """Fit an equivalent-circuit model to EIS data.

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

    # Extract features for initial guesses
    features = extract_features(
        eis_result.frequency, eis_result.z_real, eis_result.z_imag_neg
    )
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
    Z = eis_result.z_real + 1j * (-eis_result.z_imag_neg)
    freq = eis_result.frequency

    try:
        from impedance.models.circuits import CustomCircuit  # type: ignore

        model = CustomCircuit(
            config["circuit"],
            initial_guess=initial_guess,
            constants=constants or {},
        )
        if bounds:
            model.fit(freq, Z, bounds=bounds)
        else:
            model.fit(freq, Z)

        params = model.parameters_

        # Capture fitted impedance for later overlay without re-fitting.
        try:
            _z_fit = model.predict(freq)
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
        )

    except Exception as exc:
        n_params = len(initial_guess)
        return FitResult(
            model_name=model_name,
            parameters=np.full(n_params, np.nan),
            R0=np.nan,
            R1=np.nan,
            R0_guess=r0_guess,
            R1_guess=r1_guess,
            z_indices=z_idx,
            success=False,
            error_msg=str(exc),
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
