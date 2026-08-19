"""One entry point, two engines, one return shape.

``analyze_spectrum`` is the only function the rest of the system needs to know about.
``engine="legacy"`` runs exactly what the rig has always run —
:func:`softae.analysis.circuit_fitting.fit_circuit` and
:func:`~softae.analysis.circuit_fitting.z_to_sigma`, unedited — and wraps the result.
``engine="gated"`` runs the admission gates first, fits only what they admit, and
reports a value or a bound according to what the instrument can actually resolve.

Both return :class:`~softae.analysis.eis.report.SpectrumReport`. That is the point:
the DataStore, the analysis tab and the EIS browser each learn one new type and never
branch on which engine produced it, so flipping ``[eis] engine`` is the entire cutover
and it is reversible per run.

**R18 — a failed gate must surface as a labelled rejection, not a fit with 10³ %
residuals.** So a rejected spectrum is not fitted at all... *while the gates are
enabled*. With ``enabled = false`` the fit still happens, because observing-only mode
must not change behaviour — it exists so the thresholds can be watched, and a mode
that silently stopped fitting would be enforcement wearing an observer's badge.

The route lives here; its arithmetic lives next door in
:mod:`softae.analysis.eis.engine_support`, which holds seven helpers this module calls
and nothing else calls directly — the ``(f, Z)`` convention guarantee, the fixture
correction on bare arrays, the surviving-points repack, the sum-vs-split resistance
decision, the railed-fit demotion, the spectrum fingerprint and the finite-metrics
filter. All seven are re-exported below, so ``engine.<helper>`` still resolves. What
stayed behind stayed for a reason, and that reason is recorded at the two boundaries
it belongs to: ``engine_support``'s own docstring says why ``_legacy_report``,
``_sigma_from_R``, ``analyze_spectrum`` and ``_log_spectrum_metrics`` cannot live
there, and the note on ``__all__`` below says who reaches these names through this
module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from softae.analysis.eis.arc import annotate_arc_closure, arc_closure
from softae.analysis.eis.engine_support import (
    _as_eis,
    _demote_if_railed,
    _finite_metrics,
    _physics_complex,
    _resolve_reported_resistance,
    apply_correction_arrays,
    blocking_open,
    pregate_settings,
    spectrum_key,
)
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis.policy import build_context, reduce_gates
from softae.analysis.eis.report import (
    SigmaReport,
    SpectrumReport,
    decide_report_mode,
    sigma_upper_bound,
)
from softae.analysis.eis.settings import eis_settings

logger = structlog.get_logger(__name__)

#: Re-exported so ``engine.<helper>`` keeps resolving after the seven moved out for
#: length. They are still *this* module's helpers: it calls them as bare globals, so a
#: test that patches one here still lands, ``tests/test_eis_engine.py`` reaches
#: ``engine_mod._physics_complex`` and imports ``_finite_metrics`` from here,
#: ``tests/test_eis_fitter.py`` imports ``_resolve_reported_resistance`` from here, and
#: ``arc.py`` / ``recommend.py`` / ``shadow_db.py`` name ``engine._demote_if_railed``
#: and ``engine.spectrum_key`` in their prose.
__all__ = [
    "_as_eis",
    "_demote_if_railed",
    "_finite_metrics",
    "_legacy_report",
    "_log_spectrum_metrics",
    "_physics_complex",
    "_resolve_reported_resistance",
    "_sigma_from_R",
    "_two_point_fit",
    "analyze_spectrum",
    "apply_correction_arrays",
    "blocking_open",
    "pregate_settings",
    "spectrum_key",
]

#: What ``fit.estimator`` says when the two-point route produced ``R1``.
#: ``DataStore.record_fit`` reads this off the *fit* — the same deviation
#: ``_arc_columns`` makes, and for the same reason: the label has to survive to a row
#: written by a caller that passes no report.
TWO_POINT = "two_point_debye"


def _sigma_from_R(
    R_ohm: float,
    cell: CellConstant | None,
    *,
    mode: str,
    provisional: bool,
    upper_bound: float,
    phase_headroom: float,
    model_free_R: float,
    R_se: float = float("nan"),
    R_basis: str = "split_bulk",
    rho: float = float("nan"),
) -> SigmaReport:
    # A non-finite resistance is not a small conductivity, it is no conductivity:
    # `cell.sigma(nan)` would return a NaN wearing `mode="value"`, which reads as a
    # measurement to anything that branches on the mode before checking the number.
    if not (R_ohm == R_ohm):
        return SigmaReport(mode="unavailable", R_reported_ohm=float("nan"),
                           R_reported_se_ohm=R_se, R_basis=R_basis, rho=rho,
                           model_free_R_ohm=model_free_R,
                           phase_headroom=phase_headroom)
    if cell is None:
        return SigmaReport(mode="unavailable", R_reported_ohm=float(R_ohm),
                           R_reported_se_ohm=R_se, R_basis=R_basis, rho=rho,
                           model_free_R_ohm=model_free_R,
                           phase_headroom=phase_headroom)

    cross = float("nan")
    if model_free_R == model_free_R and R_ohm == R_ohm and R_ohm > 0:
        cross = abs(model_free_R - R_ohm) / R_ohm * 100.0

    rel_unc = abs(R_se / R_ohm) if (R_se == R_se and R_ohm not in (0, None)
                                    and R_ohm == R_ohm and R_ohm != 0) else float("nan")

    common = dict(
        R_reported_ohm=float(R_ohm),
        R_reported_se_ohm=float(R_se),
        R_basis=R_basis,
        rho=rho,
        K_per_cm=cell.K_per_cm,
        K_route=cell.K_route,
        thickness_method=cell.thickness_method,
        electrode_config=cell.electrode_config,
        k_config_factor=cell.k_config_factor,
        config_factor_verified=cell.config_factor_verified,
        re_contact_verified=cell.re_contact_verified,
        model_free_R_ohm=model_free_R,
        cross_check_pct=cross,
        phase_headroom=phase_headroom,
    )

    if mode in ("bound", "bound_unqualified"):
        return SigmaReport(mode=mode, upper_bound=upper_bound,
                           provisional=provisional, **common)

    # ``provisional`` carries into the value branch too.  A value that merely cleared
    # an *extrapolated* phase floor is still a value taken on trust: the floor it beat
    # was measured three decades away, on a resistive load. Dropping the flag here
    # would make an extrapolated number indistinguishable from a calibrated one.
    return SigmaReport(
        mode="value",
        value=cell.sigma(R_ohm),
        rel_uncertainty=cell.sigma_rel_uncertainty(
            rel_unc if rel_unc == rel_unc else 0.0),
        provisional=provisional,
        **common,
    )


def _legacy_report(
    eis_result: Any, cell: CellConstant | None, model_name: str
) -> SpectrumReport:
    """Exactly the pre-existing behaviour, wrapped in the new return shape."""
    from softae.analysis.circuit_fitting import fit_circuit
    from softae.analysis.quality import grade_fit

    fit = fit_circuit(eis_result, model_name)
    # The one behaviour change on this path, and it only ever fires on a fit that
    # was already reporting the model's bound rather than the sample. A converged
    # fit takes the identical route it always has.
    railed = _demote_if_railed(fit)
    arc = annotate_arc_closure(fit, eis_result)
    quality = grade_fit(getattr(fit, "quality", {}) or {},
                        success=bool(fit.success))
    if railed:
        quality.issues.append(railed)
    if not arc.closed:
        quality.issues.append(arc.detail)

    sigma = SigmaReport(mode="unavailable", R_reported_ohm=float(fit.R1),
                        R_basis="split_bulk")
    if cell is not None and fit.success:
        sigma = SigmaReport(
            mode="value",
            value=cell.sigma(fit.R1),
            R_reported_ohm=float(fit.R1),
            R_basis="split_bulk",
            K_per_cm=cell.K_per_cm,
            K_route=cell.K_route,
            thickness_method=cell.thickness_method,
            electrode_config=cell.electrode_config,
            k_config_factor=cell.k_config_factor,
            config_factor_verified=cell.config_factor_verified,
            re_contact_verified=cell.re_contact_verified,
        )

    return SpectrumReport(engine="legacy", fit=fit, sigma=sigma, quality=quality,
                          gate_log=(), mask=None, cell=cell)


def _two_point_fit(eis_result: Any, model_name: str) -> Any | None:
    """``R₁`` from the ideal Debye circle through the two lowest-frequency points.

    Returns ``None`` when the two points do not describe a physical arc, and that is
    the safety property rather than an edge case: a decline falls through to the route
    this engine takes today, so the two-point read can only ever *replace* a fit, never
    fail one.

    **The arithmetic.** A Debye arc is a semicircle centred on the real axis, so two
    points ``(x, y)`` on it — with ``y = −Z″ > 0`` — determine it outright::

        c = [(x₁² + y₁²) − (x₂² + y₂²)] / [2(x₁ − x₂)]        centre
        r = √[(x₁ − c)² + y₁²]                                 radius
        R_series = c − r,   R_bulk = 2r                        the intercepts

    Closed form, three multiplications, no optimiser — against the 87 421 residual
    evaluations the CPE fit spends on the same spectrum before returning nothing.

    **Why the two *lowest* points**, and the risk in it. They are the two nearest the
    end of the arc that was never measured, which is the end ``R₁`` lives at — and they
    are also where this rig is noisiest. That trade is deliberate and it is the reason
    [p35]'s bias measurement, not this docstring, is what licenses the route: on
    truncated arcs the two-point read's median ``R_est/R_true`` is **1.598** against the
    full CPE fit's **2.752**, so the cheap estimator is measurably the *less* biased one
    exactly where it is used.

    **Why it declines rather than clamps.** On a genuinely blocking spectrum the two
    lowest points sit on a near-vertical tail, ``x₁ ≈ x₂``, and the circle through them
    is enormous and meaningless. The physical guards below — a non-negative
    ``R_series`` that does not exceed the smallest measured ``Re Z`` — reject exactly
    that geometry. Returning a confident number there would be the failure [a53] names:
    a biased ``R₁`` flowing through unlabelled.
    """
    from softae.analysis.circuit_fitting import FitResult

    freq, Z = _physics_complex(eis_result)
    order = np.argsort(freq)
    f_s, x, y = freq[order], Z.real[order], -Z.imag[order]
    if f_s.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None

    (f1, x1, y1), (f2, x2, y2) = (f_s[0], x[0], y[0]), (f_s[1], x[1], y[1])
    if not (y1 > 0 and y2 > 0) or x1 == x2:
        return None

    centre = ((x1 * x1 + y1 * y1) - (x2 * x2 + y2 * y2)) / (2.0 * (x1 - x2))
    radius = float(np.sqrt((x1 - centre) ** 2 + y1 * y1))
    R_series, R_bulk = float(centre - radius), float(2.0 * radius)

    # The high-frequency intercept of a real arc is non-negative and cannot sit to the
    # right of the smallest resistance actually measured. A blocking tail fails both.
    if not (np.isfinite(R_series) and np.isfinite(R_bulk)) or R_bulk <= 0:
        return None
    if R_series < 0 or R_series > float(np.min(x)):
        return None

    # τ from the same first point, so the reconstructed curve passes through it and the
    # residuals downstream are the honest ones for *this* circle rather than a curve
    # fitted a second time.
    span = x1 - R_series
    if span <= 0:
        return None
    tau = float(y1 / (span * 2.0 * np.pi * f1))
    z_fit = R_series + R_bulk / (1.0 + 1j * 2.0 * np.pi * freq * tau)

    quality: dict[str, float] = {}
    try:
        from softae.analysis.quality import compute_fit_quality

        quality = compute_fit_quality(eis_result, z_fit, n_params=3)
    except Exception:      # noqa: BLE001 - grading must not cost the estimate
        quality = {}

    fit = FitResult(
        model_name=model_name,
        parameters=np.array([R_series, R_bulk, tau], dtype=float),
        R0=R_series, R1=R_bulk, R0_guess=R_series, R1_guess=R_bulk,
        z_indices=[0, 1], z_fit=z_fit, quality=quality,
    )
    # Read by ``DataStore.record_fit`` so the stored row says which estimator produced
    # its ``R1``. Attached to the fit rather than carried on the report because a row
    # can be written with ``report=None``, and an unlabelled epoch is the whole problem.
    fit.estimator = TWO_POINT
    logger.info("eis_two_point_read", channel=getattr(eis_result, "channel", None),
                R_series_ohm=R_series, R_bulk_ohm=R_bulk, tau_s=tau,
                f_low_hz=float(f1))
    return fit


def _log_spectrum_metrics(
    eis_result: Any,
    *,
    freq: np.ndarray,
    Z: np.ndarray,
    quality: Any,
    results: Any,
    mask: np.ndarray,
    enforced: bool,
    report_mode: str,
    fit_ok: bool,
) -> None:
    """One ``eis_spectrum_metrics`` line per spectrum, whatever the verdict.

    :func:`~softae.analysis.eis.policy.reduce_gates` logs ``metrics=`` only where a
    spectrum *fails*, so a shadow run records the failing tail and nothing else — and a
    fence placed on a sample of rejects is a percentile of the wrong population. This
    is the pass side, and the arming decision in ``docs/SHADOW_CAMPAIGN.md`` §8 is the
    thing it exists to supply evidence for.

    Emitted from here rather than from inside the reduction because ``r_squared`` and
    ``residual_rms_pct`` are merged into ``quality.metrics`` *after* ``reduce_gates``
    returns: an event raised one frame earlier could never carry them, and half the
    recommendable keys live in ``[quality]``.

    Purely additive — no verdict, σ, mask or return value depends on it, **and that is
    enforced rather than asserted.** Everything below runs inside a guard, because a
    spectrum is expensive and irreplaceable: it came off real hardware, in a well that
    is now used up. Losing one to a defect in its own telemetry would be the same
    mistake ``run_gates`` refuses when it says a broken gate must not discard a
    measurement. A record that cannot be built is a record not written, and the
    analysis carries on returning exactly what it would have returned anyway.
    """
    try:
        surviving = np.asarray(mask, dtype=bool)
        n_surviving = int(surviving.sum())
        logger.info(
            "eis_spectrum_metrics",
            spectrum_key=spectrum_key(getattr(eis_result, "channel", None), freq, Z),
            channel=getattr(eis_result, "channel", None),
            verdict=str(getattr(quality.verdict, "value", quality.verdict)),
            enforced=bool(enforced),
            report_mode=str(report_mode),
            fit_ok=bool(fit_ok),
            n_surviving=n_surviving,
            n_dropped=int(surviving.size - n_surviving),
            gates_run=[str(r.name) for r in results],
            gates_failed=[str(r.name) for r in results if not r.passed],
            metrics=_finite_metrics(getattr(quality, "metrics", None)),
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never cost a spectrum
        try:
            logger.warning("eis_spectrum_metrics_failed", error=str(exc),
                           msg="metrics event not emitted; the analysis is unaffected, "
                               "but this spectrum is missing from any later threshold "
                               "recommendation")
        except Exception:  # noqa: BLE001 - the logger itself is what failed
            pass


def analyze_spectrum(
    eis_result: Any,
    *,
    cell: CellConstant | None = None,
    model_name: str = "simpleSalt",
    engine: str | None = None,
    envelope: Any = None,
    gates: Any = None,
    settings: Any = None,
    blocking: bool = True,
    re_connection: str = "unverified",
    correction: Any = None,
    calibration: Any = None,
    pregate: Any = None,
) -> SpectrumReport:
    """Analyse one spectrum through whichever engine is selected.

    Parameters
    ----------
    eis_result
        An :class:`~softae.analysis.eis_data.EISResult` or anything exposing
        ``frequency`` / ``z_real`` / ``z_imag_neg``.
    cell
        Per-sample geometry. ``None`` means no thickness is known, and conductivity
        is reported as ``unavailable`` rather than computed from a nominal — the same
        posture P7.2 took for deposit area.
    engine
        ``"legacy"`` or ``"gated"``. Defaults to ``[eis] engine``.
    correction
        A :class:`~softae.analysis.eis.fixture.FixtureCorrection`. ``None`` resolves one
        from ``[eis.fixture]`` and the stored calibration. The legacy engine ignores
        this entirely — correcting there would break the parity that makes the two
        engines comparable.
    pregate
        A :class:`~softae.analysis.eis.engine_support.PregateSettings`. ``None`` resolves
        ``[eis.pregate]``, whose two flags both ship false. An override in the same shape
        as ``gates`` and ``envelope``, and for the same reason — it does not choose an
        *engine*, so it is not a second answer to the question ``engine`` asks.
    """
    cfg = settings if settings is not None else eis_settings()
    chosen = (engine or cfg.engine or "legacy").strip().lower()

    if chosen != "gated":
        return _legacy_report(eis_result, cell, model_name)

    from softae.analysis.circuit_fitting import fit_circuit
    from softae.analysis.eis.admittance import model_free_r_bulk
    from softae.analysis.eis.envelope import instrument_envelope
    from softae.analysis.eis.gates import (
        FRONT1_POST_CORRECTION,
        FRONT1_PRE_CORRECTION,
        blocked_by,
        run_gates,
    )
    from softae.analysis.quality import Verdict, grade_fit

    gate_cfg = gates if gates is not None else cfg.gates
    env = envelope if envelope is not None else instrument_envelope()
    freq, Z = _physics_complex(eis_result)

    # RE contact is read off the cell rather than accepted as a second parameter here.
    # It is a precondition on the cell constant (R26), so the cell is where it belongs;
    # a duplicate keyword on this function would be a second place to set the same fact
    # and would silently disagree with the one that actually chose K_config_factor.
    ctx = build_context(
        envelope=env, gates=gate_cfg, cell=cell, blocking=blocking,
        re_connection=re_connection,
        re_contact_verified=bool(getattr(cell, "re_contact_verified", False)),
    )

    corr = correction
    if corr is None:
        from softae.analysis.eis.fixture import resolve_correction

        corr = resolve_correction(
            int(getattr(eis_result, "channel", -1) or -1),
            settings=cfg.fixture, calibration=calibration,
        )

    # Framework §6 interleaves fixture correction *between* the Front-1 gates, at
    # step 4, and the split is load-bearing in both directions.
    #
    # Steps 1–3 judge the raw instrument record — "did this measure anything real?"
    # Correcting first would let a subtraction rescue a spectrum the measurement
    # itself failed; a railed point stays railed however much lead you remove.
    #
    # Steps 5–9 ask whether the spectrum contains the physics being extracted, which
    # is only answerable once the fixture's own contribution is gone. §6: the topology
    # triad "must run on corrected, truncated data — an uncorrected series parasitic
    # or an uncorrected HF artifact can invert the very slopes the triad tests."
    # A fixture R_short *is* a series parasitic, so gate_tand_slope is the sharp case.
    mask, results, log = run_gates(freq, Z, ctx, FRONT1_PRE_CORRECTION)

    stopper = blocked_by(results)
    if stopper is None:
        Z_work, corr_outcome = apply_correction_arrays(freq, Z, corr)
        mask, post, post_log = run_gates(
            freq, Z_work, ctx, FRONT1_POST_CORRECTION, initial_mask=mask)
        results = list(results) + list(post)
        log = list(log) + list(post_log)
    else:
        # Admission rejected it. Correcting and re-gating from here would produce
        # verdicts describing a measurement that was already inadmissible.
        from softae.analysis.eis.fixture import CorrectionOutcome

        Z_work = Z
        corr_outcome = CorrectionOutcome(applied=False, n_points=int(Z.size))
        logger.info("eis_correction_skipped", gate=stopper.name, channel=corr.channel,
                    msg="spectrum rejected before the correction stage")

    n_surviving = int(np.asarray(mask, dtype=bool).sum())
    pre = reduce_gates(results, n_surviving=n_surviving,
                       min_fit_pts=gate_cfg.min_fit_pts, enabled=gate_cfg.enabled)

    # R18: do not hand an inadmissible spectrum to an optimiser. Only when the gates
    # are actually enforcing — observing-only must not change what happens.
    if gate_cfg.enabled and pre.verdict is Verdict.REJECT:
        # Never reached in a shadow run, and recorded anyway: a spectrum that simply
        # vanishes from the population once the flag flips would bias every later
        # re-review of the same log without leaving a trace that it had.
        # ``report_mode`` is honestly absent here — the decision is taken after the
        # fit, and this spectrum never reached one.
        _log_spectrum_metrics(
            eis_result, freq=freq, Z=Z, quality=pre, results=results, mask=mask,
            enforced=True, report_mode="not_reached", fit_ok=False,
        )
        return SpectrumReport(
            engine="gated", fit=None,
            sigma=SigmaReport(mode="unavailable"),
            quality=pre, gate_log=tuple(log), mask=mask, cell=cell, envelope=env,
            # Which correction was *in force* belongs in the provenance even when the
            # spectrum never reached the fit — the outcome says whether it ran.
            correction=corr, correction_outcome=corr_outcome,
        )

    # Corrected and truncated — what steps 10–11 are specified to fit.
    surviving = (eis_result if (mask.all() and not corr_outcome.applied)
                 else _as_eis(eis_result, freq, Z_work, mask))

    # The gated path fits through the covariance-preserving fitter — which is also
    # the only one that scales the optimiser, without which curve_fit terminates at
    # iteration zero on parameters spanning fourteen orders of magnitude. Falling
    # back to the legacy fitter keeps a spectrum analysable if the backend is absent.
    from softae.analysis.eis.fitter import fit_spectrum

    # ── The pre-gate ─────────────────────────────────────────────────────────
    #
    # `arc_closure` costs ~1 ms and already knows what the optimiser needs 38 s to
    # find out. Until now it was read only *after* the fit, by `annotate_arc_closure`
    # below; reading it first is the entire change.
    #
    # **It does not decide admissibility, and it must not.** `arc.py` says outright
    # that nothing there demotes a fit, because refusing the open third "would throw
    # away most of the cold end of every temperature sweep" — 8 % open at the hot end
    # against 73 % at the cold end, which is precisely where an Arrhenius slope has
    # its leverage. So every spectrum still produces an R₁; the pre-gate only chooses
    # a cheaper route to one.
    #
    # Judged on `surviving` — the same corrected, truncated points the fit and
    # `annotate_arc_closure` see, so the route taken and the closure recorded beside
    # it can never describe different data.
    pre_cfg = pregate if pregate is not None else pregate_settings()
    fit = None
    budget = None
    if pre_cfg.engaged and blocking_open(
        arc_closure(surviving.frequency, surviving.z_imag_neg,
                    getattr(surviving, "phase", None)), pre_cfg
    ):
        if pre_cfg.two_point_open:
            # (B). Epoch-grade: this *changes* R₁ on the open population, under an
            # operator authorisation naming that change — "given that the raw data
            # will be better represented" — and on [p35]'s measurement that the CPE
            # fit is the more biased estimator here (175.2 % against 60.9 %). A
            # decline returns None and the ordinary route runs.
            fit = _two_point_fit(surviving, model_name)
        if fit is None and pre_cfg.budget_cap:
            # (A). Same estimator, bounded effort, and it cannot move a number: the
            # unbounded fit was measured to exhaust its 20 000 evaluations and return
            # nothing on this exact population, and a bounded run is a strict prefix
            # of that trajectory — so it fails identically, falls back identically,
            # and reports the identical R₁ two orders of magnitude sooner.
            budget = pre_cfg.max_nfev

    if fit is None:
        try:
            fit = fit_spectrum(surviving, model_name, max_nfev=budget)
        except ValueError:
            fit = fit_circuit(surviving, model_name)
        if not fit.success:
            fit = fit_circuit(surviving, model_name)
    # After the fallback, never before it: demoting a railed fit clears
    # ``success``, and the line above reads that flag as "try the other fitter".
    railed = _demote_if_railed(fit)
    # Judged on ``surviving`` — the corrected, truncated points R₁ actually came
    # from. Judging the raw sweep would credit the fit with a low-frequency point
    # a gate had already removed.
    arc = annotate_arc_closure(fit, surviving)

    f_ok, Z_ok = _physics_complex(surviving)
    mode, provisional, headroom = decide_report_mode(
        f_ok, Z_ok, envelope=env, cell=cell,
        tand_headroom_mult=gate_cfg.tand_headroom_mult,
    )
    bound = sigma_upper_bound(f_ok, Z_ok, envelope=env, cell=cell)

    R, R_se, basis, rho = _resolve_reported_resistance(
        fit, rho_degenerate=gate_cfg.rho_degenerate)
    sigma = _sigma_from_R(
        R if fit.success else float("nan"),
        cell, mode=mode, provisional=provisional, upper_bound=bound,
        phase_headroom=headroom, model_free_R=model_free_r_bulk(Z_ok),
        R_se=R_se, R_basis=basis, rho=rho,
    )

    # Front 2 — how well determined is the answer? These read the fit from ctx and
    # never remove points: by the time a fit exists the data has been admitted.
    from softae.analysis.eis.gates import FRONT2_GATES

    ctx["fit"] = fit
    _, front2, log2 = run_gates(f_ok, Z_ok, ctx, FRONT2_GATES)
    results = list(results) + list(front2)
    log = list(log) + list(log2)

    fit_report = grade_fit(getattr(fit, "quality", {}) or {},
                           success=bool(fit.success))
    quality = reduce_gates(
        results, n_surviving=n_surviving, min_fit_pts=gate_cfg.min_fit_pts,
        report_mode=mode, enabled=gate_cfg.enabled,
    )
    quality.issues.extend(fit_report.issues)
    if railed:
        quality.issues.append(railed)
    if not arc.closed:
        quality.issues.append(arc.detail)
    quality.metrics.update(fit_report.metrics)
    # A correction that drove points non-physical is a data-quality fact, so it travels
    # with the verdict rather than living only in the log where nothing reads it.
    quality.issues.extend(corr_outcome.issues)
    if corr_outcome.applied:
        quality.metrics["fixture_shift_pct"] = corr_outcome.max_shift_pct
    if corr_outcome.suspect and quality.verdict is Verdict.ACCEPT:
        quality.verdict = Verdict.SUSPECT
    if fit_report.verdict is Verdict.REJECT and gate_cfg.enabled:
        quality.verdict = Verdict.REJECT
    elif fit_report.verdict is Verdict.SUSPECT and quality.verdict is Verdict.ACCEPT:
        quality.verdict = Verdict.SUSPECT

    # After every merge into ``quality.metrics`` and every verdict adjustment above —
    # the event reports what this spectrum was finally judged to be, not an interim.
    _log_spectrum_metrics(
        eis_result, freq=freq, Z=Z, quality=quality, results=results, mask=mask,
        enforced=bool(gate_cfg.enabled), report_mode=mode, fit_ok=bool(fit.success),
    )

    return SpectrumReport(
        engine="gated", fit=fit, sigma=sigma, quality=quality,
        gate_log=tuple(log), mask=mask, cell=cell, envelope=env,
        correction=corr, correction_outcome=corr_outcome,
    )
