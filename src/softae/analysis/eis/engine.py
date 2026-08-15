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

from softae.analysis.eis.arc import annotate_arc_closure
from softae.analysis.eis.engine_support import (
    _as_eis,
    _demote_if_railed,
    _finite_metrics,
    _physics_complex,
    _resolve_reported_resistance,
    apply_correction_arrays,
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
    "analyze_spectrum",
    "apply_correction_arrays",
    "spectrum_key",
]


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

    try:
        fit = fit_spectrum(surviving, model_name)
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
