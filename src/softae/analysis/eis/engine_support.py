"""The engine's arithmetic, kept apart from the route it serves.

Seven helpers :mod:`softae.analysis.eis.engine` leans on and nothing else calls
directly: the ``(f, Z)`` convention guarantee, the fixture-correction call on bare
arrays, the surviving-points repack, the sum-vs-split resistance decision, the railed-fit
demotion, the spectrum fingerprint and the finite-metrics filter.

**Nothing here fits, and nothing here computes a conductivity.** That is a hard
boundary, not a stylistic one: ``tests/test_eis_universal_fit_route.py`` walks every file
under ``src/softae`` for four off-route shapes and exempts exactly four modules, of which
``engine.py`` is one and this is not. ``_legacy_report``, ``_sigma_from_R`` and
``analyze_spectrum`` stay in ``engine.py`` for precisely that reason, and a helper that
grows a fit or a conductivity has to move back there rather than join the allowlist.

``_log_spectrum_metrics`` stays there too, for a different reason: its whole subject is
that a broken **logger** must not cost a spectrum, and the test that proves it swaps the
logger bound in ``engine``'s namespace. A copy of that name over here would be a second
logger the guard could not reach, so the telemetry lives with the module whose logger it
is — and calls :func:`spectrum_key` and :func:`_finite_metrics` from here.

This module imports nothing from ``softae`` at module scope, so the arrow
``engine -> engine_support`` is one-way and the re-export in ``engine.py`` is an ordinary
top-of-module import.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ── The fitter pre-gate: which spectra get a cheaper route ───────────────────
#
# `arc_closure` knows for ~1 ms what the optimiser takes 38 s to discover, and until
# now it was consulted only *after* the fit, through `annotate_arc_closure`. Reading
# it first is the whole idea. The predicate and its thresholds live here, next to the
# engine's other arithmetic; the *routes* it selects live in `engine.py`, because a
# module whose docstring says nothing here fits does not get to grow an estimator.

#: `phase_low_deg` at or below which the response at the sweep floor is still
#: essentially capacitive.
#:
#: **The state is not the discriminator; this is** — which is `ArcClosure`'s own
#: severity/verdict separation being consumed rather than re-derived. Its docstring
#: sets out why: *"near 0° the response at the sweep floor is already resistive and a
#: modestly lower floor would close the arc; near −90° it is still essentially
#: capacitive and no realistic extension of the preset will rescue it."*
#:
#: Measured on `20260811T023757Z_equilibration_characterization`, open arcs only:
#:
#: ===============  ==========================  =========
#: `phase_low_deg`  covariance fit              cost
#: ===============  ==========================  =========
#: −92.0°           returns None (nfev spent)    51.6 s
#: −87.7°           returns None                 66.0 s
#: −81.7°           returns None                 48.1 s
#: −31.1°           **converges**                **0.05 s**
#: ===============  ==========================  =========
#:
#: The −31.1° row is why a bare `state == OPEN` test is wrong: it is an open arc that
#: fits in 50 ms, so diverting it buys no time and — under the two-point route — would
#: move an `R1` for nothing. −60° sits between the two clusters with a wide margin on
#: both sides, and it is a threshold on *severity*, so it is deliberately not a
#: threshold `arc.py` ships (§3.5 of the acquisition spec: that module keeps its
#: CLOSED-biased verdict and gains no policy).
DEFAULT_PREGATE_PHASE_MAX_DEG = -60.0

#: Residual-evaluation budget for the capped fit. Two orders below
#: :data:`~softae.analysis.eis.fitter.DEFAULT_MAX_NFEV` and still an order above the
#: cost of every fit measured to converge at all on this corpus, so the cap sits in
#: the empty band between "converges" and "never will".
DEFAULT_PREGATE_MAX_NFEV = 2_000


@dataclass(frozen=True)
class PregateSettings:
    """Whether the pre-gate runs, and what it does when it fires.

    **Two flags, not one, and they are independent on purpose.** ``budget_cap``
    changes how long the *same* estimator may run and cannot move a reported number;
    ``two_point_open`` changes *which* estimator produces ``R1`` and deliberately does.
    Flip them together and a shifted ``R1`` has two candidate causes with no way to
    separate them — the same argument that keeps ``[eis.scout] enabled`` a third,
    separate switch, since that one moves which frequencies were acquired in the first
    place.

    Both ship **false**, the posture ``[eis.gates] enabled`` and ``[purge] actuate``
    ship in and for the same reason.
    """

    budget_cap: bool = False
    two_point_open: bool = False
    phase_low_max_deg: float = DEFAULT_PREGATE_PHASE_MAX_DEG
    max_nfev: int = DEFAULT_PREGATE_MAX_NFEV

    @property
    def engaged(self) -> bool:
        """True when either route is armed — the guard that keeps the off path free.

        With both flags false the engine does not even read the arc before fitting, so
        a disabled pre-gate costs nothing at all rather than costing "almost nothing".
        """
        return bool(self.budget_cap or self.two_point_open)

    def describe(self) -> str:
        if not self.engaged:
            return "EIS fitter pre-gate off — every spectrum takes the full fit."
        routes = []
        if self.two_point_open:
            routes.append("two-point read (CHANGES R1)")
        if self.budget_cap:
            routes.append(f"optimiser capped at {self.max_nfev} nfev (R1 unchanged)")
        return (f"EIS fitter pre-gate on below {self.phase_low_max_deg:+.0f}° at the "
                f"sweep floor: " + ", ".join(routes) + ".")


def pregate_settings(config: dict[str, Any] | None = None) -> PregateSettings:
    """Read ``[eis.pregate]``. Unparseable values fall back; nothing raises.

    Same posture as :func:`~softae.analysis.eis.settings.eis_settings` and
    :func:`~softae.analysis.eis.scout.scout_settings`: a typo in a config file must not
    stop a campaign that would otherwise have run exactly as it always has. Here the
    fallback is the *shipped-off* state, so a malformed key can only ever leave the
    engine on the route it takes today.
    """
    if config is None:
        try:
            from softae.config import loader

            config = (loader.load().get("eis", {}) or {}).get("pregate", {}) or {}
        except Exception:      # noqa: BLE001 - an unreadable config must not stop a fit
            config = {}

    defaults = PregateSettings()

    def _f(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            logger.warning("eis_pregate_key_unparseable", key=key, default=default)
            return default

    return PregateSettings(
        budget_cap=bool(config.get("budget_cap", False)),
        two_point_open=bool(config.get("two_point_open", False)),
        phase_low_max_deg=_f("phase_low_max_deg", defaults.phase_low_max_deg),
        max_nfev=int(_f("max_nfev", defaults.max_nfev)),
    )


def blocking_open(arc: Any, settings: PregateSettings) -> bool:
    """Is this the population the cheap route is for? Pure, and ~µs.

    Two conditions, both read off fields :class:`~softae.analysis.eis.arc.ArcClosure`
    already computes — no new detector, and **not** the interior-apex fields the
    acquisition scout added, which answer *where should the next sweep put its points?*
    rather than *what will the optimiser do with this one?*

    1. the arc did not close in band, and
    2. the response at the sweep floor is still essentially capacitive.

    A NaN ``phase_low_deg`` — the caller supplied no phase — returns **False**. That is
    the conservative direction and it is chosen rather than defaulted into: severity is
    exactly what condition 2 is about, so a spectrum whose severity is unknown takes the
    route it takes today. The same reasoning gives ``UNKNOWN`` the same answer, through
    condition 1.
    """
    from softae.analysis.eis.arc import OPEN

    if getattr(arc, "state", None) != OPEN:
        return False
    phase = float(getattr(arc, "phase_low_deg", float("nan")))
    if phase != phase:                       # NaN: no phase was supplied
        return False
    return phase <= float(settings.phase_low_max_deg)


def _physics_complex(eis_result: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(f, Z)`` with ``Im Z < 0`` for a capacitive response.

    :attr:`~softae.analysis.eis_data.EISResult.z_complex` already returns the physics
    convention; this exists so the engine has one place that guarantees it, rather
    than each gate trusting its caller.
    """
    freq = np.asarray(eis_result.frequency, dtype=float)
    z = getattr(eis_result, "z_complex", None)
    if z is None:
        z = np.asarray(eis_result.z_real, dtype=float) - 1j * np.asarray(
            eis_result.z_imag_neg, dtype=float)
    return freq, np.asarray(z, dtype=complex)


def apply_correction_arrays(
    freq: np.ndarray, Z: np.ndarray, correction: Any
) -> tuple[np.ndarray, Any]:
    """``(Z_corrected, outcome)`` — the correction applied to bare arrays.

    Arrays rather than an ``EISResult`` because §6 applies the correction *mid-gate-
    chain*, where the working data is already a ``(f, Z)`` pair. Returns the input
    array itself when nothing applies, so an uncorrected gated run is bit-identical
    to one built before E3 existed.
    """
    from softae.analysis.eis.fixture import CorrectionOutcome, apply_series_correction

    Zc = np.asarray(Z, dtype=complex)
    if correction is None or not getattr(correction, "applies", False):
        return Zc, CorrectionOutcome(applied=False, n_points=int(Zc.size))
    return apply_series_correction(freq, Zc, correction)


def _as_eis(eis_result: Any, freq: np.ndarray, Z: np.ndarray, mask: np.ndarray) -> Any:
    """The surviving, corrected points as an ``EISResult`` the fitter can consume."""
    from softae.analysis.eis_data import EISResult

    m = np.asarray(mask, dtype=bool)
    return EISResult.from_arrays(
        channel=getattr(eis_result, "channel", 0),
        f=np.asarray(freq, dtype=float)[m],
        z_real=np.asarray(Z.real, dtype=float)[m],
        # Stored negated: the file convention is −Z″, the physics one is Im Z.
        z_imag_neg=-np.asarray(Z.imag, dtype=float)[m],
        timestamp=getattr(eis_result, "timestamp", None) or None,
        eis_params=dict(getattr(eis_result, "eis_params", {}) or {}),
    )


def _resolve_reported_resistance(
    fit: Any, *, rho_degenerate: float
) -> tuple[float, float, str, float]:
    """Pick which resistance the evidence licenses reporting. Returns
    ``(R, SE, basis, rho)``.

    **R2 is behavioural, not advisory.** The framework is explicit that the reporting
    function must select sum-vs-split from ``ρ`` rather than leaving the choice to an
    analyst, and :class:`SigmaReport` therefore carries exactly one resistance.

    Once the relaxation corner leaves the band the optimiser trades resistance between
    the two terms at near-zero cost: ``ρ → −1``, both individual variances inflate, and
    ``Var(R_series + R_bulk)`` collapses far below their sum. Measured on a synthetic
    spectrum with the corner pushed above ``F_MAX``: ρ = −0.977, individual relative
    standard errors 31 % and 7 %, but the **sum** determined to 1.4 % with 35× less
    variance. Reporting ``R_bulk`` alone there silently drops a σ-dependent fraction of
    the true resistance — overhaul §3.1 identifies exactly that as the origin of the
    apparent "non-constant cell constant".
    """
    cov = getattr(fit, "covariance", None)
    R_split = float(getattr(fit, "R1", float("nan")))
    if cov is None:
        return R_split, float("nan"), "split_bulk", float("nan")

    from softae.analysis.eis.models import roles_for

    roles = roles_for(getattr(fit, "model_name", "")) or {}
    a = roles.get("R_series", "R0")
    b = roles.get("R_bulk", "R1")

    rho = cov.rho(a, b)
    degenerate = cov.singular or (rho == rho and rho <= float(rho_degenerate))

    if degenerate:
        logger.info(
            "eis_split_degenerate", rho=rho,
            r_sum_ohm=cov.sum_value(a, b), r_sum_se_ohm=cov.sum_se(a, b),
            msg="relaxation corner out of band — reporting R_series+R_bulk only",
        )
        return cov.sum_value(a, b), cov.sum_se(a, b), "sum", rho

    return cov.value(b), cov.se(b), "split_bulk", rho


def _demote_if_railed(fit: Any) -> str:
    """Strip the measurement claim off a fit that railed. Returns why, or ``""``.

    A fit whose ``R_bulk`` came to rest on a box constraint reports the *bound*,
    not the sample, and it does so with the same ``success`` flag as a genuine
    measurement — so every consumer that reads only ``success`` (the optimiser,
    the settle criterion, the analysis tab, ``fit_results.sigma_S_per_cm``) takes
    a constant that is a property of ``CIRCUIT_MODELS`` for an observation.

    Three things change, and each closes one of those routes:

    ``success = False``
        The single flag most consumers branch on. A *distinct* quality state was
        the alternative and was rejected: it would leave ``success = 1`` on the
        row, so every reader that does not yet know about the new state keeps
        believing the number. Demoting an unidentified parameter to "not a fit"
        is also simply what it is — the optimiser reported where the wall was.
    ``error_msg``
        Names the bound and its value, so a railed row is distinguishable from a
        fit that failed to converge without re-deriving anything.
    ``R1 = NaN``
        σ follows from R₁ everywhere it is computed, including inside
        ``DataStore.record_fit``, which derives it from ``fit_result.R1`` and
        cannot see this reason. A NaN R₁ fails that guard, so no conductivity is
        stored — the ``resolve_thickness_cm`` posture: absence, not an invented
        value. The railed value itself survives in ``parameters_json``, which is
        where a diagnostic belongs.

    ``parameters``, ``z_fit`` and ``quality`` are deliberately untouched: the
    residuals against a railed model are real, and are the evidence for the
    demotion rather than a casualty of it.
    """
    from softae.analysis.eis.models import railed_measurand

    reason = railed_measurand(fit)
    if not reason:
        return ""

    fit.success = False
    fit.error_msg = f"railed fit: {reason} — parameter unidentified, no conductivity"
    fit.R1 = float("nan")
    logger.info("eis_fit_railed", model=getattr(fit, "model_name", "?"), reason=reason)
    return reason


def spectrum_key(channel: Any, freq: Any, Z: Any) -> str:
    """A content fingerprint naming one *physical* spectrum across repeat analyses.

    Every measured spectrum in an autonomous campaign passes through
    :func:`analyze_spectrum` **twice** — once for the auto-route fit that writes
    ``fit_results``, once again when the objective is extracted — so anything that
    counts the emitted metrics events sees each spectrum twice. Left uncorrected that
    doubles every sample size a threshold recommendation is weighed against, and an
    evidence floor passes at half the evidence it was set to demand.

    The fingerprint is taken over the arrays rather than the timestamp because
    :attr:`~softae.analysis.eis_data.EISResult.timestamp` is optional, two calls on one
    object necessarily share it, and so would two genuinely distinct spectra measured
    inside the same second. The channel is carried in the clear so the key stays
    readable, and so two channels measuring an identical synthetic load stay distinct.
    """
    digest = hashlib.blake2s(
        np.asarray(freq, dtype=float).tobytes()
        + np.asarray(Z, dtype=complex).tobytes(),
        digest_size=6,
    ).hexdigest()
    try:
        return f"c{int(channel):02d}:{digest}"
    except (TypeError, ValueError):
        return f"c--:{digest}"


def _finite_metrics(metrics: Any) -> dict[str, float]:
    """The loggable subset of a metrics mapping — finite floats only.

    ``repr(float("nan"))`` is a bare ``nan``, which ``ast.literal_eval`` refuses, so a
    single non-finite value degrades the *whole* rendered ``metrics={...}`` mapping to
    an unparsed string in ``softae-shadow review``. Dropping it costs nothing: a gate
    that could not compute its metric has no observation to place a threshold against,
    and the reviewer counts a metric's evidence by the records that carry it.
    """
    out: dict[str, float] = {}
    for name, raw in (metrics or {}).items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            out[str(name)] = value
    return out
