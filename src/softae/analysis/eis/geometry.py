"""The cell constant, per sample — because one nominal value across a series is a defect.

Overhaul §3.4 records what happens without this: a single nominal thickness applied
to every sample made ``K`` wrong by 2× at 100 µm and 1.33× at 150 µm, and the
resulting thickness series failed its own consistency test with a physically
impossible negative intercept. R12 exists to stop exactly that.

**The formula shape here is not new — only its inputs are.** For a coplanar cell
``K = L_gap / ((t − h) · L_stripe)``, so ``σ = K/R`` at ``h = 0`` is *the same
arithmetic* as the legacy ``z_to_sigma(L, t, w, R) = L/(R·t·w)`` — in a different
association order, so it agrees to within 1 ULP rather than bit-for-bit; see
:meth:`CellConstant.sigma`. What changes is that
``L``/``w`` are named for what they physically are (the electrode gap and the stripe
length, not a length and a width), that ``t`` is sourced per sample instead of
defaulting to a placeholder, and that a dead height can be carried. No displayed σ
moves and no database column is renamed.

.. warning::
   The legacy default ``t = 0.175 cm`` is 1.75 mm — roughly ten times thicker than
   any drop-cast film, giving ``K ≈ 5.7 /cm`` against the 50–100 /cm this cell
   geometry actually implies. It is a placeholder that has always been there, not a
   measurement. This module surfaces it (via :attr:`CellConstant.plausible` and the
   reported ``K``) rather than silently correcting it, because correcting it would
   change every historical number without anyone deciding to.

.. note::
   ``K_route`` records how the constant was obtained because R12 and framework §1.1
   look contradictory and are not. R12 objects to *one nominal value across a
   series*; §1.1 objects to *claiming absolute accuracy from a formula*, since
   coplanar cell constants are fringing-dependent. Both are satisfied by computing
   per sample and saying which route produced the number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


#: Electrode gap (cm) — the ``L`` of the legacy triple, named for what it is.
DEFAULT_L_GAP_CM = 0.2
#: Stripe length (cm) — the ``w`` of the legacy triple. Named ``L_stripe`` rather
#: than a width to avoid colliding with conductance ``G`` in the framework's algebra.
DEFAULT_L_STRIPE_CM = 0.2
#: Dead height ``h`` (µm). **Zero until a ≥4-level thickness series fits one.**
#: Overhaul §9.1 puts it near 48 µm from three levels with 1.3–2.1× replicate
#: scatter, trusted to no better than ±20 µm — and ``h = 0`` is also what
#: reproduces the legacy arithmetic exactly.
DEFAULT_DEAD_HEIGHT_UM = 0.0
#: Above this thickness (cm) a value is almost certainly a placeholder, not a film.
DEFAULT_THICKNESS_MAX_CM = 0.05

#: How a thickness was obtained, best first. Ordering is the precedence used by
#: :func:`resolve_thickness_cm`.
THICKNESS_METHODS = ("profilometry", "target", "predicted", "dispensed", "nominal")

#: Recorded electrode configuration. ``unverified`` is the shipped state and is a
#: *distinct* value from either real configuration — see :attr:`CellConstant.k_config_factor`.
ELECTRODE_CONFIGS = ("unverified", "3-electrode", "2-electrode")

#: ``K_config_factor`` by configuration, per framework §1.1 / overhaul R20.
#:
#: **The symmetry argument.** With a three-electrode measurement the sensed quantity is
#: ``ΔE(WE vs RE)/I(WE)``. When WE and CE are identical stripes and RE sits exactly
#: mid-gap, mirror antisymmetry puts RE at the mean of the two electrode potentials, so
#: ``Z_3-electrode = ½ · Z_2-electrode`` exactly, at every frequency, interfaces
#: included. The measured resistance is therefore half the full-gap cell's, and the
#: full-gap ``K_geom`` over-reports σ by the same factor. Hence
#:
#:     ``σ = K_geom / (K_config_factor · R)``
#:
#: which is why the factor divides rather than multiplies: it corrects a resistance
#: that is *too small* for the geometry ``K_geom`` describes.
CONFIG_FACTORS = {"3-electrode": 2.0, "2-electrode": 1.0, "unverified": 1.0}

#: Shipped default. **Deliberately 1.0, which changes no number.**
#:
#: R20 is right that the term must exist and be auditable, but the only direct
#: measurement in the record does not confirm its value: overhaul §3.8's back-to-back
#: comparison of the same two films gives ``Z_2el/Z_3el`` = 1.28× and 1.46×, not the
#: 2.00 the symmetry predicts. The gap is not small enough to be noise and not
#: consistent enough to be a different constant, and the symmetry result is contingent
#: on two things nobody has verified on this board — stripe symmetry and RE centring.
#:
#: So the term ships built, recorded and reported, at the value that leaves every
#: existing σ untouched. Relative trends — which a constant factor cannot affect —
#: stay valid throughout. Same posture as the phase floor and ``[quality] enabled``:
#: measure, then arm.
DEFAULT_K_CONFIG_FACTOR = 1.0


@dataclass(frozen=True)
class CellConstant:
    """Per-sample cell geometry and the conductivity it licenses.

    Replaces the loose ``(L, t, w)`` triple that was passed positionally through
    five call sites and stored in three database columns with no record of where
    the thickness came from.
    """

    L_gap_cm: float
    L_stripe_cm: float
    thickness_cm: float
    dead_height_cm: float = 0.0
    thickness_unc_cm: float = float("nan")
    thickness_method: str = "nominal"
    geometry: str = "coplanar"
    K_route: str = "geometric"
    thickness_max_cm: float = DEFAULT_THICKNESS_MAX_CM
    #: Electrode configuration as *recorded*, never inferred. This rig is wired
    #: 3-electrode permanently — the operator's decision, kept for the other
    #: electrochemical techniques that need it — so the value is a known fact and is
    #: recorded as one. Knowing the configuration is **not** the same as having
    #: verified the factor it implies; see :attr:`k_config_verified`.
    electrode_config: str = "unverified"
    #: Divides ``K_geom`` — see :data:`CONFIG_FACTORS` for the derivation and
    #: :data:`DEFAULT_K_CONFIG_FACTOR` for why it ships at 1.0.
    k_config_factor: float = DEFAULT_K_CONFIG_FACTOR
    #: Whether the factor itself has been confirmed for this board.
    #:
    #: Separate from :attr:`electrode_config` on purpose. The configuration is a
    #: wiring fact anyone can read off the board; the *factor* rests on stripe
    #: symmetry and RE centring, and overhaul §3.8's own measurement (1.28×, 1.46×)
    #: does not reproduce the predicted 2.00. Folding the two into one flag would let
    #: "we know it is 3-electrode" silently arm a correction nothing has confirmed.
    k_config_verified: bool = False
    #: Whether **this sample's** ionic path to the reference stripe was confirmed —
    #: overhaul R26.
    #:
    #: The third flag in the same family, and per-*measurement* where the other two are
    #: per-board. :attr:`electrode_config` is wiring anyone can read off the board;
    #: :attr:`k_config_verified` is stripe symmetry and RE centring, checked once per
    #: board. Contact is neither: a dry, dewetted or non-wetting film on a fully
    #: verified board still has no ionic path, and F17 shows that when the path is
    #: absent the sensing ratio is not merely unknown but *undefined* — the RE floats
    #: onto a load-dependent capacitive divider (α spanning 2.2–23.8) rather than
    #: sensing a potential at all.
    #:
    #: So this cannot be a config key. A board-level ``true`` would assert contact for
    #: every sample cast on it, which is precisely the population where it fails.
    re_contact_verified: bool = False

    # ── Derived geometry ─────────────────────────────────────────────────────

    @property
    def effective_thickness_cm(self) -> float:
        """``t − h``, floored at zero.

        A film cast below the dead height should show essentially no lateral
        conduction — overhaul §9.1 calls that the sharp falsifiable test of whether
        ``h`` is real.
        """
        return max(self.thickness_cm - self.dead_height_cm, 0.0)

    @property
    def K_geometric_per_cm(self) -> float:
        """Geometric cell constant ``L_gap / ((t − h) · L_stripe)`` in cm⁻¹.

        The full-gap constant, before any sensing-configuration correction.
        """
        t_eff = self.effective_thickness_cm
        if t_eff <= 0 or self.L_stripe_cm <= 0:
            return float("nan")
        return self.L_gap_cm / (t_eff * self.L_stripe_cm)

    @property
    def K_per_cm(self) -> float:
        """The cell constant σ is actually divided by: ``K_geom / K_config_factor``.

        At the shipped ``k_config_factor = 1.0`` this equals
        :attr:`K_geometric_per_cm`, so no existing number moves.
        """
        K = self.K_geometric_per_cm
        f = float(self.k_config_factor)
        if K != K or not (f > 0):
            return float("nan")
        return K / f

    @property
    def config_declared(self) -> bool:
        """Whether the electrode configuration is on record (a wiring fact)."""
        return self.electrode_config in ("3-electrode", "2-electrode")

    @property
    def config_factor_verified(self) -> bool:
        """Whether the *factor* has been confirmed — not merely the configuration.

        Three conditions, because there are three separable ways to be wrong:

        1. the configuration is on record at all (a wiring fact),
        2. :attr:`k_config_verified` — the board's stripe symmetry and RE centring
           were checked, and
        3. :attr:`re_contact_verified` — *this sample* has an ionic path to the RE.

        Condition 3 applies only to three-electrode measurements. A two-electrode
        measurement has no reference electrode in the sensing path, so its factor of 1
        is exact regardless of contact; requiring it there would relabel sound σ as
        unqualified without changing a number.

        False means the absolute scale of σ is unqualified: F16 is a clean ~2× error
        that leaves the spectrum, the fit and the residuals all looking perfect.
        **Relative trends are unaffected either way**, since the factor is a constant
        across a series, so a campaign comparing formulations is sound throughout.
        """
        if not (self.config_declared and bool(self.k_config_verified)):
            return False
        if self.electrode_config == "3-electrode":
            return bool(self.re_contact_verified)
        return True

    @property
    def plausible(self) -> bool:
        """False when the thickness looks like a placeholder rather than a film."""
        return (
            0.0 < self.thickness_cm <= self.thickness_max_cm
            and self.effective_thickness_cm > 0.0
            and self.L_gap_cm > 0.0
            and self.L_stripe_cm > 0.0
        )

    @property
    def measured_per_sample(self) -> bool:
        """True when the thickness is this sample's own, not a shared nominal."""
        return self.thickness_method in ("profilometry", "target", "predicted",
                                         "dispensed")

    # ── Conductivity ─────────────────────────────────────────────────────────

    def sigma(self, R_ohm: float) -> float:
        """``σ = K / R`` in S/cm.

        At ``dead_height_cm = 0`` this is the same arithmetic as
        ``circuit_fitting.z_to_sigma(L_gap, thickness, L_stripe, R)`` **in a
        different association order** — ``K/R`` with ``K = L/(t·w)`` precomputed,
        versus ``L/((R·t)·w)`` — so the two can differ in the final bit. Measured:
        41 of 90 sampled ``(L, t, w, R)`` combinations differ by exactly 1 ULP,
        worst relative 2.55e-16; see ``tests/test_gui_eis_sigma_source.py`` for the
        bound. Every precision the GUI prints is identical, so no displayed number
        moves; a re-fit can store a σ differing in the last bit.

        The word that used to be here was "exactly", and it was read as licence to
        collapse the two functions into one. Do not: ``z_to_sigma`` is the
        independent oracle those parity tests are written against, and making the
        survivor call the deprecated symbol turns the proof into ``x == x``.
        """
        try:
            R = float(R_ohm)
        except (TypeError, ValueError):
            return float("nan")
        if not (R > 0):
            return float("nan")
        K = self.K_per_cm
        return K / R if K == K else float("nan")

    def sigma_rel_uncertainty(self, R_rel_unc: float) -> float:
        """Combine the resistance and thickness relative uncertainties in quadrature.

        Thickness is normally the dominant term for the film route (framework §7.3),
        so reporting only the fit's standard error would understate the real spread.
        """
        t_eff = self.effective_thickness_cm
        t_rel = (
            abs(self.thickness_unc_cm) / t_eff
            if t_eff > 0 and self.thickness_unc_cm == self.thickness_unc_cm
            else 0.0
        )
        try:
            r_rel = abs(float(R_rel_unc))
        except (TypeError, ValueError):
            r_rel = 0.0
        return float((r_rel ** 2 + t_rel ** 2) ** 0.5)

    # ── Interop with the legacy triple ───────────────────────────────────────

    def as_legacy_triple(self) -> tuple[float, float, float]:
        """``(L, t, w)`` as the legacy path expects it.

        Returns the **nominal** thickness, not ``t − h``: the legacy columns record
        what was measured, and folding a dead-height correction into them would make
        a stored geometry mean two different things depending on when it was written.
        """
        return (self.L_gap_cm, self.thickness_cm, self.L_stripe_cm)

    @classmethod
    def from_legacy(cls, L: float, t: float, w: float, **kwargs: Any) -> "CellConstant":
        """Adopt a legacy ``(L, t, w)`` triple without changing its arithmetic."""
        return cls(L_gap_cm=float(L), L_stripe_cm=float(w), thickness_cm=float(t),
                   **kwargs)

    def describe(self) -> str:
        K = self.K_per_cm
        k_txt = f"{K:.1f} /cm" if K == K else "K unavailable"
        h_txt = (
            f", h = {self.dead_height_cm * 1e4:.0f} µm"
            if self.dead_height_cm > 0 else ""
        )
        warn = "" if self.plausible else "  ⚠ implausible thickness"
        if self.config_factor_verified:
            cfg = f", {self.electrode_config} ÷{self.k_config_factor:g}"
        elif self.config_declared:
            # Name which precondition is missing. "Unverified" alone sent the operator
            # to the wrong bench task half the time: the board check and the per-sample
            # contact check are different work by different people.
            why = (
                "board symmetry unverified"
                if not self.k_config_verified
                else "RE contact unverified for this sample (R26)"
            )
            cfg = (f", {self.electrode_config} (K_config_factor not applied — {why}; "
                   f"absolute σ unqualified)")
        else:
            cfg = ", electrode config undeclared (absolute σ unqualified)"

        return (
            f"{self.geometry} cell: gap {self.L_gap_cm:g} cm × stripe "
            f"{self.L_stripe_cm:g} cm, t = {self.thickness_cm * 1e4:.0f} µm "
            f"({self.thickness_method}){h_txt}{cfg} → {k_txt} [{self.K_route}]{warn}"
        )


def cell_from_legacy_terms(L: Any, t: Any, w: Any, **kwargs: Any) -> CellConstant | None:
    """The cell an ``(L, t, w)`` triple in cm implies, or ``None`` when it implies none.

    ``None`` for a non-numeric, missing or non-positive term. That propagates to
    ``sigma.mode == "unavailable"`` and then to a blank σ — a missing thickness yields
    *no* conductivity, never one built on a nominal. Note that a term of ``nan`` fails
    ``> 0`` and so is declined like any other unusable value.

    Four sites spelled this out separately — ``gui.eis_sigma.gui_cell``,
    ``web.data_adapter._cell_from_geometry``,
    ``workflows.temp_eis_sweep.ArrheniusSweep._cell`` and
    ``analysis.eis.router._cell_from_params`` — each deciding on its own *whether a
    conductivity exists at all*. They are now key-extraction around this one call; the
    guard itself lives here.

    **It lives in this module specifically because this module is import-cheap.**
    ``web/data_adapter``'s own docstring recorded why it refused to import
    ``gui/eis_sigma``: a web adapter reaching into ``softae.gui`` would put Qt on the
    headless import path. ``geometry.py`` has no such cost — stdlib, ``structlog``, and
    two deferred in-function imports — so every caller, headless or not, can reach it.
    Do not add a module-level import here that a headless consumer cannot afford;
    ``core/data_store.py`` imports this module and must keep working.
    """
    try:
        L_cm, t_cm, w_cm = float(L), float(t), float(w)
    except (TypeError, ValueError):
        return None
    if not (L_cm > 0 and t_cm > 0 and w_cm > 0):
        return None
    return CellConstant.from_legacy(L_cm, t_cm, w_cm, **kwargs)


def cell_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read ``[eis.cell]`` — the single parse point for the cell defaults."""
    if config is None:
        try:
            from softae.config import loader

            config = loader.load().get("eis", {}).get("cell", {}) or {}
        except Exception:
            config = {}

    def _f(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return default

    electrode_config = str(
        config.get("electrode_configuration", "unverified")).strip().lower()
    if electrode_config not in ELECTRODE_CONFIGS:
        logger.warning(
            "eis_electrode_config_unknown", value=electrode_config,
            known=ELECTRODE_CONFIGS,
            msg="treating the electrode configuration as unverified",
        )
        electrode_config = "unverified"

    # Declaring the configuration records a fact; it does not arm the correction.
    # The factor follows the configuration only once `k_config_verified` says the
    # symmetry behind it was actually checked on this board. An explicit factor still
    # wins outright, for a board whose measured ratio is neither 1 nor 2.
    verified = bool(config.get("k_config_verified", False))
    if "k_config_factor" in config:
        k_factor = _f("k_config_factor", DEFAULT_K_CONFIG_FACTOR)
    elif verified:
        k_factor = CONFIG_FACTORS.get(electrode_config, DEFAULT_K_CONFIG_FACTOR)
    else:
        k_factor = DEFAULT_K_CONFIG_FACTOR
    if not (k_factor > 0):
        k_factor = DEFAULT_K_CONFIG_FACTOR

    if electrode_config == "3-electrode" and not verified and k_factor == 1.0:
        logger.info(
            "eis_k_config_unarmed", electrode_config=electrode_config,
            predicted=CONFIG_FACTORS["3-electrode"],
            msg="3-electrode recorded; K_config_factor NOT applied (unverified) — "
                "absolute σ is unqualified, relative trends unaffected",
        )

    return {
        "L_gap_cm": _f("L_gap_cm", DEFAULT_L_GAP_CM),
        "L_stripe_cm": _f("L_stripe_cm", DEFAULT_L_STRIPE_CM),
        "dead_height_cm": _f("dead_height_um", DEFAULT_DEAD_HEIGHT_UM) * 1e-4,
        "thickness_max_cm": _f("thickness_max_cm", DEFAULT_THICKNESS_MAX_CM),
        "electrode_config": electrode_config,
        "k_config_factor": k_factor,
        "k_config_verified": verified,
    }


def resolve_thickness_cm(
    *,
    profilometry_um: float | None = None,
    target_um: float | None = None,
    predicted_um: float | None = None,
    dispensed_um: float | None = None,
) -> tuple[float, str]:
    """Pick this sample's thickness and say where it came from.

    Precedence, best first:

    ``profilometry``
        An independent measurement of the film. Framework §7.3 makes this the
        dominant uncertainty term for the geometry route, so a real one always wins.
    ``target``
        The autonomous path: the campaign's :class:`~softae.core.formulation.ThicknessTarget`
        *is* the nominal thickness, because a dry target under full solvent loss
        reduces exactly to a volume target and the solve holds it.
    ``predicted``
        The high-throughput path: computed from well geometry and stock volumes
        assuming complete drying — the existing ``formulations.predicted_thickness_um``.
    ``dispensed``
        Bare cast volume over deposit area, with nothing said about drying.

    Returns ``(nan, "unavailable")`` when nothing is known. **That is a result, not
    a failure** — the same posture P7.2 took for deposit area. An invented thickness
    silently corrupts every σ downstream, and σ divides by it.
    """
    for value, method in (
        (profilometry_um, "profilometry"),
        (target_um, "target"),
        (predicted_um, "predicted"),
        (dispensed_um, "dispensed"),
    ):
        if value is None:
            continue
        try:
            um = float(value)
        except (TypeError, ValueError):
            continue
        if um == um and um > 0:
            return um * 1e-4, method

    return float("nan"), "unavailable"


def _re_contact_established(re_contact_verified: bool, re_connection: str) -> bool:
    """Resolve the two RE records into one answer: did ions reach the reference?

    ``re_contact_verified`` is the assertion that carries the weight — R26 asks for a
    *verified* path, and a topology label is not a verification. ``re_connection`` is
    consulted only to catch the case where the two records contradict each other.

    A contradiction fails closed and is logged rather than resolved. "Contact verified"
    together with ``open_by_geometry`` (nothing spans the stripes) or ``tied_to_ce``
    (the RE is jumpered to the counter electrode and reads its potential) is an operator
    error in one record or the other, and silently believing either one is how the wrong
    one survives. ``unverified`` is not a contradiction — it means nothing was recorded.
    """
    from softae.analysis.eis.policy import RE_IONIC_CONTACT

    if not re_contact_verified:
        return False

    state = str(re_connection or "unverified").strip().lower()
    if state == "unverified" or state in RE_IONIC_CONTACT:
        return True

    logger.warning(
        "eis_re_contact_contradiction", re_connection=state,
        msg=f"RE contact reported verified, but re_connection = '{state}' says there "
            f"is no ionic path to the reference stripe. Treating contact as "
            f"unverified — one of the two records is wrong and it is not safe to "
            f"guess which",
    )
    return False


def cell_constant_for_sample(
    *,
    profilometry_um: float | None = None,
    target_um: float | None = None,
    predicted_um: float | None = None,
    dispensed_um: float | None = None,
    thickness_unc_um: float | None = None,
    re_contact_verified: bool = False,
    re_connection: str = "unverified",
    config: dict[str, Any] | None = None,
    pcb_config: dict[str, Any] | None = None,
) -> CellConstant | None:
    """Build a :class:`CellConstant` for one sample, or ``None`` if it cannot be.

    Board geometry comes from ``[pcb.*] electrode_L_cm`` / ``electrode_w_cm`` when a
    board is supplied, else from ``[eis.cell]``. Thickness resolution is
    :func:`resolve_thickness_cm`.

    ``re_contact_verified`` is a **per-sample** assertion and has deliberately no
    ``[eis.cell]`` counterpart: a board-level key would assert an ionic path for every
    sample cast on that board, and the samples where it fails — dry, dewetted, poorly
    wetting — are exactly the ones it would wrongly qualify. Unsupplied means *not
    verified*, and on a three-electrode board that demotes ``K_config_factor`` to 1.0
    (R26). Fail-closed here costs a known factor of two; fail-open costs an unknown one
    somewhere between 2.2 and 23.8 (F17).

    Returns ``None`` when no per-sample thickness exists. Callers must treat that as
    *σ unavailable* and report the resistance alone — never substitute a nominal.
    """
    cfg = cell_config(config)

    L_gap = cfg["L_gap_cm"]
    L_stripe = cfg["L_stripe_cm"]
    if pcb_config:
        try:
            L_gap = float(pcb_config.get("electrode_L_cm", L_gap))
            L_stripe = float(pcb_config.get("electrode_w_cm", L_stripe))
        except (TypeError, ValueError):
            pass

    t_cm, method = resolve_thickness_cm(
        profilometry_um=profilometry_um,
        target_um=target_um,
        predicted_um=predicted_um,
        dispensed_um=dispensed_um,
    )
    if method == "unavailable":
        logger.info(
            "eis_thickness_unavailable",
            msg="no per-sample thickness — conductivity will not be reported",
        )
        return None

    unc_cm = float("nan")
    if thickness_unc_um is not None:
        try:
            unc_cm = float(thickness_unc_um) * 1e-4
        except (TypeError, ValueError):
            unc_cm = float("nan")

    contact = _re_contact_established(re_contact_verified, re_connection)

    # R26: the factor is a *number*, and `config_factor_verified` only labels it — so
    # the precondition has to bite here, where the number is chosen, not only in the
    # property that reports it. Without this the board flag alone would still divide.
    k_factor = cfg["k_config_factor"]
    if cfg["electrode_config"] == "3-electrode" and k_factor != 1.0 and not contact:
        logger.warning(
            "eis_k_config_no_re_contact",
            requested_factor=k_factor, applied_factor=1.0,
            re_connection=str(re_connection), re_contact_verified=bool(re_contact_verified),
            msg="K_config_factor NOT applied: no verified ionic path to the reference "
                "electrode (R26). The symmetry derivation assumes RE senses a potential "
                "in the conducting medium; without it the sensing ratio is undefined, "
                "not merely unknown",
        )
        k_factor = 1.0

    cell = CellConstant(
        L_gap_cm=L_gap,
        L_stripe_cm=L_stripe,
        thickness_cm=t_cm,
        dead_height_cm=cfg["dead_height_cm"],
        thickness_unc_cm=unc_cm,
        thickness_method=method,
        thickness_max_cm=cfg["thickness_max_cm"],
        electrode_config=cfg["electrode_config"],
        k_config_factor=k_factor,
        k_config_verified=cfg["k_config_verified"],
        re_contact_verified=contact,
    )

    if not cell.plausible:
        logger.warning(
            "eis_thickness_implausible",
            thickness_cm=cell.thickness_cm,
            ceiling_cm=cell.thickness_max_cm,
            K_per_cm=cell.K_per_cm,
            method=method,
            msg="thickness looks like a placeholder, not a film — σ reported anyway, flagged",
        )
    return cell
