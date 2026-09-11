"""The arc-closure verdict, in the shape ``run_gates`` can carry — as a **flag**.

:func:`~softae.analysis.eis.arc.arc_closure` has been computed on every
``analyze_spectrum`` call on both engines since T7, and its verdict has been
**discarded** every time: it is annotated onto the fit, it is copied into three
DataStore columns, and it is a member of no gate tuple, so ``reduce_gates``,
``QualityReport.issues`` and every consumer that reads a gate log have never seen
it. This module is the predicate in the cascade's own shape, so that registering it
costs :mod:`softae.analysis.eis.gates` exactly one line — a name in
:data:`~softae.analysis.eis.gates.FRONT2_GATES` — and nothing else.

**Why a flag, and not a refusal.** Promoting the arc *state* to a Front-1
``block_spectrum`` was measured on the ten ``probe-3ch-v3`` spectra and **rejected**
(``docs/SubAgent docs/arc_closure_gate.md`` §2, reproduced independently by two other
sessions from the raw files):

- Two of the gated engine's four worst rows read ``CLOSED`` — ch32_001 at 53.5× low
  and ch32_004 at 147.2× low. The gate would have admitted both.
- Gated's single **best** row, ch22_002 at 1.05×, reads ``OPEN``. The gate would have
  refused the one answer that was right.
- ``ρ(state == OPEN, log gated error) = +0.266, p = 0.458`` over all ten.
- On a seeded 600-spectrum sample of the stored corpus, ``state == OPEN`` fires on
  39.3 % and would add **197 new refusals = 32.8 % of the corpus** against a Front-1
  baseline of 23.3 % — more than doubling what the cascade refuses, and more
  consequential alone than the other fourteen gates combined. It also deletes whole
  Arrhenius runs outright (two measured at 89.6 % and 85.4 % ``OPEN``), which is
  precisely where the slope has its leverage.

So the state is not the discriminator, and a refusal built on it would refuse the
wrong population. A ``flag`` costs **zero σ** by two independent mechanisms:
``reduce_gates`` cannot reach ``Verdict.REJECT`` from a ``flag`` severity at all, and
``FRONT2_GATES`` runs *after* σ is built, so a Front-2 verdict cannot reach the
Front-1 early return that is the only place a gate can cost a σ. Either alone is
sufficient; both hold.

**What the flag is FOR, since it refuses nothing: it builds the labelled corpus that
does not exist.** ``fit_results`` holds **zero** ``engine = 'gated'`` rows and
``arc_phase_low_deg`` is **94 % NULL**, so no validation of this verdict against
gated error is possible today — not because the measurement is hard, but because the
verdict is computed and thrown away. Nothing can be concluded about a check that was
never recorded. The flag is how the population comes into existence; the decision
about whether it should ever refuse waits on that population and on ≥ 6 channels
against an anchor outside the pipeline (spec §5).

**Registering it: WHERE the line goes is constrained, and the constraint is real.**
This module imports ``GateResult`` and ``FLAG`` from :mod:`softae.analysis.eis.gates`
at module scope, so ``gates`` importing *this* module is a cycle — benign, but only
if it resolves in the right order. A ``from softae.analysis.eis.arc_gate import
gate_arc_closure`` at the **top** of ``gates.py`` raises ``ImportError`` at interpreter
start, because ``gates`` is then in ``sys.modules`` but still empty and neither name
exists yet (verified by emulating exactly that state: it fails on the first constant
it reaches). The import belongs **immediately above** the ``FRONT2_GATES`` assignment,
where ``GateResult`` and ``FLAG`` are long since defined — a mid-module import, which
is legal and is what the ordering requires. So the registration is two adjacent lines
in one place, not one line anywhere. The failure is loud and immediate rather than
silent, but it costs a collection error across the whole suite if taken the obvious
way.

**The threshold is read, not owned, and it is recorded rather than applied.**
``phase_low_max_deg`` comes from
:func:`~softae.analysis.eis.engine_support.pregate_settings` — the live ``−60°`` cut
that :func:`~softae.analysis.eis.engine_support.blocking_open` already uses — rather
than from a new ``[eis.gates]`` key, so this module adds no config surface and
``settings.py`` is untouched. ``passed`` turns on the **state** alone, per spec §6.1;
the threshold classifies which ``OPEN`` spectra are the severe ones and travels in
the detail and the metrics. That split is deliberate: §2.4 found ``phase_low_deg`` a
far better rank predictor of gated error than the state (``ρ = −0.964`` on the seven
rows that are genuinely gated fits) but on an effective ``n`` of **two channels**,
with a permutation floor of ``p ≈ 0.057``. Recording both means the corpus can settle
which one is right without a second ship.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from softae.analysis.eis.arc import OPEN, UNKNOWN, arc_closure
from softae.analysis.eis.engine_support import blocking_open, pregate_settings
from softae.analysis.eis.gates import FLAG, GateResult

#: The gate's name in the log and in ``QualityReport.issues``. Deliberately the same
#: string :meth:`~softae.analysis.eis.arc.ArcClosure.as_record` has always written, so
#: a query over historical ``gate_log_json`` and one over new rows select the same
#: gate rather than two spellings of it.
GATE_NAME = "arc_closure"


def _pregate(ctx: dict[str, Any]) -> Any:
    """``ctx["pregate"]`` if the caller supplied one, else the live config.

    Mirrors ``engine.py``'s ``pregate if pregate is not None else pregate_settings()``.
    The override exists so a test can state its threshold instead of inheriting
    whatever ``softae_config.toml`` happens to say that hour — a gate whose expected
    verdict moves with a config file is a gate whose tests prove nothing.
    """
    supplied = ctx.get("pregate") if isinstance(ctx, dict) else None
    return supplied if supplied is not None else pregate_settings()


def gate_arc_closure(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Did the semicircle close inside the swept window? Flag only; refuses nothing.

    Three outcomes, not two, and the third is the one that matters:

    ``CLOSED``
        ``passed=True``. R₁ is bracketed by measured points.
    ``OPEN``
        ``passed=False``, severity :data:`~softae.analysis.eis.gates.FLAG`. R₁ is
        reached by extrapolating off the high-frequency side — the same number, a
        weaker claim, and until now nothing downstream said so.
    ``UNKNOWN``
        :meth:`~softae.analysis.eis.gates.GateResult.unchecked`. The sweep was too
        short, mismatched or degenerate to judge. ``passed=True`` keeps the module's
        fail-open posture and ``checked=False`` is what stops that posture being
        read as a clean result — 0.5 % of the stored corpus, and it means the
        instrument could not judge, never that the arc is fine.

    Reads nothing from the fit. The inputs are ``f`` and ``Z`` alone, so there is no
    ``pcov`` anywhere on this path and the gate cannot inherit ``fitter.py``'s
    finiteness-test-wearing-a-rank-test's-name defect that every other gate-accuracy
    number here was taken through.
    """
    f = np.asarray(f, dtype=float).ravel()
    Z = np.asarray(Z, dtype=complex).ravel()
    n = int(min(f.size, Z.size))
    ok = np.ones(n, dtype=bool)

    # `arc_closure` wants the file convention (−Z″, positive for capacitive) and
    # degrees of phase; `Z` here is the physics convention `Im Z < 0`, which is what
    # `_physics_complex` guarantees the cascade. This is the same pair
    # `annotate_arc_closure` passes, derived rather than re-measured.
    arc = arc_closure(f[:n], -Z.imag[:n], np.angle(Z[:n], deg=True))

    settings = _pregate(ctx)
    limit = float(getattr(settings, "phase_low_max_deg", float("nan")))
    severe = bool(blocking_open(arc, settings))

    metrics: dict[str, float] = {
        "arc_phase_low_deg": float(arc.phase_low_deg),
        "arc_f_peak_hz": float(arc.f_peak_hz),
        "arc_f_low_hz": float(arc.f_low_hz),
        "arc_band_below_apex_decades": float(arc.band_below_apex_decades),
        # 0/1 rather than a bool because `gate_metrics` flattens into `dict[str, float]`
        # and a bool there reads back out of JSON as a number anyway. This is the
        # column any future validation of the −60° cut is regressed against.
        "arc_blocking_open": 1.0 if severe else 0.0,
    }

    if arc.state == UNKNOWN:
        return GateResult.unchecked(GATE_NAME, FLAG, arc.detail, ok, metrics)

    if arc.state == OPEN:
        detail = arc.detail
        if severe:
            # Named, not acted on. This is the subset the live pre-gate already
            # diverts and the subset a Front-1 promotion would refuse; saying which
            # rows it is costs nothing now and is the whole point of recording.
            detail += (f" — still capacitive at the floor (below {limit:+.0f}°); "
                       f"no realistic extension of this preset closes it")
        return GateResult(GATE_NAME, FLAG, False, detail, ok, metrics)

    return GateResult(GATE_NAME, FLAG, True, arc.detail, ok, metrics)
