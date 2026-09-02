"""Commissioning sweeps: the acquisition half of E2.

A commissioning measurement is an **ordinary EIS measurement with a different tag**.
It uses the same :func:`~softae.core.deposition_steps.eis_measure_step`, the same
executor routing, the same ``EISResult.save`` and the same ``record_measurement`` — only
``role`` differs. That is why acquisition costs almost no new code here, and why a blank
is queryable, plottable and browsable with everything already built for samples.

**What to run, in order of value per hour of bench time** (framework §7.4):

1. **Short blank** — jumpered channel. Gives ``R_fixture(f)`` and ``L_lead``, which
   unlocks series correction and licenses pinning ``L ≈ 0`` in the sample fit so F5's
   400–500 µH artifact cannot recur.
2. **Load blank** — precision resistor. Validates the correction end to end (R9); until
   it exists there is no evidence the correction helps rather than harms.
3. **Reference capacitor** — low-loss, C0G/NP0. The only route to a *measured* phase
   floor where the films actually sit, and §7.4 calls it the most-skipped and
   least-substitutable of the three.

**The open blank is free but weakest**, and on this fixture it is also compromised: an
open cell inherently floats the reference electrode, which is the condition that
produced the withdrawn ``Z_φ`` (overhaul §3.7, F13). A genuine fixture open needs **RE
tied to CE at the connector** — which the commissioning board should have designed in
rather than discovered at the bench. Until it does, a bare-board open measures
inter-stripe geometry, not the fixture.

.. note::
   The useful artifacts live on a **dedicated commissioning board** — same PCB,
   populated with jumpers, one precision resistor and one low-loss capacitor. That
   costs zero AE budget and gives per-channel data across all 32 mux paths, where
   sacrificing production channels would cover two or three.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from softae.analysis.eis.calibration import COMMISSIONING_ROLES, short_is_plausible
from softae.core.deposition_steps import eis_measure_step
from softae.workflows.workflow_model import Workflow, WorkflowStep

logger = structlog.get_logger(__name__)


class CommissioningError(RuntimeError):
    """A commissioning pass produced nothing usable.

    Raised rather than returning a hollow calibration: a set with no fixture
    constants that still *looks* like a calibration is worse than none, because
    ``resolve_calibration`` would hand it out and the capability ladder would report
    a correction mode nothing supports.
    """

#: Recommended order, best value per hour of bench time first (framework §7.4).
COMMISSIONING_ORDER = ("blank_short", "blank_load", "reference_cap", "reference_r")

#: What each artifact needs physically present, for the operator prompt.
ARTIFACT_SETUP = {
    "blank_short": "a jumpered channel (CE-WE shorted)",
    "blank_load": "a precision resistor of known value across CE-WE",
    "blank_open": "a bare, uncast board — NOTE: floats RE unless RE is tied to CE",
    "reference_cap": "a low-loss C0G/NP0 capacitor (100 pF - 1 nF, tan d < 1e-3)",
    "reference_r": "a reference resistor; repeat once per impedance decade",
}


@dataclass(frozen=True)
class AcquiredSpectrum:
    """One commissioning sweep, **with the facts recorded alongside it**.

    Artifacts used to reach :func:`derive_calibration` as bare ``(channel, f, Z)``
    tuples, with the part's marked value and the electrode mode supplied *per role*.
    That collapse is a bug with a shape: the database stores ``nominal_value`` and
    ``electrode_mode`` on every measurement row, and a role-level dict can only hold
    one, so the last acquisition's facts were silently applied to all of them. On the
    real mux16 record three reference capacitors marked 1e-10, 1e-10 and 1e-9 F were
    all derived against 1e-9, and one pre-jumper sweep recorded ``electrode_mode =
    'unknown'`` was laundered into "two" by a later acquisition of the same role —
    defeating the R24/F17 refusal exactly where it was meant to bite.

    Unpacking as a 3-tuple still works, so callers and tests that pass
    ``(channel, f, Z)`` are unaffected: the extra facts default to "not recorded",
    which is the honest reading of a tuple that never carried them.
    """

    channel: int
    freq_hz: Any
    Z: Any
    #: The part's marked value as recorded *for this acquisition*, or None.
    nominal: float | None = None
    #: How this acquisition was sensed, or None when the caller did not say.
    electrode_mode: str | None = None
    #: Row this came from, for R17 provenance.
    measurement_id: int | None = None

    def __iter__(self) -> Iterator[Any]:
        return iter((self.channel, self.freq_hz, self.Z))


def _acquisitions(items: Any) -> list[AcquiredSpectrum]:
    """Normalise a role's artifacts to :class:`AcquiredSpectrum`, tuples included."""
    out: list[AcquiredSpectrum] = []
    for item in items or []:
        if isinstance(item, AcquiredSpectrum):
            out.append(item)
        else:
            ch, f, Z = item
            out.append(AcquiredSpectrum(int(ch), f, Z))
    return out


def commissioning_step_name(role: str, channel: int) -> str:
    """The step name a commissioning sweep is recorded under.

    Deliberately *not* ``measure_eis_ch<N>``: the campaign objective extractors match
    that prefix, and a blank must never be scored as a trial. A distinct prefix makes
    that impossible rather than merely unlikely.
    """
    return f"commission_{role}_ch{channel}"


def build_commissioning_workflow(
    role: str,
    channels: Sequence[int],
    *,
    fixture_id: str = "default",
    nominal: float | None = None,
    electrode_mode: str = "unknown",
    name: str | None = None,
) -> Workflow:
    """One sweep per channel for a single commissioning artifact.

    *nominal* is the part's marked value — the resistor's ohms or the capacitor's
    farads. Carried in metadata rather than assumed at analysis time, because
    overhaul §3.7 records a capacitor marked "102" (1 nF) that measured ~150 nF: the
    marking and the measurement disagreeing is exactly the check that catches an
    unusable reference, and it needs both numbers.
    """
    if role not in COMMISSIONING_ROLES:
        raise ValueError(
            f"unknown commissioning role {role!r}; expected one of "
            f"{COMMISSIONING_ROLES}"
        )
    chans = [int(c) for c in channels]
    if not chans:
        raise ValueError("a commissioning sweep needs at least one channel")

    steps: list[WorkflowStep] = []
    for ch in chans:
        step = eis_measure_step(ch, name=commissioning_step_name(role, ch))
        steps.append(
            WorkflowStep(
                name=step.name,
                instrument=step.instrument,
                method=step.method,
                params=dict(step.params),
                timeout_s=step.timeout_s,
                retry=step.retry,
                # Role, fixture and the part's marked value all ride on the step so
                # the executor's routing can record them without a second lookup.
                # The nominal in particular MUST be persisted here: derivation
                # happens later, possibly weeks later, and cannot ask the socket
                # what was in it.
                tags={
                    **step.tags,
                    "role": role,
                    "fixture_id": fixture_id,
                    # How the cell was sensed. Rides on the step for the same reason
                    # the nominal does: derivation happens later and cannot ask the
                    # bench whether the jumper was fitted (R24/F17).
                    "electrode_mode": str(electrode_mode or "unknown"),
                    **({"nominal": repr(float(nominal))}
                       if nominal is not None else {}),
                },
            )
        )

    return Workflow(
        name=name or f"commission_{role}",
        description=f"Commissioning sweep: {role} on {len(chans)} channel(s)",
        setup=steps,
        iterations=1,
        metadata={
            "source": "commissioning",
            "role": role,
            "fixture_id": fixture_id,
            "channels": chans,
            "nominal": nominal,
            "electrode_mode": electrode_mode,
            "setup_required": ARTIFACT_SETUP.get(role, ""),
        },
    )


def derive_calibration(
    artifacts: dict[str, list[Any]],
    *,
    fixture_id: str = "default",
    created_at: str = "",
    hardware_hash_value: str = "",
    nominals: dict[str, float] | None = None,
    nominal_overrides: dict[str, float] | None = None,
    sources: dict[str, int] | None = None,
    representative_channel: int | None = None,
    all_channels: Sequence[int] | None = None,
    electrode_modes: dict[str, str] | None = None,
) -> Any:
    """Turn acquired commissioning spectra into a :class:`CalibrationSet`.

    *artifacts* maps ``role -> [AcquiredSpectrum | (channel, freq, Z), ...]``. Every
    role is optional: the calibration is incremental by design, and a set with only a
    short blank is useful — it unlocks series correction — while reporting what the
    rest would add.

    **The marked value is resolved per acquisition**, in one order, most specific last
    to win:

    ===================  ============================================================
    *nominals*           role-wide fallback — the newest value recorded for the role,
                         used only for an acquisition that recorded none of its own.
    ``acq.nominal``      what *this* acquisition recorded. Normally governs.
    *nominal_overrides*  the operator asserting a part was mis-entered at acquisition
                         time. Overrides **every** acquisition of that role, which is
                         the point: the whole role's records are being corrected.
    ===================  ============================================================

    The electrode mode resolves the same way, with *electrode_modes* as the role-wide
    fallback and each acquisition's own recorded mode governing when it has one. The
    R24/F17 refusal then applies **per acquisition**, so one pre-jumper sweep is
    dropped without taking its re-measured siblings with it.

    Channels present in *all_channels* but not measured are recorded as
    ``channels_assumed``, so using one **logs the assumption** rather than silently
    extrapolating a fixture constant across the mux.
    """
    from softae.analysis.eis.calibration import (
        CalibrationSet,
        PhaseAccuracyTable,
        derive_open,
        derive_open_constants,
        derive_phase_table,
        derive_reference_cap,
        derive_reference_r,
        derive_short,
        electrode_mode_ok,
    )

    # R24: a two-terminal reference sensed in three-electrode mode is uncalibratable,
    # so it is dropped *here*, before any constant is derived from it. Refusing at the
    # derivation boundary rather than at acquisition means an already-recorded spectrum
    # can be re-imported with its mode declared, without re-measuring.
    #
    # Per acquisition, not per role. Role-level refusal meant one role's mode was
    # whatever the newest row happened to say, so an unknown-mode sweep sitting beside
    # a later two-electrode one inherited its verdict — the refusal reading as "pass"
    # on precisely the spectrum it exists to stop.
    modes = electrode_modes or {}
    refused: list[str] = []
    accepted: dict[str, list[AcquiredSpectrum]] = {}
    for role, items in (artifacts or {}).items():
        kept: list[AcquiredSpectrum] = []
        for acq in _acquisitions(items):
            mode = acq.electrode_mode or modes.get(role, "unknown")
            ok, why = electrode_mode_ok(role, mode)
            if ok:
                kept.append(acq)
                continue
            refused.append(f"{role} ch{acq.channel}: {why}")
            logger.warning("commissioning_electrode_mode_refused", role=role,
                           channel=acq.channel, mode=mode, reason=why)
        if kept:
            accepted[role] = kept
    if refused and not accepted:
        raise CommissioningError(
            "every commissioning artifact was refused on electrode mode:\n  "
            + "\n  ".join(refused)
            + "\nMeasure two-terminal references with RE tied to CE (two-electrode)."
        )
    artifacts = accepted

    nominals = nominals or {}
    overrides = nominal_overrides or {}

    def _nominal_for(role: str, acq: AcquiredSpectrum) -> float | None:
        """This acquisition's marked value: override, then its own, then the role's."""
        if role in overrides:
            return float(overrides[role])
        if acq.nominal is not None and float(acq.nominal) == float(acq.nominal):
            return float(acq.nominal)
        value = nominals.get(role)
        return float(value) if value is not None else None

    R_short: dict[int, float] = {}
    L_lead: dict[int, float] = {}
    open_usable: dict[int, bool] = {}
    C_stray: dict[int, float] = {}
    G_fixture: dict[int, Any] = {}
    measured: set[int] = set()

    rejected_shorts: list[tuple[int, str]] = []
    for acq in artifacts.get("blank_short", []):
        ch, f, Z = acq.channel, acq.freq_hz, acq.Z
        R, L = derive_short(f, Z)
        ok, reason = short_is_plausible(R, L)
        if not ok:
            # A short blank is the one artifact whose answer is known in advance, so
            # an implausible one is an operator setup error — recoverable now, and
            # invisible forever if it is simply recorded. Refuse it rather than write
            # a fixture constant nothing could have produced.
            rejected_shorts.append((int(ch), reason))
            logger.warning("commissioning_short_implausible", channel=int(ch),
                           R_ohm=R, L_H=L, reason=reason)
            continue
        R_short[int(ch)] = R
        L_lead[int(ch)] = L
        measured.add(int(ch))

    if rejected_shorts and not R_short:
        raise CommissioningError(
            "every short blank was implausible, so no fixture constants were "
            "derived:\n  " + "\n  ".join(
                f"ch{ch}: {reason}" for ch, reason in rejected_shorts)
            + "\nCheck the jumper and the channel selection, then re-run."
        )

    for acq in artifacts.get("blank_open", []):
        ch, f, Z = acq.channel, acq.freq_hz, acq.Z
        usable, over, flips = derive_open(f, Z)
        open_usable[int(ch)] = usable
        measured.add(int(ch))

        # The verdict alone used to be all this produced, which left `C_stray_F`
        # declared, serialised, read by `for_channel` — and written by nothing. Both
        # shunt constants are derivable from the artifact already in hand.
        #
        # Derived only from a *usable* open: on an unusable one the numbers would be
        # noise wearing the shape of a measurement, and the whole point of the verdict
        # is to decide whether the trace means anything.
        if usable:
            C, G = derive_open_constants(f, Z)
            if C == C:
                C_stray[int(ch)] = C
            if not G.is_empty:
                G_fixture[int(ch)] = G
            logger.info("commissioning_open_constants", channel=int(ch),
                        C_stray_pF=C * 1e12 if C == C else None,
                        exponent=G.exponent, summary=G.describe())

        logger.info("commissioning_open", channel=int(ch), usable=usable,
                    over_range_frac=over, im_flip_frac=flips)

    load_error = float("nan")
    for acq in artifacts.get("blank_load", []):
        nom = _nominal_for("blank_load", acq)
        if nom:
            _R, err, _noise = derive_reference_r(
                acq.freq_hz, acq.Z, nominal_ohm=float(nom))
            load_error = err
        measured.add(int(acq.channel))

    # Phase accuracy: every reference component contributes one (|Z|, eps) point.
    #
    # z_points is also the *only* source of the measured |Z| window below, so nothing
    # ungated may reach it. It used to receive the reference capacitor's whole sweep
    # via an ungated table, which put the instrument's ~1.0147 GΩ input rail into
    # z_max — an envelope bound that was the instrument giving up, not a magnitude the
    # fixture reproduces.
    z_points: list[float] = []
    eps_points: list[float] = []
    load_kind = "resistive"
    for acq in artifacts.get("reference_r", []):
        nom = _nominal_for("reference_r", acq)
        ref = derive_reference_r(
            acq.freq_hz, acq.Z, nominal_ohm=float(nom or 0.0) or 1.0)
        # Gated, and anchored where the gate left it. Both halves used to be wrong in
        # the same way: an ungated scatter statistic, filed at |R| rather than at the
        # |Z| it was measured at.
        #
        # An earlier comment here argued the resistive branch could not be gated,
        # because "a resistor's |Z| is flat by construction" leaves the slope-plateau
        # detector nothing to see but the instrument's rail — which it cannot tell from
        # a working part. The second half of that is true and the first half is false on
        # this fixture: above ~10^5 Ω the board's own 10-25 pF stray shunts a reference
        # resistor into a parallel RC, so roughly a third of the sweep on a 1 MΩ part
        # sits on a capacitive roll-off rather than on the resistor (9 of 28 points on
        # the deployed 4 Hz-200 kHz grid). What the slope test fires on here is the
        # STRAY, not the rail — and a railed resistor and a shunt-dominated one are both
        # non-measurements, so dropping both is the right answer either way. The ladder
        # (">=1 per decade") and the load blank's error check still guard operator
        # choice of a bad nominal; this guards a fixture artifact inside one good part's
        # sweep, which they cannot see.
        #
        # Anchoring on ref.z_at_eps_ohm rather than |R| also makes the comment above
        # z_points true again: median(Re Z) is pulled below |Z| by exactly that
        # roll-off, by 2.9x on a 10 MΩ reference, so epsilon would be filed at an
        # impedance the sweep never characterised — and at the top of the ladder that
        # gap approaches valid_decades, where the table starts claiming coverage it
        # does not have and refusing coverage it does.
        if ref.z_at_eps_ohm == ref.z_at_eps_ohm and ref.eps_deg == ref.eps_deg:
            z_points.append(ref.z_at_eps_ohm)
            eps_points.append(ref.eps_deg)
        measured.add(int(acq.channel))

    for acq in artifacts.get("reference_cap", []):
        ch = int(acq.channel)
        nom = _nominal_for("reference_cap", acq)
        # The fixture's stray shunt sits in PARALLEL with the part, so the marked-value
        # check is only a statement about the part once it is subtracted. The open
        # blanks were processed above in this same pass, so this channel's stray is
        # already in hand -- when one was measured.
        #
        # Known gap, accepted: a derive containing a reference_cap but no blank_open for
        # its channel has no same-pass stray and falls back to the raw check. Reaching
        # into a CalibrationSet on disk for one would import staleness, fixture-id and
        # hardware-hash questions into a check that currently has none, for a pass shape
        # that has not yet occurred -- deliberate YAGNI, revisit when it does.
        cap = derive_reference_cap(acq.freq_hz, acq.Z, nominal_F=nom,
                                   C_stray_F=C_stray.get(ch))
        # One capacitor spans several |Z| decades across its sweep, so it contributes a
        # *table* rather than a point — see derive_phase_table for why the per-decade
        # median is the right statistic and the sweep minimum is not a bound at all.
        #
        # On RAW Z, deliberately, and it must stay that way. The epsilon table bounds
        # the system AS USED: no production sample spectrum is shunt-corrected anywhere
        # in this codebase (the fixture correction is series-only by design), so a table
        # built from stray-corrected impedances would claim a phase floor that no sample
        # measurement ever experiences. tand_min and its |Z| are raw for the same reason.
        z_dec, e_dec = derive_phase_table(acq.freq_hz, acq.Z, load="capacitive")
        if z_dec:
            z_points.extend(z_dec)
            eps_points.extend(e_dec)
            load_kind = "capacitive"
        measured.add(ch)
        logger.info("commissioning_reference_cap", channel=ch,
                    nominal_F=nom, C_F=cap.C_F, C_raw_F=cap.C_raw_F,
                    C_corrected_F=cap.C_corrected_F if cap.corrected else None,
                    C_stray_F=cap.C_stray_F if cap.corrected else None,
                    tand_min=cap.tand_min, at_ohm=cap.z_at_tand_min_ohm,
                    tand_basis="raw_Z")

    phase = PhaseAccuracyTable()
    if z_points:
        order = sorted(range(len(z_points)), key=lambda i: z_points[i])
        phase = PhaseAccuracyTable(
            z_ohm=tuple(z_points[i] for i in order),
            eps_deg=tuple(eps_points[i] for i in order),
            load=load_kind,
        )

    # The window is the *surviving* points, because those are the only ones that were
    # measurements. With nothing left the empty-table path stands: NaN, and the ladder
    # keeps asking for the reference resistors.
    z_min = min(z_points) if z_points else float("nan")
    z_max = max(z_points) if z_points else float("nan")

    assumed: tuple[int, ...] = ()
    if all_channels:
        assumed = tuple(sorted(set(int(c) for c in all_channels) - measured))
        if assumed and representative_channel is not None:
            rep = int(representative_channel)
            for ch in assumed:
                if rep in R_short:
                    R_short.setdefault(ch, R_short[rep])
                if rep in L_lead:
                    L_lead.setdefault(ch, L_lead[rep])

    cal = CalibrationSet(
        fixture_id=fixture_id,
        hardware_hash=hardware_hash_value,
        created_at=created_at,
        channels_measured=tuple(sorted(measured)),
        channels_assumed=assumed,
        R_short_ohm=R_short,
        L_lead_H=L_lead,
        open_usable=open_usable,
        C_stray_F=C_stray,
        G_fixture=G_fixture,
        phase_acc=phase,
        z_min_ohm=z_min,
        z_max_ohm=z_max,
        load_error_pct=load_error,
        sources=dict(sources or {}),
    )
    logger.info("commissioning_derived", **{"summary": cal.describe()})
    return cal


def next_artifact(calibration: Any = None) -> str | None:
    """The most valuable commissioning artifact still missing, or ``None``.

    Encodes §7.4's priority so an operator is told *what to do next* rather than
    handed a list of everything absent.
    """
    if calibration is None:
        return COMMISSIONING_ORDER[0]
    caps = calibration.capabilities()
    if caps.correction_mode == "none":
        return "blank_short"
    if not caps.can_validate_correction:
        return "blank_load"
    if not caps.phase_floor_measured:
        return "reference_cap"
    if not caps.magnitude_window_measured:
        return "reference_r"
    return None
