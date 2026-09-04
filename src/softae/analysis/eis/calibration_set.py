"""The calibration set itself, and its persistence.

:class:`CalibrationSet` is everything commissioning derived for one fixture at one
hardware state: the per-channel fixture constants, the phase table, the magnitude
window, and the ``role -> measurement_id`` provenance behind each. It answers three
kinds of question — *what did we measure* (``has_short``, ``for_channel``), *what does
that let us do* (:meth:`~CalibrationSet.capabilities`,
:meth:`~CalibrationSet.envelope`), and *is it still about this hardware*
(:meth:`~CalibrationSet.is_stale`).

Then persistence, which is the same asset seen from disk. The canonical TOML is
version-controlled beside the code because framework 8.5 requires commissioning data to
travel with the software; :func:`resolve_calibration` is the only loader production code
should use, because it is the one that **drops the constants when the hardware hash has
moved** rather than applying another board's short blank to today's spectra.

Every name is re-exported from :mod:`softae.analysis.eis.calibration`, which is the
published spelling — import from there.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from softae.analysis.eis.calibration import (
    CalibrationCapabilities,
    FixtureConductance,
    PhaseAccuracyTable,
    hardware_hash,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CalibrationSet:
    """Everything commissioning derived, for one fixture, at one hardware state."""

    fixture_id: str = "default"
    hardware_hash: str = ""
    created_at: str = ""
    channels_measured: tuple[int, ...] = ()
    #: Channels inheriting a representative channel's constants. Listed explicitly so
    #: using one can *log the assumption* — never a silent extrapolation.
    channels_assumed: tuple[int, ...] = ()

    R_short_ohm: dict[int, float] = field(default_factory=dict)
    L_lead_H: dict[int, float] = field(default_factory=dict)
    C_stray_F: dict[int, float] = field(default_factory=dict)
    open_usable: dict[int, bool] = field(default_factory=dict)
    #: Per-channel ``Re(Y_fixture)`` over frequency, from the open blank. A table
    #: rather than a scalar because on this fixture it is dielectric loss
    #: (``d ln G/d ln f`` ≈ 1), so one number describes one frequency only.
    G_fixture: dict[int, FixtureConductance] = field(default_factory=dict)

    phase_acc: PhaseAccuracyTable = field(default_factory=PhaseAccuracyTable)
    z_min_ohm: float = float("nan")
    z_max_ohm: float = float("nan")
    load_error_pct: float = float("nan")

    #: role → measurement_id, so every derived number can be traced to its spectrum (R17).
    sources: dict[str, int] = field(default_factory=dict)

    # ── Interrogation ────────────────────────────────────────────────────────

    @property
    def has_short(self) -> bool:
        return bool(self.R_short_ohm)

    @property
    def has_load(self) -> bool:
        return self.load_error_pct == self.load_error_pct

    @property
    def has_open(self) -> bool:
        return bool(self.open_usable)

    def open_is_usable(self, channel: int | None = None) -> bool:
        """Whether the open blank is a measurement rather than noise.

        An unusable open is **not a missing artifact** — it is the positive evidence
        that shunt admittance is negligible, which is exactly the condition under
        which short-only series correction is *exact* (framework §3.9). One free
        bare-board pass therefore answers "is OSL legitimate here?" with a defensible
        no, which is worth more than a missing answer.
        """
        if not self.open_usable:
            return False
        if channel is None:
            return any(self.open_usable.values())
        return bool(self.open_usable.get(int(channel), False))

    def capabilities(self) -> CalibrationCapabilities:
        """The ladder: what is unlocked, and the one artifact that unblocks each rest."""
        blocked: dict[str, str] = {}

        if self.has_short:
            mode = "osl" if self.open_is_usable() else "series"
        else:
            mode = "none"
            blocked["fixture correction"] = "run the short blank"

        if mode == "series" and self.has_open:
            # Not a limitation worth reporting: an unmeasurable open *selects* this.
            pass
        elif mode == "series":
            blocked["OSL correction"] = (
                "run the open blank (an unusable open is itself a valid answer)")

        phase = not self.phase_acc.is_empty
        if not phase:
            blocked["qualified upper bounds"] = (
                "run the reference capacitor — bounds stay provisional without it")

        window = self.z_min_ohm == self.z_min_ohm and self.z_max_ohm == self.z_max_ohm
        if not window:
            blocked["measured |Z| window"] = (
                "run the reference resistors (≥1 per decade)")

        if not self.has_load:
            blocked["correction validation"] = "run the load blank"

        return CalibrationCapabilities(
            correction_mode=mode,
            phase_floor_measured=phase,
            magnitude_window_measured=window,
            can_validate_correction=self.has_load,
            open_is_usable=self.open_is_usable(),
            blocked=blocked,
        )

    def is_stale(self, *, current_hash: str | None = None) -> bool:
        """Whether this set belongs to different hardware than the rig now has.

        An unknown hash on either side reads as **stale**: "we do not know what this
        was taken on" must degrade capabilities, not pass silently.
        """
        current = current_hash if current_hash is not None else hardware_hash()
        if not self.hardware_hash or not current:
            return True
        return self.hardware_hash != current

    def measured_spread(self, field_name: str = "C_stray_F") -> float:
        """max/min of a per-channel constant over the channels actually measured.

        The empirical size of what :attr:`channels_assumed` is assuming away. NaN when
        fewer than two channels were measured — with one channel there is nothing to
        compare, and reporting 1.0 would read as "no variation" rather than "unknown".
        """
        mapping = getattr(self, field_name, None)
        if not isinstance(mapping, Mapping):
            return float("nan")
        vals = [float(v) for ch, v in mapping.items()
                if ch in self.channels_measured
                and float(v) == float(v) and float(v) > 0]
        if len(vals) < 2:
            return float("nan")
        return float(max(vals) / min(vals))

    def for_channel(self, channel: int) -> dict[str, float]:
        """This channel's constants, logging when they are inherited rather than its own.

        The inheritance is **not** a formality on this fixture. Seven tied open blanks
        across nominally identical stripes (ch17–23, 2026-08-06) gave ``C_stray``
        spanning 10.2–24.7 pF — a 2.4× spread — while repeating to 1% on any single
        channel. So channel-to-channel variation is real and roughly an order of
        magnitude larger than the measurement error it would otherwise be mistaken for.
        The warning carries that number, because "inherited" without a magnitude reads
        as bookkeeping and gets skimmed.
        """
        ch = int(channel)
        if ch in self.channels_assumed:
            logger.warning(
                "eis_calibration_channel_assumed", channel=ch,
                measured=self.channels_measured,
                measured_spread=self.measured_spread("C_stray_F"),
                msg="constants inherited from a representative channel — measured "
                    "channel-to-channel C_stray spread on this fixture is 2.4x "
                    "(10.2-24.7 pF over 7 identical stripes), so this is a real "
                    "uncertainty, not a formality",
            )
        return {
            "R_short_ohm": float(self.R_short_ohm.get(ch, float("nan"))),
            "L_lead_H": float(self.L_lead_H.get(ch, float("nan"))),
            "C_stray_F": float(self.C_stray_F.get(ch, float("nan"))),
        }

    def _headline_phase_row(self) -> int | None:
        """Which row of :attr:`phase_acc` becomes the envelope's headline ε and anchor.

        **The largest characterised ε, and its own |Z| as the anchor for it** — not the
        lowest-|Z| row. That was the shipped rule, justified in a comment as *"the
        lowest characterised impedance is the conservative headline value"*, and it is
        wrong on its own terms: on the committed 19-point table the lowest-|Z| row gives
        tan ε = 0.0761 while the largest-ε row gives 0.1072, so ``argmin(|Z|)`` was the
        *less* conservative of the two. ε is non-monotonic in |Z| and varies 29× across
        that table, so a rule that sorts on |Z| optimises neither field.

        The **anchor** is the larger consequence, and it is not about conservatism at
        all. ``phase_noise_at_ohm`` is what
        :meth:`~softae.analysis.eis.envelope.InstrumentEnvelope.phase_noise_valid_at`
        measures decades from, so this one index decides which spectra may quote the
        floor. ``argmin(|Z|)`` put it at 795.6 Ω — *further* from where films sit than
        the 9.9 kΩ guess it replaces, making the trust radius worse in the act of
        replacing an assumption with a measurement. ``argmax(ε)`` puts it at 10.1 MΩ — the commissioned 10 MΩ reference resistor —
        which brings 10⁶–10⁸ Ω films inside the radius for the first time (measured on
        the corpus: 896 spectra in band against 320 for the shipped default and 94 for
        ``argmin(|Z|)``).

        Selecting both fields from the *same* row is the property that matters:
        whatever else it is, the pair is then coherent.

        ``None`` when no row is usable. A NaN ε must not be selected and must not claim
        ``phase_noise_measured``, which would be "unknown" spelled as "checked and
        clean" — the envelope's own ``phase_noise_measured = True`` default is the
        original of exactly that mistake.
        """
        z = np.asarray(self.phase_acc.z_ohm, dtype=float)
        eps = np.asarray(self.phase_acc.eps_deg, dtype=float)
        n = int(min(z.size, eps.size))
        if n == 0:
            return None
        usable = np.isfinite(eps[:n]) & np.isfinite(z[:n]) & (z[:n] > 0)
        if not usable.any():
            return None
        rows = np.flatnonzero(usable)
        return int(rows[int(np.argmax(eps[:n][usable]))])

    def envelope(self, base: Any = None, *, role: str = "sample") -> Any:
        """An :class:`InstrumentEnvelope` with whatever this set actually measured.

        Unmeasured quantities keep the configured estimate *and its
        ``*_measured = False`` flag*, so a partial calibration never silently
        promotes a guess to a measurement.

        *role* names what was measured. The magnitude window is promoted **only for a
        sample** — see
        :func:`~softae.analysis.eis.envelope.magnitude_window_applies` for why, and for
        the commissioning artifact that stops being admissible under its own window if
        this carve-out is skipped. The gate lives here rather than at the call sites so
        it cannot be forgotten by one of them.
        """
        from softae.analysis.eis.envelope import (
            InstrumentEnvelope,
            instrument_envelope,
            magnitude_window_applies,
        )

        env = base if base is not None else instrument_envelope()
        updates: dict[str, Any] = {}

        idx = self._headline_phase_row()
        if idx is not None:
            updates["phase_noise_deg"] = float(self.phase_acc.eps_deg[idx])
            updates["phase_noise_at_ohm"] = float(self.phase_acc.z_ohm[idx])
            updates["phase_noise_load"] = self.phase_acc.load
            updates["phase_noise_valid_decades"] = float(self.phase_acc.valid_decades)
            updates["phase_noise_measured"] = True

        window_measured = (self.z_min_ohm == self.z_min_ohm
                           and self.z_max_ohm == self.z_max_ohm)
        if window_measured and magnitude_window_applies(role):
            updates["z_min_ohm"] = float(self.z_min_ohm)
            updates["z_max_ohm"] = float(self.z_max_ohm)
            updates["magnitude_window_measured"] = True
        elif window_measured:
            logger.info(
                "eis_magnitude_window_not_applied", role=role,
                z_min_ohm=float(self.z_min_ohm), z_max_ohm=float(self.z_max_ohm),
                msg="commissioning artifact judged against the configured window — a "
                    "window derived from reference resistors does not certify a blank",
            )

        if self.created_at:
            updates["measured_at"] = self.created_at

        if not updates:
            return env
        if isinstance(env, InstrumentEnvelope):
            return replace(env, **updates)
        return env

    def describe(self) -> str:
        caps = self.capabilities()
        n = len(self.channels_measured)
        assumed = (f" (+{len(self.channels_assumed)} assumed)"
                   if self.channels_assumed else "")
        return (
            f"calibration '{self.fixture_id}' @{self.hardware_hash or '?'} "
            f"[{self.created_at or 'undated'}], {n} channel(s){assumed}: "
            f"{caps.describe()}"
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """A plain mapping, keys stringified for TOML/JSON round-tripping."""
        return {
            "fixture_id": self.fixture_id,
            "hardware_hash": self.hardware_hash,
            "created_at": self.created_at,
            "channels_measured": list(self.channels_measured),
            "channels_assumed": list(self.channels_assumed),
            "R_short_ohm": {str(k): float(v) for k, v in self.R_short_ohm.items()},
            "L_lead_H": {str(k): float(v) for k, v in self.L_lead_H.items()},
            "C_stray_F": {str(k): float(v) for k, v in self.C_stray_F.items()},
            "open_usable": {str(k): bool(v) for k, v in self.open_usable.items()},
            "G_fixture": {
                str(k): {"freq_hz": list(v.freq_hz), "G_S": list(v.G_S),
                         "exponent": float(v.exponent)}
                for k, v in self.G_fixture.items()
            },
            "phase_acc": {
                "z_ohm": list(self.phase_acc.z_ohm),
                "eps_deg": list(self.phase_acc.eps_deg),
                "load": self.phase_acc.load,
                "valid_decades": self.phase_acc.valid_decades,
            },
            "z_min_ohm": self.z_min_ohm,
            "z_max_ohm": self.z_max_ohm,
            "load_error_pct": self.load_error_pct,
            "sources": {str(k): int(v) for k, v in self.sources.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationSet":
        """Rebuild from :meth:`to_dict`, tolerating a partial or older mapping."""
        def _imap(key: str) -> dict[int, float]:
            out: dict[int, float] = {}
            for k, v in (data.get(key) or {}).items():
                try:
                    out[int(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return out

        def _f(key: str) -> float:
            try:
                return float(data.get(key, float("nan")))
            except (TypeError, ValueError):
                return float("nan")

        pa = data.get("phase_acc") or {}
        phase = PhaseAccuracyTable(
            z_ohm=tuple(float(z) for z in (pa.get("z_ohm") or [])),
            eps_deg=tuple(float(e) for e in (pa.get("eps_deg") or [])),
            load=str(pa.get("load", "resistive")),
            valid_decades=float(pa.get("valid_decades", 1.0) or 1.0),
        )
        opens: dict[int, bool] = {}
        for k, v in (data.get("open_usable") or {}).items():
            try:
                opens[int(k)] = bool(v)
            except (TypeError, ValueError):
                continue
        gfix: dict[int, FixtureConductance] = {}
        for k, v in (data.get("G_fixture") or {}).items():
            try:
                gfix[int(k)] = FixtureConductance(
                    freq_hz=tuple(float(x) for x in (v.get("freq_hz") or [])),
                    G_S=tuple(float(x) for x in (v.get("G_S") or [])),
                    exponent=float(v.get("exponent", float("nan"))),
                )
            except (TypeError, ValueError, AttributeError):
                continue
        sources: dict[str, int] = {}
        for k, v in (data.get("sources") or {}).items():
            try:
                sources[str(k)] = int(v)
            except (TypeError, ValueError):
                continue

        return cls(
            fixture_id=str(data.get("fixture_id", "default")),
            hardware_hash=str(data.get("hardware_hash", "")),
            created_at=str(data.get("created_at", "")),
            channels_measured=tuple(
                int(c) for c in (data.get("channels_measured") or [])),
            channels_assumed=tuple(
                int(c) for c in (data.get("channels_assumed") or [])),
            R_short_ohm=_imap("R_short_ohm"),
            L_lead_H=_imap("L_lead_H"),
            C_stray_F=_imap("C_stray_F"),
            open_usable=opens,
            G_fixture=gfix,
            phase_acc=phase,
            z_min_ohm=_f("z_min_ohm"),
            z_max_ohm=_f("z_max_ohm"),
            load_error_pct=_f("load_error_pct"),
            sources=sources,
        )


# ── Persistence ──────────────────────────────────────────────────────────────

def calibration_path(fixture_id: str, *, root: Path | None = None) -> Path:
    """Where a fixture's canonical calibration lives.

    Version-controlled beside the code, because framework §8.5 requires commissioning
    data to travel with the software and be re-validated after a hardware change. The
    database keeps history; *this* file is what a checkout reproduces.
    """
    base = root if root is not None else Path("calibration") / "eis"
    return base / f"{fixture_id}.toml"


def save_calibration(
    calibration: CalibrationSet, *, root: Path | None = None
) -> Path:
    """Write the canonical TOML. Returns the path written."""
    path = calibration_path(calibration.fixture_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_to_toml(calibration.to_dict()), encoding="utf-8")
    logger.info("eis_calibration_saved", path=str(path),
                fixture=calibration.fixture_id,
                hardware_hash=calibration.hardware_hash)
    return path


def load_calibration(
    fixture_id: str = "default",
    *,
    root: Path | None = None,
    path: Path | None = None,
) -> CalibrationSet | None:
    """Load a calibration, or ``None`` when there is none to load.

    ``None`` is a legitimate state, not an error: the envelope then falls back to
    ``[eis.instrument]``'s estimates *with their ``measured = False`` flags intact*,
    which is exactly how an uncalibrated rig should behave.
    """
    target = path if path is not None else calibration_path(fixture_id, root=root)
    if not target.exists():
        logger.info("eis_calibration_absent", path=str(target),
                    msg="falling back to configured estimates, flagged unmeasured")
        return None
    try:
        import tomllib

        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("eis_calibration_unreadable", path=str(target), exc_info=True)
        return None
    return CalibrationSet.from_dict(data)


def resolve_calibration(
    fixture_id: str = "default",
    *,
    root: Path | None = None,
    path: Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> CalibrationSet | None:
    """Load a calibration and **degrade it if the hardware has changed**.

    A stale set is not applied. Returning it unchanged would let a short blank from a
    different board correct today's spectra — silently, and with every number looking
    plausible. Instead the fixture constants are dropped and the capability ladder
    falls back to "run the short blank", which is the truth.
    """
    # Resolved through the hub, per call, rather than as this module's own global —
    # the same late binding ``_legacy_report`` keeps for ``fit_circuit`` and for the
    # same reason. ``tests/test_campaign_cli.py`` patches
    # ``softae.analysis.eis.calibration.load_calibration`` to make preflight see an
    # uncommissioned rig; while the two functions shared a module that patch reached
    # here for free, and a bare call would silently stop honouring it.
    from softae.analysis.eis.calibration import load_calibration

    cal = load_calibration(fixture_id, root=root, path=path)
    if cal is None:
        return None

    current = hardware_hash(config)
    if not cal.is_stale(current_hash=current):
        return cal

    logger.warning(
        "eis_calibration_stale", fixture=fixture_id,
        recorded=cal.hardware_hash or "(none)", current=current,
        msg="hardware changed since this calibration — constants dropped rather "
            "than applied to a different fixture",
    )
    return replace(
        cal, R_short_ohm={}, L_lead_H={}, C_stray_F={}, open_usable={},
        phase_acc=PhaseAccuracyTable(), z_min_ohm=float("nan"),
        z_max_ohm=float("nan"), load_error_pct=float("nan"),
    )


def _to_toml(data: Mapping[str, Any]) -> str:
    """Minimal TOML writer — the stdlib reads TOML but does not write it.

    Deliberately not a dependency, but it must handle **nested** tables. The first
    version emitted one level and fell through to ``json.dumps(str(v))`` for anything
    deeper, so ``G_fixture`` — a mapping of per-channel tables — serialised as quoted
    Python reprs and was silently discarded on load. The file looked fine and the
    calibration came back missing a field, which is the worst shape a persistence bug
    can take. Recursion costs four lines and removes the whole class of it.
    """
    def _fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int,)):
            return str(v)
        if isinstance(v, float):
            if v != v:
                return "nan"
            return repr(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_fmt(x) for x in v) + "]"
        return json.dumps(str(v))

    def _emit(node: Mapping[str, Any], prefix: str, tables: list[str]) -> list[str]:
        """Return this table's scalar lines, appending nested tables to *tables*."""
        own: list[str] = []
        nested: list[tuple[str, Mapping[str, Any]]] = []
        for k, v in node.items():
            if isinstance(v, Mapping):
                nested.append((str(k), v))
            else:
                own.append(f"{k} = {_fmt(v)}")
        for k, v in nested:
            # TOML permits digit-only bare keys, which is what channel numbers are.
            path = f"{prefix}.{k}" if prefix else str(k)
            body = _emit(v, path, tables)
            tables.append(f"\n[{path}]\n" + "\n".join(body))
        return own

    tables: list[str] = []
    scalars = _emit(data, "", tables)

    header = (
        "# EIS commissioning calibration — generated, but version-controlled.\n"
        "# Framework §8.5: commissioning data travels with the code and is\n"
        "# re-validated after any hardware change. Staleness is by hardware_hash,\n"
        "# not by date: fixture drift is minimal, so there is no expiry.\n"
    )
    return header + "\n".join(scalars) + "\n" + "\n".join(tables) + "\n"


def describe_or_absent(calibration: CalibrationSet | None) -> str:
    """One startup line, whether or not a calibration exists."""
    if calibration is None:
        return (
            "EIS calibration: none — using configured estimates (flagged unmeasured). "
            "Run commissioning to measure the fixture: short blank → load blank → "
            "reference capacitor, in that order of value per hour of bench time."
        )
    return calibration.describe()
