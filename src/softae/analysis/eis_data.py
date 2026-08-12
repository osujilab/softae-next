"""EIS data model and I/O utilities.

Provides a structured :class:`EISResult` container for impedance data
and helper functions for parsing, saving, and loading EIS measurement
files.  This replaces the ad-hoc list-based data flow from the legacy
``ESPico_class.py``.

Data Flow
---------
1. ``AsyncESPico.sendscript_getdata()`` → raw PalmSens result
2. ``EISResult.from_raw()`` wraps the extraction + metadata tagging
3. ``EISResult.save()`` writes a standardised text file with header
4. ``EISResult.load()`` reads it back for analysis

The column format is::

    f(Hz)  Z_total(Ohm)  phase(deg)  Z'(Ohm)  -Z''(Ohm)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime xarray cost
    from softae.analysis.measurement_result import MeasurementResult

# Standard column names (matches legacy convention)
COLUMN_NAMES = ["f(Hz)", "Z_total(Ohm)", "phase(deg)", "Z'(Ohm)", "-Z''(Ohm)"]
RESIDUAL_COLUMN_NAMES = ["resid_Zreal(%)", "resid_-Z''(%)"]

#: Trailing column carrying which admission gate dropped each point.  Stored as a
#: numeric id against a legend in the header rather than as a name, so ``np.savetxt``
#: and ``np.loadtxt`` keep working unchanged on a purely numeric table.  0 = kept.
GATE_COLUMN_NAME = "gate_drop_id"

#: Registry key for this modality in the ``MeasurementResult`` contract, and the
#: value stored in the ``measurements.modality`` column from Tier 2 component 3.
EIS_MODALITY = "eis"

#: Dimension/coordinate name of the frequency axis in an EIS payload. Units are
#: in the name because a bare ``frequency`` on disk invites the mHz/Hz confusion
#: that ``f_lo_mHz`` vs ``f_hi`` already causes in :attr:`EISResult.eis_params`.
FREQ_DIM = "frequency_hz"


@dataclass
class EISResult:
    """Structured container for a single EIS measurement.

    All impedance arrays share the same length (one entry per frequency
    point).
    """

    channel: int
    frequency: np.ndarray          # Hz
    z_magnitude: np.ndarray        # |Z| (Ω)
    phase: np.ndarray              # degrees
    z_real: np.ndarray             # Z' (Ω)
    z_imag_neg: np.ndarray         # -Z'' (Ω, positive by convention)
    residual_real_pct: np.ndarray | None = None
    residual_imag_pct: np.ndarray | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    measurement_time_s: float = 0.0
    eis_params: dict[str, Any] = field(default_factory=dict)
    raw_file_path: str | None = None
    T_sp: float = float("nan")
    T_pv: float = float("nan")
    rh_sp: float = float("nan")
    rh_pv: float = float("nan")
    #: Survivor mask from the admission gates (True = kept).  ``None`` for ungated
    #: spectra, which is every file written before the gated engine existed.
    mask: np.ndarray | None = None
    #: Name of the gate that dropped each point, empty string where the point was
    #: kept.  Lives beside the frequencies rather than in the database because it is
    #: per-frequency data — and because R17 forbids removing a point without a named
    #: gate and a reason, which has to survive with the spectrum itself.
    drop_gate: np.ndarray | None = None

    # ── Convenience properties ───────────────────────────────────────

    @property
    def z_complex(self) -> np.ndarray:
        """Complex impedance Z' + jZ'' (note: Z'' is stored as -Z'')."""
        return self.z_real - 1j * self.z_imag_neg

    @property
    def npts(self) -> int:
        """Number of frequency points."""
        return len(self.frequency)

    # ── Factory methods ──────────────────────────────────────────────

    @staticmethod
    def from_raw(rawdata, channel: int, *,
                 timestamp: datetime | None = None,
                 measurement_time_s: float = 0.0,
                 eis_params: dict[str, Any] | None = None,
                 raw_file_path: str | None = None) -> "EISResult":
        """Create an :class:`EISResult` from a PalmSens parsed raw-data object.

        This is the equivalent of the legacy ``eis_extractdata()`` method
        but returns a structured object instead of a bare list.

        Parameters
        ----------
        rawdata
            Parsed output from ``palmsens.mscript.parse_result_lines``.
        channel : int
            1-based electrode channel number.
        timestamp : datetime, optional
            Measurement timestamp (defaults to now).
        measurement_time_s : float
            Wall-clock duration of the measurement.
        eis_params : dict, optional
            EIS configuration (npts, f_hi, f_lo_mHz, mv_ac, …).
        raw_file_path : str, optional
            Path to the saved raw hex file.
        """
        # If rawdata is already a numpy array (or list-of-array from mock/test),
        # read directly using the standard 5-column layout produced by the mock
        # and by the real driver's post-processing:
        #   col 0 = f, col 1 = |Z|, col 2 = phase, col 3 = Z', col 4 = -Z''
        # A 3-column array is treated as [f, Z', -Z''] for backward compat.
        _data = rawdata[0] if isinstance(rawdata, list) and len(rawdata) == 1 else rawdata
        if (
            isinstance(_data, (list, tuple))
            and len(_data) == 5
            and all(isinstance(c, np.ndarray) for c in _data)
        ):
            # [f, |Z|, phase, Z', -Z''] — the eis_extractdata contract
            # shared by AsyncESPico and MockESPico.
            f, zreal, zimg = np.asarray(_data[0]), np.asarray(_data[3]), -np.asarray(_data[4])
        elif isinstance(_data, np.ndarray):
            arr = _data
            if arr.ndim == 2 and arr.shape[1] >= 5:
                f, zreal, zimg = arr[:, 0], arr[:, 3], -arr[:, 4]
            elif arr.ndim == 2 and arr.shape[1] == 3:
                # [f, Z', -Z''] — note the negation.  This branch previously took
                # column 2 as-is while documenting it as -Z'', so a 3-column array
                # produced an EISResult with an inverted imaginary part.  The sign
                # has to be right *here*: every admission gate assumes the physics
                # convention (Im Z < 0 for a capacitive response), so an inverted
                # spectrum reaching them flips the quadrant and HF-inductive tests.
                f, zreal, zimg = arr[:, 0], arr[:, 1], -arr[:, 2]
            else:
                raise ValueError(f"Unexpected EIS array shape: {arr.shape}")
        else:
            try:
                import palmsens.mscript  # type: ignore
                f = np.array(palmsens.mscript.get_values_by_column(rawdata, 0))
                zreal = np.array(palmsens.mscript.get_values_by_column(rawdata, 1))
                zimg = np.array(palmsens.mscript.get_values_by_column(rawdata, 2))
            except ImportError:
                # Last-resort fallback: treat as [f, zreal, zimg]
                arr = np.asarray(rawdata)
                f, zreal, zimg = arr[:, 0], arr[:, 1], arr[:, 2]

        z_complex = zreal + 1j * zimg
        z_magnitude = np.abs(z_complex)
        phase = np.angle(z_complex, deg=True)
        z_imag_neg = -zimg

        return EISResult(
            channel=channel,
            frequency=f,
            z_magnitude=z_magnitude,
            phase=phase,
            z_real=zreal,
            z_imag_neg=z_imag_neg,
            timestamp=timestamp or datetime.now(),
            measurement_time_s=measurement_time_s,
            eis_params=eis_params or {},
            raw_file_path=raw_file_path,
        )

    @staticmethod
    def from_arrays(channel: int, f: np.ndarray, z_real: np.ndarray,
                    z_imag_neg: np.ndarray, **kwargs) -> "EISResult":
        """Create from pre-extracted numpy arrays (Z' and -Z'')."""
        z_complex = z_real - 1j * z_imag_neg
        return EISResult(
            channel=channel,
            frequency=f,
            z_magnitude=np.abs(z_complex),
            phase=np.angle(z_complex, deg=True),
            z_real=z_real,
            z_imag_neg=z_imag_neg,
            **kwargs,
        )

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str | Path, *, study_name: str = "") -> Path:
        """Save EIS data to a standardised text file with metadata header.

        The format matches (and improves upon) the legacy convention used
        in ``eis_multichannel_measure``.

        Parameters
        ----------
        path : str or Path
            Output file path. Parent directories are created automatically.
        study_name : str, optional
            Study identifier included in the header.

        Returns
        -------
        Path
            The path to the written file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        params = self.eis_params
        npts = params.get("npts", self.npts)
        f_lo = params.get("f_lo_mHz", "?")
        f_hi = params.get("f_hi", "?")
        mv_dc = params.get("mv_dc", 0)
        mv_ac = params.get("mv_ac", "?")

        # Format measurement time
        t = self.measurement_time_s
        time_str = (
            f"{int(t // 3600)}h{int((t % 3600) // 60)}m{int(t % 60)}s"
        )

        has_residuals = (
            self.residual_real_pct is not None
            and self.residual_imag_pct is not None
            and len(self.residual_real_pct) == self.npts
            and len(self.residual_imag_pct) == self.npts
        )
        # Gate provenance: a numeric id per point plus a legend, so a reader can say
        # *which named gate* dropped each point (R17) without breaking the numeric
        # table that np.loadtxt expects.
        gate_ids, gate_legend = self._encode_drop_gates()
        has_gates = gate_ids is not None

        column_names = list(COLUMN_NAMES)
        if has_residuals:
            column_names.extend(RESIDUAL_COLUMN_NAMES)
        if has_gates:
            column_names.append(GATE_COLUMN_NAME)

        import math as _math
        def _fmt(v: float, decimals: int = 2) -> str:
            return "nan" if _math.isnan(v) else f"{v:.{decimals}f}"

        header = (
            f"Study: {study_name}\n"
            f" Conducted: {self.timestamp}\n"
            f" Channel: {self.channel}\n"
            f" Measurement time: {time_str}\n"
            f" T_SP: {_fmt(self.T_sp)} degC   T_PV: {_fmt(self.T_pv)} degC\n"
            f" RH_SP: {_fmt(self.rh_sp)} %RH   RH_PV: {_fmt(self.rh_pv)} %RH\n"
            f" Residual columns: {'present (resid_Zreal(%), resid_-Z''(%))' if has_residuals else 'not present'}\n"
            f" Gate legend: {gate_legend if has_gates else 'not present'}\n"
            f" {npts} log points between {f_lo} mHz and {f_hi} Hz, "
            f"DC voltage (mV) = {mv_dc}, AC voltage (mV) = {mv_ac}\n"
            f" {'  '.join(column_names)}"
        )

        data_cols = [
            self.frequency,
            self.z_magnitude,
            self.phase,
            self.z_real,
            self.z_imag_neg,
        ]
        if has_residuals:
            data_cols.extend([self.residual_real_pct, self.residual_imag_pct])
        if has_gates:
            data_cols.append(gate_ids)
        data = np.column_stack(data_cols)
        np.savetxt(str(p), data, header=header)
        return p

    # ── Gate provenance encoding ─────────────────────────────────────

    def _encode_drop_gates(self) -> tuple[np.ndarray | None, str]:
        """``(ids, legend)`` for the trailing gate column, or ``(None, "")``.

        Ids are 1-based indices into the legend; 0 means the point survived.
        """
        names = self.drop_gate
        if names is None:
            if self.mask is None:
                return None, ""
            # A mask with no per-point reasons still records *that* points were
            # dropped; name them generically rather than losing the fact.
            kept = np.asarray(self.mask, dtype=bool)
            names = np.where(kept, "", "gated")

        names = np.asarray(names, dtype=object)
        if names.size != self.npts:
            return None, ""

        legend: list[str] = []
        ids = np.zeros(self.npts, dtype=float)
        for i, raw in enumerate(names):
            name = str(raw or "").strip()
            if not name:
                continue
            if name not in legend:
                legend.append(name)
            ids[i] = float(legend.index(name) + 1)

        if not legend:
            return None, ""
        return ids, ", ".join(f"{i + 1}={n}" for i, n in enumerate(legend))

    @staticmethod
    def _decode_drop_gates(
        ids: np.ndarray, legend: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert :meth:`_encode_drop_gates` into ``(mask, drop_gate)``."""
        names: dict[int, str] = {}
        for part in (legend or "").split(","):
            if "=" not in part:
                continue
            key, _, value = part.partition("=")
            try:
                names[int(key.strip())] = value.strip()
            except ValueError:
                continue

        ids = np.asarray(ids, dtype=float)
        mask = ids <= 0
        drop = np.array(
            [names.get(int(v), "gated") if v > 0 else "" for v in ids], dtype=object
        )
        return mask, drop

    @staticmethod
    def load(path: str | Path) -> "EISResult":
        """Load an EIS data file saved by :meth:`save` or legacy code.

        Handles the standard 5-column format with a multi-line ``#``
        header.

        Parameters
        ----------
        path : str or Path
            Path to the data file.

        Returns
        -------
        EISResult
        """
        p = Path(path)
        lines = p.read_text(encoding="utf-8").splitlines()

        # Parse metadata from header lines (lines starting with #)
        channel = 0
        timestamp = datetime.now()
        eis_params: dict[str, Any] = {}
        gate_legend = ""

        for line in lines:
            if not line.startswith("#"):
                break
            text = line.lstrip("# ")
            if text.startswith("Channel:"):
                try:
                    channel = int(text.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif text.startswith("Conducted:"):
                try:
                    ts_str = text.split(":", 1)[1].strip()
                    timestamp = datetime.fromisoformat(ts_str)
                except (ValueError, IndexError):
                    pass
            elif text.startswith("T_SP:"):
                m_tsp = re.search(r"T_SP:\s*([\d.nan-]+)", text)
                m_tpv = re.search(r"T_PV:\s*([\d.nan-]+)", text)
                if m_tsp:
                    eis_params["T_sp"] = float(m_tsp.group(1))
                if m_tpv:
                    eis_params["T_pv"] = float(m_tpv.group(1))
            elif text.startswith("RH_SP:"):
                m_rhsp = re.search(r"RH_SP:\s*([\d.nan-]+)", text)
                m_rhpv = re.search(r"RH_PV:\s*([\d.nan-]+)", text)
                if m_rhsp:
                    eis_params["rh_sp"] = float(m_rhsp.group(1))
                if m_rhpv:
                    eis_params["rh_pv"] = float(m_rhpv.group(1))
            elif text.startswith("Residual columns:"):
                has_residual_cols = "not present" not in text
            elif text.startswith("Gate legend:"):
                value = text.split(":", 1)[1].strip()
                gate_legend = "" if value == "not present" else value
            # Extract EIS params from the format line
            m = re.search(r"(\d+)\s+log\s+points\s+between\s+([\d.]+)\s+mHz\s+and\s+([\d.]+)\s+Hz", text)
            if m:
                eis_params["npts"] = int(m.group(1))
                eis_params["f_lo_mHz"] = float(m.group(2))
                eis_params["f_hi"] = float(m.group(3))

        # Load numeric data (skip header lines)
        data = np.loadtxt(str(p))
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # The gate column, when present, is always last — so the residual columns
        # keep the indices every older file already uses.
        ncols = data.shape[1]
        gate_col = ncols - 1 if gate_legend else None
        n_value_cols = gate_col if gate_col is not None else ncols

        # Column count alone decides the residuals, exactly as before — an older file
        # whose header predates the "Residual columns:" line must still load them.
        residual_real_pct = data[:, 5] if n_value_cols >= 6 else None
        residual_imag_pct = data[:, 6] if n_value_cols >= 7 else None

        mask = drop_gate = None
        if gate_col is not None:
            mask, drop_gate = EISResult._decode_drop_gates(data[:, gate_col],
                                                           gate_legend)

        return EISResult(
            channel=channel,
            frequency=data[:, 0],
            z_magnitude=data[:, 1],
            phase=data[:, 2],
            z_real=data[:, 3],
            z_imag_neg=data[:, 4],
            residual_real_pct=residual_real_pct,
            residual_imag_pct=residual_imag_pct,
            timestamp=timestamp,
            eis_params=eis_params,
            raw_file_path=str(p),
            T_sp=eis_params.pop("T_sp", float("nan")),
            T_pv=eis_params.pop("T_pv", float("nan")),
            rh_sp=eis_params.pop("rh_sp", float("nan")),
            rh_pv=eis_params.pop("rh_pv", float("nan")),
            mask=mask,
            drop_gate=drop_gate,
        )

    # ── MeasurementResult bridge (Tier 2 component 1) ────────────────

    def to_measurement(self) -> "MeasurementResult":
        """Render this spectrum as a modality-agnostic :class:`MeasurementResult`.

        The inverse is :meth:`from_measurement`, and the pair is **lossless**:
        every field of this dataclass survives the round trip (see the fidelity
        note below for the one boundary).

        Layout — frequency-indexed, per the spec's Tier 2 component 1::

            coord     frequency_hz
            data_vars z_real, z_imag_neg, z_mag, phase
                      [residual_real_pct, residual_imag_pct]  (if present)
                      [mask, drop_gate]                       (independently,
                                                               if present)
            attrs     channel, timestamp, measurement_time_s, T_sp/T_pv,
                      rh_sp/rh_pv, eis_params_json, [raw_file_path]

        ``mask`` and ``drop_gate`` are attached **independently** rather than as
        a pair: a gated spectrum may carry a survivor mask with no per-point
        reasons (:meth:`_encode_drop_gates` synthesises names for exactly that
        case), so requiring both would silently drop the mask on those results.

        ``attrs`` is kept netCDF-encodable so the Dataset alone reconstructs the
        result — no companion object needed once Tier 2 component 3 writes these
        to disk. That is why ``eis_params`` becomes a JSON string rather than a
        nested dict (netCDF ``attrs`` cannot hold one) and why ``raw_file_path``
        is *omitted* when ``None`` rather than stored as null.
        :attr:`MeasurementResult.meta` mirrors the same values unencoded — real
        ``dict`` and ``datetime`` — for in-memory consumers.

        Fidelity boundary: ``eis_params`` round-trips exactly for
        JSON-representable values (every value the EIS path actually
        produces — npts, f_hi, f_lo_mHz, mv_ac, mv_dc — is a number or string).
        A non-JSON-representable value is coerced to its ``str`` form rather
        than raising, so the bridge degrades to a recorded approximation
        instead of failing a measurement that physically succeeded.
        """
        # Imported here, not at module scope: `eis_data` is pulled in by the
        # DataStore, the router and several GUI tabs, and xarray is an
        # expensive import. Only callers that actually cross the contract seam
        # should pay for it.
        import json

        import xarray as xr

        from softae.analysis.measurement_result import MeasurementResult

        data_vars: dict[str, Any] = {
            "z_real": (FREQ_DIM, np.asarray(self.z_real)),
            "z_imag_neg": (FREQ_DIM, np.asarray(self.z_imag_neg)),
            "z_mag": (FREQ_DIM, np.asarray(self.z_magnitude)),
            "phase": (FREQ_DIM, np.asarray(self.phase)),
        }
        if self.residual_real_pct is not None:
            data_vars["residual_real_pct"] = (FREQ_DIM,
                                              np.asarray(self.residual_real_pct))
        if self.residual_imag_pct is not None:
            data_vars["residual_imag_pct"] = (FREQ_DIM,
                                              np.asarray(self.residual_imag_pct))
        if self.mask is not None:
            data_vars["mask"] = (FREQ_DIM, np.asarray(self.mask, dtype=bool))
        if self.drop_gate is not None:
            # Stored as fixed-width unicode ('<U…'), not object dtype: netCDF
            # has no object type, and h5netcdf would have to guess an encoding.
            # `from_measurement` restores the object dtype that
            # `_decode_drop_gates` produces, so the EISResult convention is
            # preserved on the way back. None and "" both mean "point kept" in
            # this schema (`_encode_drop_gates` already conflates them), so
            # normalising None to "" here changes no meaning.
            names = ["" if v is None else str(v) for v in self.drop_gate]
            data_vars["drop_gate"] = (FREQ_DIM, np.asarray(names, dtype=str))

        attrs: dict[str, Any] = {
            "modality": EIS_MODALITY,
            "channel": int(self.channel),
            "timestamp": self.timestamp.isoformat(),
            "measurement_time_s": float(self.measurement_time_s),
            "T_sp": float(self.T_sp),
            "T_pv": float(self.T_pv),
            "rh_sp": float(self.rh_sp),
            "rh_pv": float(self.rh_pv),
            "eis_params_json": json.dumps(self.eis_params, default=str),
        }
        if self.raw_file_path is not None:
            attrs["raw_file_path"] = str(self.raw_file_path)

        dataset = xr.Dataset(
            data_vars,
            coords={FREQ_DIM: np.asarray(self.frequency)},
            attrs=attrs,
        )

        meta: dict[str, Any] = {
            "channel": int(self.channel),
            "timestamp": self.timestamp,
            "measurement_time_s": float(self.measurement_time_s),
            "eis_params": dict(self.eis_params),
            "raw_file_path": self.raw_file_path,
            "T_sp": self.T_sp,
            "T_pv": self.T_pv,
            "rh_sp": self.rh_sp,
            "rh_pv": self.rh_pv,
            "npts": self.npts,
        }

        # Derived digest only — every value here also lives in `data`, so a
        # consumer that ignores `summary` loses nothing.
        summary: dict[str, float] | None = None
        if self.npts:
            summary = {
                "npts": float(self.npts),
                "f_min_hz": float(np.min(self.frequency)),
                "f_max_hz": float(np.max(self.frequency)),
            }

        return MeasurementResult(
            modality=EIS_MODALITY, data=dataset, meta=meta, summary=summary,
        )

    @staticmethod
    def from_measurement(measurement: "MeasurementResult") -> "EISResult":
        """Reconstruct an :class:`EISResult` from :meth:`to_measurement`'s output.

        Reads the **Dataset alone** — ``meta`` is a mirror, never the only home
        for a value — so this works identically on a payload just built in
        memory and on one read back from a netCDF file.

        Raises
        ------
        ValueError
            If *measurement* is not an EIS payload, or lacks a required
            variable. Refusing loudly is deliberate: a silent partial
            reconstruction would hand analysis a spectrum with fabricated
            columns.
        """
        import json

        if measurement.modality != EIS_MODALITY:
            raise ValueError(
                f"cannot rebuild an EISResult from modality "
                f"{measurement.modality!r} (expected {EIS_MODALITY!r})"
            )

        ds = measurement.data
        missing = [n for n in ("z_real", "z_imag_neg", "z_mag", "phase")
                   if n not in ds.data_vars]
        if missing:
            raise ValueError(f"EIS payload missing variable(s): {missing}")
        if FREQ_DIM not in ds.coords:
            raise ValueError(f"EIS payload missing coordinate {FREQ_DIM!r}")

        attrs = ds.attrs

        def _array(name: str) -> np.ndarray | None:
            return np.asarray(ds[name].values) if name in ds.data_vars else None

        timestamp = attrs.get("timestamp")
        try:
            ts = (datetime.fromisoformat(timestamp) if isinstance(timestamp, str)
                  else datetime.now())
        except ValueError:
            ts = datetime.now()

        try:
            eis_params = json.loads(attrs.get("eis_params_json", "{}"))
        except (TypeError, ValueError):
            eis_params = {}

        drop_gate = None
        if "drop_gate" in ds.data_vars:
            # Back to object dtype holding plain `str`, matching
            # `_decode_drop_gates`' output exactly — numpy hands back `np.str_`
            # here, and while that is a `str` subclass, the explicit `str()`
            # keeps a round-tripped result indistinguishable from a loaded one
            # rather than merely equal to it.
            drop_gate = np.asarray([str(v) for v in ds["drop_gate"].values],
                                   dtype=object)

        mask = _array("mask")
        raw_path = attrs.get("raw_file_path")

        return EISResult(
            channel=int(attrs.get("channel", 0)),
            frequency=np.asarray(ds.coords[FREQ_DIM].values),
            z_magnitude=np.asarray(ds["z_mag"].values),
            phase=np.asarray(ds["phase"].values),
            z_real=np.asarray(ds["z_real"].values),
            z_imag_neg=np.asarray(ds["z_imag_neg"].values),
            residual_real_pct=_array("residual_real_pct"),
            residual_imag_pct=_array("residual_imag_pct"),
            timestamp=ts,
            measurement_time_s=float(attrs.get("measurement_time_s", 0.0)),
            eis_params=eis_params,
            raw_file_path=str(raw_path) if raw_path is not None else None,
            T_sp=float(attrs.get("T_sp", float("nan"))),
            T_pv=float(attrs.get("T_pv", float("nan"))),
            rh_sp=float(attrs.get("rh_sp", float("nan"))),
            rh_pv=float(attrs.get("rh_pv", float("nan"))),
            mask=None if mask is None else mask.astype(bool),
            drop_gate=drop_gate,
        )

    # ── Conversion helpers ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (arrays as lists)."""
        return {
            "channel": self.channel,
            "npts": self.npts,
            "f_min_Hz": float(self.frequency.min()) if self.npts else None,
            "f_max_Hz": float(self.frequency.max()) if self.npts else None,
            "timestamp": self.timestamp.isoformat(),
            "measurement_time_s": self.measurement_time_s,
            "eis_params": self.eis_params,
            "raw_file_path": self.raw_file_path,
            "frequency": self.frequency.tolist(),
            "z_real": self.z_real.tolist(),
            "z_imag_neg": self.z_imag_neg.tolist(),
            "z_magnitude": self.z_magnitude.tolist(),
            "phase": self.phase.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"EISResult(ch={self.channel}, npts={self.npts}, "
            f"f=[{self.frequency.min():.1f}–{self.frequency.max():.1f}] Hz)"
            if self.npts > 0
            else f"EISResult(ch={self.channel}, empty)"
        )
