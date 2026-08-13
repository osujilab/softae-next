"""Dataset adapters: turn a raw data source into the tidy campaign schema.

Each adapter implements :meth:`DatasetAdapter.to_tidy` and returns a
:class:`pandas.DataFrame` conforming to :mod:`softae.campaigns.schema`.  The
campaign engine consumes only the tidy frame, so adding support for a new file
format or data source means writing one adapter — nothing downstream changes.

Adapters provided here:

* :class:`AggregatedTxtAdapter` — parses the hand-aggregated conductivity text
  file (PEO/LiCl/silica seed dataset).
* :class:`DataStoreAdapter` — *stub* for reading a live ``DataStore`` run
  (built out in phase P1; the signature and tidy contract are fixed now).
"""

from __future__ import annotations

import abc
import re
from pathlib import Path

import pandas as pd

from softae.analysis.conditions import resolve_temperature_C
from softae.campaigns.schema import validate_tidy
from softae.errors import CampaignError

#: Columns that are *about* a measurement rather than *where* it was taken, and
#: so must not enter a candidate's identity.  ``temp_source`` is here because a
#: provenance label is not a coordinate: two rows at the same stage temperature
#: are the same candidate whether or not one of them lost its stage read.
_NON_COORDINATE_COLUMNS: frozenset[str] = frozenset(
    {"conductivity", "fitted_Z", "adjusted_Z", "temp_source"}
)


class DatasetAdapter(abc.ABC):
    """Abstract source → tidy-frame converter."""

    #: Short tag written into the tidy ``source`` column for provenance.
    source_tag: str = "adapter"

    @abc.abstractmethod
    def to_tidy(self) -> pd.DataFrame:
        """Return a tidy DataFrame (see :mod:`softae.campaigns.schema`)."""

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag with ``source`` and validate before returning."""
        if "source" not in df.columns:
            df = df.assign(source=self.source_tag)
        validate_tidy(df)
        return df


# ---------------------------------------------------------------------------
# Aggregated text-file adapter (seed dataset)
# ---------------------------------------------------------------------------

# Block headers in the seed file.  We match on a lowercase substring so minor
# wording/spacing changes do not break parsing.
_BLOCK_KEYS: dict[str, str] = {
    "conductivity": "conductivities",
    "fitted_Z": "fitted impedances",
    "adjusted_Z": "manually adjusted impedances",
}

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class AggregatedTxtAdapter(DatasetAdapter):
    """Parse the hand-aggregated conductivity ``.txt`` file into tidy rows.

    The seed file lays out three labelled blocks (conductivity, fitted
    impedance, manually-adjusted impedance), each a flat list of
    ``n_rh * n_eo * n_silica * n_replicates`` comma-separated numbers.  The flat
    index decomposes as (slowest- to fastest-varying)::

        RH  →  EO:Li  →  silica  →  replicate

    All layout parameters are constructor arguments with seed defaults, so a
    sibling file with a different RH list or composition grid parses without
    code changes.

    Parameters
    ----------
    path
        Path to the aggregated text file.
    rh_levels
        RH setpoints (%), outermost (slowest-varying) loop.
    eo_li_levels
        EO:Li ratio levels, second loop.
    silica_levels
        Silica volume-fraction levels, third loop.
    n_replicates
        Replicates per condition, innermost (fastest-varying) loop.
    temp_C
        Temperature (°C) recorded for every row (seed file is isothermal).
    source_tag
        Provenance tag written to the ``source`` column.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        rh_levels: list[float] | None = None,
        eo_li_levels: list[float] | None = None,
        silica_levels: list[float] | None = None,
        n_replicates: int = 2,
        temp_C: float = 35.0,
        source_tag: str = "aggregated_txt",
    ) -> None:
        self.path = Path(path)
        self.rh_levels = list(rh_levels) if rh_levels is not None else [10.0, 30.0]
        self.eo_li_levels = (
            list(eo_li_levels) if eo_li_levels is not None else [40.0, 20.0, 10.0, 5.0]
        )
        self.silica_levels = (
            list(silica_levels) if silica_levels is not None else [0.0, 0.05, 0.10, 0.20]
        )
        self.n_replicates = int(n_replicates)
        self.temp_C = float(temp_C)
        self.source_tag = source_tag

    @property
    def expected_count(self) -> int:
        """Number of values each block must contain."""
        return (
            len(self.rh_levels)
            * len(self.eo_li_levels)
            * len(self.silica_levels)
            * self.n_replicates
        )

    # ── Parsing ──────────────────────────────────────────────────────────

    def _read_blocks(self) -> dict[str, list[float]]:
        """Extract the three numeric blocks keyed by tidy column name."""
        if not self.path.exists():
            raise CampaignError(f"dataset file not found: {self.path}")
        text = self.path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        # Locate each block header line by its substring.
        header_idx: dict[str, int] = {}
        for i, line in enumerate(lines):
            low = line.lower()
            for col, key in _BLOCK_KEYS.items():
                if key in low and col not in header_idx:
                    header_idx[col] = i

        missing = [c for c in _BLOCK_KEYS if c not in header_idx]
        if missing:
            raise CampaignError(
                f"could not find data block(s) {missing} in {self.path.name}; "
                f"expected headers containing {[_BLOCK_KEYS[c] for c in missing]}"
            )

        blocks: dict[str, list[float]] = {}
        for col, hidx in header_idx.items():
            values = self._scan_numbers(lines, hidx)
            if len(values) != self.expected_count:
                raise CampaignError(
                    f"block '{col}' in {self.path.name} has {len(values)} values, "
                    f"expected {self.expected_count} "
                    f"({len(self.rh_levels)} RH x {len(self.eo_li_levels)} EO:Li x "
                    f"{len(self.silica_levels)} silica x {self.n_replicates} replicates)"
                )
            blocks[col] = values
        return blocks

    def _scan_numbers(self, lines: list[str], header_idx: int) -> list[float]:
        """Collect numbers starting on the header line, across continuations.

        Handles both ``"label: 1, 2, 3"`` (inline) and ``"label:"`` followed by
        the numbers on the next non-empty line(s).  Scanning stops at the first
        blank line *after* numbers have started, or at the next block header.
        """
        other_headers = {
            key for c, key in _BLOCK_KEYS.items()
        }
        collected: list[float] = []
        started = False
        for j in range(header_idx, len(lines)):
            raw = lines[j]
            low = raw.lower()
            # Stop if we reach a *different* block header after starting.
            if started and any(k in low for k in other_headers):
                break
            # On the header line itself, strip the label before the colon.
            scan_text = raw
            if j == header_idx and ":" in raw:
                scan_text = raw.split(":", 1)[1]
            found = _NUM_RE.findall(scan_text)
            if found:
                collected.extend(float(tok) for tok in found)
                started = True
            elif started and raw.strip() == "":
                break
        return collected

    # ── Index decoding ───────────────────────────────────────────────────

    def _decode_index(self, k: int) -> tuple[float, float, float, int]:
        """Map flat index *k* → (rh, eo_li, silica, replicate)."""
        n_rep = self.n_replicates
        n_sil = len(self.silica_levels)
        n_eo = len(self.eo_li_levels)
        per_rh = n_eo * n_sil * n_rep

        rh_idx = k // per_rh
        within = k % per_rh
        eo_idx = within // (n_sil * n_rep)
        silica_idx = (within % (n_sil * n_rep)) // n_rep
        replicate = within % n_rep
        return (
            self.rh_levels[rh_idx],
            self.eo_li_levels[eo_idx],
            self.silica_levels[silica_idx],
            replicate,
        )

    def to_tidy(self) -> pd.DataFrame:
        blocks = self._read_blocks()
        rows = []
        for k in range(self.expected_count):
            rh, eo_li, silica, replicate = self._decode_index(k)
            rows.append(
                {
                    "eo_li_ratio": eo_li,
                    "silica_vol_frac": silica,
                    "rh_pct": rh,
                    "temp_C": self.temp_C,
                    "replicate": replicate,
                    "conductivity": blocks["conductivity"][k],
                    "fitted_Z": blocks["fitted_Z"][k],
                    "adjusted_Z": blocks["adjusted_Z"][k],
                    # point_id identifies a candidate = a distinct condition
                    # (composition + environment), shared across replicates.
                    "point_id": _point_id(eo_li, silica, rh, self.temp_C),
                }
            )
        df = pd.DataFrame(rows)
        return self._finalize(df)


def _point_id(eo_li: float, silica: float, rh: float, temp_C: float) -> str:
    """Stable candidate key from composition + environment."""
    return f"eo{eo_li:g}_sil{silica:g}_rh{rh:g}_T{temp_C:g}"


# ---------------------------------------------------------------------------
# DataStore adapter (stub — built out in P1)
# ---------------------------------------------------------------------------

class DataStoreAdapter(DatasetAdapter):
    """Read a softae-next ``DataStore`` run into the tidy schema.

    Joins ``measurements`` ⋈ ``fit_results`` (for σ and R₁) ⋈ ``conditions``
    (for T/RH), and merges any per-channel composition recorded in
    ``doe_parameters`` so live platform runs become a candidate pool with no
    engine change.  ``fitted_Z``/``adjusted_Z`` are both set to the fitted R₁
    (no manually-adjusted impedance exists in the DB → zero fit discrepancy),
    so the fit-quality noise source contributes nothing for DataStore data.

    A candidate (``point_id``) is the recorded composition + environment when a
    composition is present, otherwise the electrode ``channel`` + environment.
    Repeated measurements of the same candidate become replicates.

    ``temp_C`` comes from :func:`~softae.analysis.conditions.resolve_temperature_C`
    and is emitted with its ``temp_source`` label beside it — the rig has two
    thermometers and a tidy frame is exactly the artifact that outlives the
    knowledge of which one was read.
    """

    source_tag = "datastore"

    def __init__(self, data_store, run_id: str) -> None:
        self.data_store = data_store
        self.run_id = run_id

    def _composition_by_channel(self) -> dict[int, dict]:
        """First numeric/string composition dict per channel from doe_parameters."""
        import json

        out: dict[int, dict] = {}
        for row in self.data_store.query_doe_parameters(run_id=self.run_id):
            ch = row.get("channel")
            if ch in out:
                continue
            try:
                params = json.loads(row.get("parameters_json") or "{}")
            except (ValueError, TypeError):
                params = {}
            clean = {
                k: v for k, v in params.items()
                if isinstance(v, (int, float, str)) and not isinstance(v, bool)
            }
            if clean:
                out[ch] = clean
        return out

    def _conditions_by_measurement(self) -> dict[int, dict]:
        """One condition snapshot per measurement (prefer the 'measurement' stage)."""
        out: dict[int, dict] = {}
        for cond in self.data_store.query_conditions(run_id=self.run_id):
            mid = cond.get("measurement_id")
            if mid not in out or cond.get("stage") == "measurement":
                out[mid] = cond
        return out

    def to_tidy(self) -> pd.DataFrame:
        measurements = self.data_store.query_measurements(run_id=self.run_id)
        fits = {
            f["measurement_id"]: f
            for f in self.data_store.query_fits(run_id=self.run_id)
        }
        conds = self._conditions_by_measurement()
        comp_by_ch = self._composition_by_channel()
        has_comp = bool(comp_by_ch)

        rows: list[dict] = []
        for m in measurements:
            mid = m["measurement_id"]
            fit = fits.get(mid)
            if fit is None:
                continue
            sigma = fit.get("sigma_S_per_cm")
            if sigma is None:
                continue
            cond = conds.get(mid, {})
            # Which thermometer this row's temperature came from is decided in
            # one place for the whole system. This adapter used to take the
            # chamber-air probe first and never read the stage PV at all — the
            # resolver's precedence exactly inverted, worth up to 42 °C on every
            # tidy row it emitted.
            #
            # Since schema epoch 4 the answer is already on the row, resolved at
            # record time, so read it rather than re-deriving it. The fallback
            # below is the SAME authority, not a second opinion: it calls the
            # resolver the writer would have called, for rows no epoch-4 writer
            # wrote — raw-INSERT test fixtures, or a row from a stale binary
            # before the next open re-resolves it.
            temp_source = cond.get("temperature_source")
            if temp_source:
                temp = cond.get("temperature_C")
                temp = float("nan") if temp is None else float(temp)
            else:
                temp, temp_source = resolve_temperature_C(
                    stage_pv_C=cond.get("stage_temp_pv_C"),
                    stage_sp_C=cond.get("stage_temp_sp_C"),
                    chamber_air_C=cond.get("chamber_air_C"),
                )
            rh = cond.get("rh_pv_pct")
            if rh is None:
                rh = cond.get("rh_sp_pct", float("nan"))
            r1 = fit.get("R1", float("nan"))

            row = {
                "conductivity": float(sigma),
                "fitted_Z": float(r1) if r1 is not None else float("nan"),
                "adjusted_Z": float(r1) if r1 is not None else float("nan"),
                "temp_C": float(temp),
                # The number and its provenance travel together or the ambiguity
                # is simply re-created one layer out — a tidy frame is exactly the
                # artifact that gets exported, joined and re-read by someone who
                # never saw this module.
                "temp_source": temp_source,
                "rh_pct": float(rh) if rh is not None else float("nan"),
            }
            comp = comp_by_ch.get(m["channel"], {})
            row.update(comp)
            if not has_comp:
                row["channel"] = int(m["channel"])
            rows.append(row)

        if not rows:
            raise CampaignError(
                f"run '{self.run_id}' has no fitted conductivity measurements to load"
            )

        df = pd.DataFrame(rows)
        df = self._assign_point_ids_and_replicates(df, has_comp)
        return self._finalize(df)

    @staticmethod
    def _assign_point_ids_and_replicates(df: pd.DataFrame, has_comp: bool) -> pd.DataFrame:
        """Derive point_id (candidate identity) and per-candidate replicate index."""
        coord_cols = [
            c for c in df.columns
            if c not in _NON_COORDINATE_COLUMNS
        ]

        def _pid(row) -> str:
            return "|".join(
                f"{c}={row[c]:g}" if isinstance(row[c], float) else f"{c}={row[c]}"
                for c in coord_cols
            )

        df = df.copy()
        df["point_id"] = df.apply(_pid, axis=1)
        df["replicate"] = df.groupby("point_id").cumcount()
        return df
