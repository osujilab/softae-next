r"""Self-contained EIS Visualizer demo.

Launches a standalone EISVisualizerWindow populated with synthetic data
so you can evaluate layout and UX without any hardware or DataStore.

The synthetic impedance is computed analytically from the 'simpleSalt'
circuit topology (R0 - CPE0 - p(R1, C0)), using physically realistic
starting parameters and per-channel log-uniform random variation in R0
and R1.

Run from the softae-next root:
    .\.venv\Scripts\python.exe scripts\eis_visualizer_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Make the package importable without an installed editable build ──────────
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis_data import EISResult
from softae.gui.widgets.eis_visualizer_widget import (
    EISEntry,
    EISVisualizerWindow,
    ListEISSource,
)

# ── Physical EIS parameters (simpleSalt: R0 - CPE0 - p(R1, C0)) ─────────────
#
#   R0      = 4.81e+04 Ω        (bulk ionic resistance, varied per channel)
#   CPE0_Q  = 1.00e-07 Ω⁻¹s^α  (CPE pre-factor, fixed)
#   CPE0_α  = 0.70              (CPE exponent, fixed)
#   R1      = 1.84e+06 Ω        (interfacial resistance, varied per channel)
#   C0      = 1.10e-10 F        (geometric / stray capacitance, fixed)
#
CPE_Q = 1.00e-7
CPE_A = 0.70
C0    = 1.10e-10

GEOMETRY = dict(L=0.2, t=0.175, w=0.2)   # electrode geometry in cm
#: The same cell every real surface builds, so the demo exercises the shipped route.
DEMO_CELL = CellConstant.from_legacy(GEOMETRY["L"], GEOMETRY["t"], GEOMETRY["w"])
RUN_IDS  = ["run_2026-03-11_A", "run_2026-03-11_B", "run_2026-03-11_C"]

# Frequency sweep: 50 kHz → 1 Hz (41 log-spaced points, high-to-low as on instrument)
_FREQ = np.geomspace(5e4, 1.0, 41)


def _make_eis_spectrum(R0: float, R1: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Z for the R0 - CPE0 - p(R1, C0) circuit.

    Returns (f, z_real, z_imag_neg) arrays.  Small multiplicative noise is
    added to each point to simulate measurement scatter.
    """
    rng = np.random.default_rng(seed)
    omega = 2.0 * np.pi * _FREQ

    # Element impedances
    Z_CPE     = 1.0 / (CPE_Q * (1j * omega) ** CPE_A)
    Z_R1_C0   = R1 / (1.0 + 1j * omega * R1 * C0)   # R1 || C0
    Z_total   = R0 + Z_CPE + Z_R1_C0

    # Add ~0.5 % rms multiplicative noise for realism
    noise = rng.normal(1.0, 0.005, len(_FREQ))
    Z_total = Z_total * noise

    return _FREQ.copy(), Z_total.real, -Z_total.imag


# ── Build entries ─────────────────────────────────────────────────────────────
entries: list[EISEntry] = []

for run_idx, run_id in enumerate(RUN_IDS):
    for ch in range(1, 9):
        seed = run_idx * 100 + ch
        rng  = np.random.default_rng(seed)

        # Log-uniform random R0 in [1e3, 1e5] and R1 in [1e6, 1e9]
        R0 = 10.0 ** rng.uniform(3.0, 5.0)
        R1 = 10.0 ** rng.uniform(6.0, 9.0)

        f, z_real, z_imag_neg = _make_eis_spectrum(R0, R1, seed=seed + 1)

        eis = EISResult.from_arrays(
            channel=ch,
            f=f,
            z_real=z_real,
            z_imag_neg=z_imag_neg,
            raw_file_path=f"/data/{run_id}/ch{ch:02d}_eis.txt",
        )

        # Deliberately leave ch 8 unfitted to test the no-fit rendering path
        if ch == 8:
            fit, sigma = None, None
        else:
            try:
                # ``engine`` unset, like every real call site: the demo shows what
                # ``[eis] engine`` currently selects, not a frozen legacy path.
                report = analyze_spectrum(eis, cell=DEMO_CELL,
                                          model_name="simpleSalt")
                fit = report.fit
                sigma = (float(report.sigma.value)
                         if report.sigma.mode == "value" else None)
            except Exception:
                fit, sigma = None, None

        entries.append(
            EISEntry(
                label=f"Ch{ch:02d}  [{run_id[-1]}]",
                eis=eis,
                fit=fit,
                sigma=sigma,
                run_id=run_id,
            )
        )

print(f"Built {len(entries)} synthetic EIS entries across {len(RUN_IDS)} runs.")

# ── Launch ───────────────────────────────────────────────────────────────────
source = ListEISSource(entries)
EISVisualizerWindow.open(source, title="EIS Visualizer — Demo (synthetic data)")
