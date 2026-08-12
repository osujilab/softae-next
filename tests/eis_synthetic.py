"""Synthetic spectra generated from the framework's reference topology.

``docs/EIS_GATE_FRAMEWORK.md`` §8 makes this the acceptance bar: spectra generated
from §1.1 with known parameters must be recovered within stated uncertainty and pass
every gate, and each synthetic *pathology* must be caught by the intended gate **and
no other**. That is only assertable if the pathologies are constructed rather than
hunted for in archived data, so they live here.

Shared helper module rather than per-test fixtures, following ``campaign_helpers.py``.

The reference topology (framework §1.1)::

    Z(ω) = jωL_lead + [ R_leak ∥ ( R_series + (R_bulk ∥ C_par) + Z_CPE ) ]

Defaults are chosen to sit where this rig actually operates: a blocking coplanar cell
with the bulk arc in band, ``C_par`` fixture-dominated at 0.35 nF, and a sweep matching
the ``Quick`` preset's 20 Hz–200 kHz — the only shipped preset entirely above the
fixture's ~9 Hz usable floor.
"""

from __future__ import annotations

import numpy as np

#: Defaults matching a well-formed 10:1 PEO:Li film on this fixture.
DEFAULT_R_SERIES = 50.0
DEFAULT_R_BULK = 5.0e4
DEFAULT_C_PAR = 3.5e-10
DEFAULT_CPE_Q = 1.0e-7
DEFAULT_CPE_N = 0.8
DEFAULT_F_LO = 20.0
DEFAULT_F_HI = 2.0e5
DEFAULT_NPTS = 41


def log_frequencies(
    f_lo: float = DEFAULT_F_LO,
    f_hi: float = DEFAULT_F_HI,
    npts: int = DEFAULT_NPTS,
    *,
    descending: bool = True,
) -> np.ndarray:
    """Log-spaced sweep, **descending by default** — the order the instrument uses.

    ``meas_loop_eis`` sweeps high→low, and that ordering is load-bearing downstream:
    :func:`softae.analysis.circuit_fitting.extract_features` takes its ``R1`` guess
    from ``zi_local_min[-1]``, described as "the lowest-frequency local min", which is
    only the lowest frequency if the array descends. Fed ascending data the same
    spectrum yields ``r1_guess = 194`` instead of ``51259`` against a true 50 000 Ω.

    Generating ascending data here would therefore test a case the rig never produces
    while hiding how well the real one fits.
    """
    f = np.logspace(np.log10(f_lo), np.log10(f_hi), int(npts))
    return f[::-1] if descending else f


def reference_spectrum(
    freq: np.ndarray | None = None,
    *,
    R_series: float = DEFAULT_R_SERIES,
    R_bulk: float = DEFAULT_R_BULK,
    C_par: float = DEFAULT_C_PAR,
    Q: float = DEFAULT_CPE_Q,
    n: float = DEFAULT_CPE_N,
    L_lead: float = 0.0,
    C_par_exponent: float = 1.0,
    noise_pct: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(f, Z)`` in the physics convention (``Im Z < 0`` for a capacitive response).

    ``C_par_exponent < 1`` makes the parallel capacitance a CPE — a *dispersive lossy
    dielectric*, which is overhaul failure mode F4 and the thing the apparent-
    capacitance flatness gate exists to catch.
    """
    f = log_frequencies() if freq is None else np.asarray(freq, dtype=float)
    w = 2.0 * np.pi * f

    Z_cpe = 1.0 / (Q * (1j * w) ** n) if Q > 0 else 0.0
    Y_par = 1.0 / R_bulk + C_par * (1j * w) ** C_par_exponent
    Z = R_series + Z_cpe + 1.0 / Y_par
    if L_lead:
        Z = Z + 1j * w * L_lead

    if noise_pct:
        rng = np.random.default_rng(seed)
        Z = Z * (1.0 + noise_pct / 100.0 * rng.standard_normal(f.size))

    return f, np.asarray(Z, dtype=complex)


# ── Pathologies, one per framework §8.3 acceptance clause ────────────────────

def pure_series_rc(
    freq: np.ndarray | None = None, *, R_series: float = 500.0, C: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Overhaul §3.3 / F2 / F3 — the sample's conduction absent at every frequency.

    ``Z'`` flat, ``−Z'' ∝ f⁻¹``, ``tan δ`` **rising**. Must be caught by the loss-
    tangent slope *and* the series-RC test, which is the redundancy §3.5.3 asks for.
    """
    f = log_frequencies() if freq is None else np.asarray(freq, dtype=float)
    return f, R_series + 1.0 / (1j * 2.0 * np.pi * f * C)


def dispersive_dielectric(
    freq: np.ndarray | None = None, *, exponent: float = 0.80
) -> tuple[np.ndarray, np.ndarray]:
    """Overhaul F4 — a lossy dielectric whose conductance shares the measurand's axis."""
    return reference_spectrum(freq, C_par_exponent=exponent)


def hf_phase_artifact(
    freq: np.ndarray | None = None, *, n_points: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Overhaul F5 — apparent inductance at the top of the band on a blocking cell.

    Left in, this forces a fit to allocate a physically absurd ``L`` (400–500 µH
    against a measured 4.18 µH) which then distorts where ``R_series`` is determined.
    """
    f, Z = reference_spectrum(freq)
    order = np.argsort(f)[::-1][: int(n_points)]
    Z = Z.copy()
    Z[order] = Z[order].real + 1j * np.abs(Z[order].imag)
    return f, Z


def negative_real_part(
    freq: np.ndarray | None = None, *, n_points: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Overhaul F1 — ``Re Z < 0`` above the phase ceiling. Not a measurement."""
    f, Z = reference_spectrum(freq)
    Z = Z.copy()
    Z[:n_points] = -np.abs(Z[:n_points].real) + 1j * Z[:n_points].imag
    return f, Z


def stuck_instrument(freq: np.ndarray | None = None, *, value: complex = 1e5 - 1e4j):
    """No counterpart in the framework — |Z| identical at every frequency."""
    f = log_frequencies() if freq is None else np.asarray(freq, dtype=float)
    return f, np.full(f.size, value, dtype=complex)


def over_range(
    freq: np.ndarray | None = None, *, scale: float = 1e6
) -> tuple[np.ndarray, np.ndarray]:
    """Above the reproducible magnitude window — points the accuracy spec disowns."""
    f, Z = reference_spectrum(freq)
    return f, Z * scale


def as_eis_result(f: np.ndarray, Z: np.ndarray, channel: int = 1):
    """Wrap ``(f, Z)`` as an :class:`~softae.analysis.eis_data.EISResult`.

    Note ``z_imag_neg`` is ``−Im Z``, so a capacitive spectrum stores positive values —
    the convention the whole codebase uses on disk and in memory.
    """
    from softae.analysis.eis_data import EISResult

    return EISResult.from_arrays(
        channel=channel,
        f=np.asarray(f, dtype=float),
        z_real=np.asarray(Z, dtype=complex).real,
        z_imag_neg=-np.asarray(Z, dtype=complex).imag,
    )
