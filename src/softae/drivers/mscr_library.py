"""MethodSCRIPT file builders for EmStat Pico measurements.

Ported from ``SoftAE_classPkg/SoftAE_ESPico_mscr_library.py``.
Each function writes (or overwrites) a ``.mscr`` file that is then sent
to the instrument by :meth:`~softae.drivers.async_espico.AsyncESPico.sendscript_getdata`.

The file is always rewritten before each measurement so that its content
always reflects the current run parameters.

EIS preset parameter sets
-------------------------
+----------+--------+----------+----------+-------+-------+-------+
| Preset   | npts   | f_hi(Hz) | f_lo(mHz)| mVac  | mVdc  | speed |
+==========+========+==========+==========+=======+=======+=======+
| Quick    |   10   |  10 000  |   100    |   10  |   0   |   3   |
| Standard |   20   | 100 000  |   100    |   10  |   0   |   3   |
| Extended |   30   | 200 000  |    16    |   10  |   0   |   3   |
| Longest  |   40   | 200 000  |    16    |   10  |   0   |   3   |
+----------+--------+----------+----------+-------+-------+-------+
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Mantissa digits a MethodSCRIPT numeric literal may carry (int32 headroom).
MSCR_MAX_MANTISSA_DIGITS = 9

#: Scale factors and their MethodSCRIPT suffix, coarsest first. ``k``/``M`` are
#: deliberately absent: the bare integer form is canonical at and above 1 Hz, which
#: is what keeps a one-segment segmented build byte-identical to
#: :func:`eis_run_mscrbuild` and its four stopwatched timing anchors.
_MSCR_SCALES = ((1.0, ""), (1e3, "m"), (1e6, "u"))

#: Two literals are the same frequency within this relative slack. Tight enough
#: that nothing is silently rounded, loose enough to absorb ``6.475 * 1000``
#: landing on 6474.999999999999.
_EXACT_REL_TOL = 1e-9


# ── Frequency literals ────────────────────────────────────────────────────────

def mscr_freq_literal(f_hz: float) -> str:
    """Render *f_hz* as a MethodSCRIPT frequency literal.

    A MethodSCRIPT numeric literal is **an integer mantissa followed by an optional
    SI prefix character** (``mscript.SI_PREFIX_FACTOR``). There is no decimal form:
    ``0.5`` is not a valid frequency literal, ``500m`` is.

    This is the highest-risk unit in the segment emitter, because a wrong suffix is
    a **silent 1000× frequency error the instrument executes without complaint** —
    the resulting spectrum is indistinguishable from a real one measured in the
    wrong band. So it raises rather than approximates: no truncated literal, no
    zeroed mantissa, no silent rounding. A caller holding a value that may not be
    representable calls :func:`quantize_hz` first, which is what the planner does.
    """
    f = float(f_hz)
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError(f"frequency must be finite and positive, got {f_hz!r}")

    for scale, suffix in _MSCR_SCALES:
        scaled = f * scale
        mantissa = round(scaled)
        if mantissa < 1 or abs(scaled - mantissa) > _EXACT_REL_TOL * max(1.0, scaled):
            continue
        if len(str(mantissa)) > MSCR_MAX_MANTISSA_DIGITS:
            raise ValueError(
                f"{f_hz!r} Hz needs {len(str(mantissa))} mantissa digits, "
                f"over the {MSCR_MAX_MANTISSA_DIGITS}-digit budget"
            )
        return f"{mantissa}{suffix}"

    raise ValueError(
        f"{f_hz!r} Hz has no exact MethodSCRIPT literal; call quantize_hz first"
    )


def quantize_hz(f_hz: float) -> float:
    """Snap *f_hz* onto the finest grid ``mscr_freq_literal`` can render exactly.

    µHz where the mantissa fits, mHz next, whole Hz last — so the **planned** grid,
    the grid **recorded** in ``eis_params`` and the grid the **instrument runs** are
    the same numbers rather than three roundings of one intention.
    """
    f = float(f_hz)
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError(f"frequency must be finite and positive, got {f_hz!r}")

    for scale, _ in reversed(_MSCR_SCALES):
        mantissa = round(f * scale)
        if mantissa >= 1 and len(str(mantissa)) <= MSCR_MAX_MANTISSA_DIGITS:
            return mantissa / scale

    raise ValueError(f"{f_hz!r} Hz is below the µHz grid; no literal can carry it")


# ── Segments ──────────────────────────────────────────────────────────────────

def resolve_segments(
    segments: Sequence[tuple[float, float, int]],
) -> tuple[tuple[float, float, int], ...]:
    """Quantize, validate and de-duplicate segment boundaries. Pure: writes nothing.

    Each segment is ``(f_start_hz, f_end_hz, npts)`` and the sequence as a whole must
    descend, because that is what lets the parser concatenate curves in emission
    order without re-sorting: ``get_values_by_column(..., icurve=None)`` already
    extends across every curve, so descending non-overlapping segments arrive as one
    monotonic sweep.

    **The boundary is the hazard, not the ordering.** Two adjacent segments that
    share a frequency measure that point twice and break the monotonicity the
    no-resort argument rests on, so a touching start is nudged one log-step inward on
    its own grid. A *genuine* overlap is a planner bug and raises — papering over it
    would hide the bug and still emit the duplicate.
    """
    resolved: list[tuple[float, float, int]] = []
    for i, seg in enumerate(segments):
        try:
            f_start, f_end, npts = seg
        except (TypeError, ValueError):
            raise ValueError(f"segment {i}: expected (f_start, f_end, npts), got {seg!r}")
        npts = int(npts)
        if npts < 1:
            raise ValueError(f"segment {i}: npts must be >= 1, got {npts}")
        f_start, f_end = quantize_hz(f_start), quantize_hz(f_end)
        if not f_start > f_end:
            raise ValueError(
                f"segment {i}: f_start must exceed f_end, got {f_start} -> {f_end}"
            )

        if resolved:
            previous_end = resolved[-1][1]
            if f_start > previous_end * (1.0 + _EXACT_REL_TOL):
                raise ValueError(
                    f"segment {i}: starts at {f_start} Hz, above segment {i - 1}'s "
                    f"end of {previous_end} Hz — segments must descend and not overlap"
                )
            if abs(f_start - previous_end) <= _EXACT_REL_TOL * previous_end:
                step = (math.log10(f_start) - math.log10(f_end)) / npts
                f_start = quantize_hz(10.0 ** (math.log10(f_start) - step))
                if not f_start > f_end:
                    raise ValueError(
                        f"segment {i}: too narrow to nudge off segment {i - 1}'s "
                        f"boundary at {previous_end} Hz"
                    )

        resolved.append((f_start, f_end, npts))

    return tuple(resolved)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mod_channel_restart(ch: int) -> int:
    """Remap channels above 16 for a second Pico (subtract 16).

    Parameters
    ----------
    ch : int
        1-based channel number (1–32).

    Returns
    -------
    int
        Channel in the 1–16 range.
    """
    return ch - 16 if ch > 16 else ch


def _chan_hex(mux_ch: int) -> str:
    """Convert a 1-based mux channel to the ESP GPIO address string used in scripts."""
    we_bits = format(mux_ch - 1, "04b")
    ce_bits = format(mux_ch - 1, "04b")
    return hex(int(ce_bits + we_bits, 2)) + "i"


# ── Script builders ───────────────────────────────────────────────────────────

def eis_run_mscrbuild(
    filename: str,
    mux_ch: int,
    mVac: int = 10,
    f_hi: int = 100_000,
    f_lo: int = 100,
    npts: int = 20,
    mVdc: int = 0,
    inst_ch: int = 0,
    speed: int = 3,
) -> None:
    """Write (or overwrite) a MethodSCRIPT file for an EIS measurement.

    Parameters
    ----------
    filename : str
        Output ``.mscr`` file path (created/overwritten each call).
    mux_ch : int
        Multiplexer channel (1–32; automatically remapped for a second Pico).
    mVac : int
        AC signal amplitude in mV (default 10).
    f_hi : int
        Upper frequency bound in Hz, max 200 000 (default 100 000).
    f_lo : int
        Lower frequency bound in mHz, min 16 (default 100).
    npts : int
        Number of log-spaced frequency points (default 20).
    mVdc : int
        DC offset voltage in mV (default 0).
    inst_ch : int
        pgstat channel index (default 0).
    speed : int
        Measurement speed mode; 3 = high (default 3).
    """
    mux_ch = mod_channel_restart(mux_ch)
    chan = _chan_hex(mux_ch)

    with open(filename, "w") as fh:
        fh.write(
            "e\n"
            "set_gpio_cfg 0x3FFi 1\n"
            "set_gpio 0x11i\n"
            "var f\nvar r\nvar j\n"
            f"set_pgstat_chan {inst_ch}\n"
            f"set_pgstat_mode {speed}\n"
            "set_autoranging ba 1p 1\n"
            "set_autoranging ab 1p 1\n"
            "cell_on\n"
            f"set_gpio {chan}\n"
            "wait 10m \n"
            f"meas_loop_eis f r j {mVac}m {f_hi} {f_lo}m {npts} {mVdc}m\n"
            " pck_start\n pck_add f\n pck_add r\n pck_add j\n pck_end\nendloop\n"
            "on_finished:\n"
            "cell_off\n\n"
        )


def eis_segmented_mscrbuild(
    filename: str,
    mux_ch: int,
    segments: Sequence[tuple[float, float, int]],
    *,
    mVac: int = 10,
    mVdc: int = 0,
    inst_ch: int = 0,
    speed: int = 3,
) -> None:
    """Write a ``.mscr`` running one ``meas_loop_eis`` block per frequency segment.

    Same preamble and epilogue as :func:`eis_run_mscrbuild` — deliberately, and a
    characterization test holds a one-segment build byte-identical to it. That
    function's emitted bytes are the artifact four stopwatched timing anchors
    describe, so it is neither edited nor re-expressed as the one-segment case here.

    Parameters
    ----------
    filename : str
        Output ``.mscr`` file path (created/overwritten each call).
    mux_ch : int
        Multiplexer channel (1–32; automatically remapped for a second Pico).
    segments : sequence of (f_start_hz, f_end_hz, npts)
        Descending, non-overlapping bands in **Hz** — note ``eis_run_mscrbuild``
        takes its lower bound in mHz and this one does not. Passed through
        :func:`resolve_segments`, so a touching boundary is nudged and a genuine
        overlap raises.
    mVac, mVdc, inst_ch, speed
        As :func:`eis_run_mscrbuild`.
    """
    resolved = resolve_segments(segments)
    if not resolved:
        raise ValueError(
            "no segments: the script would cell_on and cell_off with nothing "
            "between them and return an empty spectrum with no error to explain it"
        )

    mux_ch = mod_channel_restart(mux_ch)
    chan = _chan_hex(mux_ch)
    blocks = "".join(
        f"meas_loop_eis f r j {mVac}m {mscr_freq_literal(f_start)} "
        f"{mscr_freq_literal(f_end)} {npts} {mVdc}m\n"
        " pck_start\n pck_add f\n pck_add r\n pck_add j\n pck_end\nendloop\n"
        for f_start, f_end, npts in resolved
    )

    with open(filename, "w") as fh:
        fh.write(
            "e\n"
            "set_gpio_cfg 0x3FFi 1\n"
            "set_gpio 0x11i\n"
            "var f\nvar r\nvar j\n"
            f"set_pgstat_chan {inst_ch}\n"
            f"set_pgstat_mode {speed}\n"
            "set_autoranging ba 1p 1\n"
            "set_autoranging ab 1p 1\n"
            "cell_on\n"
            f"set_gpio {chan}\n"
            "wait 10m \n"
            + blocks
            + "on_finished:\n"
            "cell_off\n\n"
        )


def lsv_run_mscrbuild(filename: str, mux_ch: int) -> None:
    """Write a MethodSCRIPT file for Linear Sweep Voltammetry (−1 V → +1 V).

    Parameters
    ----------
    filename : str
        Output ``.mscr`` file path.
    mux_ch : int
        Multiplexer channel (1–32).
    """
    mux_ch = mod_channel_restart(mux_ch)
    chan = _chan_hex(mux_ch)

    with open(filename, "w") as fh:
        fh.write(
            "e\n"
            "set_gpio_cfg 0x3FFi 1\n"
            "set_gpio 0x11i\n"
            "var i\nvar c\nvar p\n"
            "store_var i 0i aa\n"
            "set_pgstat_chan 0\n"
            "set_pgstat_mode 2\n"
            "set_max_bandwidth 400\n"
            "set_pot_range -1 1\n"
            "set_cr 1m\n"
            "set_autoranging 10u 1m\n"
            "cell_on\n"
            f"set_gpio {chan}\n"
            "set_e -1000m\n"
            "wait 100m\n"
            "meas_loop_lsv p c -1 1 10m 1\n"
            " pck_start\n pck_add p\n pck_add c\n pck_end\nendloop\n"
            "on_finished:\n"
            "cell_off\n\n"
        )


def ocp_run_mscrbuild(
    filename: str,
    mux_ch: int,
    ms_interval: int = 50,
    seconds_total: int = 2,
) -> None:
    """Write a MethodSCRIPT file for Open Circuit Potentiometry.

    .. note:: OCP support on the Pico MUX16 is experimental.

    Parameters
    ----------
    filename : str
        Output ``.mscr`` file path.
    mux_ch : int
        Multiplexer channel.
    ms_interval : int
        Sampling interval in ms (default 50).
    seconds_total : int
        Total measurement duration in s (default 2).
    """
    with open(filename, "w") as fh:
        fh.write(
            "e\n"
            "var p\n"
            "set_pgstat_chan 0\n"
            "set_pgstat_mode 3\n"
            "cell_off\n"
            "wait 10m\n"
            f"meas_loop_ocp p {ms_interval}m {seconds_total}\n"
            " pck_start\n pck_add p\n pck_end\nendloop\n"
            "on_finished:\n"
            "cell_off\n\n"
        )
