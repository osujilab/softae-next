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
