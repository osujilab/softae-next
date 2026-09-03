"""Time every EIS preset on one channel, to ground-truth the duration model.

``core/preflight.estimate_eis_duration`` is a two-constant model fitted to
stopwatch readings held in :data:`~softae.core.preflight.EIS_MEASURED_S_PER_CHANNEL`.
Those readings are the only thing standing between a projection and a guess, and
they go stale the moment a preset's grid moves — which is exactly what the
2026-08-17 mains-notch retune did to all four. This tool re-reads them.

What it produces is a per-channel cost per preset, in seconds, plus a paste-ready
block for the anchor table. What it deliberately does *not* do is edit that table:
which numbers become ground truth is an operator decision, and a tool that writes
its own anchors can launder a bad run into authority.

**This actuates the potentiostat.** It drives real sweeps on a real channel and so
sits behind the same arming interlock and explicit confirmation as any other
motion tool. ``--mock`` exercises the whole path against simulated instruments.

Ordering is REPEAT-MAJOR — every preset once, then again — rather than finishing
one preset before starting the next. A rig that drifts (warming enclosure, a cell
relaxing under bias) would otherwise dump all of that drift into whichever preset
happened to run last, and it would read as that preset being expensive.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

DEFAULT_REPEATS = 3
DEFAULT_WARMUP = 1
#: Spread above which a preset's samples are called unstable rather than averaged.
#: The anchors this feeds are quoted to 0.05 s; a preset swinging wider than this
#: is not reporting a cost, it is reporting that something else was happening.
SPREAD_WARN_PCT = 10.0

#: ``<kind>:<name>:<run_id>`` — the grammar ``core.rig_session`` documents, whose
#: shipped sibling is ``campaign:<name>:<run_id>``. This tool has no DataStore and
#: no run id at all (it is a pure benchmarking utility), so the third field is a
#: timestamp stamped at launch rather than a database key -- see ``run_timing``'s
#: ``stamp`` argument. Filled rather than left trailing, same reasoning as
#: ``tools/env_hold.py``'s ``CLAIM_KIND``: a bare ``tool:eis-timing:`` in a lock
#: file asserts "there is an identifier and it is blank".
CLAIM_KIND = "tool:eis-timing"


@dataclass
class PresetTiming:
    """Every sample taken for one preset, and what they add up to."""

    preset: str
    params: dict[str, int]
    samples_s: list[float] = field(default_factory=list)
    warmup_s: list[float] = field(default_factory=list)
    modelled_s: float = 0.0

    @property
    def median_s(self) -> float:
        return statistics.median(self.samples_s) if self.samples_s else 0.0

    @property
    def mean_s(self) -> float:
        return statistics.fmean(self.samples_s) if self.samples_s else 0.0

    @property
    def spread_pct(self) -> float | None:
        """Peak-to-peak as a percentage of the median — the honesty check.

        ``None`` with fewer than two timed samples. Returning 0.0 there would
        print a preset that was measured **once** as perfectly reproducible,
        which is the opposite of what one sample establishes.
        """
        if len(self.samples_s) < 2 or self.median_s <= 0:
            return None
        return (max(self.samples_s) - min(self.samples_s)) / self.median_s * 100.0

    @property
    def warmup_delta_pct(self) -> float | None:
        """How far the discarded warmup sat from the timed median.

        The warmup is thrown out of the anchor because it can carry connect and
        first-script cost, but it is still a sweep of the same grid — so it is a
        free consistency check, and the only one available at ``--repeats 1``.
        """
        if not self.warmup_s or self.median_s <= 0:
            return None
        return (statistics.median(self.warmup_s) / self.median_s - 1.0) * 100.0

    @property
    def is_unstable(self) -> bool:
        """True when replicates disagree by more than the warn threshold.

        Falls back to the warmup comparison when there is only one timed sample,
        so a single-pass run is still checked rather than silently trusted.
        """
        if self.spread_pct is not None:
            return self.spread_pct > SPREAD_WARN_PCT
        if self.warmup_delta_pct is not None:
            return abs(self.warmup_delta_pct) > SPREAD_WARN_PCT
        return False

    @property
    def model_error_pct(self) -> float:
        """How far the model sits from the bench, signed: + means model is high."""
        if self.median_s <= 0:
            return 0.0
        return (self.modelled_s / self.median_s - 1.0) * 100.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "samples_s": [round(s, 3) for s in self.samples_s],
            "warmup_s": [round(s, 3) for s in self.warmup_s],
            "median_s": round(self.median_s, 2),
            "mean_s": round(self.mean_s, 2),
            "min_s": round(min(self.samples_s), 2) if self.samples_s else 0.0,
            "max_s": round(max(self.samples_s), 2) if self.samples_s else 0.0,
            "spread_pct": (round(self.spread_pct, 2)
                           if self.spread_pct is not None else None),
            "warmup_delta_pct": (round(self.warmup_delta_pct, 3)
                                 if self.warmup_delta_pct is not None else None),
            "modelled_s": round(self.modelled_s, 2),
            "model_error_pct": round(self.model_error_pct, 1),
        }


def _resolve_presets(names: str | None) -> list[str]:
    """Named presets, or every preset in config, in a stable order."""
    from softae.config.loader import eis_presets

    available = list(eis_presets().keys())
    if not names:
        return available
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in wanted if n not in available]
    if unknown:
        raise ValueError(
            f"unknown preset(s) {unknown}; config defines {available}"
        )
    return wanted


def _time_one_sweep(pico: Any, channel: int, params: Any, script_dir: str) -> float:
    """Build the script, run it, and return wall-clock seconds for the sweep.

    The build is deliberately *outside* the timed region: writing a ``.mscr`` is
    host-side work that no anchor should carry, and including it would make the
    per-point cost look higher on short presets than on long ones.
    """
    from softae.drivers.mscr_library import eis_run_mscrbuild

    script_path = os.path.join(script_dir, f"softae_timing_ch{channel}.mscr")
    eis_run_mscrbuild(
        script_path,
        mux_ch=channel,
        mVac=params.mv_ac,
        f_hi=params.f_hi,
        f_lo=params.f_lo_mHz,
        npts=params.npts,
        mVdc=params.mv_dc,
    )
    outdir = getattr(pico, "_output_dir", None) or tempfile.gettempdir()

    start = time.monotonic()
    pico.sendscript_getdata(script_path, outdir, channel)
    return time.monotonic() - start


async def run_timing(
    channel: int,
    presets: Sequence[str],
    *,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    mock: bool = False,
    stamp: str = "",
) -> dict[str, PresetTiming]:
    """Drive every preset ``repeats`` times on one channel, repeat-major.

    ``stamp`` identifies this run inside the rig claim's ``what`` field
    (``tool:eis-timing:<stamp>``) in place of the run id a DataStore-backed tool
    would use -- this tool has neither. ``main`` derives it from ``started``
    before calling in, so a claim taken here and one read back from the lock
    file agree on when the run began. A caller that omits it (a test, or a
    script importing ``run_timing`` directly) gets a fresh one instead of a
    trailing-colon claim.
    """
    from softae.config.loader import pico_for_channel
    from softae.core.eis_scripts import EISParams
    from softae.core.hardware_safety import assert_hardware_armed
    from softae.core.preflight import estimate_eis_duration
    from softae.core.rig_session import held_rig_session
    from softae.drivers.factory import create_manager

    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resolved = {p: EISParams.from_preset(p) for p in presets}
    results = {
        p: PresetTiming(
            preset=p,
            params={
                "npts": e.npts, "f_hi": e.f_hi, "f_lo_mHz": e.f_lo_mHz,
                "mv_ac": e.mv_ac, "mv_dc": e.mv_dc,
            },
            modelled_s=estimate_eis_duration(e),
        )
        for p, e in resolved.items()
    }

    manager = create_manager(mock=mock)
    assert_hardware_armed(manager, action=f"run EIS timing sweeps on channel {channel}")

    # `--mock` claims nothing, and the gate is **this tool's own `mock` flag**
    # rather than `held_rig_session`'s internal `session_is_simulated` exemption
    # alone. Same shape as `tools/eis_validate.py`'s `_rig_claim` and
    # `tools/env_hold.py`'s claim gate -- see either docstring for the reasoning
    # in full; summarised for the two reasons that still apply to a tool this
    # small: (1) `session_is_simulated` places a driver by checking it against a
    # hand-maintained registry of shipped mock classes, and every failure mode
    # there -- a mock added to `softae.drivers` and forgotten in the registry, an
    # unreadable driver, an enumeration that raises -- answers "real" and claims
    # the rig anyway; `mock` needs no registry, so it cannot go stale that way.
    # (2) it is the one flag this tool's own actuation prompt in `main` is
    # already gated on, so claiming the same flag keeps "this run touches no
    # hardware" a single fact instead of two that could disagree. Do not
    # collapse this into an unconditional `held_rig_session` call.
    claim = (contextlib.nullcontext() if mock
             else held_rig_session(manager, what=f"{CLAIM_KIND}:{stamp}"))
    # Claimed before `connect_all`, released after `disconnect_all` --
    # `core.rig_session`'s rule is "acquire when the ports open, release when
    # they close" -- so the claim wraps the whole connect/sweep/disconnect
    # block rather than sitting inside it.
    with claim:
        await manager.connect_all()
        try:
            pico = manager.get(pico_for_channel(channel))
            script_dir = tempfile.gettempdir()

            for pass_index in range(warmup + repeats):
                is_warmup = pass_index < warmup
                tag = ("warmup" if is_warmup
                       else f"pass {pass_index - warmup + 1}/{repeats}")
                for preset in presets:
                    elapsed = _time_one_sweep(
                        pico, channel, resolved[preset], script_dir
                    )
                    bucket = results[preset]
                    (bucket.warmup_s if is_warmup else bucket.samples_s).append(elapsed)
                    print(f"  [{tag}] {preset:<10} {elapsed:8.2f} s", flush=True)
        finally:
            await manager.disconnect_all()

    return results


def _print_report(
    results: dict[str, PresetTiming], channel: int, *, mock: bool = False
) -> None:
    """The table, then the paste block, then the caveats.

    ASCII only. This prints to whatever console the operator has, and a report
    whose warnings arrive as mojibake is a report whose warnings get skipped.
    """
    print(f"\nEIS preset timing -- channel {channel}{' (MOCK)' if mock else ''}")
    print(f"{'preset':<10} {'median':>9} {'mean':>9} {'spread':>8} {'vs warm':>8} "
          f"{'modelled':>10} {'model err':>10}")
    print("-" * 69)
    for timing in results.values():
        flag = "  <-- UNSTABLE" if timing.is_unstable else ""
        spread = (f"{timing.spread_pct:7.1f}%" if timing.spread_pct is not None
                  else "    n/a ")
        warm = (f"{timing.warmup_delta_pct:+7.2f}%"
                if timing.warmup_delta_pct is not None else "    n/a ")
        print(f"{timing.preset:<10} {timing.median_s:8.2f}s {timing.mean_s:8.2f}s "
              f"{spread} {warm} {timing.modelled_s:9.2f}s "
              f"{timing.model_error_pct:+9.1f}%{flag}")

    if any(t.spread_pct is None for t in results.values()):
        print("\nNOTE: 'spread n/a' means ONE timed pass -- no replicate scatter was "
              "measured.\n      The 'vs warm' column is then the only cross-check: it "
              "compares the\n      discarded warmup against the timed pass on the same "
              "grid.")

    if mock:
        print("\nNO PASTE BLOCK: these are simulated sweeps. A mock run measures "
              "how fast this host can write a .mscr file and nothing whatever "
              "about the rig. Re-run without --mock to produce anchors.")
        return

    unstable = [t.preset for t in results.values() if t.is_unstable]
    print("\nPaste into core/preflight.EIS_MEASURED_S_PER_CHANNEL "
          "(median, not mean -- one stalled sweep should not move an anchor):\n")
    print("EIS_MEASURED_S_PER_CHANNEL: dict[str, float] = {")
    for timing in results.values():
        note = "  # UNSTABLE -- re-run before trusting" if timing.preset in unstable else ""
        print(f'    "{timing.preset}": {timing.median_s:.2f},{note}')
    print("}")

    if unstable:
        print(f"\nWARNING: {', '.join(unstable)} varied by more than "
              f"{SPREAD_WARN_PCT:.0f}% peak-to-peak. That is a rig or cell telling "
              f"you something, not a cost. Do not enshrine it as an anchor.")
    print("\nThese are ONE channel's costs. The anchor table is per-channel by "
          "definition, but a round's cost is not simply N x this -- per-round "
          "executor and mscr-rebuild work sits outside the timed region "
          "(see equilibration.ROUND_BUFFER_S).")


def _write_json(path: str, results: dict[str, PresetTiming], meta: dict) -> None:
    payload = {
        "schema": "softae.eis_timing/1",
        **meta,
        "presets": {name: t.as_dict() for name, t in results.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="softae-eis-timing",
        description="Time every EIS preset on one channel to ground-truth the "
                    "duration model.",
    )
    parser.add_argument("--channel", type=int, default=1,
                        help="1-based channel to sweep (default 1)")
    parser.add_argument("--presets", default=None,
                        help="comma-separated subset; default is every preset "
                             "in [eis_presets]")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"timed passes per preset (default {DEFAULT_REPEATS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="untimed passes first, discarded "
                             f"(default {DEFAULT_WARMUP})")
    parser.add_argument("--out", default=None, help="write results as JSON here")
    parser.add_argument("--mock", action="store_true",
                        help="simulated instruments; exercises the path, times nothing real")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="skip the actuation confirmation prompt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from softae.core.run_lock import RunLockHeld
    from softae.tools import use_utf8_console

    use_utf8_console()
    args = build_parser().parse_args(argv)

    if args.repeats < 1:
        print("--repeats must be at least 1")
        return 2
    try:
        presets = _resolve_presets(args.presets)
    except ValueError as exc:
        print(str(exc))
        return 2
    if not presets:
        print("no presets defined in [eis_presets]")
        return 2

    total = (args.warmup + args.repeats) * len(presets)
    print(f"Channel {args.channel}: {len(presets)} preset(s) x "
          f"{args.warmup} warmup + {args.repeats} timed = {total} sweeps"
          f"{' (MOCK)' if args.mock else ''}")

    if not args.mock and not args.yes:
        print("\nThis drives REAL sweeps on the potentiostat. The cell must be "
              "connected and settled -- a channel still relaxing will time long "
              "and poison the anchor.")
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    try:
        results = asyncio.run(run_timing(
            args.channel, presets,
            repeats=args.repeats, warmup=args.warmup, mock=args.mock,
            stamp=stamp,
        ))
    except RunLockHeld as held:
        from softae.core.run_lock import busy_rig_message

        print(f"\nNOT STARTING: "
              f"{busy_rig_message(held.lock, action='This timing run')}")
        return 1
    _print_report(results, args.channel, mock=args.mock)

    if args.out:
        _write_json(args.out, results, {
            "channel": args.channel,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "mock": args.mock,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
