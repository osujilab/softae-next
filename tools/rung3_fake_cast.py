#!/usr/bin/env python
"""rung3_fake_cast.py — one real campaign cycle with the cast made inert.

NOTE the shebang: `python`, not `python3`, for the reason `tools/dbq.py` gives.

Rung 3 is one campaign cycle on the real bench. 3a — this — skips the physical
cast and drives wells that are ALREADY CAST (the films from the eis-validate
runs of 2026-09-13/14) through anneal -> equilibrate -> production read on the
real campaign path. 3b casts live, later.

Spec: `docs/SubAgent docs/rung3a_fake_cast.md`.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Real and live: the heater, the humidifier, pico1/pico2. Registered and never
actuated: the piezo — `softae_config.toml` gives it a real driver, and the spec
carries no ``[piezo]`` plan, so ``DepositionSettings.piezo`` stays ``None``.
Inert: the syringe and the stage, both swapped for mocks before anything
connects.

Because the piezo IS a real motion instrument, `hardware_safety.probe_motion`
returns ``real = ["piezo"]`` and `run_lock.rig_is_simulated` is **False**. This
run is therefore an ARMED REAL RUN: `WorkflowExecutor.execute` calls
`assert_hardware_armed` before every workflow, so ``SOFTAE_ALLOW_HARDWARE=1``
must be in the environment or the first workflow raises.

HOW IT REACHES THE CLI WITHOUT EDITING IT
-----------------------------------------
`softae.tools.campaign._cmd_run` imports `create_manager` and
`run_autonomous_campaign` **function-locally**, and says in a comment that the
locality is load-bearing precisely so a monkeypatch applied beforehand is seen.
This module patches those two module attributes, calls `campaign.main(...)`, and
restores both in a `finally`. `campaign.py` itself is never edited.

The dispense goes inert because `AsyncLiquidHandler` resolves its
sub-instruments through ``self.manager.get(name)`` at CALL time, not at
construction: `single_drop_simul` and `precondition_flush` are catalog tasks on
``instrument = "liquid_handler"``, but every `move_to`, `head_descend`,
`head_retract` and `single_pump` inside them lands on whatever is registered
under "stage" and "syringe" — i.e. on the mocks.

TWO THINGS THIS RUN WRITES THAT ARE NOT MEASUREMENTS
----------------------------------------------------
1. A NOMINAL `measured_thickness` row per channel. These wells have none
   (verified: zero rows for channels 1/11/13/14), and with no thickness there is
   no sigma and the settle criterion can only ever report `not_evaluable`. The
   row must be written AFTER `run_started`, carrying that run id, because
   `make_thickness_lookup` calls ``thickness_for(ch, run_id=run_id)`` and the SQL
   appends ``AND run_id = ?``. Sigma is then nominal in SCALE; the `rate`
   criterion is decades per hour, a ratio of sigma to sigma, and is
   scale-invariant. No absolute conductivity from this run may be quoted.
2. `formulations` rows describing a cast that did not occur.
   `_record_trial_formulations` runs on the batch path regardless of whether a
   pump turned. They carry this run's id, so they can be found and discounted;
   `RUNG3A_FAKE_CAST.txt` in the run directory says so in words.
"""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

NOTE_FILENAME = "RUNG3A_FAKE_CAST.txt"

#: Written into every nominal thickness row, and into the run-directory note.
NOMINAL_NOTE = (
    "rung 3a fake cast: NOMINAL thickness, no material dispensed. Sigma is "
    "nominal in SCALE; the rate criterion (decades/hour) is scale-invariant, so "
    "the settle verdict survives what the absolute number does not."
)

RUN_NOTE = (
    "RUNG 3A — FAKE CAST\n"
    "===================\n\n"
    "Nothing was dispensed and nothing moved. The syringe and the stage were\n"
    "mocked before this run connected; the heater, the humidifier and the picos\n"
    "were real and live.\n\n"
    f"{NOMINAL_NOTE}\n\n"
    "The `reservoir_levels` ledger was DEBITED for stock that was never\n"
    "dispensed: the campaign attaches `attach_reservoir_ledger` to the syringe,\n"
    "and the mock syringe reports every phantom dispense exactly as a real one\n"
    "would. Re-assert the true levels before the next run that depends on them.\n\n"
    "The `formulations` rows for this run describe a cast that DID NOT OCCUR.\n"
    "`_record_trial_formulations` runs on the batch path regardless of whether a\n"
    "pump turned, so the volumes and the twin's predicted thickness recorded\n"
    "against this run id are the solver's intent, not an event. Discount them.\n\n"
    "Spec: docs/SubAgent docs/rung3a_fake_cast.md\n"
)


def _add_src_to_path() -> None:
    """Repo `src/` fallback, for the bare `py` launcher. See `tools/dbq.py`."""
    try:
        import softae  # noqa: F401
    except ImportError:
        src = Path(__file__).resolve().parent.parent / "src"
        if not src.is_dir():
            raise
        sys.path.insert(0, str(src))


def _parse_channels(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    return tuple(int(part) for part in raw.replace(" ", "").split(",") if part)


# ── the two patches ──────────────────────────────────────────────────────────

@contextmanager
def _patched_manager_factory(*, emit=print) -> Iterator[None]:
    """`create_manager` -> the same manager, with syringe and stage mocked.

    The caller's ``mock=`` is passed through unchanged: ``--mock`` yields an
    all-mock manager and the swap below is then a no-op in substance, which is
    what makes the test path exercise the real composition rather than a
    different one.

    The two replacements follow the `install_mock_picos` idiom — construct with
    the displaced instrument's own config and assign into ``_instruments`` —
    because a driver swapped through the public `register` would be re-connected
    by a manager that has already connected.

    The piezo, the picos and the two conditions controllers are NOT touched.
    Mocking the piezo would flip `rig_is_simulated` to True on this rig and
    silently disarm both the run lock and the hardware interlock for a run whose
    heater and humidifier are live.
    """
    from softae.drivers import factory
    from softae.drivers.mock_stage import MockStage
    from softae.drivers.mock_syringe import MockSyringe

    original = factory.create_manager

    def _wrapped(*args: Any, **kwargs: Any):
        manager = original(*args, **kwargs)
        replaced: dict[str, str] = {}
        for name, cls in (("syringe", MockSyringe), ("stage", MockStage)):
            try:
                existing = manager.get(name)
            except Exception:
                continue
            replaced[name] = type(existing).__name__
            config = dict(getattr(existing, "config", {}) or {})
            manager._instruments[name] = cls(name, config)
        emit(f"   fake_cast_instruments_mocked: {replaced or 'nothing to replace'}")
        return manager

    factory.create_manager = _wrapped
    try:
        yield
    finally:
        factory.create_manager = original


@contextmanager
def _patched_campaign(channels: tuple[int, ...], thickness_um: float,
                      *, emit=print) -> Iterator[None]:
    """`run_autonomous_campaign` -> the same coroutine, with `on_event` chained.

    The chain does two things on `run_started`, and both need the run id, which
    is why they cannot be done before the call: the nominal thickness rows, and
    the run-directory note. The store is taken from the ``data_store=`` keyword
    the CLI passes, so nothing here has to guess at a project path.
    """
    from softae.core import autonomous_wiring

    original = autonomous_wiring.run_autonomous_campaign

    async def _wrapped(spec, **kwargs: Any):
        store = kwargs.get("data_store")
        inner = kwargs.get("on_event")

        def _on_event(event: dict[str, Any]) -> None:
            if event.get("type") == "run_started" and store is not None:
                # NEVER raises. An exception escaping `on_event` at `run_started`
                # is an unknown path through the campaign loop, and a bookkeeping
                # failure must not be the thing that finds out where it goes.
                try:
                    _annotate_run(store, str(event.get("run_id")), channels,
                                  thickness_um, emit=emit)
                except BaseException as exc:  # noqa: BLE001 - see above
                    emit(f"!! RUNG 3A: run annotation failed entirely: {exc}. "
                         f"Assume no nominal thickness was recorded — every well "
                         f"will be excluded 'sigma_null'. Abort with "
                         f"`softae-campaign control abort`.")
            if inner is not None:
                inner(event)

        kwargs["on_event"] = _on_event
        return await original(spec, **kwargs)

    autonomous_wiring.run_autonomous_campaign = _wrapped
    try:
        yield
    finally:
        autonomous_wiring.run_autonomous_campaign = original


def _annotate_run(store: Any, run_id: str, channels: tuple[int, ...],
                  thickness_um: float, *, emit=print) -> None:
    """Nominal thickness per channel, plus the note. Best-effort, loud on failure.

    `DataStore.start_run` takes an ``annotation``, but the campaign path calls it
    itself and `CampaignSpec` has no field that reaches it — so the note goes to
    the run directory instead, where the transcript already lives.
    """
    for channel in channels:
        try:
            store.record_thickness(int(channel), float(thickness_um),
                                   run_id=run_id, instrument="nominal",
                                   notes=NOMINAL_NOTE)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            emit(f"   !! could not record nominal thickness for ch{channel}: {exc}")

    # READ BACK THROUGH THE LOOKUP THE CAMPAIGN WILL USE, not through the writer.
    # `thickness_for` filters `AND run_id = ?`, so a row written to the wrong run —
    # or not written at all — is invisible to the criterion while the write above
    # reported success. "Wrote it" and "the criterion can see it" are different
    # statements, and only the second one matters.
    for channel in channels:
        found = None
        try:
            found = store.thickness_for(int(channel), run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            emit(f"   !! thickness read-back failed for ch{channel}: {exc}")
        if found is None or abs(float(found) - float(thickness_um)) > 1e-9:
            emit(f"!! RUNG 3A: sigma will be unavailable on ch{channel} — "
                 f"read back {found!r}, expected {float(thickness_um):g} um. "
                 f"That well has no sigma, so it is excluded 'sigma_null' every "
                 f"round and the settle criterion reports 'not_evaluable' for it. "
                 f"Abort with `softae-campaign control abort` rather than spend "
                 f"the cure.")
        else:
            emit(f"   nominal thickness ch{channel}: {float(found):g} um")
    try:
        note = Path(store.run_dir(run_id)) / NOTE_FILENAME
        note.write_text(RUN_NOTE, encoding="utf-8")
        emit(f"   fake-cast note: {note}")
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        emit(f"   !! could not write {NOTE_FILENAME}: {exc}")


# ── refusals ─────────────────────────────────────────────────────────────────

def _refuse(spec: Any, requested: tuple[int, ...] | None,
            thickness_um: float | None = None) -> str | None:
    """Why this must not start, or ``None``."""
    if thickness_um is not None and not (
            math.isfinite(float(thickness_um)) and float(thickness_um) > 0.0):
        return (f"--thickness-um {thickness_um!r} is not a positive length. "
                f"NOMINAL does not mean arbitrary: sigma divides by this number, "
                f"so zero, a negative, a NaN or an infinity gives the settle "
                f"criterion a sigma that is undefined rather than merely "
                f"mis-scaled, and every well is then excluded 'sigma_null'. Give "
                f"a plausible film thickness in micrometres.")
    if getattr(spec, "electrode_capacity", None) is not None:
        return (f"the spec sets electrode_capacity="
                f"{spec.electrode_capacity}. In allocator mode the board is "
                f"walked sequentially from electrode_start and spec.channels "
                f"stops deciding anything — this run would measure wells that "
                f"were never cast. Remove it.")
    if requested is not None and tuple(spec.channels) != requested:
        return (f"--channels {list(requested)} does not match the spec's "
                f"channels {list(spec.channels)}. The thickness rows follow "
                f"--channels and the measurement follows the spec, so a "
                f"mismatch records a nominal thickness against a well nobody "
                f"reads and leaves the wells that ARE read with no sigma.")
    return None


def _banner(spec_path: str, project: str, channels: tuple[int, ...],
            thickness_um: float, *, mock: bool, emit=print) -> None:
    emit("")
    emit("  RUNG 3A — FAKE CAST" + ("  [--mock: nothing here is real]" if mock else ""))
    emit(f"  spec {spec_path} -> project {project}")
    emit(f"  wells {list(channels)} at a NOMINAL {thickness_um:g} um")
    emit("")
    emit("  REAL and live ....... heater, humidifier, pico1/pico2"
         if not mock else "  REAL and live ....... nothing (--mock)")
    emit("  REGISTERED, never actuated ... piezo (the spec carries no [piezo] plan)")
    emit("  INERT ............... syringe and stage, both mocked:")
    emit("                        no material is dispensed and nothing moves.")
    emit("")
    if not mock:
        emit("  This is an ARMED REAL RUN. SOFTAE_ALLOW_HARDWARE must be set.")
    emit("  The measured_thickness rows this run writes are NOMINAL.")
    emit("  The formulations rows this run writes describe a cast that did not occur.")
    emit("  The reservoir_levels ledger WILL BE DEBITED for stock never dispensed —")
    emit("  the mock syringe reports each phantom dispense. Re-assert stock after.")
    emit("")


# ── entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    _add_src_to_path()
    parser = argparse.ArgumentParser(
        prog="rung3_fake_cast",
        description="One real campaign cycle on already-cast wells, with the "
                    "cast made inert.")
    parser.add_argument("spec", help="campaign spec (.toml)")
    parser.add_argument("--project", required=True, help="DataStore project dir")
    parser.add_argument("--thickness-um", type=float, required=True,
                        help="NOMINAL per-channel thickness (um). These wells "
                             "have no profilometry; this exists to give the "
                             "settle criterion a sigma at all.")
    parser.add_argument("--channels", default=None,
                        help="comma-separated, e.g. 1,11,13,14. Must equal the "
                             "spec's channels when given.")
    parser.add_argument("--mock", action="store_true",
                        help="all-mock run: claims no rig and touches nothing.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="pass -y through to the campaign CLI.")
    args = parser.parse_args(argv)

    from softae.core.campaign_spec_io import SpecLoadError, load_campaign_spec
    from softae.tools import campaign

    requested = _parse_channels(args.channels)
    try:
        spec = load_campaign_spec(args.spec)
    except SpecLoadError as exc:
        print(f"Spec error: {exc}", file=sys.stderr)
        return 2

    refusal = _refuse(spec, requested, args.thickness_um)
    if refusal is not None:
        print(f"\n!! NOT STARTING — {refusal}", file=sys.stderr)
        return 2

    channels = requested or tuple(int(c) for c in spec.channels)
    _banner(args.spec, args.project, channels, args.thickness_um, mock=args.mock)

    with ExitStack() as stack:
        stack.enter_context(_patched_manager_factory())
        stack.enter_context(_patched_campaign(channels, args.thickness_um))
        if not args.mock:
            # Held for the whole call, and only for a real one: a mock run
            # holding the rig turns a dry run into an outage for a real one
            # (`run_lock.rig_is_simulated`'s own reasoning). On THIS rig the
            # claim is redundant and re-entrant — the real piezo makes the CLI
            # take its own — but it is kept for the configuration that has no
            # real piezo, where the CLI would read the rig as simulated and skip
            # its claim while the heater and humidifier ran live.
            from softae.core.run_lock import (
                RunLockHeld,
                busy_rig_message,
                held_run_lock,
            )

            try:
                stack.enter_context(
                    held_run_lock(what="campaign:rung3a-fake-cast"))
            except RunLockHeld as busy:
                # The CLI's own words and the CLI's own exit code, imported
                # rather than restated: two wordings for one situation is how a
                # wrapper learns to parse stdout, and a traceback here reads as a
                # crash when it is only somebody else holding the rig. Returning
                # through the `with ExitStack()` restores both patches on the way
                # out, which is the half that would be easy to lose here.
                print(f"\n!! NOT STARTING '{spec.name}'\n\n"
                      f"{busy_rig_message(busy.lock, action='This campaign')}",
                      flush=True)
                return campaign.EXIT_BUSY

        argv_out = ["run", args.spec, "--project", args.project, "--head-up"]
        if args.yes:
            argv_out.append("-y")
        if args.mock:
            argv_out.append("--mock")
        return campaign.main(argv_out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
