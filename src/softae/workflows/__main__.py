"""CLI workflow runner for SoftAE.

Run an experiment workflow from YAML/JSON without launching the GUI::

    softae-run workflows/standard_eis_sweep.yaml
    softae-run workflows/single_drop_and_measure.yaml --mock --dry-run
    softae-run my_experiment.json --log-dir ./logs

Flags
-----
--mock       Force mock instruments (no real hardware).
--real       Require real instruments (fail if unavailable).
--dry-run    Parse and validate the workflow without executing it.
--validate   Like --dry-run, but also checks instrument & method names.
--log-dir    Directory for JSON-lines experiment logs (default: ``./logs``).
--verbose    Print step-by-step progress to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import structlog

from softae.config import loader as cfg
from softae.drivers.factory import create_manager
from softae.tools import use_utf8_console
from softae.workflows.experiment_logger import ExperimentLogger
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import WorkflowStep
from softae.workflows.workflow_parser import parse_file

logger = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``softae-run`` argument parser."""
    p = argparse.ArgumentParser(
        prog="softae-run",
        description="Execute a SoftAE workflow from a YAML or JSON file.",
    )
    p.add_argument(
        "workflow",
        help="Path to the workflow definition file (.yaml / .json).",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Force all mock instruments (no hardware required).",
    )
    mode.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="Require real instruments (error if unavailable).",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse and validate the workflow without executing it.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Like --dry-run, plus check instrument names and method signatures against mock drivers.",
    )
    p.add_argument(
        "--log-dir",
        default="./logs",
        help="Directory for experiment log files (default: ./logs).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print detailed step progress to stdout.",
    )
    return p


# ── Callback factories (for --verbose) ──────────────────────────────────


def _make_step_start_cb(verbose: bool):
    def _cb(step: WorkflowStep, index: int, total: int) -> None:
        if verbose:
            pct = int((index / total) * 100) if total else 0
            print(f"  [{index + 1}/{total}] ({pct}%) Starting: {step.name}")
    return _cb


def _make_step_complete_cb(verbose: bool):
    def _cb(step: WorkflowStep, index: int, total: int, result, elapsed: float = 0.0) -> None:
        if verbose:
            print(f"  [{index + 1}/{total}] \u2713 {step.name} ({elapsed:.1f}s)")
    return _cb


def _make_step_error_cb(verbose: bool):
    def _cb(step: WorkflowStep, index: int, total: int, error) -> None:
        msg = f"  [{index + 1}/{total}] ✗ {step.name}: {error}"
        print(msg, file=sys.stderr)
    return _cb


def _make_state_cb(verbose: bool):
    def _cb(old: ExecutorState, new: ExecutorState) -> None:
        if verbose:
            print(f"  State: {old.name} → {new.name}")
    return _cb


# ── Validation helper (for --validate) ──────────────────────────────────


def _validate_steps(steps: list[WorkflowStep]) -> list[str]:
    """Check that each step's instrument and method exist on mock drivers.

    Returns a list of human-readable error strings (empty = all valid).
    """
    mgr = create_manager(mock=True)
    registered = set(mgr.names)
    errors: list[str] = []

    for step in steps:
        if step.instrument not in registered:
            errors.append(
                f"Step '{step.name}': instrument '{step.instrument}' not registered"
            )
            continue
        inst = mgr.get(step.instrument)
        if not hasattr(inst, step.method):
            errors.append(
                f"Step '{step.name}': method '{step.method}' not found on '{step.instrument}'"
            )

    return errors


# ── Main ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``softae-run``.

    Parameters
    ----------
    argv : list[str] | None
        Override ``sys.argv[1:]`` for testing.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for workflow error, 2 for bad args,
        3 for validation failures.
    """
    use_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── 1. Parse workflow ────────────────────────────────────────────
    try:
        workflow = parse_file(args.workflow)
    except (FileNotFoundError, Exception) as exc:
        print(f"Error loading workflow: {exc}", file=sys.stderr)
        return 2

    total = workflow.total_steps
    print(f"Workflow: {workflow.name}")
    print(f"  Description: {workflow.description}")
    print(f"  Steps: {total} ({len(workflow.setup)} setup, "
          f"{len(workflow.loop_steps)}×{workflow.iterations} loop, "
          f"{len(workflow.teardown)} teardown)")

    # ── 1b. Config hash ──────────────────────────────────────────────
    try:
        c_hash = cfg.config_hash()
        c_path = cfg.config_path()
    except FileNotFoundError:
        c_hash = "<no config file found>"
        c_path = None

    if args.verbose or args.dry_run or args.validate:
        print(f"  Config hash: {c_hash}")
        if c_path:
            print(f"  Config path: {c_path}")

    # ── 2. Dry-run: validate and exit ────────────────────────────────
    if args.dry_run or args.validate:
        resolved = workflow.resolve_steps()
        print("\n  [dry-run] Resolved steps:")
        for i, step in enumerate(resolved):
            print(f"    {i + 1}. {step.name} → {step.instrument}.{step.method}()")

        if args.validate:
            errors = _validate_steps(resolved)
            if errors:
                print(f"\n  [validate] {len(errors)} error(s):")
                for err in errors:
                    print(f"    ✗ {err}", file=sys.stderr)
                return 3
            print("\n  [validate] ✓ All instruments and methods valid.")

        print("\n  Validation passed. No execution.")
        return 0

    # ── 3. Create instrument manager ─────────────────────────────────
    if args.real:
        mock_flag: bool | None = False
    elif args.mock:
        mock_flag = True
    else:
        mock_flag = None  # auto-detect

    manager = create_manager(mock=mock_flag)

    # ── 4. Connect instruments ───────────────────────────────────────
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(manager.connect_all())
    except Exception as exc:
        print(f"Error connecting instruments: {exc}", file=sys.stderr)
        return 1

    # ── 5. Set up executor + logger ──────────────────────────────────
    exp_logger = ExperimentLogger(args.log_dir, workflow.name)
    exp_logger.log_event(
        "config_hash",
        hash=c_hash,
        path=str(c_path) if c_path else None,
    )
    executor = WorkflowExecutor(manager, experiment_logger=exp_logger)

    executor.on_step_start = _make_step_start_cb(args.verbose)
    executor.on_step_complete = _make_step_complete_cb(args.verbose)
    executor.on_step_error = _make_step_error_cb(args.verbose)
    executor.on_state_change = _make_state_cb(args.verbose)

    # ── 6. Run ───────────────────────────────────────────────────────
    t0 = time.monotonic()
    exit_code = 0
    try:
        loop.run_until_complete(executor.run(workflow))
        elapsed = time.monotonic() - t0
        print(f"\n  Workflow completed in {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\n  Interrupted — aborting workflow...")
        executor.abort()
        # Give teardown a chance to run
        try:
            loop.run_until_complete(asyncio.sleep(0.5))
        except Exception:
            pass
        exit_code = 130  # standard SIGINT exit code

    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"\n  Workflow failed after {elapsed:.1f}s: {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        exp_logger.close()
        loop.run_until_complete(manager.disconnect_all())
        loop.close()

    if exit_code == 0:
        print(f"  Log: {exp_logger.log_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
