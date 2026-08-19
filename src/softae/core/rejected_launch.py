"""What a refused launch leaves behind, so a refusal costs a file — not a setup.

Only one campaign may own the rig at a time, and a second launch attempt is
**refused outright**: never queued, never offered a takeover. That ruling is
easy to state and easy to make expensive. An operator who has just spent twenty
minutes entering a parameter space, a composition axis set, seed observations
and a board plan, and who is then told "the rig is busy", has been charged
twenty minutes for a schedule collision they could not have known about.

So the refusal writes the configuration down first, and this module is that
write. Two things go to ``<project>/rejected/``:

``<name>_<stamp>.json``
    The **panel state** — always written, always lossless. It is the same
    payload the tab's *Save Config…* button produces, so *Load Config…* restores
    the screen exactly, composition axes and all.

``<name>_<stamp>.toml``
    A **campaign spec**, written only when :func:`spec_toml_completeness` can
    prove the file carries the whole spec, and accompanied by the exact
    ``softae-campaign run`` command that would run it.

**The order of those two is the point.** The TOML is the more useful artifact —
it relaunches without a GUI — and it is also the one that can lie: a
composition campaign written as TOML loses ``general_formulation``, reloads with
no ``vol_params`` either, and searches its composition axes as raw µL volumes
without raising anything. Handing an operator a command that runs a different
experiment is worse than handing them no command at all, so the command is
offered only against a proof, and the panel state is what is always there.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from softae.core.campaign_spec_io import (
    SpecCompleteness,
    spec_toml_completeness,
    write_campaign_spec_toml,
)

logger = structlog.get_logger(__name__)

#: Where a refused launch is preserved, under the project directory.
REJECTED_DIRNAME = "rejected"


@dataclass(frozen=True)
class PreservedLaunch:
    """The files a refused launch left, and how to get back to it."""

    panel_state_path: Path | None = None
    toml_path: Path | None = None
    #: The exact command line, or ``None`` when the spec is not fully writable.
    command: str | None = None
    completeness: SpecCompleteness | None = None
    #: Anything that could not be written, said rather than swallowed.
    errors: tuple[str, ...] = ()

    def describe(self) -> str:
        """The operator-facing paragraph: nothing was lost, and here is the way back."""
        lines = ["Nothing was started, and nothing you entered was lost."]
        if self.panel_state_path is not None:
            lines += [
                "",
                f"Your configuration is saved at:\n    {self.panel_state_path}",
                'Reload it with "Load Config…" in this tab and press Run again '
                "once the rig is free.",
            ]
        if self.command:
            lines += [
                "",
                "Or run it from a terminal without the GUI, once the rig is free:",
                f"    {self.command}",
                "(`run` asks for the dispenser-head position; answer it up front "
                "with --head-up or --head-down.)",
            ]
        elif self.completeness is not None and not self.completeness.complete:
            lines += [
                "",
                "This campaign has no terminal command, because a spec file "
                "cannot carry all of it:",
                self.completeness.explain(),
                "Relaunching it through this tab is the only way to run the "
                "campaign you configured.",
            ]
        if self.errors:
            lines += ["", "Could not be written:"] + [f"  - {e}" for e in self.errors]
        return "\n".join(lines)


def _slug(text: str) -> str:
    """A filename stem that cannot escape the directory it is written into."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("._")
    return cleaned[:60] or "campaign"


def _stamp(now: datetime | None) -> str:
    return (now or datetime.now(tz=timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def _quoted(path: Any) -> str:
    text = str(path)
    return f'"{text}"' if any(c.isspace() for c in text) else text


def _relaunch_command(toml_path: Path, project_dir: Path, *, mock: bool,
                      head_up: bool | None) -> str:
    """The command that runs *toml_path*, in the form the CLI actually parses.

    Positional spec, no ``--spec`` flag (``tools/campaign.py`` ``build_parser``).
    ``--head-up`` / ``--head-down`` appear only when the head position is
    genuinely known: the refusal happens *before* the head gate, precisely so a
    rejected launch asks the operator nothing, so at that point it is not.
    """
    parts = ["softae-campaign", "run", _quoted(toml_path),
             "--project", _quoted(project_dir), "--yes"]
    if mock:
        parts.append("--mock")
    if head_up is not None:
        parts.append("--head-up" if head_up else "--head-down")
    return " ".join(parts)


def preserve_rejected_launch(
    *,
    project_dir: "str | Path",
    panel_state: dict[str, Any] | None = None,
    spec: Any = None,
    mock: bool = False,
    head_up: bool | None = None,
    now: datetime | None = None,
) -> PreservedLaunch:
    """Write a refused launch to disk and say how to get it back.

    Never raises. This runs on a path where the operator has already been told
    "no", and a traceback there would take away the one thing the refusal was
    supposed to leave them.
    """
    base = Path(project_dir).expanduser() / REJECTED_DIRNAME
    stem = _slug((panel_state or {}).get("name") or getattr(spec, "name", ""))
    stem = f"{stem}_{_stamp(now)}"
    errors: list[str] = []

    panel_path: Path | None = None
    if panel_state is not None:
        panel_path = base / f"{stem}.json"
        try:
            base.mkdir(parents=True, exist_ok=True)
            panel_path.write_text(json.dumps(panel_state, indent=2),
                                  encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{panel_path}: {exc}")
            panel_path = None

    if spec is None:
        return PreservedLaunch(panel_state_path=panel_path, errors=tuple(errors))

    completeness = spec_toml_completeness(spec)
    toml_path: Path | None = None
    command: str | None = None
    if completeness.complete:
        toml_path = base / f"{stem}.toml"
        try:
            write_campaign_spec_toml(spec, toml_path)
            command = _relaunch_command(toml_path, Path(project_dir).expanduser(),
                                        mock=mock, head_up=head_up)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{toml_path}: {exc}")
            toml_path = None

    logger.info("rejected_launch_preserved", campaign=getattr(spec, "name", ""),
                panel_state=str(panel_path or ""), spec_file=str(toml_path or ""),
                toml_complete=completeness.complete,
                missing=list(completeness.missing))
    return PreservedLaunch(panel_state_path=panel_path, toml_path=toml_path,
                           command=command, completeness=completeness,
                           errors=tuple(errors))
