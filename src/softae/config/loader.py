"""Configuration loader for SoftAE.

Reads ``softae_config.toml`` and exposes its sections as typed helpers.
Uses stdlib ``tomllib`` (Python 3.11+) for reading; ``tomli_w`` for writing.

Lookup order for the config file:
    1. Explicit path passed to :func:`load`
    2. ``SOFTAE_CONFIG`` environment variable
    3. ``softae_config.toml`` in the current working directory
    4. ``softae_config.toml`` next to this package's install root
"""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_FILENAME = "softae_config.toml"

# Cached parsed config (populated by load())
_config: dict[str, Any] | None = None
_config_path: Path | None = None
_config_hash: str | None = None


def _find_config_file(explicit_path: str | Path | None = None) -> Path:
    """Resolve the config file location using the lookup order."""
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Config file not found: {p}")

    env = os.environ.get("SOFTAE_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"SOFTAE_CONFIG points to missing file: {p}")

    cwd = Path.cwd() / _DEFAULT_FILENAME
    if cwd.is_file():
        return cwd

    # Fall back to the repo root (two levels up from src/softae/config/)
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent / _DEFAULT_FILENAME
    if pkg_root.is_file():
        return pkg_root

    raise FileNotFoundError(
        f"Cannot find {_DEFAULT_FILENAME}. "
        "Set SOFTAE_CONFIG env var or place the file in the working directory."
    )


def load(path: str | Path | None = None, *, reload: bool = False) -> dict[str, Any]:
    """Load and cache the TOML configuration.

    Parameters
    ----------
    path : str or Path, optional
        Explicit path to a config file.  Overrides automatic lookup.
    reload : bool
        If *True*, re‑read from disk even if already cached.

    Returns
    -------
    dict
        The parsed configuration dictionary.
    """
    global _config, _config_path, _config_hash
    if _config is not None and not reload:
        return _config

    resolved = _find_config_file(path)
    logger.info("loading_config", path=str(resolved))

    raw = resolved.read_bytes()
    _config_path = resolved
    _config_hash = hashlib.sha256(raw).hexdigest()
    _config = tomllib.loads(raw.decode("utf-8"))

    return _config


def get(section: str, key: str | None = None, *, default: Any = None) -> Any:
    """Retrieve a value from the cached config.

    Parameters
    ----------
    section : str
        Dot-separated section path, e.g. ``"instruments.stage"`` or ``"safety"``.
    key : str, optional
        Key within the section.  If *None*, returns the entire section dict.
    default : Any
        Fallback if the key is missing.

    Examples
    --------
    >>> cfg.get("instruments.stage", "port")
    'ASRL7::INSTR'
    >>> cfg.get("eis_presets.Quick")
    {'npts': 25, 'f_hi': 200000, ...}
    """
    cfg = load()  # ensure loaded
    parts = section.split(".")
    node: Any = cfg
    for part in parts:
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return default
        if node is None:
            return default

    if key is None:
        return node
    if isinstance(node, dict):
        return node.get(key, default)
    return default


def instruments() -> dict[str, dict[str, Any]]:
    """Return the ``[instruments.*]`` section as a flat dict of instrument configs."""
    cfg = load()
    return cfg.get("instruments", {})


def pcb_configs() -> dict[str, dict[str, Any]]:
    """Return all ``[pcb.*]`` entries."""
    cfg = load()
    return cfg.get("pcb", {})


def default_pcb_name() -> str | None:
    """The PCB selected by default (GUI open + headless campaigns).

    Reads the top-level ``default_pcb`` config key when it names a known board;
    otherwise falls back to the first PCB alphabetically.  Single source of truth
    so every surface (Init, HT, Live BO, ``resolve_pcb``) opens on the same board.
    """
    cfg = load()
    pcbs = cfg.get("pcb", {})
    if not pcbs:
        return None
    configured = cfg.get("default_pcb")
    if isinstance(configured, str) and configured in pcbs:
        return configured
    return sorted(pcbs)[0]


def eis_presets() -> dict[str, dict[str, Any]]:
    """Return all ``[eis_presets.*]`` entries."""
    cfg = load()
    return cfg.get("eis_presets", {})


def safety() -> dict[str, Any]:
    """Return the ``[safety]`` section."""
    cfg = load()
    return cfg.get("safety", {})


def optimizer_tuning() -> dict[str, Any]:
    """Return the ``[optimizer]`` section — the SITE default for T1.3's knobs.

    Absent section → an empty dict, which
    :func:`~softae.core.autonomous_wiring.build_optimizer` reads as "unset" and
    therefore as today's constructor defaults. It does **not** substitute values
    here: a second place that knows what the default is would agree today and
    drift the moment the rule changes in one of them.
    """
    cfg = load()
    return cfg.get("optimizer", {}) or {}


def feasibility_config() -> dict[str, Any]:
    """Return the ``[feasibility]`` section (T3.1). Absent → ``{}`` → off.

    Same posture as :func:`optimizer_tuning`: this reports what the file says and
    nothing more. The defaults, the floor validation and the derived clamp all
    live in :class:`~softae.optimizers.feasibility.FeasibilityConfig`, so a knob
    cannot mean one thing in the config layer and another in the optimizer.
    """
    cfg = load()
    return cfg.get("feasibility", {}) or {}


def campaign_config() -> dict[str, Any]:
    """Return the ``[campaign]`` section (stage 5, S5.F). Absent → ``{}``.

    The section is **optional and currently absent from the shipped file**: a
    headless campaign must run identically with it missing, so every accessor
    below resolves its own default rather than requiring the block to exist.
    """
    cfg = load()
    section = cfg.get("campaign", {})
    return section if isinstance(section, dict) else {}


def _campaign_seconds(key: str, default: float) -> float:
    """Read a ``[campaign]`` cadence in seconds. ``0`` disables; junk → default.

    ``0`` is a *value*, not an absence — it is how both cadences are switched
    off — so it must survive the fallback, which is why this tests for presence
    rather than for truthiness.
    """
    raw = campaign_config().get(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("campaign_cadence_not_a_number", key=key, value=raw)
        return float(default)


def campaign_conditions_poll_s() -> float:
    """Seconds between ``conditions.json`` republishes. ``0`` disables it.

    The default lives with the publisher, not here — a second copy of the number
    would agree today and drift the day one of them changes.
    """
    from softae.core.campaign_events import DEFAULT_CONDITIONS_POLL_S

    return _campaign_seconds("conditions_poll_s", DEFAULT_CONDITIONS_POLL_S)


def campaign_heartbeat_s() -> float:
    """Seconds between ``events.jsonl`` heartbeats. ``0`` disables the beat.

    Surfaced from config for the first time here. Note that the three-beat
    staleness rule a watcher applies is expressed in *beats*, so moving this
    moves the verdict with it — which is exactly why the conditions cadence is a
    separate knob on a separate clock.
    """
    from softae.core.campaign_events import DEFAULT_HEARTBEAT_S

    return _campaign_seconds("heartbeat_s", DEFAULT_HEARTBEAT_S)


def channel_routing() -> dict[str, Any]:
    """Return the ``[channel_routing]`` section.

    Defaults to pico1 for channels 1–16 and pico2 for 17–32 if the
    section is absent from the config file.
    """
    cfg = load()
    return cfg.get("channel_routing", {
        "pico1_range": [1, 16],
        "pico2_range": [17, 32],
    })


def liquid_handling_config() -> dict[str, Any]:
    """Return the ``[liquid_handling]`` section with safe defaults."""
    cfg = load()
    section = cfg.get("liquid_handling", {})

    defaults: dict[str, Any] = {
        "enabled": False,
        "beta": 0.30,
        "eta_ref_mpas": 1.0,
        "alpha_growth_per_run": 0.0,
        "pump_line": {"0": 0, "1": 1},
        "line": {
            "0": {
                "cracking_kpa_per_valve": 8.0,
                "compliance_uL_per_kpa": 0.55,
                "alpha_base": 0.20,
                "viscosity_mpas": 1.0,
            },
            "1": {
                "cracking_kpa_per_valve": 8.0,
                "compliance_uL_per_kpa": 0.55,
                "alpha_base": 0.20,
                "viscosity_mpas": 1.0,
            },
        },
    }

    merged = dict(defaults)
    merged.update(section)

    pump_line = dict(defaults["pump_line"])
    pump_line.update(section.get("pump_line", {}))
    merged["pump_line"] = pump_line

    line_cfg = dict(defaults["line"])
    for line_id, values in section.get("line", {}).items():
        base = dict(defaults["line"].get(str(line_id), {}))
        base.update(values)
        line_cfg[str(line_id)] = base
    merged["line"] = line_cfg
    return merged


def dropcast_config() -> dict[str, Any]:
    """Return the ``[dropcast]`` section with safe defaults.

    Drives the two-phase cast (precondition flush → deposition) in the HT and
    Autonomous tabs: default flow rates, the precondition preload multiplier, the
    derived settling-wait multiplier, and the per-pump start-flush volumes.  All
    keys are overridable in ``softae_config.toml``; the defaults reproduce the
    values the HT tab shipped with before the section existed.
    """
    cfg = load()
    section = cfg.get("dropcast", {})
    if not isinstance(section, dict):
        section = {}

    defaults: dict[str, Any] = {
        "dispense_rate_uL_min": 75.0,
        "line_flush_rate_uL_min": 500.0,
        "flush_factor": 3.0,
        "settle_factor": 2.0,
        "settle_base_s": 0.0,
        "start_flush_uL": [80.0, 80.0, 80.0],
        # Default deposition recipe the HT tab starts on: "legacy" (role-based
        # per-pump path), "single_drop", or "two_phase". Flip to "two_phase" after
        # rig validation to make the two-phase cast the default (cutover Runway A).
        "default_recipe": "legacy",
        # Deprecated alias honored only when default_recipe is unset: true → two_phase.
        "default_two_phase": False,
    }
    merged = dict(defaults)
    merged.update(section)
    # start_flush_uL must be a list; coerce a scalar or bad value to the default.
    sf = merged.get("start_flush_uL")
    if not isinstance(sf, (list, tuple)) or not sf:
        merged["start_flush_uL"] = list(defaults["start_flush_uL"])
    else:
        merged["start_flush_uL"] = [float(v) for v in sf]
    return merged


def set_dropcast_default_recipe(recipe: str) -> None:
    """Persist ``[dropcast].default_recipe`` to the config file, comment-preserving.

    Replaces the ``default_recipe`` line inside ``[dropcast]`` (appending it to the
    section, or creating the section, if absent), then reloads the config cache.
    """
    cfg_path = config_path()
    lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_line = f'default_recipe = "{recipe}"\n'

    result: list[str] = []
    in_section = False
    written = False
    for line in lines:
        stripped = line.strip()
        is_header = stripped.startswith("[") and stripped.endswith("]")
        if is_header and in_section and not written:
            # Leaving [dropcast] without a default_recipe line → append it here.
            result.append(new_line)
            written = True
        if is_header:
            in_section = stripped == "[dropcast]"
        if in_section and stripped.startswith("default_recipe"):
            indent = line[: len(line) - len(line.lstrip())]
            result.append(f"{indent}{new_line}")
            written = True
            continue
        result.append(line)
    if in_section and not written:      # file ended inside [dropcast]
        result.append(new_line)
        written = True
    if not written:                     # no [dropcast] section at all
        if result and not result[-1].endswith("\n"):
            result.append("\n")
        result.append(f"\n[dropcast]\n{new_line}")

    cfg_path.write_text("".join(result), encoding="utf-8")
    load(path=cfg_path, reload=True)


def piezo_config() -> dict[str, Any]:
    """Return effective piezo config with canonical enablement semantics.

    Canonical enable source is ``[instruments.piezo].enabled`` when present.
    Legacy ``[piezo].enabled`` is still honored only when the canonical key is
    absent, with a warning to aid migration.
    """
    cfg = load()
    instruments_section = cfg.get("instruments", {})
    if not isinstance(instruments_section, dict):
        instruments_section = {}
    inst_section = instruments_section.get("piezo", {})
    if not isinstance(inst_section, dict):
        inst_section = {}

    section = cfg.get("piezo", {})
    if not isinstance(section, dict):
        section = {}

    defaults: dict[str, Any] = {
        "enabled": False,
        "channel": "A",
        "frequency_hz": 500,
        "sweep_on_s": 2.0,
        "sweep_rest_s": 3.0,
        "liquid_events": {
            "enabled": False,
            "settings_source": "manual_profile",
            "channel_a": True,
            "frequency_hz": 500,
            "sweep_on_s": 2.0,
            "sweep_rest_s": 3.0,
        },
    }

    merged = dict(defaults)
    merged.update(inst_section)
    merged.update({k: v for k, v in section.items() if k != "liquid_events"})

    if "enabled" in inst_section:
        merged["enabled"] = bool(inst_section.get("enabled", False))
    elif "enabled" in section:
        # Legacy compatibility path for pre-canonical configs.
        logger.warning(
            "piezo_legacy_enabled_key_used",
            key="piezo.enabled",
            canonical_key="instruments.piezo.enabled",
        )
        merged["enabled"] = bool(section.get("enabled", False))

    events = dict(defaults["liquid_events"])
    sec_events = section.get("liquid_events", {})
    if isinstance(sec_events, dict):
        events.update(sec_events)
    merged["liquid_events"] = events
    return merged


def liquid_line_for_pump(pump_id: int) -> int:
    """Return line-id mapped to the given pump-id (defaults to identity)."""
    sec = liquid_handling_config()
    mapping = sec.get("pump_line", {})
    mapped = mapping.get(str(pump_id), mapping.get(pump_id, pump_id))
    try:
        return int(mapped)
    except (TypeError, ValueError):
        return int(pump_id)


def pico_for_channel(ch: int) -> str:
    """Return the pico instrument name for a 1-based channel number.

    Uses ``[channel_routing]`` ranges from config.  Raises ``ValueError``
    if the channel is outside all configured ranges.
    """
    routing = channel_routing()
    p1 = routing.get("pico1_range", [1, 16])
    p2 = routing.get("pico2_range", [17, 32])
    if p1[0] <= ch <= p1[1]:
        return "pico1"
    if p2[0] <= ch <= p2[1]:
        return "pico2"
    raise ValueError(
        f"Channel {ch} is not mapped to any pico instrument. "
        f"Configured ranges: pico1={p1}, pico2={p2}"
    )


# ── Data directory helpers ──────────────────────────────────────────────


def web_port() -> int:
    """Return the ``[web] port`` value (default ``8050``)."""
    cfg = load()
    return int(cfg.get("web", {}).get("port", 8050))


def data_root() -> Path:
    """Return the catalog data directory as an ABSOLUTE, expanded path.

    Resolution order:
      1. Read ``[paths] data_root`` from config (default ``"./data"``).
      2. ``expanduser()`` it (honours a leading ``~``).
      3. If the resulting path is absolute, return it resolved as-is.
      4. Otherwise (relative, e.g. the default ``./data``) anchor it at the
         config file's directory: ``config_path().parent / raw``, resolved.
      5. If no config file can be found (``load()`` raises ``FileNotFoundError``),
         fall back to ``Path.cwd() / raw`` (default ``./data``), resolved.

    Never raises: a missing config degrades to the CWD-anchored default so a
    standalone launch from an arbitrary directory still yields a usable path.
    The directory is NOT created here (callers create it on save).
    """
    raw = "./data"
    anchor = Path.cwd()
    try:
        raw = str(get("paths", "data_root", default="./data") or "./data")
        anchor = config_path().parent
    except FileNotFoundError:
        pass  # no config anywhere → CWD-anchored default
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (anchor / p).resolve()


def tasks_toml_path() -> Path:
    """Return the canonical task-catalog file path (``data_root()/tasks.toml``).

    Centralises the location the Process Configuration tab reads/writes its
    :class:`~softae.core.task_catalog.TaskCatalog`.  The file need not exist;
    ``TaskCatalog.load_toml`` degrades a missing file to an empty catalog.
    """
    return data_root() / "tasks.toml"


def recipes_toml_path() -> Path:
    """Return the canonical recipe-registry file path (``data_root()/recipes.toml``).

    Holds named runnable recipes with lifecycle metadata (see
    :class:`~softae.core.recipe_registry.RecipeRegistry`).  The file need not
    exist; ``RecipeRegistry.load_toml`` degrades a missing file to an empty
    registry.
    """
    return data_root() / "recipes.toml"


def recipe_workflows_dir() -> Path:
    """Directory holding the workflow YAMLs backing registered recipes.

    ``data_root()/recipe_workflows/`` — the Process Studio writes ``<name>.yaml``
    here when registering a built recipe, and the recipe entry's
    ``workflow_path`` points at it.  Created on first save.
    """
    return data_root() / "recipe_workflows"


def data_project_dir() -> str:
    """Return the ``[data] project_dir`` value (default ``~/softae_data``)."""
    cfg = load()
    return cfg.get("data", {}).get("project_dir", "~/softae_data")


def data_db_filename() -> str:
    """Return the ``[data] db_filename`` value (default ``softae.db``)."""
    cfg = load()
    return cfg.get("data", {}).get("db_filename", "softae.db")


def data_auto_save_eis() -> bool:
    """Return the ``[data] auto_save_eis`` flag (default ``True``)."""
    cfg = load()
    return cfg.get("data", {}).get("auto_save_eis", True)


def config_path() -> Path:
    """Return the resolved path of the loaded config file."""
    load()  # ensure loaded
    assert _config_path is not None
    return _config_path


def config_hash() -> str:
    """Return the SHA-256 hex digest of the loaded config file."""
    load()  # ensure loaded
    assert _config_hash is not None
    return _config_hash


def log_level() -> str:
    """Return the ``[logging] level`` value (default ``"INFO"``)."""
    cfg = load()
    return cfg.get("logging", {}).get("level", "INFO").upper()


def save_pico_ports(pico1_port: str, pico2_port: str) -> None:
    """Write pico1/pico2 port assignments back to ``softae_config.toml``.

    Performs a line-by-line replacement so all comments are preserved.
    Reloads the in-memory config cache afterwards.

    Parameters
    ----------
    pico1_port : str
        COM port (or ``"auto"``) to assign to the ``pico1`` instrument.
    pico2_port : str
        COM port (or ``"auto"``) to assign to the ``pico2`` instrument.

    Raises
    ------
    FileNotFoundError
        If the config file cannot be located.
    """
    global _config

    cfg_path = config_path()  # use the already-resolved cached path
    lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

    port_map: dict[str, str] = {"pico1": pico1_port, "pico2": pico2_port}
    current_section: str | None = None
    result: list[str] = []

    for line in lines:
        # Track which [instruments.picoN] section we're inside
        m = re.match(r"^\[instruments\.(pico\d+)\]", line.strip())
        if m:
            current_section = m.group(1)
        elif line.strip().startswith("["):
            current_section = None

        # Replace port value when inside a targeted section
        if current_section in port_map and re.match(r"\s*port\s*=", line):
            new_port = port_map[current_section]
            # Preserve leading whitespace and any inline comment
            indent_m = re.match(r"^(\s*)", line)
            indent = indent_m.group(1) if indent_m else ""
            comment_m = re.search(r"(#.*)$", line)
            comment = "  " + comment_m.group(1) if comment_m else ""
            line = f'{indent}port     = "{new_port}"{comment}\n'

        result.append(line)

    cfg_path.write_text("".join(result), encoding="utf-8")

    # Reload from the same file (do NOT re-discover; pass explicit path).
    _config = None
    load(path=cfg_path, reload=True)
    logger.info("pico_ports_saved", pico1=pico1_port, pico2=pico2_port)


def stage_calibration() -> dict[str, float]:
    """Return Home, Dep-1, Flush, and Wick calibration coordinates from config.

    Defaults to ``home=(0.0, 0.0)``, ``dep1=(43.5, 50.0)``, and
    ``flush=(0.0, 0.0)`` / ``wick=(0.0, 0.0)`` if the ``[stage_calibration]``
    section (or an individual key) is absent.
    """
    cfg = load()
    sec = cfg.get("stage_calibration", {})
    return {
        "home_x": float(sec.get("home_x", 0.0)),
        "home_y": float(sec.get("home_y", 0.0)),
        "dep1_x": float(sec.get("dep1_x", 43.5)),
        "dep1_y": float(sec.get("dep1_y", 50.0)),
        "flush_x": float(sec.get("flush_x", 0.0)),
        "flush_y": float(sec.get("flush_y", 0.0)),
        "wick_x": float(sec.get("wick_x", 0.0)),
        "wick_y": float(sec.get("wick_y", 0.0)),
    }


def syringe_parallel_count(pump_id: int | None = None) -> int:
    """Return configured parallel syringe count.

    When *pump_id* is omitted, returns the legacy global count. When
    *pump_id* is provided, returns that pump's explicit count if present,
    otherwise falls back to the legacy global count.
    """
    cfg = load()
    sec = cfg.get("instruments", {}).get("syringe", {})
    fallback = 1
    try:
        fallback = max(1, int(sec.get("parallel_syringes", 1)))
    except (TypeError, ValueError):
        fallback = 1
    if pump_id is None:
        return fallback
    key = f"parallel_syringes_pump{int(pump_id)}"
    try:
        return max(1, int(sec.get(key, fallback)))
    except (TypeError, ValueError):
        return fallback


def syringe_parallel_counts() -> dict[int, int]:
    """Return configured parallel syringe count per pump (defaults to the global count)."""
    cfg = load()
    sec = cfg.get("instruments", {}).get("syringe", {})
    fallback = syringe_parallel_count()
    counts: dict[int, int] = {}
    for pump_id in range(3):
        key = f"parallel_syringes_pump{pump_id}"
        try:
            counts[pump_id] = max(1, int(sec.get(key, fallback)))
        except (TypeError, ValueError):
            counts[pump_id] = fallback
    return counts


def save_syringe_parallel_counts(counts: dict[int, int]) -> None:
    """Persist per-pump ``parallel_syringes_pump<N>`` values in config."""
    global _config

    normalised: dict[int, int] = {}
    for pump_id in range(3):
        value = counts.get(pump_id, syringe_parallel_count())
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parallel syringe count for pump {pump_id} must be an integer") from exc
        if value < 1:
            raise ValueError(f"parallel syringe count for pump {pump_id} must be >= 1")
        normalised[pump_id] = value

    cfg_path = config_path()
    lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

    result: list[str] = []
    in_section = False
    section_found = False
    updated: set[int] = set()
    key_pattern = re.compile(r"^\s*parallel_syringes_pump([0-2])\s*=\s*")

    for line in lines:
        stripped = line.strip()
        if re.match(r"^\[instruments\.syringe\]", stripped):
            in_section = True
            section_found = True
        elif stripped.startswith("["):
            in_section = False

        if in_section:
            m = key_pattern.match(line)
            if m:
                pump_id = int(m.group(1))
                indent_m = re.match(r"^(\s*)", line)
                indent = indent_m.group(1) if indent_m else ""
                line = f"{indent}parallel_syringes_pump{pump_id} = {normalised[pump_id]}\n"
                updated.add(pump_id)

        result.append(line)

    def _section_insert_index() -> int | None:
        for idx, line in enumerate(result):
            if re.match(r"^\[instruments\.syringe\]", line.strip()):
                insert_idx = idx + 1
                while insert_idx < len(result) and not result[insert_idx].strip().startswith("["):
                    insert_idx += 1
                return insert_idx
        return None

    if not section_found:
        if result and not result[-1].endswith("\n"):
            result[-1] = result[-1] + "\n"
        result.append("\n[instruments.syringe]\n")
        for pump_id in range(3):
            result.append(f"parallel_syringes_pump{pump_id} = {normalised[pump_id]}\n")
    else:
        insert_idx = _section_insert_index()
        missing = [pump_id for pump_id in range(3) if pump_id not in updated]
        if missing:
            block = [f"parallel_syringes_pump{pump_id} = {normalised[pump_id]}\n" for pump_id in missing]
            if insert_idx is None:
                result.append("\n[instruments.syringe]\n")
                result.extend(block)
            else:
                for offset, line in enumerate(block):
                    result.insert(insert_idx + offset, line)

    # Keep the legacy global key aligned to pump 0 for backward compatibility.
    global_value = normalised[0]
    global_updated = False
    result2: list[str] = []
    in_section = False
    for line in result:
        stripped = line.strip()
        if re.match(r"^\[instruments\.syringe\]", stripped):
            in_section = True
        elif stripped.startswith("["):
            in_section = False
        if in_section and re.match(r"^\s*parallel_syringes\s*=\s*", line):
            indent_m = re.match(r"^(\s*)", line)
            indent = indent_m.group(1) if indent_m else ""
            line = f"{indent}parallel_syringes = {global_value}\n"
            global_updated = True
        result2.append(line)
    if not global_updated:
        inserted = False
        final: list[str] = []
        in_section = False
        for line in result2:
            final.append(line)
            if re.match(r"^\[instruments\.syringe\]", line.strip()):
                in_section = True
                continue
            if in_section and line.strip().startswith("parallel_syringes_pump") and not inserted:
                final.insert(len(final) - 1, f"parallel_syringes = {global_value}\n")
                inserted = True
                in_section = False
        if not inserted:
            final.extend([f"parallel_syringes = {global_value}\n"])
        result2 = final

    cfg_path.write_text("".join(result2), encoding="utf-8")
    _config = None
    load(path=cfg_path, reload=True)
    logger.info("syringe_parallel_saved", parallel_syringes=normalised)


def save_liquid_handling_config(section: dict[str, Any]) -> None:
    """Persist the ``[liquid_handling]`` correction parameters in config."""
    global _config

    cfg_path = config_path()
    lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

    system_keys = {"enabled", "beta", "eta_ref_mpas", "alpha_growth_per_run", "valves_in_series"}
    line_keys = {"cracking_kpa_per_valve", "compliance_uL_per_kpa", "alpha_base", "viscosity_mpas"}
    section_values = dict(section)
    line_values = section_values.get("line", {}) if isinstance(section_values.get("line", {}), dict) else {}

    def _fmt(key: str, value: Any) -> str:
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, str):
            literal = f'"{value}"'
        else:
            literal = f"{value}"
        return literal

    def _update_section(section_name: str, values: dict[str, Any], allowed_keys: set[str], text_lines: list[str]) -> list[str]:
        out: list[str] = []
        in_section = False
        section_found = False
        seen: set[str] = set()
        for line in text_lines:
            stripped = line.strip()
            if re.match(rf"^\[{re.escape(section_name)}\]", stripped):
                in_section = True
                section_found = True
            elif stripped.startswith("["):
                in_section = False
            if in_section:
                m = re.match(r"^(\s*)([A-Za-z0-9_]+)\s*=\s*", line)
                if m and m.group(2) in allowed_keys and m.group(2) in values:
                    indent = m.group(1)
                    key = m.group(2)
                    line = f"{indent}{key} = {_fmt(key, values[key])}\n"
                    seen.add(key)
            out.append(line)
        if not section_found:
            if out and not out[-1].endswith("\n"):
                out[-1] = out[-1] + "\n"
            out.append(f"\n[{section_name}]\n")
            for key in allowed_keys:
                if key in values:
                    out.append(f"{key} = {_fmt(key, values[key])}\n")
            return out
        missing = [key for key in allowed_keys if key in values and key not in seen]
        if missing:
            insert_idx = None
            for idx, line in enumerate(out):
                if re.match(rf"^\[{re.escape(section_name)}\]", line.strip()):
                    insert_idx = idx + 1
                    while insert_idx < len(out) and not out[insert_idx].strip().startswith("["):
                        insert_idx += 1
                    break
            if insert_idx is None:
                out.extend([f"{key} = {_fmt(key, values[key])}\n" for key in missing])
            else:
                for offset, key in enumerate(missing):
                    out.insert(insert_idx + offset, f"{key} = {_fmt(key, values[key])}\n")
        return out

    result = _update_section("liquid_handling", {k: v for k, v in section_values.items() if k in system_keys}, system_keys, lines)
    for line_id, values in line_values.items():
        if not isinstance(values, dict):
            continue
        result = _update_section(
            f"liquid_handling.line.{line_id}",
            {k: v for k, v in values.items() if k in line_keys},
            line_keys,
            result,
        )

    cfg_path.write_text("".join(result), encoding="utf-8")
    _config = None
    load(path=cfg_path, reload=True)
    logger.info("liquid_handling_saved")


def save_piezo_config(section: dict[str, Any]) -> None:
    """Persist piezo settings and keep canonical/legacy enable flags in sync.

    Root piezo settings are written to ``[piezo]`` and event settings to
    ``[piezo.liquid_events]``. The effective ``enabled`` flag is mirrored to
    both ``[instruments.piezo].enabled`` (canonical) and ``[piezo].enabled``
    (legacy compatibility).
    """
    global _config

    cfg_path = config_path()
    lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

    root_keys = {"enabled", "channel", "frequency_hz", "sweep_on_s", "sweep_rest_s"}
    event_keys = {"enabled", "settings_source", "channel_a", "frequency_hz", "sweep_on_s", "sweep_rest_s"}

    root_values = {k: v for k, v in dict(section).items() if k in root_keys}
    effective_enabled = bool(root_values.get("enabled", piezo_config().get("enabled", False)))
    root_values["enabled"] = effective_enabled
    event_values_raw = section.get("liquid_events", {})
    if not isinstance(event_values_raw, dict):
        event_values_raw = {}
    event_values = {k: v for k, v in event_values_raw.items() if k in event_keys}

    def _fmt(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        return f"{value}"

    def _update_section(section_name: str, values: dict[str, Any], keys: set[str], text_lines: list[str]) -> list[str]:
        out: list[str] = []
        in_section = False
        section_found = False
        seen: set[str] = set()

        for line in text_lines:
            stripped = line.strip()
            if re.match(rf"^\[{re.escape(section_name)}\]", stripped):
                in_section = True
                section_found = True
            elif stripped.startswith("["):
                in_section = False

            if in_section:
                m = re.match(r"^(\s*)([A-Za-z0-9_]+)\s*=\s*", line)
                if m and m.group(2) in keys and m.group(2) in values:
                    indent = m.group(1)
                    key = m.group(2)
                    line = f"{indent}{key} = {_fmt(values[key])}\n"
                    seen.add(key)
            out.append(line)

        if not section_found:
            if out and not out[-1].endswith("\n"):
                out[-1] = out[-1] + "\n"
            out.append(f"\n[{section_name}]\n")
            for key in sorted(keys):
                if key in values:
                    out.append(f"{key} = {_fmt(values[key])}\n")
            return out

        missing = [key for key in sorted(keys) if key in values and key not in seen]
        if not missing:
            return out

        insert_idx = None
        for idx, line in enumerate(out):
            if re.match(rf"^\[{re.escape(section_name)}\]", line.strip()):
                insert_idx = idx + 1
                while insert_idx < len(out) and not out[insert_idx].strip().startswith("["):
                    insert_idx += 1
                break

        if insert_idx is None:
            for key in missing:
                out.append(f"{key} = {_fmt(values[key])}\n")
            return out

        for offset, key in enumerate(missing):
            out.insert(insert_idx + offset, f"{key} = {_fmt(values[key])}\n")
        return out

    result = _update_section("piezo", root_values, root_keys, lines)
    result = _update_section("piezo.liquid_events", event_values, event_keys, result)
    result = _update_section(
        "instruments.piezo",
        {"enabled": effective_enabled},
        {"enabled"},
        result,
    )

    cfg_path.write_text("".join(result), encoding="utf-8")
    _config = None
    load(path=cfg_path, reload=True)
    logger.info("piezo_config_saved")



def save_stage_calibration(
    home_x: float,
    home_y: float,
    dep1_x: float,
    dep1_y: float,
    flush_x: float = 0.0,
    flush_y: float = 0.0,
    wick_x: float = 0.0,
    wick_y: float = 0.0,
) -> None:
    """Persist stage calibration coordinates to ``softae_config.toml``.

    Writes the Home, Dep-1, Flush, and Wick positions. If the
    ``[stage_calibration]`` section does not yet exist it is appended to the
    file; otherwise the values are updated in-place so that all other comments
    and sections are preserved. Keys absent from an existing section are
    appended to it.
    """
    global _config

    cfg_path = config_path()
    lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

    new_values = {
        "home_x": home_x,
        "home_y": home_y,
        "dep1_x": dep1_x,
        "dep1_y": dep1_y,
        "flush_x": flush_x,
        "flush_y": flush_y,
        "wick_x": wick_x,
        "wick_y": wick_y,
    }
    ordered_keys = (
        "home_x", "home_y", "dep1_x", "dep1_y",
        "flush_x", "flush_y", "wick_x", "wick_y",
    )
    key_pattern = "|".join(ordered_keys)
    updated_keys: set[str] = set()
    in_section = False
    result: list[str] = []
    section_end_idx: int | None = None

    for line in lines:
        if re.match(r"^\[stage_calibration\]", line.strip()):
            in_section = True
        elif line.strip().startswith("["):
            if in_section:
                # Record where the section ends so new keys land inside it.
                section_end_idx = len(result)
            in_section = False

        if in_section:
            m = re.match(rf"^(\s*)({key_pattern})\s*=", line)
            if m:
                key = m.group(2)
                indent = m.group(1)
                line = f"{indent}{key} = {new_values[key]!r}\n"
                updated_keys.add(key)

        result.append(line)

    # If the section or individual keys were missing, append them.
    missing = set(new_values) - updated_keys
    if missing:
        has_section = any(
            re.match(r"^\[stage_calibration\]", ln.strip()) for ln in result
        )
        new_key_lines = [
            f"{key} = {new_values[key]!r}\n" for key in ordered_keys if key in missing
        ]
        if not has_section:
            result.append("\n[stage_calibration]\n")
            result.extend(new_key_lines)
        elif section_end_idx is not None:
            # Section exists but ends before EOF — insert missing keys at its end.
            result[section_end_idx:section_end_idx] = new_key_lines
        else:
            # Section runs to EOF.
            result.extend(new_key_lines)

    cfg_path.write_text("".join(result), encoding="utf-8")
    _config = None
    load(path=cfg_path, reload=True)
    logger.info(
        "stage_calibration_saved",
        home_x=home_x, home_y=home_y, dep1_x=dep1_x, dep1_y=dep1_y,
        flush_x=flush_x, flush_y=flush_y, wick_x=wick_x, wick_y=wick_y,
    )
