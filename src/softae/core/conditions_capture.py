"""Capture environmental setpoints/process-values at measurement time.

The DataStore ``conditions`` table has always had room for temperature and
humidity SP/PVs, but nothing populated it.  :func:`read_environment` reads the
five available values off the live instrument manager so the EIS write sites can
persist them via :meth:`softae.core.data_store.DataStore.record_conditions`.

Mapping (confirmed against the rig): the temperature controller reads the
**stage** over Modbus (one SP, one PV); the humidity controller's onboard
air-temperature reading is the **chamber** PV.

======================  =================================  ================
DataStore column        Driver read                         Physical value
======================  =================================  ================
``stage_temp_sp_C``     ``temp_controller.get_sp()``        stage SP
``stage_temp_pv_C``     ``temp_controller.get_pv()``        stage PV (Modbus)
``chamber_air_C``       ``rh_controller.get_T()``           chamber air (RH sensor)
``rh_sp_pct``           ``rh_controller.status()['setpoint']``  RH SP
``rh_pv_pct``           ``rh_controller.get_H()``           RH PV
======================  =================================  ================

The NI-DAQ surface thermocouple (``temp_controller.get_pv_surf()``) is *not*
used — it returns ``NaN`` when no DAQ thermocouple is wired, which is the rig's
configuration.  Every read is best-effort: a missing controller, a driver
error, or a NaN reading maps to ``None`` so a snapshot is always safe to record
and never blocks a run.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import structlog

logger = structlog.get_logger(__name__)


class Environment(TypedDict):
    """Temp/humidity SP+PVs, keyed to match ``record_conditions`` kwargs exactly.

    A caller can splat the result: ``store.record_conditions(mid, stage, **env)``.
    """

    stage_temp_sp_C: float | None
    chamber_air_C: float | None
    stage_temp_pv_C: float | None
    rh_sp_pct: float | None
    rh_pv_pct: float | None


# Keys mirror ``DataStore.record_conditions`` keyword arguments exactly.
ENV_KEYS = (
    "stage_temp_sp_C", "chamber_air_C", "stage_temp_pv_C", "rh_sp_pct", "rh_pv_pct",
)

TEMP_CONTROLLER = "temp_controller"
RH_CONTROLLER = "rh_controller"


def _clean(value: Any) -> float | None:
    """Coerce a driver reading to a finite float, or ``None``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _get(manager: Any, name: str) -> Any | None:
    """Return the named instrument, or ``None`` if it is not registered."""
    if manager is None:
        return None
    try:
        if name not in manager.names:
            return None
        return manager.get(name)
    except Exception:  # defensive: never let capture interfere with a run
        return None


def read_environment(manager: Any) -> Environment:
    """Read the five temp/humidity SP/PVs off *manager*.

    Returns an :class:`Environment` (all five keys always present); any value
    that can't be read is ``None``.  Never raises.
    """
    env: Environment = {
        "stage_temp_sp_C": None,
        "chamber_air_C": None,
        "stage_temp_pv_C": None,
        "rh_sp_pct": None,
        "rh_pv_pct": None,
    }

    temp = _get(manager, TEMP_CONTROLLER)
    if temp is not None:
        env["stage_temp_sp_C"] = _safe_call(temp, "get_sp")  # stage SP
        env["stage_temp_pv_C"] = _safe_call(temp, "get_pv")  # stage PV (Modbus)

    rh = _get(manager, RH_CONTROLLER)
    if rh is not None:
        env["chamber_air_C"] = _safe_call(rh, "get_T")       # chamber air (RH sensor)
        env["rh_pv_pct"] = _safe_call(rh, "get_H")           # RH PV
        env["rh_sp_pct"] = _rh_setpoint(rh)                  # RH SP

    return env


def _safe_call(obj: Any, method: str) -> float | None:
    """Call ``obj.method()`` and clean the result; ``None`` on any failure."""
    fn = getattr(obj, method, None)
    if fn is None:
        return None
    try:
        return _clean(fn())
    except Exception:
        logger.debug("env_read_failed", method=method)
        return None


def _rh_setpoint(rh: Any) -> float | None:
    """Read the RH setpoint via ``status()['setpoint']`` (no public getter)."""
    try:
        status = rh.status()
    except Exception:
        return None
    if isinstance(status, dict):
        return _clean(status.get("setpoint"))
    return None
