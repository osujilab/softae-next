"""Which thermometer a temperature came from — one authority, because there are two.

This rig has **two independent temperature reads**, and the column names in the
``conditions`` table actively mislead about which is which:

======================  ====================================  ==========================
DataStore column        Instrument                            What it actually is
======================  ====================================  ==========================
``temp_sp_C``           temperature controller (Modbus)       stage **setpoint**
``stage_temp_pv_C``     temperature controller (Modbus)       stage **PV** — the sample
``temp_pv_C``           humidity controller (I²C sensor)      **chamber air**, not the sample
======================  ====================================  ==========================

``temp_sp_C`` and ``temp_pv_C`` read like an SP/PV pair off one instrument. They are
not: they come from two different instruments, and the stage's own PV sits in a third
column under a different prefix. That trap has already been sprung once — the
equilibration series selected ``temp_pv_C``, and on run ``20260811T023757Z`` the air
probe read up to 42 °C below the stage at the same setpoint. An activation energy
fitted on it is inflated by ~2.4×, because the 1/T span across the up leg compresses
from 5.34e-4 to 2.20e-4 while ln σ is unchanged. **The fit's R² stays healthy**, so
nothing downstream reports a problem.

So the decision "which thermometer is this temperature" lives here and nowhere else.
This project has already paid once for one physical quantity spelled several ways —
the σ arc — and the way that recurs is a second authority growing quietly inside a
consumer. Do not re-implement this precedence inside an analysis module.

.. note::
   **No column is renamed.** ``temp_pv_C`` still means chamber air despite its name,
   and every historical row keeps the meaning it was written with. Renaming the
   columns is a separate migration with its own compatibility problem; this module
   makes the naming survivable in the meantime by forcing every consumer to carry a
   *source label* alongside the number.
"""

from __future__ import annotations

import math

import structlog

logger = structlog.get_logger(__name__)

#: Nothing below this can be a temperature. A reading at or under absolute zero is a
#: sensor fault or an uninitialised register, never a measurement.
ABSOLUTE_ZERO_C = -273.15

#: Where a temperature came from, **best first**. The ordering *is* the precedence
#: used by :func:`resolve_temperature_C`, exactly as ``THICKNESS_METHODS`` is for
#: :func:`~softae.analysis.eis.geometry.resolve_thickness_cm`.
#:
#: **The SQL mirror of this precedence is retired (schema epoch 4, 2026-08-12).**
#: ``conditions`` now carries ``temperature_C`` + ``temperature_source``, written by
#: this module's :func:`resolve_temperature_C` inside ``DataStore.record_conditions``
#: and backfilled across every historical row, so ``query_measurements(temp_range=…)``
#: filters one column that *is* this resolver's answer. Changing this tuple now changes
#: what gets **written** — new rows follow immediately; historical rows keep the labels
#: they were written with, and a re-resolve of old rows would be a new data epoch, not
#: an edit here. Pre-epoch-4 rows carry the backfill's answer *at* epoch 4, not one
#: contemporaneous with the measurement; the resolver is deterministic on stored
#: values, so those coincide only for as long as this tuple does.
TEMPERATURE_SOURCES = ("stage_pv", "stage_sp", "chamber_air")

#: No thermometer had anything to say. A result, not an error — see the resolver.
TEMPERATURE_UNAVAILABLE = "unavailable"

#: More than one thermometer fed a single aggregate. Spelled apart from every real
#: source because an aggregate whose rows came from different instruments is the
#: original defect wearing a new coat, and it must be visible as such rather than
#: inheriting the label of whichever row happened to be first.
TEMPERATURE_MIXED = "mixed"


def resolve_temperature_C(
    *,
    stage_pv_C: float | None = None,
    stage_sp_C: float | None = None,
    chamber_air_C: float | None = None,
) -> tuple[float, str]:
    """Pick the temperature of the sample and say which thermometer read it.

    Precedence, best first:

    ``stage_pv``
        The Modbus temperature controller's process value. This is the stage the
        sample sits on, and the operator has confirmed it is the authoritative read.
        A real one always wins.
    ``stage_sp``
        The commanded stage setpoint. Correct whenever the approach actually reached
        tolerance, and wrong by exactly the approach error when it did not — on run
        ``20260811T023757Z`` the down leg's last setpoint was commanded 27.5 °C and
        the stage sat at 30.7 °C median, because the chamber cannot cool that far
        unaided. Second, not first: a setpoint is an intention, a PV is a measurement.
    ``chamber_air``
        The humidity controller's onboard air-temperature reading. **Last**, and it
        must be labelled as such wherever it is used, because it is not the sample's
        temperature at all: it is air in the enclosure, and it ran up to 42 °C below
        the stage on the run above. It is kept in the precedence only because a
        series with no stage read at all is better characterised by the air than by
        nothing — never because the two are interchangeable.

    Returns ``(nan, "unavailable")`` when nothing is known. **That is a result, not a
    failure** — the same posture :func:`~softae.analysis.eis.geometry.resolve_thickness_cm`
    takes for a missing thickness. An invented temperature is worse than an absent
    one: it enters an Arrhenius fit as ``1/T`` and comes back out as an activation
    energy that nothing flags.

    Non-finite readings and anything at or below absolute zero are rejected rather
    than passed through, and the next source in the precedence is tried instead.
    """
    for value, source in (
        (stage_pv_C, "stage_pv"),
        (stage_sp_C, "stage_sp"),
        (chamber_air_C, "chamber_air"),
    ):
        if value is None:
            continue
        try:
            celsius = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(celsius) and celsius > ABSOLUTE_ZERO_C:
            return celsius, source

    return float("nan"), TEMPERATURE_UNAVAILABLE


def combine_temperature_sources(sources: object) -> str:
    """One label for a group of rows: their common source, ``mixed``, or ``unavailable``.

    Rows that resolved to nothing contribute nothing — a series where two rounds lost
    the stage read is still a stage-PV series, not a mixed one. But two rounds that
    resolved to *different* thermometers make the group's temperature a quantity with
    no single provenance, and :data:`TEMPERATURE_MIXED` says so instead of letting the
    first row's label stand for all of them.
    """
    distinct = {
        str(s) for s in (sources or ())
        if str(s) not in (TEMPERATURE_UNAVAILABLE, "")
    }
    if not distinct:
        return TEMPERATURE_UNAVAILABLE
    if len(distinct) > 1:
        logger.warning(
            "temperature_sources_mixed", sources=sorted(distinct),
            msg="rows in one group came from different thermometers; the aggregate "
                "temperature has no single provenance and is labelled 'mixed'",
        )
        return TEMPERATURE_MIXED
    return distinct.pop()
