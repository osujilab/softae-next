"""Shared lock for the single MCP2221 USB-to-I²C bridge.

Every driver that talks to a device on the MCP2221 — the SHT31-D
humidity/temperature sensor (:mod:`async_ht_sensor`) and the MCP4728
DAC switches such as the lamp (:mod:`async_dac_switch`) — ultimately
shares **one** physical USB HID handle.  Blinka routes every
``busio.I2C(board.SCL, board.SDA)`` instance through a single global
MCP2221 singleton, so concurrent transactions issued from different
threads corrupt one another (typically surfacing as an I²C read /
"couldn't get I²C status" error).

Before the lamp was migrated from ``nidaqmx`` to the MCP4728 quad-DAC it
lived on a separate NI-DAQ device and never touched this bus.  Post-migration,
a lamp write issued **synchronously on the GUI thread** (e.g. the Emergency
Stop button calling ``lamp.off()``) can race the background humidity poll that
reads the SHT31-D every couple of seconds.  Wrapping every I²C transaction in
``with I2C_BUS_LOCK:`` serialises access across all drivers and threads and
eliminates the collision at its root.

The lock is intentionally module-level (process-wide) and re-entrant so that a
driver method holding it may call another helper that also acquires it without
deadlocking.  Hold it only for the duration of a single transaction — never
across ``time.sleep`` or ``await`` — to keep contention low.
"""

from __future__ import annotations

import threading

#: Process-wide re-entrant lock guarding all MCP2221 I²C transactions.
I2C_BUS_LOCK = threading.RLock()
