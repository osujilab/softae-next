"""Shared shutdown constants for GUI daemon-thread teardown.

``DAEMON_JOIN_TIMEOUT`` bounds how long each daemon-tab ``cleanup()`` waits when
joining an in-progress ``threading.Thread`` runner on window close.  Long enough
for a cooperative runner to notice its abort between steps, short enough never to
freeze the close (daemon threads survive a non-join and die at process exit).
"""

from __future__ import annotations

DAEMON_JOIN_TIMEOUT = 3.0
