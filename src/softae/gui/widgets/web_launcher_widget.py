"""Web launcher widget — starts the Dash server and opens the browser.

Placed in ``tab_analysis.py`` sidebar.  The server runs in a daemon thread
so it exits automatically when the Qt application closes.
"""

from __future__ import annotations

import threading
import webbrowser
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget


class WebLauncherWidget(QGroupBox):
    """Minimal widget that starts the Dash EIS web visualizer.

    On first button click the Dash server is started in a background
    daemon thread and the default browser is opened.  Subsequent clicks
    just open a new browser tab.

    Parameters
    ----------
    db_path : str or None
        Path to pass to the Dash server as the default DataStore.
    port : int
        Port for the Dash server (default 8050).
    parent : QWidget or None
    """

    def __init__(
        self,
        db_path: str | None = None,
        port: int = 8050,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("EIS Web Visualizer", parent)
        self._db_path = db_path
        self._port = port
        self._server_thread: threading.Thread | None = None
        self._running = False

        self._launch_btn = QPushButton("Launch Web Viewer")
        self._launch_btn.clicked.connect(self._on_launch)

        self._status_label = QLabel("Status: stopped")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self._launch_btn)
        layout.addWidget(self._status_label)

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"

    def set_db_path(self, db_path: str | None) -> None:
        """Update the DataStore path passed to the Dash server."""
        self._db_path = db_path

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_launch(self) -> None:
        if not self._running:
            self._start_server()
            # Wait a moment before opening browser to let server bind
            QTimer.singleShot(1500, lambda: webbrowser.open(self.url))
        else:
            webbrowser.open(self.url)

    # ── Private ──────────────────────────────────────────────────────────

    def _start_server(self) -> None:
        try:
            from softae.web.app import create_app
        except ImportError:
            self._status_label.setText(
                "dash / plotly not installed.\n"
                "Run: pip install 'softae[web]'"
            )
            return

        app = create_app(db_path=self._db_path)

        def _run() -> None:
            app.run(host="127.0.0.1", port=self._port, debug=False, use_reloader=False)

        self._server_thread = threading.Thread(target=_run, daemon=True, name="softae-web")
        self._server_thread.start()
        self._running = True
        self._launch_btn.setText(f"Open in Browser ({self.url})")
        self._status_label.setText(f"Running at {self.url}")
        self._status_label.setStyleSheet("color: #2a7; font-size: 11px; font-weight: bold;")
