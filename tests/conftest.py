from __future__ import annotations

import os

# Run Qt headless during the test suite so widgets that call ``.show()``
# (e.g. the EIS Browser pop-out in test_popout_creates_window) never flash a
# real window on screen. Must be set before any QApplication is created; pytest
# imports conftest before collecting test modules, so this runs first.
# ``setdefault`` lets a developer override it (e.g. to watch a test visually).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
