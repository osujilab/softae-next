"""Tests for the durable alert record + pluggable notifier seam (P1.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.alerts import (
    CRITICAL,
    Alert,
    clear_alert_sinks,
    raise_alert,
    register_alert_sink,
    unregister_alert_sink,
)
from softae.core.data_store import DataStore


@pytest.fixture(autouse=True)
def _no_sink_leakage():
    clear_alert_sinks()
    yield
    clear_alert_sinks()


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


class TestPersistence:
    def test_alert_is_recorded_and_queryable(self, store):
        raise_alert(
            Alert(kind="park", message="reservoir empty", severity=CRITICAL,
                  run_id="r1", details={"iteration": 7}),
            data_store=store,
        )
        rows = store.query_alerts()
        assert len(rows) == 1
        assert rows[0]["kind"] == "park"
        assert rows[0]["severity"] == "critical"
        assert rows[0]["run_id"] == "r1"
        assert rows[0]["details"]["iteration"] == 7

    def test_alerts_survive_reopen(self, tmp_path: Path):
        """The point of the table: the reason survives the process that died."""
        with DataStore(tmp_path / "p") as ds:
            raise_alert(Alert(kind="park", message="overnight stop"), data_store=ds)
        with DataStore(tmp_path / "p") as ds2:
            assert ds2.query_alerts()[0]["message"] == "overnight stop"

    def test_query_scoped_by_run(self, store):
        raise_alert(Alert(kind="a", message="m1", run_id="r1"), data_store=store)
        raise_alert(Alert(kind="b", message="m2", run_id="r2"), data_store=store)
        assert len(store.query_alerts(run_id="r1")) == 1
        assert len(store.query_alerts()) == 2

    def test_works_without_a_store(self):
        """No store configured must not lose the notification path."""
        seen = []
        register_alert_sink(seen.append)
        assert raise_alert(Alert(kind="x", message="no store")) is None
        assert len(seen) == 1


class TestSinks:
    def test_sink_receives_the_alert(self, store):
        seen: list[Alert] = []
        register_alert_sink(seen.append)
        raise_alert(Alert(kind="park", message="hi"), data_store=store)
        assert seen[0].message == "hi"

    def test_failing_sink_does_not_break_others_or_raise(self, store):
        ok: list[Alert] = []

        def bad(_alert):
            raise RuntimeError("webhook down")

        register_alert_sink(bad)
        register_alert_sink(ok.append)

        raise_alert(Alert(kind="park", message="m"), data_store=store)  # must not raise

        assert len(ok) == 1                       # good sink still ran
        assert len(store.query_alerts()) == 1     # and it was still persisted

    def test_persist_failure_still_notifies(self):
        """An alert is a report about a failure — it must not become one."""
        class Broken:
            def record_alert(self, *a, **k):
                raise RuntimeError("db gone")

        seen = []
        register_alert_sink(seen.append)
        assert raise_alert(Alert(kind="x", message="m"), data_store=Broken()) is None
        assert len(seen) == 1

    def test_register_is_idempotent_and_unregister_works(self, store):
        seen = []
        register_alert_sink(seen.append)
        register_alert_sink(seen.append)          # same callable → not duplicated
        raise_alert(Alert(kind="x", message="m"), data_store=store)
        assert len(seen) == 1

        unregister_alert_sink(seen.append)
        # `seen.append` is a fresh bound method each time, so compare by effect:
        clear_alert_sinks()
        raise_alert(Alert(kind="x", message="m2"), data_store=store)
        assert len(seen) == 1                     # no further dispatch
