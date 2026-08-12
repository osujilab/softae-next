"""Anti-clog purge scheduling (P8).

Particulate stock clogs its check valve when it sits still — on a ~10 min scale,
and *being mid-run is not protection* because a campaign can go many trials
without drawing from that stock. Time is injected, so multi-day schedules run
instantly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.data_store import DataStore
from softae.core.purge import (
    PurgeScheduler,
    PurgeSettings,
    load_purge_settings,
    purge_settings,
    save_purge_settings,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


def _settings(**kw) -> PurgeSettings:
    base = dict(interval_s=900.0, particulate_uL=20.0, other_uL=10.0,
                particulate_pumps=(0,), pumps=(0, 1, 2))
    base.update(kw)
    return PurgeSettings(**base)


# ── Settings ─────────────────────────────────────────────────────────────────

class TestSettings:
    def test_particulate_line_gets_the_larger_volume(self):
        s = _settings()
        assert s.volume_for(0) == 20.0
        assert s.volume_for(1) == 10.0

    def test_all_lines_purge_not_just_the_particulate_one(self):
        """Purging only one line makes another the new stagnation site."""
        assert set(_settings().per_purge_uL()) == {0, 1, 2}

    def test_daily_rate_matches_the_operator_figures(self):
        """20 + 10 + 10 µL every 15 min = 96 purges/day = 3.84 mL."""
        assert _settings().total_uL_per_day() == pytest.approx(3840.0)

    def test_per_pump_daily_rates(self):
        rates = _settings().uL_per_day()
        assert rates[0] == pytest.approx(1920.0)
        assert rates[1] == pytest.approx(960.0)

    def test_disabled_consumes_nothing(self):
        assert _settings(enabled=False).uL_per_day() == {}

    def test_describe_states_the_rate_that_actually_matters(self):
        text = _settings().describe()
        assert "15 min" in text
        assert "mL/day" in text

    def test_invalid_settings_are_rejected(self):
        with pytest.raises(ValueError):
            _settings(interval_s=0.0).validated()
        with pytest.raises(ValueError):
            _settings(particulate_uL=-1.0).validated()
        with pytest.raises(ValueError):
            _settings(pumps=()).validated()

    def test_disabled_settings_skip_validation(self):
        _settings(enabled=False, interval_s=0.0).validated()   # must not raise


class TestConfig:
    def test_reads_the_purge_section(self):
        s = purge_settings({"interval_s": 600.0, "particulate_uL": 30.0,
                            "other_uL": 5.0, "particulate_pumps": [1]})
        assert s.interval_s == 600.0
        assert s.volume_for(1) == 30.0
        assert s.volume_for(0) == 5.0

    def test_bad_values_fall_back_to_defaults(self):
        assert purge_settings({"interval_s": "soon"}).interval_s == 900.0

    def test_missing_section_uses_operator_defaults(self):
        s = purge_settings({})
        assert s.interval_s == 900.0 and s.particulate_uL == 20.0


class TestOverrides:
    def test_config_supplies_defaults_when_nothing_is_saved(self, store):
        assert load_purge_settings(store).interval_s == purge_settings().interval_s

    def test_a_saved_override_wins(self, store):
        """The bench is where clogging is observed; retuning must not need TOML."""
        save_purge_settings(store, _settings(interval_s=300.0, particulate_uL=40.0))

        loaded = load_purge_settings(store)
        assert loaded.interval_s == 300.0
        assert loaded.particulate_uL == 40.0

    def test_an_override_survives_reopen(self, tmp_path: Path):
        ds = DataStore(tmp_path / "proj")
        save_purge_settings(ds, _settings(interval_s=123.0))
        ds.close()

        reopened = DataStore(tmp_path / "proj")
        try:
            assert load_purge_settings(reopened).interval_s == 123.0
        finally:
            reopened.close()

    def test_disabling_persists(self, store):
        save_purge_settings(store, _settings(enabled=False))
        assert load_purge_settings(store).enabled is False

    def test_no_store_means_config_only(self):
        assert load_purge_settings(None).interval_s == purge_settings().interval_s


# ── Scheduling ───────────────────────────────────────────────────────────────

class TestScheduler:
    def test_nothing_is_due_immediately(self):
        clock = _Clock()
        assert PurgeScheduler(_settings(), now=clock).due() is None

    def test_a_purge_becomes_due_after_the_interval(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        clock.t = 901.0

        due = sched.due()
        assert due is not None
        assert due.volumes_uL == {0: 20.0, 1: 10.0, 2: 10.0}
        assert due.total_uL == 40.0

    def test_performing_a_purge_clears_it(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        clock.t = 901.0
        sched.note_purged()
        assert sched.due() is None

    def test_a_recently_used_pump_does_not_need_purging(self):
        """This is what keeps an active campaign off the full idle rate."""
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)

        clock.t = 800.0
        for pump in (0, 1, 2):
            sched.note_dispense(pump)

        clock.t = 1000.0            # past the interval from t=0, not from t=800
        assert sched.due() is None

    def test_one_idle_line_triggers_a_purge_of_all(self):
        """Leaving the others idle through a cycle just moves the problem."""
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)

        clock.t = 800.0
        sched.note_dispense(0)      # pump 0 stays fresh
        clock.t = 901.0             # pumps 1 and 2 are now overdue

        due = sched.due()
        assert due is not None
        assert set(due.volumes_uL) == {0, 1, 2}

    def test_overdue_time_is_reported(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        clock.t = 1000.0
        assert sched.due().overdue_s == pytest.approx(100.0)

    def test_next_due_counts_down(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        assert sched.next_due_in_s() == pytest.approx(900.0)
        clock.t = 500.0
        assert sched.next_due_in_s() == pytest.approx(400.0)

    def test_disabled_never_becomes_due(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(enabled=False), now=clock)
        clock.t = 100_000.0
        assert sched.due() is None
        assert sched.next_due_in_s() is None

    def test_a_multi_day_idle_stays_due_rather_than_drifting(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        clock.t = 3 * 86400.0
        assert sched.due() is not None


class TestProjectionIntegration:
    def test_purge_rates_feed_the_preflight_projection(self):
        """Purge accrues with elapsed time, so it belongs in the runway."""
        rates = _settings().uL_per_day()
        assert rates and all(v > 0 for v in rates.values())
        assert sum(rates.values()) == pytest.approx(3840.0)
