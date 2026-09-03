"""The rate criterion: is sigma *moving*, or is this cell merely noisy?

Run ``20260820T183625Z_eis_validate`` is the whole reason this module exists.
Two channels, four rounds, one verdict between them:

====  =========================  ======================  =================
cell  relative deviation 3 -> 6  what it is              what the gate said
====  =========================  ======================  =================
ch30  28 -> 14 -> 11 -> 9        still drying            "not yet"
ch25  52 -> 88 -> 92 -> 87       flat mean, ~90 % noise  "not yet"
====  =========================  ======================  =================

``settle_check`` measures ``max|sigma - mean| / |mean|``, which for a 3-round
window IS the window's noise floor to within 13 %, and then compares it to a
*drift* tolerance. It cannot separate the two, so ch25 -- which was going
nowhere -- held a fifteen-channel board at its ceiling for an hour while ch25's
own scatter made the tolerance unreachable by any hold length whatsoever.

Every fixture below is shaped from that run, and the assertions are about the
**separation**: a decaying cell must come back moving, a scattering cell must
never come back moving, and the two must be told apart by a quantity that does
not care how many channels the board carries or how the rounds happened to be
spaced.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from softae.analysis.equilibration import (
    DEFAULT_SETTLE_MIN_FIT_POINTS,
    EXCLUDED_SIGMA_NULL,
    EXCLUDED_UNSETTLEABLE,
    RATE_MOVING,
    RATE_SPAN_TOO_SHORT,
    RATE_TOO_FEW_POINTS,
    RATE_UNDETECTABLE,
    SETTLE_MIN_FIT_POINTS,
    RoundFit,
    SettleTracker,
    channel_noise_floors,
    log_rate,
    rate_check,
    settle_check,
    window_noise_floor,
)

#: The shipped 15-channel ``Quick`` pacing: 17.5 s per channel plus a 300 s
#: period. Every span below is a real one, so a tolerance that looks reachable
#: here is reachable at the bench.
ROUND_PERIOD_S = 562.5


def _times(n: int, period_s: float = ROUND_PERIOD_S) -> list[float]:
    return [i * float(period_s) for i in range(n)]


# ── The three shapes, in R1 space, exactly as the gate will see them ─────────

@pytest.fixture
def ch25_flat_and_noisy_r1() -> list[float]:
    """R1 swinging ~2x about a **constant** mean with no trend at all.

    Seeded from ch25's 52/88/92/87 % deviations. The mean does not move; the
    scatter is enormous. A magnitude test reads this as "still changing" and is
    wrong, and no hold length fixes it.
    """
    return [5.00e3, 2.60e3, 5.40e3, 2.70e3, 5.20e3, 2.65e3]


@pytest.fixture
def ch30_decaying_transient_r1() -> list[float]:
    """R1 rising toward a plateau -- a film drying, so **sigma falls**.

    Seeded from ch30's 28/14/11/9 %. The sign is load-bearing and is asserted
    rather than assumed: a fixture with the sign backwards would still pass a
    magnitude test, which is precisely the failure this module is about.
    """
    t = np.asarray(_times(6))
    return [float(1.0 / s) for s in 1.0e-4 * (1.0 + np.exp(-t / 1500.0))]


@pytest.fixture
def quiet_r1() -> list[float]:
    """A settled cell: flat, with ~2 % scatter and no meaningful slope."""
    swing = np.array([1.0, -1.0, 0.6, -0.6, 1.0, -1.0])
    return [float(1.0 / s) for s in 2.0e-4 * (1.0 + 0.02 * swing)]


def _window(series: dict[int, list[float]], *, cell_constant: float = 1.0):
    """``{channel: [R1 per round]}`` -> the window the criterion reads.

    ``sigma = K / R1``. *cell_constant* exists so the K-cancellation algebra the
    module docstring asserts can be exercised as a test rather than trusted as a
    claim: every statistic here is relative, so K must fall out exactly.
    """
    rounds = max(len(values) for values in series.values())
    return [[RoundFit(channel=ch, sigma=cell_constant / values[i], r1_ohms=values[i])
             for ch, values in sorted(series.items())] for i in range(rounds)]


def _board(noisy: dict[int, list[float]], quiet: list[float], *, n_quiet: int = 3):
    """*noisy* plus *n_quiet* well-behaved cells -- a realistic 15-cell board in
    miniature. The quiet cells are what make "the window was too short" and
    "this one cell is too noisy" distinguishable at all."""
    free = (ch for ch in range(18, 100) if ch not in noisy)
    return _window({**noisy,
                    **{next(free): list(quiet) for _ in range(n_quiet)}})


# ── The separation, which is the entire deliverable ──────────────────────────

class TestTheSeparation:
    def test_rate_check_a_flat_but_noisy_channel_is_undetectable_not_moving(
            self, ch25_flat_and_noisy_r1, quiet_r1):
        window = _board({25: ch25_flat_and_noisy_r1}, quiet_r1)
        check = rate_check(window, _times(6), tol_per_hour=0.30, min_channels=3)

        assert check.undetectable == [25]
        assert check.moving == []
        ch25 = check.by_channel[25]
        assert ch25.refusal == RATE_UNDETECTABLE
        assert ch25.evaluable is False and ch25.settled is False
        # The bound is blown by scatter, not by the slope -- which is the whole
        # claim, said as an inequality.
        assert abs(ch25.rate_per_hour) < ch25.t_multiplier * ch25.stderr_per_hour

    def test_rate_check_a_flat_but_noisy_channel_is_never_moving_at_any_tolerance(
            self, ch25_flat_and_noisy_r1, quiet_r1):
        """The strong form. Tightening or loosening the tolerance changes which
        refusal fires; it must never turn a stationary cell into a moving one,
        because `rate_moving` is the one verdict that BLOCKS a board."""
        window = _board({25: ch25_flat_and_noisy_r1}, quiet_r1)
        for tol in (0.01, 0.06, 0.30, 1.0, 3.0):
            check = rate_check(window, _times(6), tol_per_hour=tol, min_channels=3)
            assert 25 not in check.moving, tol
            assert check.by_channel[25].refusal != RATE_MOVING, tol

    def test_rate_check_a_decaying_channel_is_moving_not_undetectable(
            self, ch30_decaying_transient_r1, quiet_r1):
        window = _board({30: ch30_decaying_transient_r1}, quiet_r1)
        check = rate_check(window, _times(6), tol_per_hour=0.30, min_channels=3)

        assert check.moving == [30]
        ch30 = check.by_channel[30]
        assert ch30.refusal == RATE_MOVING
        assert ch30.evaluable is True and ch30.settled is False
        # A drying film's sigma FALLS, so the rate is negative. A fixture that
        # got this backwards would pass every magnitude assertion above.
        assert ch30.rate_per_hour < 0.0
        assert check.evaluable is True and check.settled is False
        assert "ch30" in check.reason

    def test_rate_check_a_decaying_channel_is_not_unsettleable_because_the_residual_drops_the_trend(
            self, ch30_decaying_transient_r1, quiet_r1):
        """The regression is what makes the per-cell endorsement usable at all.

        ch30's RAW scatter is 21.6 % -- five times its 4.3 % residual -- because
        `std/|mean|` over a drifting series measures the drift. Endorsing a
        10 % tolerance against the raw floor would condemn every cell that is
        merely still drying as one that can never settle, which is the same
        conflation the criterion exists to remove, wearing the other hat.
        """
        window = _board({30: ch30_decaying_transient_r1}, quiet_r1)
        check = rate_check(window, _times(6), tol_per_hour=0.30, tol_rel=0.10,
                           min_channels=3)
        ch30 = check.by_channel[30]

        assert ch30.noise_floor_rel > 0.10        # the raw floor would refuse
        assert ch30.resid_rel < 0.10              # the residual does not
        assert ch30.refusal == RATE_MOVING
        assert check.unsettleable == []

    def test_rate_check_a_flat_but_noisy_channel_is_unsettleable_when_a_relative_tolerance_is_given(
            self, ch25_flat_and_noisy_r1, quiet_r1):
        """ch25's residual survives the trend removal, because there is no trend
        to remove: 40 % scatter against a 10 % tolerance is unachievable by any
        hold length, and that is a different instruction from "wait longer"."""
        window = _board({25: ch25_flat_and_noisy_r1}, quiet_r1)
        check = rate_check(window, _times(6), tol_per_hour=0.30, tol_rel=0.10,
                           min_channels=3)

        assert check.unsettleable == [25]
        assert check.by_channel[25].refusal == EXCLUDED_UNSETTLEABLE
        assert check.by_channel[25].evaluable is False
        assert "no hold length can satisfy it" in check.by_channel[25].reason

    def test_rate_check_a_genuinely_quiet_channel_is_settled(self, quiet_r1):
        check = rate_check(_window({18: quiet_r1, 19: quiet_r1, 20: quiet_r1}),
                           _times(6), tol_per_hour=0.30, min_channels=3)

        assert check.evaluable is True and check.settled is True
        assert check.quiet == [18, 19, 20]
        assert check.max_upper_bound_per_hour <= 0.30

    def test_rate_check_one_moving_cell_blocks_a_board_of_quiet_ones(
            self, ch30_decaying_transient_r1, quiet_r1):
        """Per-cell, not population. The validator's endpoints are per cell --
        H3 is a per-cell drift, D3 a per-cell deviation -- so a population
        certificate does not carry to them, and probe-3ch-v3 is the run where it
        demonstrably did not."""
        check = rate_check(_board({30: ch30_decaying_transient_r1}, quiet_r1),
                           _times(6), tol_per_hour=0.30, min_channels=3)

        assert len(check.quiet) == 3 >= 3      # the population would have passed
        assert check.settled is False
        assert check.pooled_rate_per_hour is not None    # reported, never routed


# ── Pacing, spacing and the arithmetic ───────────────────────────────────────

class TestTheEstimator:
    def test_rate_check_uneven_round_spacing_gives_the_same_verdict_as_even(self):
        """`t` is a REGRESSOR, not an index, so this holds by construction --
        which is exactly why it is worth pinning. Sweep time varies with channel
        count and a round can overrun; a criterion that counted rounds instead
        would silently rescale itself when a board gained a cell.
        """
        rate_per_s = -0.30 / 3600.0
        even, uneven = _times(6, 600.0), [0.0, 500.0, 1100.0, 1650.0, 2500.0, 3000.0]
        series = lambda axis: [1.0 / (1.0e-4 * np.exp(rate_per_s * t))  # noqa: E731
                               for t in axis]

        checks = [rate_check(_window({7: series(axis), 8: series(axis),
                                      9: series(axis)}),
                             axis, tol_per_hour=0.10, min_channels=3)
                  for axis in (even, uneven)]

        assert [c.moving for c in checks] == [[7, 8, 9], [7, 8, 9]]
        for check in checks:
            assert check.by_channel[7].rate_per_hour == pytest.approx(-0.30,
                                                                      abs=1e-9)

    def test_rate_check_three_points_is_refused_because_one_degree_of_freedom(
            self, quiet_r1):
        """k=3 leaves df=1, where t(0.975, 1) = 12.706 -- 6.5x the z a reader
        assumes. An interval that wide makes `rate_undetectable` the universal
        verdict, so it is refused outright and the refusal is NOT the noisy-cell
        one: too few points is a fact about the observation, not about the cell.
        """
        window = _window({18: quiet_r1[:3], 19: quiet_r1[:3], 20: quiet_r1[:3]})
        # Asked for 3 explicitly: the hard floor still refuses.
        check = rate_check(window, _times(3), tol_per_hour=0.30,
                           min_fit_points=3, min_channels=3)

        assert check.evaluable is False and check.settled is False
        assert all(rate.refusal == RATE_TOO_FEW_POINTS
                   for rate in check.by_channel.values())
        assert check.undetectable == [] and check.moving == []
        assert SETTLE_MIN_FIT_POINTS == 4 and DEFAULT_SETTLE_MIN_FIT_POINTS == 6

    def test_rate_check_confidence_multiplier_is_t_not_z_at_small_k(self, quiet_r1):
        """The arithmetic the external proposal got wrong. At k=4 the multiplier
        is 4.303, not 1.96, and a bound quoted at z would be less than half the
        width the residual actually justifies."""
        window = _window({18: quiet_r1[:4], 19: quiet_r1[:4], 20: quiet_r1[:4]})
        check = rate_check(window, _times(4), tol_per_hour=0.60,
                           min_fit_points=4, min_channels=3)
        rate = check.by_channel[18]

        assert rate.t_multiplier == pytest.approx(4.303, abs=5e-4)
        assert rate.upper_bound_per_hour == pytest.approx(
            abs(rate.rate_per_hour) + 4.303 * rate.stderr_per_hour, rel=1e-3)
        assert (abs(rate.rate_per_hour) + 1.96 * rate.stderr_per_hour
                < rate.upper_bound_per_hour)

    def test_rate_check_a_span_too_short_for_the_noise_is_not_evaluable(
            self, quiet_r1):
        """Where `MIN_WINDOWS_PER_TAU`'s discipline survives without its
        constant: "a window shorter than the dynamics is an extrapolation"
        becomes span-vs-noise. A cell at this window's median residual could
        certify no better than 0.088 ln/h, so a 0.06 tolerance was never on
        offer -- a statement about the observation, and so NOT a verdict.
        """
        check = rate_check(_window({18: quiet_r1, 19: quiet_r1, 20: quiet_r1}),
                           _times(6), tol_per_hour=0.06, min_channels=3)

        assert check.evaluable is False and check.settled is False
        assert check.moving == []
        assert all(rate.refusal == RATE_SPAN_TOO_SHORT
                   for rate in check.by_channel.values())
        # Loosen only the tolerance and the same window certifies: the refusal
        # was about what was asked of the observation, not about the cells.
        assert rate_check(_window({18: quiet_r1, 19: quiet_r1, 20: quiet_r1}),
                          _times(6), tol_per_hour=0.30,
                          min_channels=3).settled is True

    def test_log_rate_refuses_a_window_with_no_span_rather_than_dividing_by_zero(
            self):
        assert log_rate([100.0] * 5, [2.0e-4] * 5) is None
        assert log_rate([0.0, 1.0], [2.0e-4, 1.0e-4]) is None      # k < 3
        assert log_rate([0.0, 1.0, 2.0], [2.0e-4, 0.0, 1.0e-4]) is None  # ln(0)

    def test_log_rate_recovers_a_known_fractional_rate_to_four_digits(self):
        axis = _times(8, 400.0)
        sigmas = [1.0e-4 * np.exp(-0.25 / 3600.0 * t) for t in axis]
        slope, stderr, resid = log_rate(axis, sigmas)

        assert slope * 3600.0 == pytest.approx(-0.25, rel=1e-4)
        assert stderr == pytest.approx(0.0, abs=1e-12)
        assert resid == pytest.approx(0.0, abs=1e-12)


# ── The invariants that make the gate independent of the cell constant ───────

class TestCellConstantCancels:
    def test_rate_check_on_r1_equals_rate_check_on_sigma_because_k_cancels(
            self, ch30_decaying_transient_r1, quiet_r1):
        """`sigma = K/R1` with K constant, so `d ln sigma/dt = -d ln R1/dt`: the
        cell constant is an additive offset on `ln sigma`, absorbed by the
        intercept. The module docstring asserts this algebra to justify a tau
        cross-check that has since been retired; the algebra is independent of
        any fit and is pinned here so it stays a guarantee rather than a claim.
        """
        series = {30: ch30_decaying_transient_r1, 18: quiet_r1, 19: quiet_r1}
        verdicts = [
            rate_check(_window(series, cell_constant=k), _times(6),
                       tol_per_hour=0.30, tol_rel=0.10, min_channels=3)
            for k in (1.0, 3.7e-4, 812.0)
        ]

        for check in verdicts[1:]:
            assert check.moving == verdicts[0].moving
            assert check.quiet == verdicts[0].quiet
            assert check.unsettleable == verdicts[0].unsettleable
            for channel, rate in check.by_channel.items():
                assert rate.rate_per_hour == pytest.approx(
                    verdicts[0].by_channel[channel].rate_per_hour, rel=1e-9)
                assert rate.resid_rel == pytest.approx(
                    verdicts[0].by_channel[channel].resid_rel, rel=1e-9)

    def test_settle_check_on_r1_equals_settle_check_on_sigma_because_k_cancels(
            self, ch30_decaying_transient_r1, quiet_r1):
        """The same invariance for the DEVIATION criterion, so that a later move
        of the gate's observable onto the fitted R1 is provably neutral with
        respect to K -- though not, and this is the point of saying it, with
        respect to raw-versus-fitted."""
        series = {30: ch30_decaying_transient_r1, 18: quiet_r1, 19: quiet_r1}
        checks = [settle_check(_window(series, cell_constant=k), tol_rel=0.10,
                               min_channels=3) for k in (1.0, 3.7e-4, 812.0)]

        for check in checks[1:]:
            assert check.settled == checks[0].settled
            assert check.participating == checks[0].participating
            assert check.max_deviation_rel == pytest.approx(
                checks[0].max_deviation_rel, rel=1e-9)


# ── What it refuses to be handed ─────────────────────────────────────────────

class TestRefusals:
    def test_rate_check_reads_no_setpoint(self):
        """The prohibition above `SETTLE_SETTLED`, pinned as a signature test.

        The sigma criterion never reads a setpoint and no gate beside it may make
        sigma wait on a PV *reaching* one. This one compares a series to itself
        exactly as `rh_window_spread` does; `tol_per_hour` is a tolerance and the
        time axis is a duration since the phase began. If a later edit adds a
        parameter that names a target, this test is what says the prohibition is
        back in force.
        """
        names = set(inspect.signature(rate_check).parameters)

        assert names == {"window", "times_s", "tol_per_hour", "tol_rel",
                         "min_fit_points", "min_channels", "r1_bound_ohms"}
        for forbidden in ("setpoint", "target", "command", "_pv", "pv_"):
            assert not any(forbidden in name for name in names), forbidden

    def test_rate_check_a_missing_round_time_refuses_rather_than_assuming_even_spacing(
            self, quiet_r1):
        window = _window({18: quiet_r1, 19: quiet_r1, 20: quiet_r1})
        axis = _times(6)
        axis[2] = None

        check = rate_check(window, axis, tol_per_hour=0.30, min_channels=3)

        assert check.evaluable is False and check.settled is False
        assert "will not assume even spacing" in check.reason
        assert check.by_channel == {}

    def test_rate_check_reuses_the_participation_rule_rather_than_restating_it(
            self, quiet_r1):
        """Absent, NULL-sigma and railed cells are excluded by the same
        `_exclusion` the deviation criterion uses, so a channel can never
        participate in one criterion and not the other."""
        railed = _window({18: quiet_r1, 19: quiet_r1, 20: quiet_r1,
                          9: [100.0] * 6})
        check = rate_check(railed, _times(6), tol_per_hour=0.30, min_channels=3,
                           r1_bound_ohms=100.0)

        assert 9 not in check.participating
        assert check.excluded[9] == "railed_R1"
        assert check.participating == settle_check(
            railed, min_channels=3, r1_bound_ohms=100.0).participating

    def test_rate_check_a_non_positive_conductance_is_excluded_not_logged(
            self, quiet_r1):
        """`ln sigma` has no value at sigma <= 0, and neither has a conducting
        cell. Excluded under the existing name, because the cause is the
        existing one: the fit produced no usable sigma."""
        window = _window({18: quiet_r1, 19: quiet_r1, 20: quiet_r1})
        window[2] = [RoundFit(18, -1.0e-4, 5.0e3)] + window[2][1:]

        check = rate_check(window, _times(6), tol_per_hour=0.30, min_channels=2)

        assert check.excluded[18] == EXCLUDED_SIGMA_NULL
        assert 18 not in check.by_channel

    def test_rate_check_too_few_participants_is_not_evaluable_rather_than_unsettled(
            self, quiet_r1):
        check = rate_check(_window({18: quiet_r1, 19: quiet_r1}), _times(6),
                           tol_per_hour=0.30, min_channels=3)

        assert check.evaluable is False and check.settled is False
        assert "cannot be evaluated" in check.reason

    def test_rate_check_an_empty_window_is_never_settled(self):
        check = rate_check([], [], tol_per_hour=0.30)
        assert not check.evaluable and not check.settled


# ── Per-channel floors: the number the median hides ──────────────────────────

class TestPerChannelFloors:
    def test_channel_noise_floors_names_the_noisy_cell_the_median_hides(
            self, ch25_flat_and_noisy_r1, quiet_r1):
        """`window_noise_floor` medians across participants on purpose, so one
        cell does not condemn the setpoint; `settle_check` maxes across them, so
        one cell decides the verdict. They point opposite ways BY DESIGN, and
        that gap is how a cell at 90 % scatter held a board at 10 % tolerance
        without ever being named.
        """
        window = _board({25: ch25_flat_and_noisy_r1}, quiet_r1, n_quiet=14)
        participants = sorted({int(fit.channel) for fit in window[0]})

        floors = channel_noise_floors(window, participants)
        median = window_noise_floor(window, participants)

        assert floors[25] > 0.30 > 0.10                 # far above the tolerance
        assert median < 0.10                            # and the median hides it
        assert median == pytest.approx(float(np.median(
            [value for value in floors.values() if value is not None])))

    def test_channel_noise_floors_reports_absence_rather_than_a_floor_of_zero(self):
        window = [[RoundFit(4, None, None)], [RoundFit(4, None, None)]]
        assert channel_noise_floors(window, [4]) == {4: None}

    def test_window_noise_floor_is_unchanged_by_the_per_channel_refactor(
            self, quiet_r1, ch25_flat_and_noisy_r1):
        """The regression proof for the Stage 0b refactor: the median of the
        per-channel floors is the number `window_noise_floor` returned before it
        was expressed in terms of them."""
        window = _board({25: ch25_flat_and_noisy_r1}, quiet_r1, n_quiet=4)
        participants = sorted({int(fit.channel) for fit in window[0]})
        series = {ch: [float(fit.sigma) for rounds in window for fit in rounds
                       if int(fit.channel) == ch] for ch in participants}

        by_hand = [float(np.std(sigmas, ddof=1) / abs(np.mean(sigmas)))
                   for sigmas in series.values()]

        assert window_noise_floor(window, participants) == pytest.approx(
            float(np.median(by_hand)))
        assert channel_noise_floors(window, participants)[25] == pytest.approx(
            by_hand[participants.index(25)])


# ── The tracker's new axis, and the cell-level endorsement ───────────────────

class TestTrackerTimeAxis:
    def test_tracker_times_s_stays_aligned_with_rounds_when_t_s_is_omitted(self):
        """The `rh_medians` precedent, copied exactly: one entry per round, and
        the default records the ABSENCE rather than a plausible zero."""
        tracker = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0)
        for _ in range(4):
            tracker.observe([RoundFit(1, 2.0e-4, 5.0e3)])

        assert len(tracker.times_s) == len(tracker.rounds) == 4
        assert tracker.times_s == [None] * 4

    def test_tracker_times_s_records_the_elapsed_seconds_it_is_given(self):
        tracker = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0)
        for index in range(4):
            tracker.observe([RoundFit(1, 2.0e-4, 5.0e3)], t_s=index * 562.5)

        assert tracker.times_s == [0.0, 562.5, 1125.0, 1687.5]

    def test_tracker_verdicts_are_identical_with_and_without_a_time_axis(self):
        """Stage 1 is verdict-neutral by construction: nothing reads the axis
        under the deviation criterion. Replayed field by field, because "the
        same answer" has to mean every field and not just `settled`."""
        series = [1.0e-4, 1.4e-4, 1.42e-4, 1.43e-4, 1.44e-4]
        without = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0)
        with_axis = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0)

        for index, sigma in enumerate(series):
            fits = [RoundFit(1, sigma, 5.0e3), RoundFit(2, sigma * 1.1, 4.5e3)]
            plain = without.observe(list(fits))
            timed = with_axis.observe(list(fits), t_s=index * 562.5)
            assert plain == timed

        assert without.outcome(stopped_early=False) == with_axis.outcome(
            stopped_early=False)


class TestTrackerPerChannelEndorsement:
    def _tracker(self, noisy: list[float], quiet: list[float]):
        tracker = SettleTracker(tol_rel=0.10, n_rounds=3, min_channels=3,
                                r1_bound_ohms=100.0)
        for index in range(3):
            tracker.observe([RoundFit(25, 1.0 / noisy[index], noisy[index]),
                             *(RoundFit(ch, 1.0 / quiet[index], quiet[index])
                               for ch in (18, 19, 20))])
        return tracker

    def test_tracker_per_channel_endorsement_names_the_cell_the_board_median_hides(
            self, ch25_flat_and_noisy_r1, quiet_r1):
        tracker = self._tracker(ch25_flat_and_noisy_r1, quiet_r1)

        board_ok, _why, board_floor = tracker.endorsement()
        per_channel = tracker.per_channel_endorsement()

        assert board_ok is True and board_floor < 0.10   # the board looks fine
        assert per_channel[25][0] is False               # ch25 never can be
        assert "no hold length can satisfy it" in per_channel[25][1]
        assert per_channel[25][2] > 0.10
        assert all(per_channel[ch][0] is True for ch in (18, 19, 20))

    def test_tracker_per_channel_endorsement_fires_at_the_first_judged_window(
            self, ch25_flat_and_noisy_r1, quiet_r1):
        """28 minutes before the ceiling, not at it. Nothing is reported before
        a window exists, because two rounds are a coincidence."""
        tracker = SettleTracker(tol_rel=0.10, n_rounds=3, min_channels=3,
                                r1_bound_ohms=100.0)
        seen = []
        for index in range(3):
            tracker.observe([RoundFit(25, 1.0 / ch25_flat_and_noisy_r1[index],
                                      ch25_flat_and_noisy_r1[index]),
                             *(RoundFit(ch, 1.0 / quiet_r1[index], quiet_r1[index])
                               for ch in (18, 19, 20))])
            seen.append(sorted(ch for ch, (ok, _w, _f)
                               in tracker.per_channel_endorsement().items()
                               if ok is False))

        assert seen == [[], [], [25]]

    def test_tracker_per_channel_endorsement_is_none_rather_than_false_without_a_floor(
            self):
        """"Not checked" is not "checked and fine", and it is not "checked and
        refused" either -- the same third state `endorsement` already keeps."""
        tracker = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0)
        for _ in range(3):
            tracker.observe([RoundFit(9, 0.5, 100.0)])

        assert tracker.per_channel_endorsement() == {}


# ── The criterion selector, and the window the two criteria disagree about ───

#: The tolerance the tracker tests judge a rate against, ln-units per hour. Well
#: above every quiet fixture's own bound and well below the decaying one's, so a
#: verdict that flips is the criterion and not the number.
RATE_TOL_LN_PER_H = 0.30


def _feed(tracker: SettleTracker, series: dict[int, list[float]],
          *, period_s: float = ROUND_PERIOD_S, with_time: bool = True):
    """Replay ``{channel: [R1 per round]}`` through a tracker, one round at a
    time, and hand back every verdict it produced."""
    rounds = max(len(values) for values in series.values())
    seen = []
    for index in range(rounds):
        fits = [RoundFit(channel=ch, sigma=1.0 / values[index],
                         r1_ohms=values[index])
                for ch, values in sorted(series.items())]
        seen.append(tracker.observe(
            fits, t_s=index * period_s if with_time else None))
    return seen


@pytest.fixture
def ch18_one_fit_excursion_r1() -> list[float]:
    """Flat R1 with **one** round at 4x -- a fitted R1 that went wrong once.

    Seeded from ``20260821T173111Z_eis_validate`` ch18, whose relative deviations
    ran 13, 11, **164, 90**, 7 %. No film moves 164 % and back to 7 % in three
    rounds; that is one bad fit entering and leaving a trailing 3-round window.
    The excursion is the case the two criteria disagree about most sharply.
    """
    return [1.0e4, 1.0e4, 4.0e4, 1.0e4, 1.0e4, 1.0e4]


class TestCriterionSelector:
    def test_tracker_default_criterion_gives_byte_identical_verdicts(
            self, ch30_decaying_transient_r1, quiet_r1):
        """The selector defaults to today's branch, and "identical" means every
        field of every round's verdict -- not merely the same `settled`."""
        series = {30: ch30_decaying_transient_r1,
                  18: quiet_r1, 19: quiet_r1, 20: quiet_r1}
        shipped = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0)
        named = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0,
                              criterion="deviation")

        assert _feed(shipped, series) == _feed(named, series)
        assert shipped.criterion == "deviation" and shipped.last_rate is None

    def test_tracker_both_mode_gates_on_deviation_and_reports_the_rate(
            self, ch30_decaying_transient_r1, quiet_r1):
        """Shadow mode: the verdict is the shipped one, byte for byte, and the
        rate rides beside it with no routing power at all.

        This is the configuration a bench run uses to compare the two criteria on
        one board before either is trusted, so the *pair* is the deliverable --
        an identical verdict is not enough if no rate came back with it.
        """
        series = {30: ch30_decaying_transient_r1,
                  18: quiet_r1, 19: quiet_r1, 20: quiet_r1}
        shipped = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0)
        shadow = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0,
                               criterion="both",
                               rate_tol_per_hour=RATE_TOL_LN_PER_H)

        assert _feed(shipped, series) == _feed(shadow, series)
        assert shadow.judged_rounds == shadow.n_rounds == 3
        # ...and the rate was computed anyway, over its own longer window.
        assert shadow.last_rate is not None
        assert shadow.last_rate.moving == [30]
        assert shadow.last_rate.pooled_rate_per_hour is not None

    def test_tracker_rate_criterion_routes_on_the_rate_not_on_the_deviation(
            self, ch30_decaying_transient_r1, quiet_r1):
        series = {30: ch30_decaying_transient_r1,
                  18: quiet_r1, 19: quiet_r1, 20: quiet_r1}
        tracker = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0,
                                criterion="rate",
                                rate_tol_per_hour=RATE_TOL_LN_PER_H)
        seen = _feed(tracker, series)

        # No verdict until the RATE window is full -- longer than n_rounds,
        # because df = 1 at k = 3 makes an interval meaningless.
        assert tracker.judged_rounds == 6
        assert seen[:5] == [None] * 5
        assert seen[5] is not None and not seen[5].settled
        # A deviation number would be a number nobody measured.
        assert seen[5].max_deviation_rel is None
        assert "still moving" in seen[5].reason

    def test_tracker_rate_criterion_without_a_time_axis_refuses_rather_than_guessing(
            self, quiet_r1):
        tracker = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0,
                                criterion="rate",
                                rate_tol_per_hour=RATE_TOL_LN_PER_H)
        seen = _feed(tracker, {18: quiet_r1}, with_time=False)

        assert seen[-1] is not None
        assert not seen[-1].evaluable and not seen[-1].settled
        assert "will not assume even spacing" in seen[-1].reason
        assert tracker.outcome(stopped_early=False) == "not_evaluable"

    def test_tracker_rate_criterion_without_a_tolerance_never_certifies(
            self, quiet_r1):
        """A gate handed no tolerance has nothing to compare against, so it
        returns no verdict rather than inventing one. The tool refuses this
        combination at `validate_plan`; the tracker refuses it in arithmetic."""
        tracker = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0,
                                criterion="rate")

        assert _feed(tracker, {18: quiet_r1}) == [None] * 6
        assert tracker.last_rate is None and not tracker.settled

    def test_tracker_an_unknown_criterion_is_refused_at_construction(self):
        """Falling back would fall back to the criterion the caller was trying
        to leave, and the run would look exactly like a correct one."""
        with pytest.raises(ValueError, match="not one of"):
            SettleTracker(criterion="deviaton")

    def test_tracker_rate_criterion_judges_the_room_over_the_window_it_judged_sigma(
            self, quiet_r1):
        """The RH clause spans the SIGMA window, which under the rate criterion
        is six rounds and not three. A room judged over half the window leaves
        the other half unwatched, and "sigma flat under a moving room is not
        evidence" is a claim about one window or it is not a claim."""
        tracker = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0,
                                rh_stability_pct=1.5, criterion="rate",
                                rate_tol_per_hour=RATE_TOL_LN_PER_H)
        # RH walks 4 %RH across six rounds and 0.6 %RH across the last three, so
        # only the longer window can see it.
        for index, r1 in enumerate(quiet_r1):
            tracker.observe([RoundFit(18, 1.0 / r1, r1)],
                            rh_median_pct=10.0 + 0.8 * index,
                            t_s=index * ROUND_PERIOD_S)

        assert tracker.rh_spread_pct == pytest.approx(4.0)
        assert tracker.rh_blocked_settle is True
        assert not tracker.settled


class TestTheFitExcursion:
    """The window the whole build exists to keep refusing.

    ``20260821T173111Z_eis_validate`` and ``20260821T192508Z_eis_validate`` both
    ran 11 rounds to `ceiling` with `n_recorded: 0`, and the failure is not
    drift: ch18 went 13, 11, **164, 90**, 7 % and ch28 was 14 % in one run and
    94 % in the other. A film cannot do that. It is a fitted R1 excursion moving
    through a trailing 3-round window -- so selecting "stable" channels from a
    prior run selects noise, and the gate that certifies once the excursion ages
    out certifies a board carrying it.
    """

    def _board(self, excursion, quiet):
        return {18: excursion, 19: quiet, 20: quiet, 21: quiet}

    def test_rate_check_a_fit_excursion_is_droppable_and_never_moving(
            self, ch18_one_fit_excursion_r1, quiet_r1):
        """The separation, on the real failure rather than on a synthetic one.

        `rate_moving` BLOCKS and the other refusals may be dropped, so a cell
        classified moving on the strength of one bad fit would hold a board
        forever -- and one classified quiet would let a wild cell into the
        dataset. It must be neither.
        """
        window = _window(self._board(ch18_one_fit_excursion_r1, quiet_r1))
        check = rate_check(window, _times(6), tol_per_hour=RATE_TOL_LN_PER_H,
                           tol_rel=0.10, min_channels=3)
        judged = check.by_channel[18]

        assert judged.refusal != RATE_MOVING
        assert judged.refusal in (RATE_UNDETECTABLE, EXCLUDED_UNSETTLEABLE,
                                  RATE_SPAN_TOO_SHORT)
        assert judged.evaluable is False and judged.settled is False
        assert 18 not in check.moving and 18 not in check.quiet

    def test_rate_check_a_genuine_decay_still_blocks_beside_the_excursion(
            self, ch18_one_fit_excursion_r1, ch30_decaying_transient_r1,
            quiet_r1):
        """The other half of the separation, asserted on ONE board so the two
        classifications are made by the same call over the same window."""
        window = _window({18: ch18_one_fit_excursion_r1,
                          30: ch30_decaying_transient_r1,
                          19: quiet_r1, 20: quiet_r1, 21: quiet_r1})
        check = rate_check(window, _times(6), tol_per_hour=RATE_TOL_LN_PER_H,
                           tol_rel=0.10, min_channels=3)

        assert check.moving == [30]
        assert check.by_channel[30].rate_per_hour < 0     # a drying film
        assert check.evaluable and not check.settled
        assert "ch30" in check.reason and "ch18" not in check.reason

    def test_settle_check_certifies_the_excursion_window_the_rate_refuses(
            self, ch18_one_fit_excursion_r1, quiet_r1):
        """**The behaviour this build exists to preserve, pinned.**

        The deviation criterion reads a trailing THREE rounds, so once the
        excursion has aged out of it the same board certifies -- today's gate
        would license a run on a board carrying a 4x fit excursion two rounds
        earlier. The rate criterion reads the longer window and does not.

        If this test ever goes green in both directions, the two criteria have
        stopped disagreeing and the selector has stopped being worth having.
        """
        window = _window(self._board(ch18_one_fit_excursion_r1, quiet_r1))

        deviation = settle_check(window[-3:], tol_rel=0.10, min_channels=3)
        rate = rate_check(window, _times(6), tol_per_hour=RATE_TOL_LN_PER_H,
                          tol_rel=0.10, min_channels=3)

        assert deviation.settled is True                  # ...and it is wrong
        assert deviation.max_deviation_rel < 0.10
        assert 18 in rate.unsettleable or 18 in rate.undetectable
        assert rate.by_channel[18].settled is False
