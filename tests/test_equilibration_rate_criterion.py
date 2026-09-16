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
    EXCLUDED_ABSENT,
    EXCLUDED_RAILED,
    EXCLUDED_SIGMA_NULL,
    EXCLUDED_UNSETTLEABLE,
    RATE_MOVING,
    RATE_SPAN_TOO_SHORT,
    RATE_TOO_FEW_POINTS,
    RATE_UNDETECTABLE,
    SETTLE_CEILING,
    SETTLE_CONSENSUS_DROP_BUCKET,
    SETTLE_MIN_FIT_POINTS,
    SETTLE_NOT_EVALUABLE,
    SETTLE_SETTLED,
    WELL_FIT_FAILED,
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
#
# Each is a function of the round count with a six-round fixture in front of it,
# because the ARMED rate window is seven rounds (`SETTLE_CONSENSUS_MAX_EXCLUDED`
# on top of `DEFAULT_SETTLE_MIN_FIT_POINTS`, operator ruling 2026-09-14) while
# every `rate_check` test below still calls the window directly at six. A
# hand-appended seventh point would change each fixture's SHAPE, which is the one
# thing these fixtures are for; generated points do not.


def _flat_and_noisy_r1(n: int = 6) -> list[float]:
    """R1 swinging ~2x about a **constant** mean with no trend at all."""
    cycle = [5.00e3, 2.60e3, 5.40e3, 2.70e3, 5.20e3, 2.65e3]
    return [cycle[i % len(cycle)] for i in range(n)]


def _decaying_transient_r1(n: int = 6) -> list[float]:
    """R1 rising toward a plateau -- a film drying, so **sigma falls**."""
    t = np.asarray(_times(n))
    return [float(1.0 / s) for s in 1.0e-4 * (1.0 + np.exp(-t / 1500.0))]


def _quiet_r1(n: int = 6) -> list[float]:
    """A settled cell: flat, with ~2 % scatter and no meaningful slope."""
    cycle = [1.0, -1.0, 0.6, -0.6]
    swing = np.array([cycle[i % len(cycle)] for i in range(n)])
    return [float(1.0 / s) for s in 2.0e-4 * (1.0 + 0.02 * swing)]


@pytest.fixture
def ch25_flat_and_noisy_r1() -> list[float]:
    """Seeded from ch25's 52/88/92/87 % deviations. The mean does not move; the
    scatter is enormous. A magnitude test reads this as "still changing" and is
    wrong, and no hold length fixes it.
    """
    return _flat_and_noisy_r1()


@pytest.fixture
def ch30_decaying_transient_r1() -> list[float]:
    """Seeded from ch30's 28/14/11/9 %. The sign is load-bearing and is asserted
    rather than assumed: a fixture with the sign backwards would still pass a
    magnitude test, which is precisely the failure this module is about.
    """
    return _decaying_transient_r1()


@pytest.fixture
def quiet_r1() -> list[float]:
    return _quiet_r1()


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
                         "min_fit_points", "min_channels", "r1_bound_ohms",
                         # A switch for the consensus rule, and it names no
                         # target either: the shape it compares is the WINDOW's
                         # own mode, so this stays a series compared to itself.
                         "consensus_exclude",
                         # T11.33's policy. A COUNT of participating wells, and
                         # `None` for no count at all -- a statement about how
                         # wide the board must be, never about where any PV
                         # should be. The prohibition is untouched by it.
                         "board_minimum"}
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

    def test_tracker_both_mode_gates_on_deviation_and_reports_the_rate(self):
        """Shadow mode: the verdict is the shipped one, byte for byte, and the
        rate rides beside it with no routing power at all.

        This is the configuration a bench run uses to compare the two criteria on
        one board before either is trusted, so the *pair* is the deliverable --
        an identical verdict is not enough if no rate came back with it.

        **Seven rounds, not six**, because the armed consensus rule buys its
        exclusion budget in observation: see `rate_window_rounds`.
        """
        quiet = _quiet_r1(7)
        series = {30: _decaying_transient_r1(7),
                  18: quiet, 19: quiet, 20: quiet}
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

    def test_tracker_rate_criterion_routes_on_the_rate_not_on_the_deviation(self):
        quiet = _quiet_r1(7)
        series = {30: _decaying_transient_r1(7),
                  18: quiet, 19: quiet, 20: quiet}
        tracker = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0,
                                criterion="rate",
                                rate_tol_per_hour=RATE_TOL_LN_PER_H)
        seen = _feed(tracker, series)

        # No verdict until the RATE window is full -- longer than n_rounds,
        # because df = 1 at k = 3 makes an interval meaningless, and one longer
        # again while the consensus rule is armed.
        assert tracker.judged_rounds == 7
        assert seen[:6] == [None] * 6
        assert seen[6] is not None and not seen[6].settled
        # A deviation number would be a number nobody measured.
        assert seen[6].max_deviation_rel is None
        assert "still moving" in seen[6].reason

    def test_tracker_rate_criterion_without_a_time_axis_refuses_rather_than_guessing(
            self):
        tracker = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0,
                                criterion="rate",
                                rate_tol_per_hour=RATE_TOL_LN_PER_H)
        seen = _feed(tracker, {18: _quiet_r1(7)}, with_time=False)

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
            self):
        """The RH clause spans the SIGMA window, which under the rate criterion
        is seven rounds and not three. A room judged over half the window leaves
        the other half unwatched, and "sigma flat under a moving room is not
        evidence" is a claim about one window or it is not a claim."""
        tracker = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0,
                                rh_stability_pct=1.5, criterion="rate",
                                rate_tol_per_hour=RATE_TOL_LN_PER_H)
        # RH walks 4.8 %RH across seven rounds and 1.6 %RH across the last three,
        # so only the longer window can see it.
        for index, r1 in enumerate(_quiet_r1(7)):
            tracker.observe([RoundFit(18, 1.0 / r1, r1)],
                            rh_median_pct=10.0 + 0.8 * index,
                            t_s=index * ROUND_PERIOD_S)

        assert tracker.rh_spread_pct == pytest.approx(4.8)
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


# ── The physical consensus rule ──────────────────────────────────────────────

#: The armed rate window. Seven and not six, so a channel that spends its one
#: exclusion still regresses on `DEFAULT_SETTLE_MIN_FIT_POINTS` points.
ARMED_WINDOW = 7


def _shaped(series: dict[int, list[float]],
            states: dict[int, list[str]] | None = None,
            dropped: dict[int, list[int | None]] | None = None):
    """``{channel: [R1]}`` plus the fit SHAPE the consensus rule reads.

    ``arc_state`` defaults to ``closed`` on every round of every channel, which
    is the ordinary bench shape and the one the rule must never fire on; a test
    that wants a disagreement says so, per channel, per round.
    """
    rounds = max(len(values) for values in series.values())
    return [
        [RoundFit(channel=ch, sigma=1.0 / values[i], r1_ohms=values[i],
                  arc_state=(states or {}).get(ch, ["closed"] * rounds)[i],
                  n_points_dropped=(dropped or {}).get(ch, [0] * rounds)[i])
         for ch, values in sorted(series.items())]
        for i in range(rounds)]


def _one_open_round(index: int, n: int = ARMED_WINDOW) -> list[str]:
    return ["open" if i == index else "closed" for i in range(n)]


class TestPhysicalConsensus:
    """A round taken mid-transient is a different CELL, not a noisier reading.

    The operator ruled against statistical outlier removal by name (`[a243]`), so
    the discriminator is the shape of the fit -- did the arc close, and did the
    fitter withhold a materially different part of the sweep -- and never the size
    of the residual. Every fixture below is a real r6 shape.
    """

    #: ch18's own failure, moved inside the armed window: flat but for one round
    #: at 4x, and the fit on that round says why -- the arc did not close.
    EXCURSION_ROUND = 2

    def _excursion_board(self, *, state_disagrees: bool = True):
        excursion = _quiet_r1(ARMED_WINDOW)
        excursion[self.EXCURSION_ROUND] *= 4.0
        quiet = _quiet_r1(ARMED_WINDOW)
        series = {18: excursion, 19: list(quiet), 20: list(quiet)}
        states = ({18: _one_open_round(self.EXCURSION_ROUND)}
                  if state_disagrees else None)
        return _shaped(series, states)

    def test_consensus_excludes_one_disagreeing_round_and_certifies(self):
        """The whole deliverable: five rounds agreed, one did not, and the one
        that did not leaves ch18's regression -- sigma AND its time point.

        Without it ch18 is `rate_undetectable` on the strength of a round that
        measured a different cell, the board is two quiet channels short of its
        minimum, and it then waits for that round to age out of a trailing
        window it has not finished passing.
        """
        window = self._excursion_board()
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.settled is True
        assert check.quiet == [18, 19, 20]
        assert check.consensus_unavailable is False
        assert [(e.channel, e.round_index) for e in check.excluded_rounds] == [
            (18, self.EXCURSION_ROUND)]
        assert check.excluded_rounds[0].reason == \
            "arc_state open vs consensus closed"
        # BOTH halves left: six points regressed, over the span the six that
        # survived actually cover, not the seven-round span.
        assert check.by_channel[18].n_points == ARMED_WINDOW - 1

    def test_the_same_board_without_the_rule_does_not_certify(self):
        """**The positive control for the two tests above it.**

        Identical window, identical tolerances, rule disarmed -- and if this ever
        goes green the exclusion has stopped doing anything and the test that
        certifies is passing for some other reason.
        """
        window = self._excursion_board()
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3,
                           consensus_exclude=False)

        assert check.settled is False
        assert 18 not in check.quiet
        assert check.excluded_rounds == []

    def test_an_excursion_whose_shape_agrees_is_not_excluded(self):
        """The rule is not an outlier filter, and this is where the two part.

        Same 4x excursion, same everything -- but every round's fit closed, so
        nothing on the fit says a different cell was measured and the round
        stays in. A residual-sized rule would drop it; this one must not.
        """
        window = self._excursion_board(state_disagrees=False)
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.excluded_rounds == []
        assert check.settled is False and 18 not in check.quiet

    def test_alternating_states_exclude_nothing(self):
        """The r6 ch4/5/6 shape: the arc opens and closes round to round.

        Two or more disagreements is not an outlier, it is a channel that is not
        consensus-stable, and picking the minority half of a flapping cell would
        be choosing a verdict rather than measuring one. Left whole, for the
        survivors mechanism -- so the verdict must be what it is with the rule
        off, field for field.
        """
        flapping = _quiet_r1(ARMED_WINDOW)
        flapping[2] *= 4.0
        quiet = _quiet_r1(ARMED_WINDOW)
        window = _shaped(
            {4: flapping, 19: list(quiet), 20: list(quiet)},
            {4: ["closed", "open"] * 3 + ["closed"]})

        armed = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)
        off = rate_check(window, _times(ARMED_WINDOW),
                         tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3,
                         consensus_exclude=False)

        assert armed.excluded_rounds == []
        assert armed.by_channel == off.by_channel
        assert (armed.settled, armed.quiet, armed.reason) == \
            (off.settled, off.quiet, off.reason)
        assert 4 not in armed.quiet

    def test_monotone_same_state_trend_is_untouched(self):
        """The r6 ch12/ch13 shape: a cell genuinely drifting, one shape
        throughout. No exclusion rule may touch a channel that is moving --
        there is no round to blame, and blaming one would certify a drying film.
        """
        window = _shaped({12: _decaying_transient_r1(ARMED_WINDOW),
                          19: _quiet_r1(ARMED_WINDOW),
                          20: _quiet_r1(ARMED_WINDOW)})

        armed = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)
        off = rate_check(window, _times(ARMED_WINDOW),
                         tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3,
                         consensus_exclude=False)

        assert armed.excluded_rounds == []
        assert armed.moving == off.moving == [12]
        assert armed.settled is False and armed.reason == off.reason

    def test_consensus_is_inert_when_no_round_carries_an_arc_state(self):
        """``SUBAGENT_RULES.md`` §3.1(a), as an assertion.

        A feeder that has not shipped its half sends rounds with no shape at all.
        The mode is then a unanimous `""`, nothing can ever disagree, and an
        empty `excluded_rounds` would say "no round disagreed" using the same
        token as "no round was asked". The two are separate fields, and the
        second one is the only evidence that the check was even able to run.
        """
        bare = _window({18: _quiet_r1(ARMED_WINDOW),
                        19: _quiet_r1(ARMED_WINDOW),
                        20: _quiet_r1(ARMED_WINDOW)})
        check = rate_check(bare, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.consensus_unavailable is True
        assert check.excluded_rounds == []
        # ...and a window that DID carry the shape says so with the other token,
        # which is what makes the pair readable.
        shaped = rate_check(
            _shaped({18: _quiet_r1(ARMED_WINDOW), 19: _quiet_r1(ARMED_WINDOW),
                     20: _quiet_r1(ARMED_WINDOW)}),
            _times(ARMED_WINDOW), tol_per_hour=RATE_TOL_LN_PER_H,
            min_channels=3)
        assert shaped.consensus_unavailable is False
        assert shaped.excluded_rounds == []

    # -- the bucket, which exists so mask jitter cannot manufacture a drop ----

    def test_one_masked_point_of_jitter_is_inside_the_bucket(self):
        """13 and 14 dropped points are the same bucket, and must be.

        `n_points_dropped` moves by +/-1 when one frequency crosses a finiteness
        or phase test between rounds of an identically behaving cell. Exact
        equality would exclude a round for that, which is the residual-sized
        behaviour the operator ruled out.
        """
        quiet = _quiet_r1(ARMED_WINDOW)
        window = _shaped(
            {18: list(quiet), 19: list(quiet), 20: list(quiet)},
            dropped={18: [13, 13, 14, 13, 13, 13, 13]})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.excluded_rounds == []
        assert SETTLE_CONSENSUS_DROP_BUCKET == 5

    def test_a_whole_decade_withheld_on_one_round_crosses_the_bucket(self):
        """The case the bucket exists to catch: the state agreed and the fit
        nonetheless saw a materially different spectrum."""
        quiet = _quiet_r1(ARMED_WINDOW)
        window = _shaped(
            {18: list(quiet), 19: list(quiet), 20: list(quiet)},
            dropped={18: [0, 0, 13, 0, 0, 0, 0]})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert [(e.channel, e.round_index, e.reason)
                for e in check.excluded_rounds] == [
            (18, 2, "drop_bucket 2 vs consensus 0")]

    # -- the window the exclusion is paid for out of -------------------------

    def test_the_rate_window_grows_by_one_while_the_rule_is_armed(self):
        """Operator ruling 2026-09-14, and it is a collision being paid rather
        than a margin: the shipped window is 6 and `_channel_rate` refuses below
        6, so a 6-round window plus an exclusion makes the rule a no-op that
        returns the SAFE answer -- the channel goes non-evaluable instead of
        quiet, and nothing goes red.
        """
        tracker = SettleTracker(criterion="rate", rate_tol_per_hour=0.30)

        assert tracker.min_fit_points == DEFAULT_SETTLE_MIN_FIT_POINTS
        assert tracker.rate_window_rounds == DEFAULT_SETTLE_MIN_FIT_POINTS + 1
        assert tracker.judged_rounds == tracker.rate_window_rounds == ARMED_WINDOW

    def test_the_rate_window_is_unchanged_when_the_rule_is_off(self):
        tracker = SettleTracker(criterion="rate", rate_tol_per_hour=0.30,
                                settle_consensus_exclude=False)

        assert tracker.rate_window_rounds == DEFAULT_SETTLE_MIN_FIT_POINTS == 6
        assert tracker.judged_rounds == 6

    def test_an_excluded_channel_still_clears_the_minimum_fit_points(self):
        """Why the extra round was bought, stated as the thing it buys: the
        survivors of an exclusion must still be a window worth quoting, or the
        rule trades a wrong verdict for an absent one."""
        window = self._excursion_board()
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3,
                           min_fit_points=DEFAULT_SETTLE_MIN_FIT_POINTS)

        assert check.by_channel[18].refusal != RATE_TOO_FEW_POINTS
        assert check.by_channel[18].n_points >= DEFAULT_SETTLE_MIN_FIT_POINTS

    # -- the cosmetic defect `[a243]` named ----------------------------------

    def test_the_settled_reason_quotes_the_worst_bound_among_the_certifiers(self):
        """`[a243]`: "worst 95 % bound 17.12 ln/h is within 0.23 ln/h".

        `bounds` spans every PARTICIPANT, and a participant that came back
        unjudgeable has a bound above the tolerance by definition -- so the
        sentence quoted a number that is not within the band it claims to be
        within, on the only certification in six runs. The number that certified
        is the worst one over the QUIET channels.
        """
        quiet = _quiet_r1(ARMED_WINDOW)
        window = _window({25: _flat_and_noisy_r1(ARMED_WINDOW),
                          18: list(quiet), 19: list(quiet), 20: list(quiet)})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.settled is True and 25 not in check.quiet
        certifying = max(check.by_channel[ch].upper_bound_per_hour
                         for ch in check.quiet)
        assert f"worst 95 % bound {certifying:.4f} ln/h" in check.reason
        assert certifying <= RATE_TOL_LN_PER_H
        # The number that used to be printed is still recorded -- it is the
        # board's worst participant and a real fact -- it is simply not the one
        # the certifying sentence may quote.
        assert check.max_upper_bound_per_hour > certifying
        assert f"{check.max_upper_bound_per_hour:.4f} ln/h is" not in check.reason


# ── T11.32: one round the fitter would not speak for is a ROUND, not a well ───


def _graded(series: dict[int, list[float]],
            verdicts: dict[int, list[str]] | None = None,
            modes: dict[int, list[str]] | None = None,
            states: dict[int, list[str]] | None = None):
    """`_shaped`, plus the two provenance words -- and the sigma that follows.

    A round whose verdict is anything but `ok` or `""`, or whose mode is anything
    but `value` or `""`, carries `sigma=None` beside an INTACT `r1_ohms`. That is
    not a fixture convenience: it is exactly what `_report_sigma` hands the
    tracker for a REJECT-graded or a bounded report (`autonomous_wiring.py`), and
    the one shape the round-level forgiveness exists for. A round with no R1
    either is genuine ABSENCE and is a different test.
    """
    rounds = max(len(values) for values in series.values())
    window = []
    for i in range(rounds):
        row = []
        for ch, values in sorted(series.items()):
            verdict = (verdicts or {}).get(ch, [""] * rounds)[i]
            mode = (modes or {}).get(ch, [""] * rounds)[i]
            usable = verdict in ("", "ok") and mode in ("", "value")
            row.append(RoundFit(
                channel=ch, sigma=(1.0 / values[i]) if usable else None,
                r1_ohms=values[i],
                arc_state=(states or {}).get(ch, ["closed"] * rounds)[i],
                n_points_dropped=0, sigma_mode=mode, quality_verdict=verdict))
        window.append(row)
    return window


def _one_round(index: int, word: str, otherwise: str,
               n: int = ARMED_WINDOW) -> list[str]:
    return [word if i == index else otherwise for i in range(n)]


class TestForgivableRound:
    """Rung-3a ch13 and ch22, as the two shapes that cost seven rounds each.

    ch13 had ONE round whose sigma came back a bound and was whole-channel-
    excluded for it r10-r16; ch22 had a CONVERGED fit graded `reject` on a 16.8 %
    residual against a 15 % threshold. Both are "the fit ran and reported, and
    declined to speak for the number" -- which is a fact about one round, and was
    being spent as a fact about the well. Participation fell to 2 against a
    minimum of 3 on a board whose R1 was moving smoothly the whole time.

    The budget is unchanged: at most ONE round per channel per window, shared
    across all four elements of the shape.
    """

    FORGIVEN_ROUND = 2

    def _board(self, verdicts=None, modes=None, states=None):
        quiet = _quiet_r1(ARMED_WINDOW)
        return _graded({18: list(quiet), 19: list(quiet), 20: list(quiet)},
                       verdicts=verdicts, modes=modes, states=states)

    def test_rate_check_drops_one_reject_graded_round_not_the_channel(self):
        """ch22's shape. The whole deliverable on the legacy settle route, where
        `grade_fit`'s residual REJECT is enforced unconditionally and is
        therefore the DISCRIMINATING axis -- there is no bound mode to read.
        """
        window = self._board(
            verdicts={18: _one_round(self.FORGIVEN_ROUND, "reject", "ok")})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert 18 not in check.excluded
        assert check.quiet == [18, 19, 20] and check.settled is True
        assert [(e.channel, e.round_index, e.reason)
                for e in check.excluded_rounds] == [
            (18, self.FORGIVEN_ROUND,
             "quality_verdict reject vs consensus ok")]
        # Six points regressed, not seven: the round left with its time point.
        assert check.by_channel[18].n_points == ARMED_WINDOW - 1

    def test_the_same_reject_graded_board_without_the_rule_loses_the_channel(
            self):
        """**The positive control.** Identical window, rule disarmed -- and the
        pre-T11.32 verdict is what comes back: one round with no usable sigma
        takes the whole channel, the board falls under its minimum, and the
        phase runs to its ceiling. If this ever goes green the forgiveness has
        stopped being what makes the test above certify.
        """
        window = self._board(
            verdicts={18: _one_round(self.FORGIVEN_ROUND, "reject", "ok")})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3,
                           consensus_exclude=False)

        assert check.excluded[18] == EXCLUDED_SIGMA_NULL
        assert check.participating == [19, 20]
        assert check.evaluable is False and check.settled is False

    def test_rate_check_drops_one_bound_mode_round_not_the_channel(self):
        """ch13's shape, kept as a regression guard even though T11.15 made it
        structurally unreachable on the campaign's settle feeder: the legacy
        engine builds only `unavailable` or `value`. `eis-validate`'s gated
        path, and any campaign path that re-admits bound mode, still produce it.
        """
        window = self._board(
            modes={18: _one_round(self.FORGIVEN_ROUND, "bound", "value")})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert 18 not in check.excluded and check.settled is True
        assert [(e.channel, e.round_index, e.kind, e.reason)
                for e in check.excluded_rounds] == [
            (18, self.FORGIVEN_ROUND, "sigma_mode",
             "sigma_mode bound vs consensus value")]

    def test_rate_check_two_forgivable_rounds_still_excludes_the_channel(self):
        """The budget, asserted as the thing it refuses.

        One round on each axis -- one `reject`, one `bound` -- is TWO
        disagreements out of a budget of one, and `rate_window_rounds = 7` can
        only pay for one drop before the regression falls under
        `min_fit_points`. Two is not an outlier; it is a channel that is not
        consensus-stable, and it goes whole, exactly as it did before T11.32.
        """
        window = self._board(
            verdicts={18: _one_round(2, "reject", "ok")},
            modes={18: _one_round(4, "bound", "value")})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.excluded[18] == EXCLUDED_SIGMA_NULL
        assert check.excluded_rounds == []
        assert check.participating == [19, 20] and check.evaluable is False

    def test_rate_check_one_absent_round_not_forgiven_by_consensus(self):
        """Absence is not forgivable, and this is the fixture that separates it.

        The placeholder `_window_series` inserts for a missing channel carries no
        `arc_state`, so it IS a lone shape outlier and consensus would drop it if
        it were ever asked. It must not be: a round that never ran and a fit that
        declined to speak send an operator to different places, and folding the
        first into a budget sized for the second spends a whole round's missing
        evidence on nothing.
        """
        window = self._board()
        window[self.FORGIVEN_ROUND] = [
            fit for fit in window[self.FORGIVEN_ROUND] if fit.channel != 18]
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.excluded[18] == EXCLUDED_ABSENT
        assert check.excluded_rounds == []
        assert check.participating == [19, 20]

    def test_rate_check_reject_and_arc_shape_share_one_budget(self):
        """One round, two divergent elements, ONE drop -- the shared 4-tuple.

        A round taken mid-transient opens the arc and grades `reject` for the
        same reason, and counting that as two disagreements would exclude the
        channel for being consistent with itself. The kind reported is the
        element nearest the spectrum, which is the tuple's own order.
        """
        window = self._board(
            verdicts={18: _one_round(self.FORGIVEN_ROUND, "reject", "ok")},
            states={18: _one_open_round(self.FORGIVEN_ROUND)})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert [(e.channel, e.round_index, e.kind)
                for e in check.excluded_rounds] == [
            (18, self.FORGIVEN_ROUND, "arc_state")]
        assert check.by_channel[18].n_points == ARMED_WINDOW - 1
        assert check.settled is True

    def test_a_second_problem_among_the_survivors_still_takes_the_channel(self):
        """The forgiveness buys ONE round, and the gate then runs on what is
        left. A railed R1 among the surviving six is still a railed channel."""
        quiet = _quiet_r1(ARMED_WINDOW)
        railed = list(quiet)
        railed[5] = 1.0e-2
        window = _graded({18: railed, 19: list(quiet), 20: list(quiet)},
                         verdicts={18: _one_round(2, "reject", "ok")})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3,
                           r1_bound_ohms=1.0e-2)

        assert check.excluded[18] == EXCLUDED_RAILED
        assert check.excluded_rounds == []

    def test_a_quality_verdict_alone_makes_the_rule_available(self):
        """`SUBAGENT_RULES.md` §3.1(a), widened with the tuple.

        `consensus_unavailable` asks whether the feeder shipped its half, and
        after T11.32 the shape has three words in it rather than one. A feeder
        that records a verdict and no `arc_state` HAS answered; reading only
        `arc_state` would call that window unaskable and silently decline to
        forgive anything on it.
        """
        blank = [""] * ARMED_WINDOW
        window = self._board(
            verdicts={18: _one_round(self.FORGIVEN_ROUND, "reject", "ok")},
            states={ch: list(blank) for ch in (18, 19, 20)})

        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.consensus_unavailable is False
        assert [(e.channel, e.round_index) for e in check.excluded_rounds] == [
            (18, self.FORGIVEN_ROUND)]

    def test_settle_check_is_untouched_by_the_round_level_forgiveness(self):
        """T11.32 is scoped to `rate_check`. The deviation criterion has no
        per-round-drop mechanism, was named in neither ruling, and a bound round
        reaching it is a separate, open question -- flagged rather than silently
        decided. So the SAME window the rate criterion now forgives must still
        cost the deviation criterion the whole channel.
        """
        window = self._board(
            verdicts={18: _one_round(self.FORGIVEN_ROUND, "reject", "ok")})
        check = settle_check(window, tol_rel=0.10, min_channels=3)

        assert check.excluded[18] == EXCLUDED_SIGMA_NULL
        assert check.participating == [19, 20]
        assert check.evaluable is False


# ── T11.33: no board minimum, and a word for every well ──────────────────────
#
# Operator ruling 2026-09-16 (`[a282]`), on an autonomous optimization run:
#
#   "it is important to follow through regardless of participating well count
#    and extract the highest fidelity information on each well to feed back to
#    the optimizer. Therefore, this constraint needs to be removed, each
#    individual well valued equally, and proper handling of its state relayed
#    faithfully to the autonomous machinery."
#
# Two halves, and they are one parameter. `board_minimum=None` takes the count
# gate off AND replaces `len(quiet) >= needed` with the early-stop rule ruled in
# `[a289]` -- no well moving, no well still resolving, at least one well quiet.
# `WellVerdict` is the "relayed faithfully" half: a word, a slope and a bound for
# every well on the board, including the ones nothing could be said about.
#
# Every fixture below is a FOUR-well board, because four is what the campaign
# runs and three was the old minimum -- so a four-well board is exactly where a
# count gate and a per-well rule give different answers.


def _campaign_board(series: dict[int, list[float]], absent: tuple[int, ...] = ()):
    """*series* wells that were measured, *absent* wells that never swept.

    Absence is spelled as a `RoundFit` with no sigma and no R1 in every round --
    which is what `_window_series` inserts for a channel missing from a round,
    and what the campaign feeder writes for a channel whose sweep never
    completed. The well is on the board, and that is the point: a well the
    criterion cannot speak for must still leave the phase with a word.
    """
    rounds = []
    for i in range(ARMED_WINDOW):
        row = [RoundFit(channel=ch, sigma=1.0 / values[i], r1_ohms=values[i])
               for ch, values in series.items()]
        row += [RoundFit(channel=ch) for ch in absent]
        rounds.append(sorted(row, key=lambda fit: fit.channel))
    return rounds


class TestNoBoardMinimum:
    """The count gate, off -- and what takes its place."""

    def test_rate_check_two_quiet_wells_of_four_settle_the_board(self):
        """The ruling, in one assertion. Two absent wells used to take a
        four-well board under `min_channels=3` to its ceiling every trial; now
        the two that reported are judged on their own evidence and certify.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW)}, absent=(3, 4))
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        assert check.settled is True and check.evaluable is True
        assert check.quiet == [1, 2]
        assert check.excluded == {3: EXCLUDED_ABSENT, 4: EXCLUDED_ABSENT}

    def test_rate_check_the_same_board_still_refuses_under_an_int_minimum(self):
        """**The positive control**, and the regression guard for the two tool
        paths the ruling did not touch. The identical window under
        `min_channels=3` must come back exactly as it does today: two
        participants, not evaluable, run to the ceiling. If this ever goes green
        the policy has stopped being a policy.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW)}, absent=(3, 4))
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.participating == [1, 2]
        assert check.evaluable is False and check.settled is False
        assert "< 3 required" in check.reason

    def test_rate_check_one_moving_well_holds_the_board(self):
        """The clause that survives the ruling untouched. `rate_moving` is the
        one refusal that is evidence about the SAMPLE, so three quiet wells
        cannot outvote it -- the board is still conditioning.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW),
                                  3: _quiet_r1(ARMED_WINDOW),
                                  4: _decaying_transient_r1(ARMED_WINDOW)})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        assert check.moving == [4] and check.settled is False
        assert check.quiet == [1, 2, 3]
        assert check.by_well[4].word == RATE_MOVING
        # Recorded even under the refusal: the slope is what T11.17 wants on the
        # row, and "how fast was it moving when we stopped waiting?" is a
        # question a refusal does not answer.
        assert check.by_well[4].rate_per_hour < 0.0

    def test_rate_check_one_undetectable_well_holds_the_board(self):
        """The clause the cheaper rule (`quiet >= 1 and moving == []`) would
        have dropped. A well too noisy to judge at this tolerance is still
        RESOLVING -- more rounds is exactly what it asks for -- so stopping now
        would spend the hold on three wells and abandon the fourth.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW),
                                  3: _quiet_r1(ARMED_WINDOW),
                                  4: _flat_and_noisy_r1(ARMED_WINDOW)})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        assert check.undetectable == [4] and check.settled is False
        assert check.quiet == [1, 2, 3]
        assert "still resolving" in check.reason and "ch4" in check.reason
        # Still EVALUABLE: ch4 produced a rate estimate, it just could not be
        # certified. The board word is `ceiling`, never `not_evaluable`.
        assert check.evaluable is True

    def test_rate_check_one_unsettleable_well_does_not_hold_the_board(self):
        """The asymmetry, and it is the whole reason the rule names two classes
        rather than one. The SAME noisy well, now given a relative tolerance its
        own residual cannot meet: no hold length can ever certify it, so waiting
        for it is a hold spent on nothing and the board stops.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW),
                                  3: _quiet_r1(ARMED_WINDOW),
                                  4: _flat_and_noisy_r1(ARMED_WINDOW)})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, tol_rel=0.10,
                           board_minimum=None)

        assert check.unsettleable == [4] and check.settled is True
        assert check.by_well[4].word == EXCLUDED_UNSETTLEABLE
        # It neither blocked nor certified: the certificate is quoted over the
        # wells that were quiet, and ch4 is not one of them.
        assert check.quiet == [1, 2, 3]

    def test_rate_check_board_minimum_none_never_reports_not_evaluable_on_count(
            self):
        """One well is a board. `not_evaluable` is reserved for a board where
        NOTHING could be judged, and a count is no longer one of the ways to get
        there -- which is delta 2 of the spec, stated as the pair.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW)},
                                 absent=(2, 3, 4))
        times = _times(ARMED_WINDOW)

        alone = rate_check(window, times, tol_per_hour=RATE_TOL_LN_PER_H,
                           board_minimum=None)
        assert alone.evaluable is True and alone.settled is True
        assert "required" not in alone.reason

        counted = rate_check(window, times, tol_per_hour=RATE_TOL_LN_PER_H,
                             min_channels=3)
        assert counted.evaluable is False

    def test_rate_check_a_board_with_no_judgeable_well_is_still_not_evaluable(
            self):
        """The floor under the ruling. Removing the count gate does not make an
        empty board evaluable -- and every well still leaves with its word, which
        is what makes that verdict attributable instead of a bare refusal.
        """
        window = [[RoundFit(channel=ch) for ch in (1, 2, 3, 4)]
                  for _ in range(ARMED_WINDOW)]
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        assert check.evaluable is False and check.settled is False
        assert {ch: verdict.word for ch, verdict in check.by_well.items()} == {
            ch: EXCLUDED_ABSENT for ch in (1, 2, 3, 4)}


class TestTooFewPointsSplits:
    """`[a289]`'s refinement, which is a distinction between two shortfalls.

    A well short of points because the WINDOW has not filled is still resolving
    and holds the board. A well short because ITS OWN rounds went missing is
    not, and must not be -- otherwise a well whose sweeps keep failing reports
    `rate_too_few_points` on every window for the life of the run, no board ever
    stops early, and every trial spends its whole `max_hold_s` on a heater-only
    rig whose post-anneal descent is already the rate limiter.
    """

    def test_rate_check_a_short_window_holds_the_board(self):
        """The blocking half. Every well counted every round and is STILL short,
        so the shortfall belongs to the observation and more rounds fix it.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW),
                                  3: _quiet_r1(ARMED_WINDOW)})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H,
                           min_fit_points=ARMED_WINDOW + 2, board_minimum=None)

        assert check.settled is False
        assert "still resolving" in check.reason
        assert RATE_TOO_FEW_POINTS in check.reason
        assert check.by_well[1].n_points == ARMED_WINDOW

    def test_rate_check_a_well_short_of_its_own_rounds_does_not_hold_the_board(
            self):
        """The non-blocking half, on the fixture T11.32 built: ch18 spends the
        consensus rule's forgiven round and regresses on six points while its
        neighbours regress on seven. At `min_fit_points=7` that is a shortfall
        of ch18's own making, so the two wells that CAN be judged certify and
        the board stops instead of holding for a round ch18 will never gain.
        """
        window = _graded(
            {18: _quiet_r1(ARMED_WINDOW), 19: _quiet_r1(ARMED_WINDOW),
             20: _quiet_r1(ARMED_WINDOW)},
            verdicts={18: _one_round(2, "reject", "ok")})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H,
                           min_fit_points=ARMED_WINDOW, board_minimum=None)

        assert check.by_well[18].word == RATE_TOO_FEW_POINTS
        assert check.by_well[18].n_points == ARMED_WINDOW - 1
        assert check.quiet == [19, 20] and check.settled is True


class TestWellVerdict:
    """A word, a slope and a bound for every well -- design C."""

    def test_rate_check_reports_a_verdict_for_every_well_including_absent_ones(
            self):
        """`by_well` covers the BOARD and not the participants. The four lists
        beside it name the wells that fell into each bucket; a well that fell
        into none of them -- absent, or whose fit never converged -- left no
        trace at all before this, which is what "every well followed through"
        is named for.
        """
        quiet = _quiet_r1(ARMED_WINDOW)
        window = _graded({1: list(quiet),
                          2: _decaying_transient_r1(ARMED_WINDOW),
                          3: list(quiet)},
                         verdicts={3: ["reject"] * ARMED_WINDOW})
        window = [row + [RoundFit(channel=4)] for row in window]
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        assert {ch: verdict.word for ch, verdict in check.by_well.items()} == {
            1: SETTLE_SETTLED, 2: RATE_MOVING,
            3: WELL_FIT_FAILED, 4: EXCLUDED_ABSENT}
        # The word is re-spelled for the operator; the reason keeps the token
        # `RateCheck.excluded` and every stored run already use, so the two
        # records still join.
        assert check.excluded[3] == EXCLUDED_SIGMA_NULL
        assert EXCLUDED_SIGMA_NULL in check.by_well[3].reason

    def test_rate_check_carries_the_slope_and_bound_of_a_well_that_did_not_certify(
            self):
        """T11.17, as the thing a refusal does not excuse. A well that came back
        `rate_undetectable` still measured a slope, and the optimizer's record
        wants it -- the decision to WEIGHT anything by certification is a
        separate ruling, and it can only be made on evidence that was kept.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW),
                                  3: _quiet_r1(ARMED_WINDOW),
                                  4: _flat_and_noisy_r1(ARMED_WINDOW)})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        noisy = check.by_well[4]
        assert noisy.word == RATE_UNDETECTABLE
        for value in (noisy.rate_per_hour, noisy.upper_bound_per_hour,
                      noisy.stderr_per_hour, noisy.resid_rel):
            assert value is not None and np.isfinite(value)
        assert noisy.span_s == pytest.approx(_times(ARMED_WINDOW)[-1])

    def test_rate_check_one_reject_graded_round_is_forgiven_once_and_counted(
            self):
        """The two halves of the same round, and they must not collapse. The
        consensus rule FORGIVES it -- ch18 keeps its place in the window -- and
        `quality_rejects` COUNTS it, so a well carried by forgiveness is not
        indistinguishable from one that never needed any.
        """
        window = _graded(
            {18: _quiet_r1(ARMED_WINDOW), 19: _quiet_r1(ARMED_WINDOW),
             20: _quiet_r1(ARMED_WINDOW)},
            verdicts={18: _one_round(2, "reject", "ok")})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        verdict = check.by_well[18]
        assert verdict.word == SETTLE_SETTLED and check.settled is True
        assert verdict.quality_rejects == 1
        assert verdict.forgiven_round is not None
        assert verdict.forgiven_round.round_index == 2
        assert check.by_well[19].quality_rejects == 0
        assert check.by_well[19].forgiven_round is None

    def test_rate_check_two_reject_graded_rounds_still_exclude_the_channel(self):
        """The budget, unchanged by the ruling. Two disagreements is not an
        outlier, so ch18 goes whole -- and under `board_minimum=None` that costs
        the BOARD nothing: `fit_failed` carries no information, so it neither
        blocks nor certifies, and its two neighbours certify on their own.
        """
        window = _graded(
            {18: _quiet_r1(ARMED_WINDOW), 19: _quiet_r1(ARMED_WINDOW),
             20: _quiet_r1(ARMED_WINDOW)},
            verdicts={18: ["reject" if i in (2, 4) else "ok"
                           for i in range(ARMED_WINDOW)]})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        assert check.excluded[18] == EXCLUDED_SIGMA_NULL
        assert check.excluded_rounds == []
        assert check.by_well[18].word == WELL_FIT_FAILED
        assert check.by_well[18].quality_rejects == 2
        assert check.by_well[18].rate_per_hour is None
        assert check.quiet == [19, 20] and check.settled is True

    def test_rate_check_carries_the_last_sigma_mode_a_well_recorded(self):
        """Provenance, and the absence it must not be confused with. The
        placeholder round for a channel that missed one carries `""` = never
        asked; letting it overwrite a mode the well DID report would spell an
        absence with the token for an answer.
        """
        window = _graded(
            {18: _quiet_r1(ARMED_WINDOW), 19: _quiet_r1(ARMED_WINDOW),
             20: _quiet_r1(ARMED_WINDOW)},
            modes={18: _one_round(2, "bound", "value")})
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, board_minimum=None)

        assert check.by_well[18].sigma_mode == "value"
        assert check.by_well[18].forgiven_round.kind == "sigma_mode"

    def test_rate_check_by_well_omits_a_participant_no_verdict_was_reached_on(
            self):
        """`SUBAGENT_RULES.md` §3.1(a), pointed at the new field.

        A window an int `board_minimum` refuses on its COUNT has participants
        that were never judged, and no word in the vocabulary is true of them:
        they are not settled, not moving, not resolving and not excluded. So
        they get no entry rather than a word somebody invented. The wells that
        WERE excluded still carry theirs, because that is a fact the window
        established before the count gate fired.

        Under `board_minimum=None` this branch is reachable only with no
        participants at all, so `by_well` is complete on the campaign path.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW)}, absent=(3, 4))
        check = rate_check(window, _times(ARMED_WINDOW),
                           tol_per_hour=RATE_TOL_LN_PER_H, min_channels=3)

        assert check.participating == [1, 2]
        assert {ch: verdict.word for ch, verdict in check.by_well.items()} == {
            3: EXCLUDED_ABSENT, 4: EXCLUDED_ABSENT}


class TestBoardMinimumPolicyReachesBothCriteria:
    """One policy, two criteria, and three construction sites."""

    def test_settle_check_board_minimum_none_judges_one_participant(self):
        """The deviation criterion gains the policy and NOTHING else: its
        settled decision is already per channel and aggregates with `max`, so
        one participant is judged by exactly the rule fifteen would have been.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW)},
                                 absent=(2, 3, 4))

        assert settle_check(window, tol_rel=0.10, board_minimum=None).settled
        assert settle_check(window, tol_rel=0.10,
                            min_channels=3).evaluable is False

    def test_settle_check_board_minimum_none_still_refuses_an_empty_board(self):
        """`max` over no participants is not a settled board, it is an
        exception -- and an empty board is an absence of evidence under every
        reading of the ruling.
        """
        window = [[RoundFit(channel=ch) for ch in (1, 2)]
                  for _ in range(ARMED_WINDOW)]
        check = settle_check(window, tol_rel=0.10, board_minimum=None)

        assert check.evaluable is False and check.settled is False

    def test_settle_tracker_board_minimum_defaults_to_min_channels(self):
        """Backward compatibility, asserted rather than assumed. Twelve
        references in `eis_validate_hold.py` and a hardware-actuating CLI in
        `workflows/equilibration.py` construct this without the parameter, and
        the ruling is scoped away from both.
        """
        assert SettleTracker(min_channels=4).board_minimum == 4
        assert SettleTracker(min_channels=4,
                             board_minimum=None).board_minimum is None
        assert SettleTracker(min_channels=4, board_minimum=1).board_minimum == 1

    def test_settle_tracker_board_minimum_none_settles_a_two_well_board_on_rate(
            self):
        """End to end through the tracker, on the criterion the campaign routes
        on. Two quiet wells of four over seven rounds: settled under the policy,
        and the identical stream is not evaluable without it.
        """
        window = _campaign_board({1: _quiet_r1(ARMED_WINDOW),
                                  2: _quiet_r1(ARMED_WINDOW)}, absent=(3, 4))
        times = _times(ARMED_WINDOW)

        def run(**policy):
            tracker = SettleTracker(
                criterion="rate", rate_tol_per_hour=RATE_TOL_LN_PER_H,
                tol_rel=0.10, min_channels=3, **policy)
            for fits, t_s in zip(window, times):
                tracker.observe(fits, t_s=t_s)
            return tracker

        assert run(board_minimum=None).settled is True
        assert run().settled is False

    def test_settle_tracker_board_minimum_none_separates_ceiling_from_not_evaluable(
            self):
        """The board word, which is what `ever_evaluable` carries. A board whose
        only well is too noisy to judge ran to its CEILING -- the criterion was
        evaluable and said no. A board where nothing swept at all is
        `not_evaluable`, and after this ruling that is the only way to get there.
        """
        times = _times(ARMED_WINDOW)

        def run(window):
            tracker = SettleTracker(
                criterion="rate", rate_tol_per_hour=RATE_TOL_LN_PER_H,
                min_channels=3, board_minimum=None)
            for fits, t_s in zip(window, times):
                tracker.observe(fits, t_s=t_s)
            return tracker.outcome(stopped_early=False)

        noisy = _campaign_board({1: _flat_and_noisy_r1(ARMED_WINDOW)},
                                absent=(2, 3, 4))
        nothing = [[RoundFit(channel=ch) for ch in (1, 2, 3, 4)]
                   for _ in range(ARMED_WINDOW)]

        assert run(noisy) == SETTLE_CEILING
        assert run(nothing) == SETTLE_NOT_EVALUABLE
