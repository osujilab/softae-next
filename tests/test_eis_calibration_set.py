"""Per-field provenance in :meth:`CalibrationSet.measured_spread`.

``channels_measured`` is the union over every commissioning role, so on its own it
answers "was anything recorded on this channel", not "was *this constant* measured
here". Since ``f282e2f`` the two questions have different answers: a channel measured
open, and inheriting its short blank, is in ``channels_measured`` **and** in
``channels_assumed``. Filtering on the union alone therefore counted eight bit-identical
inherited copies as eight independent measurements and reported ``max/min = 1.0`` --
"no channel-to-channel variation" -- which is the precise reading ``measured_spread``'s
own docstring says its NaN exists to prevent.

The sibling tests in ``tests/test_eis_calibration.py``
(``TestAssumedChannelsCarryTheirMagnitude``) cover the *union* filter and are unchanged
by this; these cover the *per-field* dimension they do not reach.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from softae.analysis.eis.calibration import CalibrationSet


def _post_rederive(**over):
    """The shape ``derive_calibration`` produces after ``f282e2f``.

    Written to match the real deriver rather than invented: ``channels_assumed`` is
    ``all_channels - set(measured shorts)`` computed *before* the representative's pair
    is copied in, so ch17-23 -- measured open, never shorted -- are in **both** sets.
    """
    base = dict(
        fixture_id="mux16", hardware_hash="h1", created_at="2026-09-09",
        channels_measured=(17, 18, 19, 20, 21, 22, 23, 25),
        channels_assumed=tuple(c for c in range(1, 33) if c != 25),
        # ch25 is the one plausible short; every other channel carries its copy.
        R_short_ohm={c: 7.732418 for c in range(1, 33)},
        L_lead_H={c: 1.7294984e-6 for c in range(1, 33)},
        # Opens are never inherited: only the channels that measured one have a key.
        C_stray_F={17: 10.20e-12, 18: 14.56e-12, 19: 24.70e-12, 20: 18.51e-12,
                   21: 23.34e-12, 22: 14.50e-12, 23: 18.85e-12, 25: 53.22e-12},
        open_usable={c: True for c in (17, 18, 19, 20, 21, 22, 23, 25)},
    )
    base.update(over)
    return CalibrationSet(**base)


class TestSpreadCountsOnlyChannelsThatMeasuredTheField:
    """The regression ``f282e2f`` armed: inherited copies are not measurements."""

    def test_inherited_short_copies_report_unknown_rather_than_no_variation(self):
        # Eight identical copies of ch25's short. max/min over them is exactly 1.0,
        # and 1.0 here means "the assumption flattering itself", not "the fixture is
        # uniform". Only ch25 measured its own short, so the honest answer is NaN.
        cal = _post_rederive()
        assert cal.measured_channels("R_short_ohm") == (25,)
        assert math.isnan(cal.measured_spread("R_short_ohm"))
        assert math.isnan(cal.measured_spread("L_lead_H"))

    def test_two_genuinely_measured_shorts_report_their_ratio(self):
        # Discrimination: the method must not answer NaN unconditionally for a
        # short-derived field. ch25 and ch32 are the only two plausible short blanks
        # that have ever existed (7.732418 and 6.4581 ohm, measurements 3490/1931).
        cal = _post_rederive(
            channels_measured=(25, 32),
            channels_assumed=tuple(c for c in range(1, 33) if c not in (25, 32)),
            R_short_ohm={**{c: 7.732418 for c in range(1, 33)}, 32: 6.4581},
        )
        assert cal.measured_channels("R_short_ohm") == (25, 32)
        assert cal.measured_spread("R_short_ohm") == pytest.approx(
            7.732418 / 6.4581, rel=1e-9)

    def test_open_derived_spread_survives_channels_assumed(self):
        # The trap in the obvious fix. Subtracting `channels_assumed` from every field
        # would drop ch17-23 -- which DID measure their own opens and are only assumed
        # about their SHORT -- leaving ch25 alone and reporting NaN. That would destroy
        # the one channel-to-channel number this codebase actually cites.
        cal = _post_rederive()
        assert cal.measured_channels("C_stray_F") == (17, 18, 19, 20, 21, 22, 23, 25)
        assert cal.measured_spread("C_stray_F") == pytest.approx(
            53.22 / 10.20, rel=1e-6)

    def test_a_channel_outside_channels_measured_is_still_excluded(self):
        # The union filter is kept as well as narrowed: it can only remove.
        cal = _post_rederive(
            C_stray_F={17: 10.20e-12, 18: 14.56e-12, 99: 1.0e-9})
        assert 99 not in cal.measured_channels("C_stray_F")
        assert cal.measured_spread("C_stray_F") == pytest.approx(
            14.56 / 10.20, rel=1e-6)


class TestUndeclaredProvenanceRefusesRatherThanGuesses:
    """Unknown provenance must not be spelled the same way as "measured everywhere"."""

    def test_a_field_with_no_declared_source_reports_unknown(self):
        # `open_usable` is per-channel and a Mapping, so the old membership filter
        # would happily have taken max/min over booleans. It declares no source role,
        # so it gets NaN -- and a per-channel constant added later gets NaN too, until
        # somebody says which artifact it comes from.
        cal = _post_rederive()
        assert cal.measured_channels("open_usable") == ()
        assert math.isnan(cal.measured_spread("open_usable"))

    def test_an_absent_or_unknown_field_reports_unknown_rather_than_raising(self):
        cal = _post_rederive()
        assert cal.measured_channels("not_a_field") == ()
        assert math.isnan(cal.measured_spread("not_a_field"))
        assert math.isnan(cal.measured_spread("phase_acc"))

    def test_an_empty_calibration_reports_unknown_for_every_constant(self):
        cal = CalibrationSet()
        for name in ("R_short_ohm", "L_lead_H", "C_stray_F"):
            assert cal.measured_channels(name) == ()
            assert math.isnan(cal.measured_spread(name))


class TestTheContractHoldsAgainstTheRealDeriver:
    """SUBAGENT_RULES 3.2: the branch above is reached by what the deriver emits.

    The fixtures above encode a claim about another module -- that ``channels_assumed``
    is exactly the set which did not measure its own short. This runs the real deriver
    so the claim is checked rather than asserted, and goes red if that contract moves.
    """

    def test_derive_calibration_marks_inherited_shorts_so_the_spread_refuses(self):
        from softae.workflows.commissioning import (
            AcquiredSpectrum,
            derive_calibration,
        )

        f = np.logspace(2, 5, 24)
        w = 2 * np.pi * f
        C = {17: 10.20e-12, 18: 14.56e-12, 19: 24.70e-12, 20: 18.51e-12,
             21: 23.34e-12, 22: 14.50e-12, 23: 18.85e-12, 25: 53.22e-12}

        cal = derive_calibration(
            {
                # One plausible short, on ch25 -- the live artifact's own situation.
                "blank_short": [AcquiredSpectrum(
                    25, f, 7.732418 + 1j * w * 1.7294984e-6, electrode_mode="two")],
                "blank_open": [AcquiredSpectrum(ch, f, 1.0 / (1j * w * c),
                                                electrode_mode="two")
                               for ch, c in C.items()],
            },
            fixture_id="synthetic", created_at="2026-09-09",
            hardware_hash_value="hsyn", representative_channel=25,
            all_channels=list(range(1, 33)),
        )

        # The condition that arms the defect: measured and assumed are NOT disjoint.
        assert set(range(17, 24)) <= set(cal.channels_measured)
        assert set(range(17, 24)) <= set(cal.channels_assumed)
        assert len(cal.R_short_ohm) == 32          # every channel carries a value ...
        assert cal.measured_channels("R_short_ohm") == (25,)   # ... one is a measurement

        assert math.isnan(cal.measured_spread("R_short_ohm"))
        assert math.isnan(cal.measured_spread("L_lead_H"))
        # ... while the open-derived spread is untouched by the same re-derive.
        assert cal.measured_spread("C_stray_F") == pytest.approx(
            53.22 / 10.20, rel=1e-6)
