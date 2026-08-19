"""EIS sweep parameters and script generation (P2.3 / data trust).

Before this, an autonomous campaign's measurement step pointed at a temp-dir
``.mscr`` that **nothing on the campaign path ever wrote** — so it measured with
whatever some earlier HT or manual session left behind, while recording
``eis_preset`` in its metadata as though that preset had been applied.
"""

from __future__ import annotations

import pytest

from softae.core.eis_scripts import (
    DEFAULT_NPTS,
    EISParams,
    build_eis_scripts,
    mscr_path_for_channel,
)


class TestResolution:
    def test_preset_values_are_read_from_config(self):
        # Literals track [eis_presets.Standard]; last moved 2026-08-17 by the
        # mains-notch retune (npts 35 -> 34, f_lo 4000 -> 3912 mHz).
        p = EISParams.from_preset("Standard")
        assert p.npts == 34
        assert p.f_hi == 200_000
        assert p.f_lo_mHz == 3_912

    def test_explicit_overrides_beat_the_preset(self):
        p = EISParams.from_preset("Standard", npts=99)
        assert p.npts == 99
        assert p.f_hi == 200_000      # untouched keys still come from the preset

    def test_none_overrides_are_ignored(self):
        """An unset widget/field must not clobber a preset value with None."""
        expected = EISParams.from_preset("Standard").npts
        assert EISParams.from_preset("Standard", npts=None).npts == expected

    def test_unknown_preset_falls_back_to_defaults_without_raising(self):
        """A 3 a.m. campaign must not refuse to start over a preset typo."""
        assert EISParams.from_preset("NoSuchPreset").npts == DEFAULT_NPTS

    def test_no_preset_gives_defaults(self):
        assert EISParams.from_preset(None) == EISParams()

    def test_metadata_reports_what_was_actually_applied(self):
        """Provenance must describe the sweep that ran, not the one requested."""
        meta = EISParams.from_preset("NoSuchPreset").as_metadata()
        assert meta["eis_npts"] == DEFAULT_NPTS


class TestScriptBuilding:
    def test_writes_one_script_per_channel(self, tmp_path, monkeypatch):
        written = build_eis_scripts([1, 2, 3], EISParams())
        assert len(written) == 3
        for ch in (1, 2, 3):
            assert mscr_path_for_channel(ch) in written

    def test_overwrites_a_stale_script(self):
        """The whole point: last session's parameters must not survive."""
        path = mscr_path_for_channel(7)
        with open(path, "w") as fh:
            fh.write("STALE CONTENT FROM A PREVIOUS SESSION")

        build_eis_scripts([7], EISParams(npts=41))

        with open(path) as fh:
            assert "STALE" not in fh.read()

    def test_a_failing_channel_does_not_abort_the_rest(self, monkeypatch):
        """Better to lose one channel's script than the whole campaign start."""
        import softae.drivers.mscr_library as lib

        real = lib.eis_run_mscrbuild

        def flaky(filename, mux_ch, **kw):
            if mux_ch == 2:
                raise OSError("disk full")
            return real(filename, mux_ch, **kw)

        monkeypatch.setattr(lib, "eis_run_mscrbuild", flaky)
        written = build_eis_scripts([1, 2, 3], EISParams())
        assert len(written) == 2

    def test_params_reach_the_builder(self, monkeypatch):
        import softae.drivers.mscr_library as lib

        seen = {}
        monkeypatch.setattr(
            lib, "eis_run_mscrbuild",
            lambda filename, mux_ch, **kw: seen.update(kw))

        build_eis_scripts([1], EISParams(f_hi=12345, f_lo_mHz=678, npts=9, mv_ac=11))

        assert seen["f_hi"] == 12345
        assert seen["f_lo"] == 678      # note: builder's kwarg is f_lo (mHz)
        assert seen["npts"] == 9
        assert seen["mVac"] == 11


class TestCampaignIntegration:
    def test_campaign_spec_carries_eis_overrides(self):
        from softae.core.autonomous_wiring import CampaignSpec

        spec = CampaignSpec(
            name="c", channels=(1,), eis_preset="Standard",
            eis_overrides={"npts": 77},
        )
        p = EISParams.from_preset(spec.eis_preset, **spec.eis_overrides)
        assert p.npts == 77
        assert p.f_hi == 200_000


class TestRecipeName:
    """P2.3: the spec can name any registered recipe, not just two of them."""

    def test_recipe_name_wins(self):
        from softae.core.autonomous_wiring import CampaignSpec

        spec = CampaignSpec(name="c", channels=(1,), recipe_name="two_phase")
        assert spec.resolved_recipe_name() == "two_phase"

    def test_legacy_bool_still_honoured(self):
        from softae.core.autonomous_wiring import CampaignSpec

        assert CampaignSpec(
            name="c", channels=(1,), two_phase=True
        ).resolved_recipe_name() == "two_phase"
        assert CampaignSpec(
            name="c", channels=(1,)
        ).resolved_recipe_name() == "single_drop"

    def test_flush_rate_follows_the_resolved_recipe_not_the_bool(self):
        """A spec naming two_phase must get the two-phase line rate."""
        from softae.core.autonomous_wiring import CampaignSpec

        spec = CampaignSpec(
            name="c", channels=(1,), recipe_name="two_phase", line_flush_rate=444.0)
        assert spec.deposition_settings(pcb={}).flush_rate == 444.0


# ── Piecewise sweeps: frequency literals, segments, and the emitter ───────────
#
# `mscr_freq_literal` is the highest-risk unit in the piecewise work. A MethodSCRIPT
# frequency is an integer mantissa plus an optional SI prefix, so a wrong suffix is a
# silent 1000x band error the instrument executes without complaint, and the spectrum
# that comes back is indistinguishable from a real one measured somewhere else. Hence
# the exhaustive table rather than a few representative cases.

class TestFrequencyLiterals:
    @pytest.mark.parametrize("f_hz,expected", [
        (20, "20"),                 # bare integer form is canonical at and above 1 Hz
        (0.5, "500m"),
        (200_000, "200000"),        # NOT "200k" — see the byte-identity test below
        (0.016, "16m"),             # the instrument floor
        (6.475, "6475m"),           # Quick
        (1.351, "1351m"),           # Extended
        (0.228, "228m"),            # Longest
        (0.05, "50m"),
        (1.7, "1700m"),
        (1033, "1033"),
        (0.0005, "500u"),
    ])
    def test_literal_renders_the_coarsest_exact_form(self, f_hz, expected):
        from softae.drivers.mscr_library import mscr_freq_literal

        assert mscr_freq_literal(f_hz) == expected

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
    def test_unrepresentable_values_raise_rather_than_render(self, bad):
        from softae.drivers.mscr_library import mscr_freq_literal

        with pytest.raises(ValueError):
            mscr_freq_literal(bad)

    def test_over_the_digit_budget_raises_and_names_the_value(self):
        from softae.drivers.mscr_library import mscr_freq_literal

        with pytest.raises(ValueError, match="1234567890"):
            mscr_freq_literal(1_234_567_890)

    def test_a_value_off_the_grid_raises_rather_than_rounding_silently(self):
        # 20.0000001 Hz is renderable in neither Hz, mHz nor uHz. Rounding it would
        # be a small lie, and the only defensible small lie is `quantize_hz`, where
        # the caller asked for one.
        from softae.drivers.mscr_library import mscr_freq_literal

        with pytest.raises(ValueError, match="quantize_hz"):
            mscr_freq_literal(20.0000001)

    def test_literal_round_trips_through_the_instrument_parser(self):
        # The inverse is the instrument's own prefix table, imported HERE and never
        # in `mscr_library` — that module stays stdlib-only and config-free.
        import numpy as np

        from softae.drivers.mscr_library import mscr_freq_literal, quantize_hz
        from softae.drivers.palmsens.mscript import SI_PREFIX_FACTOR

        for f in np.logspace(np.log10(0.016), np.log10(200_000), 200):
            target = quantize_hz(f)
            literal = mscr_freq_literal(target)
            suffix = literal[-1] if literal[-1].isalpha() else " "
            mantissa = literal[:-1] if suffix != " " else literal
            assert (int(mantissa) * SI_PREFIX_FACTOR[suffix]
                    == pytest.approx(target, rel=1e-9))

    def test_no_literal_in_the_band_carries_a_zero_mantissa(self):
        import numpy as np

        from softae.drivers.mscr_library import mscr_freq_literal, quantize_hz

        for f in np.logspace(np.log10(0.016), np.log10(200_000), 200):
            literal = mscr_freq_literal(quantize_hz(f))
            assert int(literal.rstrip("mu")) > 0

    def test_the_thousandfold_trap(self):
        """20 Hz and 20 mHz differ by one character and three decades."""
        from softae.drivers.mscr_library import mscr_freq_literal

        assert mscr_freq_literal(20) == "20"
        assert mscr_freq_literal(20) != "20m"
        assert mscr_freq_literal(0.02) == "20m"

    def test_quantize_makes_every_value_in_the_band_representable(self):
        import numpy as np

        from softae.drivers.mscr_library import mscr_freq_literal, quantize_hz

        for f in np.logspace(np.log10(0.016), np.log10(200_000), 977):
            mscr_freq_literal(quantize_hz(f))       # must not raise

    def test_quantize_keeps_whole_hertz_whole(self):
        from softae.drivers.mscr_library import quantize_hz

        assert quantize_hz(200_000) == 200_000.0
        assert quantize_hz(6.475) == pytest.approx(6.475, abs=1e-9)


class TestSegmentResolution:
    def test_a_touching_boundary_is_nudged_off_the_previous_end(self):
        # Sharing a frequency would measure that point twice AND break the
        # monotonicity that lets the parser concatenate curves without re-sorting.
        from softae.drivers.mscr_library import resolve_segments

        out = resolve_segments([(200_000, 20, 10), (20, 2, 25)])
        assert out[1][0] < out[0][1]

    def test_the_nudge_is_one_log_step_of_the_segments_own_grid(self):
        # Not a fixed epsilon: the right distance depends on how dense that segment
        # is, and a fixed one would be invisible on a coarse grid and a hole in a
        # fine one.
        import math

        from softae.drivers.mscr_library import resolve_segments

        out = resolve_segments([(200_000, 20, 10), (20, 2, 25)])
        step = (math.log10(20) - math.log10(2)) / 25
        assert out[1][0] == pytest.approx(10 ** (math.log10(20) - step), rel=1e-6)

    def test_an_ascending_sequence_raises(self):
        from softae.drivers.mscr_library import resolve_segments

        with pytest.raises(ValueError, match="descend"):
            resolve_segments([(20, 2, 10), (200_000, 20, 10)])

    def test_a_genuine_overlap_raises_rather_than_being_papered_over(self):
        from softae.drivers.mscr_library import resolve_segments

        with pytest.raises(ValueError, match="overlap"):
            resolve_segments([(200_000, 20, 10), (100, 2, 25)])

    @pytest.mark.parametrize("bad", [
        [(20, 200, 10)],            # inverted
        [(20, 20, 10)],             # zero width
        [(20, 2, 0)],               # no points
        [(20, 0, 10)],              # zero bound
    ])
    def test_malformed_segments_raise(self, bad):
        from softae.drivers.mscr_library import resolve_segments

        with pytest.raises(ValueError):
            resolve_segments(bad)

    def test_every_resolved_bound_is_representable(self):
        from softae.drivers.mscr_library import mscr_freq_literal, resolve_segments

        for f_start, f_end, _ in resolve_segments(
                [(200_000, 20, 10), (20, 2, 25), (2, 0.016, 5)]):
            mscr_freq_literal(f_start)
            mscr_freq_literal(f_end)

    def test_resolution_is_pure(self, tmp_path):
        import os

        from softae.drivers.mscr_library import resolve_segments

        before = set(os.listdir(tmp_path))
        segments = [(200_000.0, 20.0, 10), (20.0, 2.0, 25)]
        resolve_segments(segments)
        assert segments == [(200_000.0, 20.0, 10), (20.0, 2.0, 25)]
        assert set(os.listdir(tmp_path)) == before


class TestSegmentedScriptEmission:
    PRESETS = [("Quick", 27, 6_475), ("Standard", 34, 3_912),
               ("Extended", 53, 1_351), ("Longest", 39, 228)]

    @pytest.mark.parametrize("name,npts,f_lo_mHz", PRESETS)
    def test_one_segment_build_is_byte_identical_to_eis_run_mscrbuild(
        self, tmp_path, name, npts, f_lo_mHz
    ):
        """All four timing anchors in `EIS_MEASURED_S_PER_CHANNEL` were stopwatched
        against the bytes `eis_run_mscrbuild` emits, so this is what licenses a
        second emitter to stand beside it: one whitespace character of drift and
        those anchors describe an artifact that no longer exists. It is also why
        `mscr_freq_literal` never emits `200k`."""
        from softae.drivers.mscr_library import (
            eis_run_mscrbuild,
            eis_segmented_mscrbuild,
        )

        one, many = tmp_path / f"{name}_one.mscr", tmp_path / f"{name}_many.mscr"
        eis_run_mscrbuild(str(one), 5, 10, 200_000, f_lo_mHz, npts, 0)
        eis_segmented_mscrbuild(str(many), 5, [(200_000.0, f_lo_mHz / 1000.0, npts)])
        assert many.read_bytes() == one.read_bytes()

    def test_one_meas_loop_block_per_segment(self, tmp_path):
        from softae.drivers.mscr_library import eis_segmented_mscrbuild

        path = tmp_path / "seg.mscr"
        eis_segmented_mscrbuild(
            str(path), 1, [(200_000, 20, 10), (20, 2, 25), (2, 0.5, 5)])
        text = path.read_text()
        assert text.count("meas_loop_eis") == 3
        assert text.count("endloop") == 3

    def test_the_preamble_matches_the_single_segment_builder(self, tmp_path):
        from softae.drivers.mscr_library import (
            eis_run_mscrbuild,
            eis_segmented_mscrbuild,
        )

        one, many = tmp_path / "one.mscr", tmp_path / "many.mscr"
        eis_run_mscrbuild(str(one), 3, 10, 200_000, 6_475, 27, 0)
        eis_segmented_mscrbuild(str(many), 3, [(200_000, 20, 10), (20, 6.475, 17)])
        head = "".join(one.read_text().splitlines(keepends=True)[:11])
        assert many.read_text().startswith(head)
        assert many.read_text().endswith("on_finished:\ncell_off\n\n")

    def test_channels_above_sixteen_remap_to_the_second_pico(self, tmp_path):
        from softae.drivers.mscr_library import _chan_hex, eis_segmented_mscrbuild

        path = tmp_path / "seg.mscr"
        eis_segmented_mscrbuild(str(path), 20, [(200_000, 20, 10)])
        assert f"set_gpio {_chan_hex(4)}\n" in path.read_text()

    def test_no_segments_raises_rather_than_emitting_an_empty_sweep(self, tmp_path):
        # cell_on then cell_off with nothing between returns an empty spectrum and
        # no error to explain it.
        from softae.drivers.mscr_library import eis_segmented_mscrbuild

        with pytest.raises(ValueError):
            eis_segmented_mscrbuild(str(tmp_path / "seg.mscr"), 1, [])

    def test_the_written_grid_is_the_resolved_grid(self, tmp_path):
        """What the file says must be what `resolve_segments` decided — including
        the nudge, which is the whole reason the boundary is not duplicated."""
        from softae.drivers.mscr_library import (
            eis_segmented_mscrbuild,
            mscr_freq_literal,
            resolve_segments,
        )

        segments = [(200_000, 20, 10), (20, 2, 25)]
        path = tmp_path / "seg.mscr"
        eis_segmented_mscrbuild(str(path), 1, segments)
        emitted = [line.split()[5:8] for line in path.read_text().splitlines()
                   if line.startswith("meas_loop_eis")]
        expected = [[mscr_freq_literal(a), mscr_freq_literal(b), str(n)]
                    for a, b, n in resolve_segments(segments)]
        assert emitted == expected
        assert emitted[0][1] != emitted[1][0]      # no shared boundary frequency
