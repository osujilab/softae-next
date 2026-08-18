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
