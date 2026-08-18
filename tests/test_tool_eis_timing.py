"""The timing harness that re-reads ``EIS_MEASURED_S_PER_CHANNEL`` from the bench.

The thing under test is not arithmetic — it is that a tool which produces
*authority* refuses to produce it carelessly: it never writes the anchor table
itself, it discards warmup, it interleaves presets so drift cannot be attributed
to one of them, and it says so loudly when the samples do not agree.
"""

import json

import pytest

from softae.tools import eis_timing
from softae.tools.eis_timing import PresetTiming, build_parser, main


class TestPresetTimingStats:
    def test_median_is_immune_to_one_stalled_sweep(self):
        """The reason the paste block quotes median and not mean."""
        t = PresetTiming("Quick", {}, samples_s=[20.0, 20.5, 95.0])
        assert t.median_s == 20.5
        assert t.mean_s > 40.0

    def test_spread_is_peak_to_peak_over_the_median(self):
        t = PresetTiming("Quick", {}, samples_s=[20.0, 22.0])
        assert t.spread_pct == pytest.approx(2.0 / 21.0 * 100.0)

    def test_model_error_is_signed_positive_when_the_model_runs_high(self):
        t = PresetTiming("Quick", {}, samples_s=[20.0], modelled_s=22.0)
        assert t.model_error_pct == pytest.approx(10.0)

    def test_a_single_sample_reports_no_spread_rather_than_zero_spread(self):
        """The distinction the 2026-08-17 run needed: it ran --repeats 1, and a
        printed 0.0 % would have read as 'verified perfectly reproducible' when
        one sample establishes no reproducibility at all."""
        assert PresetTiming("Quick", {}, samples_s=[20.0]).spread_pct is None

    def test_an_empty_run_reports_zeroes_rather_than_raising(self):
        """A preset that never completed must not take the report down with it."""
        t = PresetTiming("Quick", {})
        assert (t.median_s, t.mean_s, t.model_error_pct) == (0, 0, 0)
        assert t.spread_pct is None and t.warmup_delta_pct is None
        assert t.is_unstable is False

    def test_the_warmup_is_a_free_replicate_when_only_one_pass_was_timed(self):
        t = PresetTiming("Quick", {}, samples_s=[20.0], warmup_s=[20.1])
        assert t.warmup_delta_pct == pytest.approx(0.5)

    def test_a_single_pass_is_still_checked_against_its_warmup(self):
        """Otherwise --repeats 1 would bypass the stability gate entirely."""
        steady = PresetTiming("Quick", {}, samples_s=[20.0], warmup_s=[20.1])
        drifting = PresetTiming("Quick", {}, samples_s=[20.0], warmup_s=[30.0])
        assert steady.is_unstable is False
        assert drifting.is_unstable is True


class TestPresetResolution:
    def test_no_argument_takes_every_preset_from_config(self):
        assert set(eis_timing._resolve_presets(None)) == {
            "Quick", "Standard", "Extended", "Longest"
        }

    def test_a_named_subset_is_honoured_in_the_order_given(self):
        assert eis_timing._resolve_presets("Longest,Quick") == ["Longest", "Quick"]

    def test_an_unknown_preset_names_what_config_actually_defines(self):
        """A typo must not silently time three presets and report four."""
        with pytest.raises(ValueError, match="Nonesuch"):
            eis_timing._resolve_presets("Quick,Nonesuch")


class TestActuationPosture:
    def test_a_real_run_requires_confirmation_and_aborts_on_refusal(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *_: "n")
        called = []
        monkeypatch.setattr(eis_timing, "run_timing",
                            lambda *a, **k: called.append(1))
        assert main(["--channel", "1"]) == 1
        assert not called, "declining the prompt must not reach the instrument"

    def test_mock_runs_need_no_confirmation(self, monkeypatch):
        def _refuse(*_a, **_k):
            raise AssertionError("mock must not prompt")

        monkeypatch.setattr("builtins.input", _refuse)
        assert main(["--mock", "--presets", "Quick", "--repeats", "1",
                     "--warmup", "0"]) == 0

    def test_a_bad_repeat_count_is_refused_before_anything_connects(self):
        assert main(["--mock", "--repeats", "0"]) == 2


class TestMockRun:
    def test_a_mock_run_times_every_preset_and_reports_each(self, capsys):
        assert main(["--mock", "--repeats", "2", "--warmup", "1"]) == 0
        out = capsys.readouterr().out
        for preset in ("Quick", "Standard", "Extended", "Longest"):
            assert preset in out

    def test_a_mock_run_refuses_to_emit_a_paste_block(self, capsys):
        """A simulated sweep times this host's filesystem, not the rig. Emitting
        anchors from it is how a 0.1 s number reaches a projection that then
        promises an overnight run will finish before breakfast."""
        main(["--mock", "--repeats", "1", "--warmup", "0"])
        out = capsys.readouterr().out
        assert "EIS_MEASURED_S_PER_CHANNEL" not in out
        assert "NO PASTE BLOCK" in out

    def test_the_report_is_ascii_so_warnings_survive_the_console(self, capsys):
        """UTF-8 hardening, same reason as the 2026-08 console work: a mangled
        warning is a skipped warning."""
        main(["--mock", "--repeats", "1", "--warmup", "0"])
        capsys.readouterr().out.encode("ascii")  # raises if any non-ASCII crept in

    def test_warmup_passes_are_recorded_but_kept_out_of_the_samples(self, capsys):
        """Discarded, not merely unlabelled — the first sweep carries connect cost."""
        main(["--mock", "--presets", "Quick", "--repeats", "2", "--warmup", "1"])
        assert capsys.readouterr().out.count("[warmup]") == 1

    def test_the_report_never_writes_the_anchor_table_itself(self, capsys):
        """It hands the operator a block to paste; it does not self-authorise."""
        from softae.core import preflight

        before = dict(preflight.EIS_MEASURED_S_PER_CHANNEL)
        main(["--mock", "--presets", "Quick", "--repeats", "1", "--warmup", "0"])
        assert preflight.EIS_MEASURED_S_PER_CHANNEL == before

    def test_json_output_carries_the_params_each_number_was_measured_at(self, tmp_path):
        """An anchor without its grid is the exact defect this tool exists to fix."""
        path = tmp_path / "timing.json"
        main(["--mock", "--presets", "Quick", "--repeats", "1", "--warmup", "0",
              "--out", str(path)])
        payload = json.loads(path.read_text())

        assert payload["schema"] == "softae.eis_timing/1"
        assert payload["mock"] is True
        quick = payload["presets"]["Quick"]
        assert quick["params"]["f_lo_mHz"] == 6_475
        assert quick["params"]["npts"] == 27
        assert len(quick["samples_s"]) == 1

    def test_an_unstable_preset_is_flagged_rather_than_averaged_away(self, capsys):
        timings = {
            "Quick": PresetTiming("Quick", {}, samples_s=[20.0, 60.0], modelled_s=21.0)
        }
        eis_timing._print_report(timings, channel=1)
        out = capsys.readouterr().out
        assert "UNSTABLE" in out
        assert "re-run before trusting" in out


class TestParser:
    def test_channel_defaults_to_one(self):
        assert build_parser().parse_args([]).channel == 1

    def test_defaults_leave_a_warmup_in_place(self):
        args = build_parser().parse_args([])
        assert args.warmup >= 1 and args.repeats >= 3
