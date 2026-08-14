"""The ``softae-shadow`` review helper — bench queue item 7's reading half.

Every fixture in this file is **synthetic text and a synthetic database**: no hardware,
no analysis engine, no campaign. The helper's whole job is to read what a shadow run
left behind, so the tests state what a shadow run leaves behind and assert the reading.

The log fragments below are copied from the shape structlog's default console renderer
actually produces under ``softae-campaign`` (unconfigured structlog → ``PrintLogger`` →
stdout), including the quoted prose, the bracketed ``issues`` list and the nested
``metrics`` dict, because a parser tested only against text it also invented proves
nothing about the format it will meet.
"""

from __future__ import annotations

import pytest

from softae.tools.shadow_review import (
    ShadowReview,
    _cmd_status,
    _split_kv,
    arc_summary,
    build_parser,
    db_summary,
    main,
    parse_line,
    railed_summary,
    recommendations,
    render,
    summarize,
)

# ── Fixtures: log lines in the exact rendering the CLI emits ──────────────────

ROUTED = ("2026-08-10 14:47:55 [info     ] eis_autorouted                 "
          "channel={ch} measurement_id={ch} role=sample step=measure_ch{ch}")

WOULD_REJECT_PRE = (
    "2026-08-10 14:47:55 [warning  ] eis_gate_would_reject          "
    "issues=['cap_flatness: d log C_app/d log f = -1.96 (DISPERSIVE)', "
    "\"valley_feature: no interior valley — do NOT fall back to the |Z| minimum\", "
    "'kk_truncation dropped 3 points'] "
    "metrics={'n_surviving': 38.0, 'tand_slope': -0.99} "
    "msg='gates observing only — spectrum used despite failing checks'"
)
# The post-fit reduction appends the Front-2 gates, so its gate set is a superset.
WOULD_REJECT_POST = (
    "2026-08-10 14:47:55 [warning  ] eis_gate_would_reject          "
    "issues=['cap_flatness: d log C_app/d log f = -1.96 (DISPERSIVE)', "
    "\"valley_feature: no interior valley — do NOT fall back to the |Z| minimum\", "
    "'pegged_parameters: pegged at a bound: CPE0_1', "
    "'kk_truncation dropped 3 points', "
    "'resolution-limited — σ reported as an upper bound'] "
    "metrics={'n_surviving': 38.0, 'n_pegged': 1.0, 'residual_rms_pct': 3.16} "
    "msg='gates observing only — spectrum used despite failing checks'"
)
GATE_REJECTED = ("2026-08-10 14:47:55 [warning  ] eis_gate_rejected              "
                 "detail=\"no interior valley\" gate=valley_feature")
POINTS_DROPPED = ("2026-08-10 14:47:52 [info     ] eis_gate_points_dropped        "
                  "detail='truncated 3 low-f point(s)' gate=kk_truncation n=3")
DECLINED_BOUND = ("2026-08-10 14:47:55 [info     ] objective_declined_bound       "
                  "mode=bound_unqualified upper_bound=4.2e-07 "
                  "msg='σ is an upper bound, not a value'")
SHADOW_SIGMA = ("2026-08-10 14:47:55 [info     ] eis_objective_shadow           "
                "mean_abs_z=203540.7 sigma=0.000132 "
                "msg='σ objective observed; mean|Z| in use'")
QUALITY_WOULD_REJECT = (
    "2026-08-10 14:47:55 [warning  ] quality_gate_would_reject      "
    "issues=['only 4 points survived'] metrics={'n_points': 4.0} "
    "msg='gate disabled — measurement used despite failing checks'")
CLI_PRINT = "  [3] -> 1.234e+05"
LEGACY_RUN = [ROUTED.format(ch=1), SHADOW_SIGMA, CLI_PRINT]


def one_rejected_spectrum(ch: int) -> list[str]:
    return [ROUTED.format(ch=ch), GATE_REJECTED, POINTS_DROPPED,
            WOULD_REJECT_PRE, WOULD_REJECT_POST]


# ── Parsing ──────────────────────────────────────────────────────────────────

class TestLineParsing:
    def test_a_console_line_yields_its_event_and_fields(self):
        parsed = parse_line(ROUTED.format(ch=7))
        assert parsed == {"event": "eis_autorouted", "level": "info", "channel": 7,
                          "measurement_id": 7, "role": "sample",
                          "step": "measure_ch7"}

    def test_a_json_line_parses_too_so_a_configured_renderer_still_reviews(self):
        parsed = parse_line('{"event": "eis_gate_would_reject", "issues": ["a: b"]}')
        assert parsed is not None
        assert parsed["event"] == "eis_gate_would_reject"

    def test_a_non_log_line_is_none_rather_than_a_half_parsed_event(self):
        # The campaign CLI's own print() output interleaves with the log stream. A
        # parser that coerced it would invent events.
        assert parse_line(CLI_PRINT) is None
        assert parse_line("") is None
        assert parse_line('  File "engine.py", line 318, in analyze_spectrum') is None

    def test_ansi_colour_is_stripped_before_matching(self):
        coloured = "\x1b[2m2026-08-10\x1b[0m [\x1b[33mwarning\x1b[0m  ] ev a=1"
        assert parse_line(coloured) == {"event": "ev", "level": "warning", "a": 1}

    def test_prose_containing_equals_and_spaces_stays_in_one_value(self):
        # `detail=` routinely contains both. A whitespace split shreds it; a naive
        # `\\w+=` regex finds keys inside the quoted prose.
        kv = _split_kv("gate=cap_flatness detail='d log C/d log f = -1.96 (n=3)' n=2")
        assert kv == {"gate": "cap_flatness",
                      "detail": "d log C/d log f = -1.96 (n=3)", "n": 2}

    def test_a_bracketed_list_survives_its_own_spaces(self):
        kv = _split_kv("issues=['a: b', 'c: d'] metrics={'x': 1.0}")
        assert kv["issues"] == ["a: b", "c: d"]
        assert kv["metrics"] == {"x": 1.0}

    def test_an_unquotable_value_degrades_to_text_instead_of_raising(self):
        assert _split_kv("path=C:\\Users\\rig\\softae.db")["path"] == \
            "C:\\Users\\rig\\softae.db"


# ── Counting ─────────────────────────────────────────────────────────────────

class TestWouldRejectCounting:
    def test_the_two_verdicts_one_spectrum_logs_are_counted_as_one_spectrum(self):
        # THE money test. `analyze_spectrum` reduces the gate log twice — admission,
        # then post-fit — and with gates.enabled=false neither call short-circuits, so
        # every rejected spectrum logs the event TWICE. Counting lines doubles the
        # would-reject rate, which is the number the arming decision turns on.
        rv = summarize(one_rejected_spectrum(1))
        assert rv.would_reject == 1
        assert rv.would_reject_verdicts == 2

    def test_four_rejected_spectra_count_four(self):
        lines = [ln for ch in (1, 2, 3, 4) for ln in one_rejected_spectrum(ch)]
        rv = summarize(lines)
        assert rv.would_reject == 4
        assert rv.n_routed == 4

    def test_an_unpaired_verdict_still_counts_as_a_spectrum(self):
        # A truncated log, or a post-fit reduction whose Front-2 gates added nothing.
        rv = summarize([ROUTED.format(ch=1), WOULD_REJECT_PRE])
        assert rv.would_reject == 1

    def test_two_verdicts_split_by_a_fresh_measurement_are_two_spectra(self):
        rv = summarize([ROUTED.format(ch=1), WOULD_REJECT_PRE,
                        ROUTED.format(ch=2), WOULD_REJECT_PRE])
        assert rv.would_reject == 2
        assert rv.channel_would_reject == {1: 1, 2: 1}

    def test_gate_names_come_from_both_issue_shapes(self):
        rv = summarize(one_rejected_spectrum(1))
        # "<gate>: <detail>" and "<gate> dropped N points" both name a gate.
        assert rv.gate_would_reject["cap_flatness"] == 1
        assert rv.gate_would_reject["kk_truncation"] == 1
        assert rv.gate_would_reject["pegged_parameters"] == 1

    def test_a_policy_level_issue_is_not_reported_as_a_gate(self):
        # "resolution-limited — σ reported as an upper bound" has no threshold to
        # calibrate; listing it as a gate would invite someone to tune one.
        rv = summarize(one_rejected_spectrum(1))
        assert not any("resolution" in g for g in rv.gate_would_reject)
        assert any("resolution-limited" in i for i in rv.other_issues)

    def test_blocking_failures_and_dropped_points_are_separate_denominators(self):
        rv = summarize(one_rejected_spectrum(1))
        assert rv.gate_blocking_fail["valley_feature"] == 1
        assert rv.gate_points_dropped["kk_truncation"] == 3

    def test_the_quality_gate_is_counted_apart_from_the_eis_gates(self):
        rv = summarize([ROUTED.format(ch=1), QUALITY_WOULD_REJECT])
        assert rv.quality_would_reject == 1
        assert rv.would_reject == 0


class TestEngineEvidence:
    def test_gated_events_are_the_proof_the_flip_took(self):
        assert summarize(one_rejected_spectrum(1)).is_shadow_run is True

    def test_a_legacy_log_is_not_a_shadow_run_however_it_is_labelled(self):
        # eis_objective_shadow fires under BOTH engines — the σ path runs either way —
        # so it must not be mistaken for evidence that the gated engine ran.
        rv = summarize(LEGACY_RUN)
        assert rv.is_shadow_run is False
        assert rv.sigma_shadow == [(203540.7, 0.000132)]

    def test_reviewing_a_legacy_log_exits_nonzero(self, tmp_path, capsys):
        log = tmp_path / "run.log"
        log.write_text("\n".join(LEGACY_RUN), encoding="utf-8")
        assert main(["review", str(log)]) == 2
        assert "NO — not one gated-engine event" in capsys.readouterr().out


class TestBoundDemotions:
    def test_a_declined_bound_is_counted_with_its_mode(self):
        rv = summarize([ROUTED.format(ch=5), DECLINED_BOUND])
        assert rv.bound_modes == {"bound_unqualified": 1}
        assert rv.channel_bound == {5: 1}

    def test_a_bound_before_any_channel_line_is_unattributed_not_guessed(self):
        rv = summarize([DECLINED_BOUND])
        assert rv.unattributed == 1
        assert rv.channel_bound == {}


# ── DataStore half ───────────────────────────────────────────────────────────

@pytest.fixture()
def shadow_project(tmp_path):
    """A project whose fit rows are what a *gated* run actually writes today."""
    import numpy as np

    from softae.analysis.circuit_fitting import FitResult
    from softae.analysis.eis_data import EISResult
    from softae.core.data_store import DataStore

    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("shadow_gated_observation", mode="campaign")
    f = np.logspace(0, 5, 12)
    for ch in (1, 2):
        eis = EISResult.from_arrays(channel=ch, f=f, z_real=np.full(12, 2000.0),
                                    z_imag_neg=np.full(12, 50.0))
        mid = store.record_measurement(run_id, eis)
        # `report=` is deliberately NOT passed — the router does not pass it either
        # (P.18 open), which is exactly the condition this half of the tool reports on.
        store.record_fit(mid, FitResult(model_name="simpleSalt", parameters=[],
                                        R0=50.0, R1=2000.0, R0_guess=50.0,
                                        R1_guess=2000.0, z_indices=[], success=True),
                         L_cm=0.2, t_cm=0.015, w_cm=0.2)
    store.close()
    return tmp_path / "proj", run_id


class TestDataStoreHalf:
    def test_it_reports_one_row_per_measurement_with_the_stored_sigma(
            self, shadow_project):
        project, run_id = shadow_project
        summary = db_summary(str(project), run_id)
        assert [r["channel"] for r in summary["rows"]] == [1, 2]
        assert all(r["sigma"] is not None for r in summary["rows"])

    def test_the_engine_column_reads_legacy_even_though_the_gated_engine_ran(
            self, shadow_project):
        # Not a bug in the tool — the point of it. `_fit_report_columns(None)` stamps
        # 'legacy' because the router does not pass a report, so the column cannot
        # distinguish "the config said legacy" from "this site never asked".
        project, run_id = shadow_project
        summary = db_summary(str(project), run_id)
        assert set(summary["engines"]) == {"legacy"}
        assert all(r["gate_verdict"] is None for r in summary["rows"])

    def test_the_report_says_the_engine_column_is_stamped_not_observed(
            self, shadow_project):
        project, run_id = shadow_project
        text = render(summarize(one_rejected_spectrum(1)),
                      db_summary(str(project), run_id), "run.log")
        assert "STAMPED DEFAULT, not an observation" in text

    def test_the_latest_run_is_used_when_none_is_named(self, shadow_project):
        project, run_id = shadow_project
        assert db_summary(str(project), None)["run_id"] == run_id

    def test_an_unreadable_project_costs_the_db_half_and_not_the_log_half(
            self, tmp_path):
        # The log half carries the verdicts; losing it to a bad --project would be the
        # tool refusing the review over its optional input.
        summary = db_summary(str(tmp_path / "empty"), None)
        assert "error" in summary
        assert "unavailable" in render(summarize(one_rejected_spectrum(1)),
                                       summary, "run.log")


# ── CLI surface ──────────────────────────────────────────────────────────────

class TestCommandLine:
    def test_the_console_script_target_matches_the_parser_prog(self):
        assert build_parser().prog == "softae-shadow"

    def test_status_reads_the_config_and_names_both_flags(self, capsys):
        # Read-only: `status` is the pre-check AND the post-revert check, so it must
        # never write. Asserted by the flags still reading what the file says after.
        from softae.analysis.eis.settings import eis_settings

        before = eis_settings()
        _cmd_status(build_parser().parse_args(["status"]))
        out = capsys.readouterr().out
        assert "[eis] engine" in out and "[eis.gates] enabled" in out
        assert "[quality] enabled" in out
        assert eis_settings() == before

    def test_an_unarmed_config_is_not_reported_as_ready(self, capsys):
        code = _cmd_status(build_parser().parse_args(["status"]))
        out = capsys.readouterr().out
        armed = "ARMED FOR A SHADOW RUN" in out
        assert armed == (code == 0)

    def test_a_missing_log_fails_without_a_traceback(self, tmp_path, capsys):
        assert main(["review", str(tmp_path / "nope.log")]) == 1
        assert "No such log file" in capsys.readouterr().err

    def test_the_review_renders_every_numbered_section(self, tmp_path, capsys):
        log = tmp_path / "run.log"
        log.write_text("\n".join(one_rejected_spectrum(1)), encoding="utf-8")
        assert main(["review", str(log)]) == 0
        out = capsys.readouterr().out
        for heading in ("1. DID THE GATED ENGINE RUN?", "2. WOULD-REJECT VERDICTS",
                        "3. VALUE-VS-BOUND DEMOTIONS", "4. CHANNEL ATTRIBUTION",
                        "5. PER-CHANNEL", "6. ARM / DON'T-ARM DECISION INPUTS"):
            assert heading in out


class TestRendering:
    def test_an_empty_review_renders_without_raising(self):
        assert "1. DID THE GATED ENGINE RUN?" in render(ShadowReview(), None, "x.log")

    def test_the_report_states_that_point_drops_happen_even_while_observing(self):
        # The front's "removes nothing" is true at the SPECTRUM level only:
        # `run_gates` applies block_point masks unconditionally.
        text = render(summarize(one_rejected_spectrum(1)), None, "run.log")
        assert "block_point masks are not" in text

    def test_the_channel_column_is_labelled_inferred(self):
        text = render(summarize(one_rejected_spectrum(1)), None, "run.log")
        assert "POSITIONAL — INFERRED, NOT RECORDED" in text


# ── T7.1: the metrics event, and the thresholds it supports ──────────────────

METRICS = (
    "2026-08-10 14:47:55 [info     ] eis_spectrum_metrics           "
    "channel={ch} enforced=False fit_ok=True gates_failed={failed} "
    "gates_run=['finiteness', 'cap_flatness', 'kk_truncation'] "
    "metrics={{'cap_slope': {cap}, 'kk_max_resid_pct': {kk}, 'r_squared': {r2}, "
    "'n_surviving': 41.0}} "
    "n_dropped=0 n_surviving=41 report_mode=value "
    "spectrum_key=c{ch:02d}:{key} verdict={verdict}"
)


def metric_lines(n: int = 30, *, bad: int = 6) -> list[str]:
    """A population with a healthy bulk and a dispersive tail, as the engine logs it."""
    out = []
    for i in range(n):
        dispersive = i < bad
        out.append(METRICS.format(
            ch=i % 8 + 1, key=f"{i:012x}",
            cap=round(1.5 + 0.05 * i if dispersive else 0.02 + 0.001 * i, 4),
            kk=round(0.3 + 0.01 * i, 4), r2=round(0.999 - 0.0005 * i, 6),
            failed="['cap_flatness']" if dispersive else "[]",
            verdict="suspect" if dispersive else "accept"))
    return out


class TestMetricsEventParsing:
    def test_the_metrics_event_becomes_a_spectrum_record(self):
        rv = summarize(metric_lines(3, bad=1))
        assert len(rv.metric_events) == 3
        assert rv.metric_events[0].metrics["cap_slope"] == 1.5
        assert rv.metric_events[0].channel == 1

    def test_the_metrics_event_is_gated_engine_evidence_on_its_own(self):
        # It is emitted only from the gated branch, so it strengthens section 1 and
        # cannot make a legacy log look gated.
        assert summarize(metric_lines(2)).is_shadow_run is True

    def test_repeat_analyses_of_one_spectrum_count_once_as_a_spectrum(self):
        # router.py and autonomous_wiring.py both call analyze_spectrum on the same
        # arrays. Without the fingerprint every n in section 7 would be doubled.
        lines = metric_lines(20)
        rv = summarize(lines + lines)
        assert len(rv.metric_events) == 40
        assert len(rv.spectra) == 20

    def test_the_report_shows_events_and_spectra_separately(self):
        lines = metric_lines(20)
        text = render(summarize(lines + lines), None, "run.log",
                      recommendations(summarize(lines + lines)))
        assert "20 spectra (40 events, deduplicated by content fingerprint)" in text


class TestSpectrumCountFallback:
    """A rehearsal routes nothing, and the report used to read as if it saw nothing.

    ``eis_autorouted`` is a campaign event: ``rehearse`` replays spectra already on
    disk, so a rehearsal log has metrics events by the hundred and no router lines at
    all. Sections 2 and 6 both sized the run from the router count, so the review of a
    real rehearsal announced ``over 0 spectrum(s)`` — the exact number an operator uses
    to decide whether the run is worth reading.
    """

    def test_a_rehearsal_shaped_log_counts_its_spectra_from_the_metrics_events(self):
        rv = summarize(metric_lines(30, bad=6))
        assert rv.n_routed == 0 and rv.n_spectra_seen == 30
        text = render(rv, None, "rehearse.log")
        assert "over 30 spectrum(s)" in text
        assert "of 30 seen" in text

    def test_the_fallback_count_says_where_it_came_from(self):
        text = render(summarize(metric_lines(30, bad=6)), None, "rehearse.log")
        assert "counted from metrics events" in text and "no router" in text

    def test_a_campaign_shaped_log_still_counts_routed_spectra_and_says_routed(self):
        lines = [line for ch in (1, 2, 3, 4) for line in one_rejected_spectrum(ch)]
        rv = summarize(lines)
        assert rv.n_routed == 4 and rv.n_spectra_seen == 4
        text = render(rv, None, "run.log")
        assert "of 4 routed" in text and "over 4 spectrum(s)" in text
        assert "counted from metrics events" not in text

    def test_adding_the_fallback_changed_no_byte_of_a_campaign_report(self):
        # The pins in TestBackwardCompatibility guard the section boundaries; this one
        # guards the two lines actually touched, against a log that has both router
        # anchors and metrics events, where the two counts could disagree.
        lines = ([line for ch in (1, 2) for line in one_rejected_spectrum(ch)]
                 + metric_lines(30, bad=6))
        rv = summarize(lines)
        assert rv.n_routed == 2 and rv.n_spectra_seen == 2  # router wins where it exists
        text = render(rv, None, "run.log")
        assert "of 2 routed" in text and "counted from metrics events" not in text

    def test_a_log_with_neither_routes_nor_metrics_renders_the_bare_line(self):
        text = render(ShadowReview(), None, "empty.log")
        assert "spectra that WOULD have been discarded : 0\n" in text
        assert "counted from metrics events" not in text


class TestSectionSeven:
    def _text(self, lines, **kwargs):
        rv = summarize(lines)
        return render(rv, None, "run.log", recommendations(rv, **kwargs))

    def test_the_section_names_the_rule_and_the_counts_behind_each_value(self):
        text = self._text(metric_lines(30, bad=6))
        assert "7. RECOMMENDED THRESHOLDS" in text
        assert "cap_flatness_max" in text and "upper-fence" in text

    def test_a_behavioural_key_is_marked_so_the_warning_is_not_buried(self):
        text = self._text(metric_lines(30))
        assert "! = changing this key changes a stored NUMBER" in text
        assert "!kk_resid_pct" in text

    def test_the_section_refuses_to_recommend_arming(self):
        text = self._text(metric_lines(30))
        assert "ARMING IS NOT RECOMMENDED HERE" in text
        assert "enabled and [quality] enabled stay false" in text

    def test_the_out_of_scope_keys_are_named_with_a_reason_each(self):
        text = self._text(metric_lines(30))
        assert "NOT RECOMMENDABLE FROM A SHADOW RUN" in text
        for key in ("kk_c", "bound_tol", "plateau_tol_pct", "tand_headroom_mult"):
            assert key in text

    def test_the_joint_reject_count_is_reported_rather_than_the_column_sum(self):
        text = self._text(metric_lines(30, bad=6))
        assert "Per-key counts do not add" in text
        assert "would reject" in text

    def test_a_sixteen_well_run_recommends_nothing_and_says_why(self):
        # examples/shadow_campaign.toml ships budget = 16, which sits BELOW the floor
        # by design: the honest output is "run 32 wells", not a fence from 16 samples.
        text = self._text(metric_lines(16, bad=4))
        assert "REFUSED" in text and "need 20" in text

    def test_lowering_the_evidence_floor_is_visible_in_the_output(self):
        assert "cap_flatness_max" in self._text(metric_lines(16, bad=4),
                                                min_evidence=12)


class TestBackwardCompatibility:
    """§6.3 — an old log must review exactly as it did, byte for byte."""

    def test_an_old_log_without_the_metrics_event_renders_sections_one_to_six_unchanged(
            self):
        rv = summarize(one_rejected_spectrum(1))
        before = render(rv, None, "run.log")               # the pre-T7.1 call shape
        after = render(rv, None, "run.log", recommendations(rv))
        assert after.startswith(before)
        assert after[len(before):].lstrip("\n").startswith("7. RECOMMENDED THRESHOLDS")

    def test_every_recommendation_refuses_on_an_old_log_with_the_one_sided_reason(self):
        recs = recommendations(summarize(one_rejected_spectrum(1)))
        assert recs and all(r.status == "refused" for r in recs)
        assert all("pre-T7.1 log" in r.reason for r in recs)

    def test_reviewing_an_old_log_still_exits_zero(self, tmp_path, capsys):
        log = tmp_path / "run.log"
        log.write_text("\n".join(one_rejected_spectrum(1)), encoding="utf-8")
        assert main(["review", str(log)]) == 0
        assert "7. RECOMMENDED THRESHOLDS" in capsys.readouterr().out


class TestEmitToml:
    @pytest.fixture()
    def gated_log(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("\n".join(metric_lines(30, bad=6)), encoding="utf-8")
        return log

    def test_emit_toml_writes_a_paste_ready_block_that_arms_nothing(
            self, gated_log, tmp_path, capsys):
        out = tmp_path / "proposed.toml"
        assert main(["review", str(gated_log), "--emit-toml", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "[eis.gates]" in text and "[quality]" in text
        assert "enabled          = false" in text or "enabled" in text
        assert "= true" not in text.replace("`enabled = true`", "")

    def test_emit_toml_refuses_to_write_to_the_live_config_path(
            self, gated_log, tmp_path, monkeypatch, capsys):
        from softae.config import loader

        live = tmp_path / "softae_config.toml"
        monkeypatch.setattr(loader, "config_path", lambda: str(live))
        assert main(["review", str(gated_log), "--emit-toml", str(live)]) == 1
        assert not live.exists()
        assert "Refusing to write to the live config" in capsys.readouterr().err

    def test_emit_toml_refuses_to_overwrite_an_existing_file(
            self, gated_log, tmp_path, capsys):
        out = tmp_path / "proposed.toml"
        out.write_text("previous run's proposal", encoding="utf-8")
        assert main(["review", str(gated_log), "--emit-toml", str(out)]) == 1
        assert out.read_text(encoding="utf-8") == "previous run's proposal"

    def test_the_review_flags_are_on_the_parser(self):
        args = build_parser().parse_args(["review", "x.log", "--min-evidence", "12"])
        assert args.min_evidence == 12 and args.emit_toml is None


# ── DataStore detectors ──────────────────────────────────────────────────────

ARC_RECORD = {"gate": "arc_closure", "severity": "annotate", "passed": False,
              "n_dropped": 0, "detail": "no descending branch", "state": "open"}


@pytest.fixture()
def railed_project(tmp_path):
    """Four fit rows: the two railed *eras*, a healthy fit, and an unbounded model."""
    from types import SimpleNamespace

    import numpy as np

    from softae.analysis.circuit_fitting import FitResult
    from softae.analysis.eis_data import EISResult
    from softae.core.data_store import DataStore

    def fit(model, R1, success, error_msg=""):
        return FitResult(model_name=model, parameters=[], R0=50.0, R1=R1,
                         R0_guess=50.0, R1_guess=R1, z_indices=[], success=success,
                         error_msg=error_msg)

    rows = [
        # Historical: success=1, R1 exactly on the simpleSalt floor, sigma ~ seawater,
        # and NOTHING in the row marking it. 325 of 1440 in run 20260811T023757Z.
        (1, fit("simpleSalt", 100.0, True), SimpleNamespace(gate_log=[ARC_RECORD])),
        # New: the demotion clears success, NaNs R1 and names the bound.
        (2, fit("simpleSalt", float("nan"), False,
                "railed fit: R1 rests on the 100 ohm bound — parameter unidentified"),
         SimpleNamespace(gate_log=[ARC_RECORD])),
        (3, fit("simpleSalt", 2000.0, True),
         SimpleNamespace(gate_log=[{**ARC_RECORD, "state": "closed", "passed": True}])),
        # flexSalt declares no bounds, so no fit against it can be *railed*.
        (4, fit("flexSalt", 100.0, True), None),
    ]

    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("railed_detectors", mode="campaign")
    f = np.logspace(0, 5, 12)
    for channel, fit_result, report in rows:
        eis = EISResult.from_arrays(channel=channel, f=f, z_real=np.full(12, 2000.0),
                                    z_imag_neg=np.full(12, 50.0))
        mid = store.record_measurement(run_id, eis)
        store.record_fit(mid, fit_result, L_cm=0.2, t_cm=0.015, w_cm=0.2,
                         report=report)
    store.close()
    return tmp_path / "proj", run_id


class TestRailedDetectors:
    """Two detectors, because they see different eras and neither alone sees the run."""

    @staticmethod
    def _by_channel(project, run_id):
        return {r["channel"]: r for r in railed_summary(str(project), run_id)["rows"]}

    def test_a_historical_railed_row_is_detected_from_its_r1_at_the_model_bound(
            self, railed_project):
        rows = self._by_channel(*railed_project)
        assert rows[1]["railed_historical"] == 1 and rows[1]["railed_new"] == 0
        assert rows[1]["n_success"] == 1          # it still wears success = 1

    def test_a_new_railed_row_is_detected_from_its_error_message(self, railed_project):
        rows = self._by_channel(*railed_project)
        assert rows[2]["railed_new"] == 1 and rows[2]["railed_historical"] == 0
        assert rows[2]["n_success"] == 0

    def test_a_healthy_fit_is_counted_in_neither_railed_column(self, railed_project):
        rows = self._by_channel(*railed_project)
        assert rows[3]["railed_new"] == rows[3]["railed_historical"] == 0
        assert rows[3]["median_R1"] == 2000.0

    def test_a_model_with_no_declared_bound_reports_unknown_not_zero(
            self, railed_project):
        # A zero would read as "the bound is 0 ohm and nothing can rail", which is a
        # claim the registry never made.
        rows = self._by_channel(*railed_project)
        assert rows[4]["bound_ohm"] == "unknown"
        assert rows[4]["railed_historical"] == 0
        assert rows[1]["bound_ohm"] == 100.0

    def test_the_railed_bound_comes_from_the_registry_not_a_literal(
            self, railed_project, monkeypatch):
        # Move the bound in CIRCUIT_MODELS and the detector must move with it: the
        # 2000 ohm fit becomes railed, the 100 ohm one stays railed.
        from softae.analysis import circuit_fitting

        spec = dict(circuit_fitting.CIRCUIT_MODELS["simpleSalt"])
        lower, upper = spec["bounds"]
        raised = list(lower)
        raised[spec["z_indices"][1]] = 5000.0
        monkeypatch.setitem(circuit_fitting.CIRCUIT_MODELS, "simpleSalt",
                            {**spec, "bounds": (raised, upper)})
        rows = self._by_channel(*railed_project)
        assert rows[3]["railed_historical"] == 1
        assert rows[3]["bound_ohm"] == 5000.0

    def test_an_unreadable_project_returns_an_error_rather_than_raising(self, tmp_path):
        assert "error" in railed_summary(str(tmp_path / "nope"), None)


@pytest.fixture()
def mixed_era_project(tmp_path):
    """One row from each of the three eras the counter has to keep apart.

    Column era (T7.7, verdict in `fit_results.arc_state`), shim era (verdict only
    in `gate_log_json`), and pre-shim (nothing recorded). A run in the wild will
    hold all three at once, so the counter is only trustworthy if it can read a
    mixture without double-counting or dropping any of it.
    """
    from types import SimpleNamespace

    import numpy as np

    from softae.analysis.circuit_fitting import FitResult
    from softae.analysis.eis.arc import ArcClosure
    from softae.analysis.eis_data import EISResult
    from softae.core.data_store import DataStore

    def fit():
        return FitResult(model_name="simpleSalt", parameters=[], R0=50.0, R1=2000.0,
                         R0_guess=50.0, R1_guess=2000.0, z_indices=[], success=True)

    column_era = fit()
    column_era.arc_closure = ArcClosure("closed", 1000.0, 20.0, -41.5)

    rows = [
        # T7.7: the column is written from the fit, and the shim also fills the JSON
        # — the duplication that the one-increment rule has to survive.
        (1, column_era, SimpleNamespace(gate_log=[{**ARC_RECORD, "state": "closed",
                                                   "passed": True}])),
        (2, fit(), SimpleNamespace(gate_log=[ARC_RECORD])),   # shim era: JSON only
        (3, fit(), None),                                      # pre-shim: nothing
    ]

    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("mixed_era", mode="campaign")
    f = np.logspace(0, 5, 12)
    for channel, fit_result, report in rows:
        eis = EISResult.from_arrays(channel=channel, f=f, z_real=np.full(12, 2000.0),
                                    z_imag_neg=np.full(12, 50.0))
        mid = store.record_measurement(run_id, eis)
        store.record_fit(mid, fit_result, L_cm=0.2, t_cm=0.015, w_cm=0.2,
                         report=report)
    store.close()
    return tmp_path / "proj", run_id


class TestArcSummary:
    def test_arc_states_are_counted_from_the_column_when_it_is_populated(
            self, mixed_era_project):
        # `fit_results.arc_state` is the first thing asked since T7.7, and it comes
        # free with `query_fits`' SELECT * — no `json.loads` per row. This fixture's
        # column-era row carries the verdict in BOTH places, so a count of 1 is also
        # what says the JSON copy was not read a second time; the whole-run version
        # of that invariant is `test_a_mixed_era_project_counts_every_row_exactly_once`.
        project, run_id = mixed_era_project
        assert arc_summary(str(project), run_id)["states"]["closed"] == 1

    def test_arc_states_fall_back_to_the_gate_log_for_pre_t7_7_rows(
            self, railed_project):
        # The `railed_project` fixture builds a plain FitResult with no
        # `.arc_closure` and hands `record_fit` a hand-built gate log, so its rows
        # ARE the pre-T7.7 era: NULL column, populated JSON. These three counts are
        # unchanged from before T7.7, which is the point.
        project, run_id = railed_project
        summary = arc_summary(str(project), run_id)
        assert summary["states"] == {"open": 2, "closed": 1}

    def test_a_mixed_era_project_counts_every_row_exactly_once(
            self, mixed_era_project):
        # For one release the verdict lives in both places on a T7.7 row. Reading
        # both would double it and break `sum(states) + no_record == rows`, which is
        # the property that makes these counts a row count rather than a tally of
        # sightings.
        project, run_id = mixed_era_project
        summary = arc_summary(str(project), run_id)
        assert summary["states"] == {"closed": 1, "open": 1}
        assert summary["no_record"] == 1
        assert sum(summary["states"].values()) + summary["no_record"] == 3

    def test_a_row_with_neither_a_column_nor_a_gate_log_is_counted_as_no_record(
            self, railed_project):
        # Never folded into an outcome it never reported.
        project, run_id = railed_project
        assert arc_summary(str(project), run_id)["no_record"] == 1

    def test_sigma_is_bound_is_reported_as_a_stamped_default(self, railed_project):
        project, run_id = railed_project
        assert "stamped default" in arc_summary(str(project), run_id)["sigma_is_bound"]
        assert "P.18" in arc_summary(str(project), run_id)["sigma_is_bound"]


def test_db_summary_is_importable_from_both_modules_after_the_move():
    # A pure move must not break a caller. The review re-exports it because that is
    # where its public surface was first.
    from softae.tools import shadow_db, shadow_review

    assert shadow_review.db_summary is shadow_db.db_summary


class TestStatusCostAdvisory:
    def test_an_armed_config_warns_that_observing_mode_is_the_slow_one(
            self, monkeypatch, capsys):
        # The flag that reads as "observe only, change nothing" is what makes the run
        # expensive: with gates enforcing, a blocking spectrum is rejected before the
        # fitter; with them observing, the optimiser grinds a parallel-R model onto
        # data that has no arc. An operator sizing a shadow run by well count alone
        # would under-budget the clock by orders of magnitude.
        from softae.analysis.eis import settings as eis_settings_mod
        from softae.config import loader

        armed = eis_settings_mod.EISSettings(
            engine="gated", gates=eis_settings_mod.GateSettings(enabled=False))
        monkeypatch.setattr(eis_settings_mod, "eis_settings", lambda *a, **k: armed)
        monkeypatch.setattr(loader, "load", lambda *a, **k: {"quality": {}})

        code = _cmd_status(build_parser().parse_args(["status"]))
        out = capsys.readouterr().out
        assert code == 0
        assert "ARMED FOR A SHADOW RUN" in out
        assert "SLOWEST analysis setting" in out
        # The invariant is that the advisory quotes MEASURED medians and keeps the
        # clock line — not any particular constant. T7.8 replaced the synthetic
        # ~78 s / ~0.07 s pair with the 192-spectrum distribution; a later rehearsal
        # may move these numbers again, and should update this assertion with them.
        assert "38 s" in out and "0.16 s" in out
        assert "192 real" in out
        assert "clock, not by the well count" in out


class TestModuleDocstringMatchesReality:
    def test_the_docstring_names_the_arc_columns_as_the_recorded_verdict(self):
        # The arc verdict reaches the DataStore as four real arc_* columns that
        # record_fit writes from fit_result.arc_closure — not as a JSON payload, and
        # not via the retired arc_provenance shim. shadow_db.arc_summary reads those
        # columns, so a docstring claiming the verdict goes unrecorded would argue the
        # tool's own evidence away.
        from softae.tools import shadow_review

        doc = shadow_review.__doc__ or ""
        assert "arc_state" in doc
        assert "annotate_arc_closure" in doc and "record_fit" in doc
        assert "arc_closure" in doc
        # P.18 is open in substance and the docstring must still say so.
        assert "P.18" in doc and "gate_verdict" in doc


class TestSectionFiveBIsReachableFromTheCli:
    """The DB detectors are only worth building if the report actually prints them.

    Driven through ``main(["review", ...])`` rather than through ``render`` directly:
    a helper that is importable but never called is the defect these tests exist to
    prevent, and only the real command path proves the wiring.
    """

    @pytest.fixture()
    def gated_log(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("\n".join(metric_lines(24, bad=5)), encoding="utf-8")
        return log

    def test_the_cli_prints_the_railed_table_when_a_project_is_given(
            self, gated_log, railed_project, capsys):
        project, run_id = railed_project
        assert main(["review", str(gated_log), "--project", str(project),
                     "--run-id", run_id]) == 0
        out = capsys.readouterr().out
        assert "5b. RAILED FITS AND ARC CLOSURE" in out
        assert "railed (new)" in out and "railed (historical)" in out
        # Two detectors, one row each, plus the unbounded model reporting 'unknown'.
        assert "unknown" in out

    def test_the_cli_prints_the_arc_states_and_the_stamped_default_posture(
            self, gated_log, railed_project, capsys):
        project, run_id = railed_project
        main(["review", str(gated_log), "--project", str(project), "--run-id", run_id])
        out = capsys.readouterr().out
        assert "arc_closure states" in out and "'open': 2" in out
        assert "no record: 1" in out
        # sigma_is_bound must keep the same posture section 5 takes on `engine`.
        assert "sigma_is_bound" in out and "stamped default" in out

    def test_section_5b_is_absent_without_a_project_so_the_old_report_is_unchanged(
            self, gated_log, capsys):
        assert main(["review", str(gated_log)]) == 0
        out = capsys.readouterr().out
        assert "5b." not in out
        assert "5. PER-CHANNEL" in out and "6. ARM / DON'T-ARM" in out

    def test_an_unreadable_project_degrades_each_half_separately(
            self, gated_log, tmp_path, capsys):
        # The log half carries the verdicts; a bad --project must not cost it, and one
        # unanswerable question must not suppress the others.
        main(["review", str(gated_log), "--project", str(tmp_path / "nope")])
        out = capsys.readouterr().out
        assert "5b. RAILED FITS AND ARC CLOSURE" in out
        assert "railed: (unavailable:" in out and "arc closure: (unavailable:" in out
        assert "7. RECOMMENDED THRESHOLDS" in out          # unaffected


class TestHoldKindIsStructural:
    """The rendered hold label comes from a field, not from parsing English."""

    def test_a_unimodal_hold_renders_as_such_even_when_the_gate_never_fired(self):
        from softae.analysis.eis.recommend import (
            HOLD_UNIMODAL,
            SpectrumRecord,
            recommend_all,
        )
        from softae.tools.shadow_render import _status_label

        # rho sits far below its -0.95 default, so nothing fires; and it is unimodal,
        # so the gap rule finds no split. Both holds are live and the gap must win.
        values = [-0.10 + 0.001 * i for i in range(30)]
        recs = recommend_all([SpectrumRecord(key=f"c01:{i:012x}", metrics={"rho": v})
                              for i, v in enumerate(values)])
        rho = next(r for r in recs if r.key == "rho_degenerate")
        assert rho.status == "hold"
        assert rho.hold_kind == HOLD_UNIMODAL
        assert rho.fired_at_default == 0          # the other hold would also have fired
        assert _status_label(rho) == "hold (unimodal)"

    def test_an_unexercised_hold_renders_as_such(self):
        from softae.analysis.eis.recommend import (
            HOLD_UNEXERCISED,
            SpectrumRecord,
            recommend_all,
        )
        from softae.tools.shadow_render import _status_label

        recs = recommend_all([
            SpectrumRecord(key=f"c01:{i:012x}", metrics={"cap_slope": 0.01 + 1e-4 * i})
            for i in range(30)])
        cap = next(r for r in recs if r.key == "cap_flatness_max")
        assert cap.status == "hold" and cap.hold_kind == HOLD_UNEXERCISED
        assert _status_label(cap) == "hold (unexercised)"

    def test_a_recommended_key_that_measures_the_rig_says_so_in_the_status_column(self):
        from softae.analysis.eis.recommend import SpectrumRecord, recommend_all
        from softae.tools.shadow_render import _status_label

        recs = recommend_all([
            SpectrumRecord(key=f"c01:{i:012x}", metrics={"cap_slope": 2.0 + 0.01 * i})
            for i in range(30)])
        cap = next(r for r in recs if r.key == "cap_flatness_max")
        assert _status_label(cap) == "recommended (measures-the-rig)"
        assert cap.hold_kind == ""


def test_the_unconfigurable_table_references_the_gate_constants_it_reports():
    # A restated constant is a constant that disagrees with the gate after the first
    # edit. Only the two that genuinely have no module-level name stay literal.
    from softae.analysis.eis import gates
    from softae.analysis.eis.recommend_report import UNCONFIGURABLE

    by_metric = {m: threshold for m, _owner, threshold, _sense in UNCONFIGURABLE}
    assert by_metric["frac_quadrant_violation"] == gates.RE_SUSPICION_FRAC
    assert by_metric["plateau_n_points"] == float(gates.MIN_PLATEAU_POINTS)
