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
    build_parser,
    db_summary,
    main,
    parse_line,
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
