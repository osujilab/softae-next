"""Turning a reviewed shadow run into text an operator reads.

Presentation only.  Nothing here parses a log, queries a DataStore, or decides an exit
code — it receives a :class:`~softae.tools.shadow_review.ShadowReview`, whatever the
DataStore half supplied, and the recommendations, and lays them out.  The split is
three ways and each module has one job::

    shadow_review.py   parse the console log, aggregate it, run the CLI
    shadow_db.py       ask the DataStore only what it can honestly answer
    shadow_render.py   this — the report, the section-7 table, the status screen

**Sections 1-6 are frozen.**  A log written before T7.1 must render exactly as it did
before T7.1, byte for byte, because the arming decision in ``docs/SHADOW_CAMPAIGN.md``
§8 is taken by reading them and a report that quietly reflowed would invalidate a
review already performed against it.  Section 7 is appended, never interleaved, and
:func:`render` omits it entirely when no recommendations are passed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from softae.analysis.eis.recommend import Recommendation, joint_would_reject
from softae.analysis.eis.recommend_report import OUT_OF_SCOPE, observed_only

if TYPE_CHECKING:  # importing ShadowReview at runtime would be a cycle
    from softae.tools.shadow_review import ShadowReview


def _table(header: "tuple[str, ...]", rows: "list[tuple[Any, ...]]") -> str:
    if not rows:
        return "   (none)"
    cols = [max(len(str(header[i])), *(len(str(r[i])) for r in rows))
            for i in range(len(header))]
    out = ["   " + "  ".join(str(h).ljust(cols[i]) for i, h in enumerate(header))]
    out.append("   " + "  ".join("-" * c for c in cols))
    for r in rows:
        out.append("   " + "  ".join(str(v).ljust(cols[i]) for i, v in enumerate(r)))
    return "\n".join(out)



def _status_label(rec: "Recommendation") -> str:
    """``recommended`` alone says nothing about whether the gate was ever tested.

    A key held for want of a firing, and one recommended off a gate that fired on every
    spectrum in the run, are the two failure shapes ``docs/SHADOW_CAMPAIGN.md`` §8 names
    in prose. Both belong in the column an operator scans, not only in the reason text.
    """
    if rec.status == "hold":
        return f"hold ({rec.hold_kind})" if rec.hold_kind else "hold"
    if rec.status == "recommended" and rec.exercise == "measures-the-rig":
        return "recommended (measures-the-rig)"
    return rec.status


def _render_recommendations(rv: ShadowReview, recs: "list[Recommendation]") -> str:
    """Section 7 — values, never a decision.

    Split from :func:`render` because sections 1–6 are pinned byte-identical against
    pre-T7.1 logs and a section that grows must not be able to disturb them.
    """
    records = rv.spectra
    out = ["7. RECOMMENDED THRESHOLDS",
           f"   Evidence: {len(records)} spectra ({len(rv.metric_events)} events, "
           f"deduplicated by content fingerprint).",
           "   ! = changing this key changes a stored NUMBER, not only a verdict.",
           ""]

    if not rv.metric_events:
        out.append("   No eis_spectrum_metrics events: this log predates T7.1. The "
                   "engine logged metrics")
        out.append("   only where a spectrum FAILED, so there is no pass-side "
                   "distribution to place a fence")
        out.append("   against, and every key below refuses for that reason. Re-run "
                   "with the current engine.")
        out.append("")

    shown = [r for r in recs if r.status != "refused"]
    out.append(_table(
        ("section", "key", "default", "recommended", "rule", "n", "fired@def",
         "rej@rec", "status"),
        [(r.section, ("!" if r.behavioural else " ") + r.key, _g(r.default),
          _g(r.value) if r.value is not None else "-", r.rule, r.n,
          r.fired_at_default, r.would_reject_at_value, _status_label(r))
         for r in shown]))

    if any(r.exercise == "measures-the-rig" for r in shown):
        out.append("   A gate marked 'measures-the-rig' fired on ≈every spectrum: at "
                   "its default it is measuring")
        out.append("   the rig, not the sample. The proposed value is what the rig's "
                   "own population supports.")

    refused = [r for r in recs if r.status == "refused"]
    if refused:
        out.append("")
        out.append("   REFUSED (evidence insufficient — the default stands):")
        for r in refused:
            out.append(f"     {r.section}.{r.key}{'!' if r.behavioural else ''}")
            out.append(f"       {r.reason}")

    out.append("")
    out.append("   NOT RECOMMENDABLE FROM A SHADOW RUN:")
    for key, why in OUT_OF_SCOPE:
        out.append(f"     {key} — {why}")
    out.append("   gates.DUPLICATE_RTOL is series-level, not per spectrum, so no "
               "per-spectrum distribution exists.")

    observed = observed_only(records)
    if observed:
        out.append("")
        out.append("   OBSERVED BUT UNCONFIGURABLE (no config line — report only):")
        out.append(_table(
            ("metric", "owner", "hardcoded", "n", "P50", "P95", "fired"),
            [(o["metric"], o["owner"],
              "-" if o["threshold"] is None else _g(o["threshold"]),
              o["n"], _g(o["p50"]), _g(o["p95"]),
              "-" if o["threshold"] is None else o["fired"]) for o in observed]))

    out.append("")
    out.append("   ⚠ ARMING IS NOT RECOMMENDED HERE. These are values, not a "
               "decision.")
    out.append("     [eis.gates] enabled and [quality] enabled stay false until §8 of")
    out.append("     docs/SHADOW_CAMPAIGN.md is satisfied gate by gate.")
    joint = joint_would_reject(recs, records)
    n_applied = sum(1 for r in recs if r.applied)
    pct = f" ({joint / len(records):.0%})" if records else ""
    out.append(f"   Applying every 'recommended' value above ({n_applied} key(s)) "
               f"would reject {joint} of {len(records)} spectra{pct}.")
    out.append("   Per-key counts do not add: one spectrum routinely fails several "
               "gates.")
    return "\n".join(out)


def _render_db_evidence(railed: dict[str, Any] | None,
                        arc: dict[str, Any] | None) -> str:
    """Section 5b — the two things ``fit_results`` records honestly.

    Section 5 has to caveat almost everything it prints, because the router stamps
    defaults into most of the columns. These two are different. A railed fit is
    detectable from the stored numbers whatever wrote them, and the arc state is a real
    per-row observation carried by the provenance shim — so this section states them
    without hedging, and hedges only the one column that is still a default.
    """
    out = ["5b. RAILED FITS AND ARC CLOSURE  (DataStore — the honest columns)"]

    if railed is None:
        out.append("   railed: (not read)")
    elif "error" in railed:
        out.append(f"   railed: (unavailable: {railed['error']})")
    else:
        rows = railed.get("rows") or []
        out.append(_table(
            ("channel", "fits", "ok", "railed (new)", "railed (historical)",
             "R1 bound (Ω)", "median σ", "median R1"),
            [(r["channel"], r["n_fits"], r["n_success"], r["railed_new"],
              r["railed_historical"], _g_or(r["bound_ohm"]), _g(r["median_sigma"]),
              _g(r["median_R1"])) for r in rows]))
        total = sum(r["railed_new"] + r["railed_historical"] for r in rows)
        fits = sum(r["n_fits"] for r in rows)
        out.append(f"   {total} of {fits} fit(s) came to rest on the model's own R₁ "
                   "floor rather than on the data.")
        out.append("   TWO detectors, because they see different eras. 'new' rows carry "
                   "success=0 and an")
        out.append("   error_msg naming the bound; 'historical' rows carry success=1 "
                   "with R1 AT the bound and")
        out.append("   nothing marking them — a σ of roughly seawater from a dry film, "
                   "wearing a success flag.")
        out.append("   The bound is read from CIRCUIT_MODELS, never restated here; a "
                   "model declaring none")
        out.append("   reports 'unknown' rather than 0.")

    if arc is None:
        out.append("   arc closure: (not read)")
    elif "error" in arc:
        out.append(f"   arc closure: (unavailable: {arc['error']})")
    else:
        states = dict(arc.get("states") or {})
        out.append(f"   arc_closure states: {states or '(none recorded)'}"
                   f"   no record: {arc.get('no_record', 0)}")
        out.append("   Honest evidence: the verdict is a real column "
                   "(fit_results.arc_state), written from the fit")
        out.append("   itself, so nothing stamps a default into it. Older rows are "
                   "read from the arc_provenance")
        out.append("   record in their gate log instead; each row counts once, "
                   "whichever way it was read.")
        out.append("   Rows counted as 'no record' predate both.")
        out.append(f"   sigma_is_bound: {arc.get('sigma_is_bound', '')}")
    return "\n".join(out)


def _g_or(value: Any) -> str:
    """A number, or whatever non-numeric marker the detector chose to report."""
    return value if isinstance(value, str) else _g(value)


def render(rv: ShadowReview, db: dict[str, Any] | None, source: str,
           recs: "list[Recommendation] | None" = None, *,
           railed: dict[str, Any] | None = None,
           arc: dict[str, Any] | None = None) -> str:
    """The operator-readable report.

    Sections 1–6 are byte-identical to what this tool produced before T7.1, for any
    log.  Everything added since is **opt-in by argument**: ``recs`` appends section 7,
    ``railed``/``arc`` insert section 5b, and a call that passes none of them renders
    the pre-T7.1 report exactly. That is what lets a review performed against the old
    output stay valid rather than merely look similar.
    """
    out: list[str] = [f"SHADOW CAMPAIGN REVIEW — {source}",
                      f"   {rv.n_lines} line(s) → {rv.n_events} structured event(s)",
                      ""]

    out.append("1. DID THE GATED ENGINE RUN?")
    if rv.is_shadow_run:
        out.append(f"   YES — {rv.n_gated_events} gated-engine event(s), "
                   f"{rv.n_routed} spectrum(s) routed to the store.")
    else:
        out.append("   NO — not one gated-engine event in this log. Either the config "
                   "flip did not take,")
        out.append("   or this is a legacy run. Check `softae-shadow status` and rerun; "
                   "reviewing this")
        out.append("   log would arm a gate against evidence it never produced.")
    out.append("")

    out.append("2. WOULD-REJECT VERDICTS  ([eis.gates] enabled = false)")
    # A rehearsal routes nothing, so its denominator comes from the metrics events
    # instead. Where it does, the report says where it came from: a count whose
    # provenance is invisible is read as a routed count and quietly over-trusted.
    seen = rv.n_spectra_seen
    from_metrics = not rv.n_routed and seen > 0
    denominator = f"  of {seen} routed" if rv.n_routed else \
        (f"  of {seen} seen" if from_metrics else "")
    out.append(f"   spectra that WOULD have been discarded : {rv.would_reject}"
               + denominator)
    if from_metrics:
        out.append("                                             (counted from metrics "
                   "events — no router")
        out.append("                                             anchors in this log)")
    out.append(f"   verdict lines logged                   : "
               f"{rv.would_reject_verdicts} — the engine reduces the gate log twice "
               f"per spectrum")
    out.append("                                             (pre-fit admission, then "
               "post-fit). The two are")
    out.append("                                             paired; the count above "
               "is spectra, not lines.")
    out.append(f"   [quality] would-reject                 : {rv.quality_would_reject}")
    gates = sorted(set(rv.gate_would_reject) | set(rv.gate_blocking_fail)
                   | set(rv.gate_points_dropped))
    out.append("")
    out.append(_table(
        ("gate", "would-reject", "blocking-fail", "points-dropped"),
        [(g, rv.gate_would_reject[g], rv.gate_blocking_fail[g],
          rv.gate_points_dropped[g]) for g in gates]))
    out.append("   'would-reject' counts SPECTRA; 'blocking-fail' counts GATE failures "
               "(a spectrum can fail")
    out.append("   several); 'points-dropped' are removed from the fit EVEN NOW — "
               "block_point masks are not")
    out.append("   behind the enabled flag.")
    if rv.other_issues:
        out.append("")
        out.append("   policy-level issues (no gate, so no threshold to calibrate):")
        for issue, n in rv.other_issues.most_common():
            out.append(f"     {n:>4}  {issue}")
    if rv.gates_raised:
        out.append(f"   ⚠ gates that RAISED and were skipped: {dict(rv.gates_raised)} "
                   "— those checks did not run.")
    out.append("")

    out.append("3. VALUE-VS-BOUND DEMOTIONS")
    total_bound = sum(rv.bound_modes.values())
    out.append(f"   σ declined as an upper bound: {total_bound} "
               f"{dict(rv.bound_modes) if rv.bound_modes else ''}")
    out.append("   Not governed by [eis.gates] enabled. In a σ-objective campaign each "
               "of these is an")
    out.append("   UNMEASURED trial; in this volume-mode spec the objective is mean|Z| "
               "and none of them")
    out.append("   costs the search anything.")
    pairs = [p for p in rv.sigma_shadow if p[1] is not None]
    out.append(f"   eis_objective_shadow pairs (mean|Z| in use, σ observed): "
               f"{len(rv.sigma_shadow)}, of which {len(pairs)} produced a σ")
    out.append("")

    out.append("4. CHANNEL ATTRIBUTION  (POSITIONAL — INFERRED, NOT RECORDED)")
    channels = sorted(set(rv.channel_would_reject) | set(rv.channel_bound)
                      | set(rv.routed_channels))
    out.append(_table(
        ("channel", "routed", "would-reject", "bound"),
        [(c, rv.routed_channels[c], rv.channel_would_reject[c], rv.channel_bound[c])
         for c in channels]))
    out.append(f"   unattributed (no preceding channel= line): {rv.unattributed}")
    out.append("   The would-reject event carries no channel. These come from the "
               "nearest preceding")
    out.append("   channel= line: sound for auto-fit gate events, UNSOUND for "
               "objective-side ones, which")
    out.append("   run after the round and land on whichever channel was routed last.")
    out.append("")

    out.append("5. PER-CHANNEL σ  (DataStore)")
    if db is None:
        out.append("   (not read — pass --project to include it)")
    elif "error" in db:
        out.append(f"   (unavailable: {db['error']})")
    else:
        out.append(f"   run_id {db['run_id']}")
        out.append(_table(
            ("channel", "meas", "R1 (Ω)", "σ (S/cm)", "engine col", "gate_verdict"),
            [(r["channel"], r["measurement_id"], _g(r["R1"]), _g(r["sigma"]),
              r["engine"] or "-", r["gate_verdict"] or "-") for r in db["rows"]]))
        out.append(f"   engine column: {dict(db['engines'])} — a STAMPED DEFAULT, not "
                   "an observation.")
        out.append("   The router does not pass report= to record_fit (P.18 open), so "
                   "a gated run still")
        out.append("   writes engine='legacy' and a NULL gate_verdict. Section 1 is the "
                   "engine evidence.")
    out.append("")

    if railed is not None or arc is not None:
        out.append(_render_db_evidence(railed, arc))
        out.append("")

    out.append("6. ARM / DON'T-ARM DECISION INPUTS")
    out.append("   DEVELOPMENT_FRONTS asks two things of this review, and neither is a "
               "number this tool")
    out.append("   can decide for you:")
    out.append(f"     [eis.gates] enabled — needs 'a reviewed would-reject log'. "
               f"You have {rv.would_reject} would-reject")
    out.append("       verdict(s) over "
               + f"{rv.n_spectra_seen} spectrum(s). Read section 2 "
               "gate by gate and ask of each: would")
    out.append("       discarding those samples have been right? Every threshold "
               "shipped is an engineering")
    out.append("       default chosen without reference to this rig.")
    out.append(f"     [quality] enabled — needs threshold calibration. "
               f"{rv.quality_would_reject} would-reject verdict(s) here.")
    out.append("     E6 cutover criterion 2 — 'a campaign's worth of reviewed "
               "eis_gate_would_reject'. Met by")
    out.append("       reviewing this log; criterion 1 (a version-controlled "
               "calibration set) is separate and")
    out.append("       still blocked on the RE→CE jumper.")
    if recs is None:
        return "\n".join(out)
    out.append("")
    return "\n".join(out) + _render_recommendations(rv, recs)


def _g(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4g}"


# ── The status screen ────────────────────────────────────────────────────────

def render_status(eis: Any, quality_enabled: bool, *,
                  config_path: str) -> tuple[str, str]:
    """``(text, state)`` for ``softae-shadow status``.

    ``state`` is this module's classification of the config — ``"armed"``,
    ``"enforcing"`` or ``"not_armed"`` — and the caller maps it to an exit code.  The
    text and the state come from one branch here so the screen and the shell can never
    disagree about what the config says, which is the whole value of a pre-flight check
    that is also run post-revert.
    """
    out = [f"config: {config_path}",
           f"   {eis.describe()}",
           f"   [eis] engine          = {eis.engine!r}",
           f"   [eis] objective       = {eis.objective!r}",
           f"   [eis.gates] enabled   = {str(eis.gates.enabled).lower()}",
           f"   [quality] enabled     = {str(quality_enabled).lower()}",
           f"   [eis.fixture] mode    = {eis.fixture.mode!r} on "
           f"{eis.fixture.fixture_id!r}",
           ""]

    if eis.engine == "gated" and not eis.gates.enabled and not quality_enabled:
        out.append("ARMED FOR A SHADOW RUN — gated physics, every check observing, "
                   "nothing removed at the")
        out.append("spectrum level. Revert `engine` to \"legacy\" when the run is "
                   "reviewed.")
        out.append("")
        # Observing-only is the *most expensive* configuration the rig has, and it is
        # the one that reads as free.  Enforcing gates reject a blocking spectrum before
        # the fitter runs; observing gates do not, so the optimiser grinds a parallel-R
        # model onto data with no arc and takes the long way to failing.
        #
        # These are measurements, not estimates.  The T7.8 rehearsal replayed 192 real
        # spectra (2026-08-14, run 20260811T023757Z_equilibration_characterization) and
        # the distribution is bimodal on arc closure: an open arc has no in-band feature
        # for the fitter to converge onto, and that — not the gate verdict — is what
        # costs the time.  They replace the synthetic ~78 s / ~0.07 s pair this advisory
        # used to quote.
        out.append("BUDGET WALL-TIME. Observing mode is the SLOWEST analysis setting, "
                   "not the cheapest:")
        out.append("a spectrum the gates would have rejected pre-fit still reaches the "
                   "fitter, and a fit")
        out.append("that has no arc to find takes the long way to failing. Measured over "
                   "192 real")
        out.append("spectra (2026-08-14): open-arc median ~38 s (max ~58 s) against "
                   "closed-arc ~0.16 s.")
        out.append("Size the run by the clock, not by the well count.")
        return "\n".join(out), "armed"

    if eis.engine == "gated":
        out.append("GATED AND ENFORCING — this is a cutover, not a shadow run. A "
                   "shadow run needs")
        out.append("[eis.gates] enabled = false and [quality] enabled = false.")
        return "\n".join(out), "enforcing"

    out.append("NOT ARMED — the shipped legacy engine. Set [eis] engine = \"gated\" in "
               "the config above")
    out.append("to run a shadow campaign (docs/SHADOW_CAMPAIGN.md).")
    return "\n".join(out), "not_armed"
