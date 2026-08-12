"""Which failures are evidence about a composition, and which are not (T3.1 §3).

**The governing rule: ``unmeasured`` != ``infeasible``.** A NULL objective is the
honest record of "suggested, cast, but not measured". Labelling every NULL row as
a bad composition would convert this codebase's honesty about missing data into a
fabricated fact about chemistry — the exact mirror of the ``0.0``-coercion bug
``_is_unmeasured`` exists to prevent. **The infeasible class is a strict subset of
the NULL rows, never all of them.** A stage timeout is not evidence about a
composition; a film that never formed is.

So this module's job is mostly to *decline*. Exactly one signal earns a label:

    a quality-gate REJECT of open-circuit or shorted/dead, which
    (1) reproduces on an immediate same-channel confirmation sweep with the
        *same* signature, on a run where
    (2) at least one other measurement on the same board ACCEPTed, and
    (3) the channel is not already the subject of a §3.3 pattern report.

**These are label-hygiene conditions, not hardware checks (user decision (vi)).**
Each is a reason to *withhold* a label; none is a claim about the state of the
rig. Condition 2 in particular is weaker than it reads: an ACCEPT elsewhere on the
board says only that at least one measurement succeeded during this run, and it
neither tests nor certifies this channel's own route. Verification of the board,
the connectors and everything upstream of them is the operator's, and nothing here
attempts it — there is no diagnosis, no self-test sweep, no channel-health model,
and no allocator gating.

**Nothing in this module touches an instrument.** It takes no manager and holds no
driver; it consumes reports that already exist and emits decisions. §3.3's report
is one ``alerts`` row plus a retraction, and that is its entire mandate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

import structlog

from softae.optimizers.feasibility import FEASIBLE, INFEASIBLE, FeasibilityModel

logger = structlog.get_logger(__name__)

__all__ = [
    "CONFIRM_MEASUREMENT",
    "MAX_CONFIRMATION_SWEEPS",
    "OUTCOME_INFEASIBLE",
    "OUTCOME_MEASURED",
    "OUTCOME_UNKNOWN",
    "FailureLabelEngine",
    "LabelDecision",
    "OutcomeSink",
    "reject_signature",
]

#: ``doe_parameters.outcome`` vocabulary (user decision (ii)). NULL is a
#: pre-feature row — "undeclared is unknown, never empty".
OUTCOME_MEASURED = "measured"
OUTCOME_INFEASIBLE = "infeasible"
OUTCOME_UNKNOWN = "unknown"

#: There is deliberately **no** ``'hardware_suspect'`` outcome. A §3.3 flag is a
#: statement about a *channel across runs*, not about this trial, and the trial's
#: own honest outcome is ``'unknown'``. Encoding a channel-level suspicion in a
#: trial-level column would put two different subjects in one field — and a
#: retraction would then have to rewrite history rather than simply stop counting.
_FORBIDDEN_OUTCOME = "hardware_suspect"

#: The ``tags["measurement"]`` value carried by a confirmation repeat. A dedicated
#: role rather than reusing ``"secondary"``, so a genuine secondary probe and a
#: confirmation retry stay distinguishable in the record. ``is_primary_measurement``
#: requires ``"primary"``, so this is excluded from objective selection **for
#: free, with no edit to that predicate** — the mechanism T2.7's ``image``
#: modality used.
CONFIRM_MEASUREMENT = "confirm"

#: Up to two repeats, per §3.2. The rig's asymmetry is the whole argument: a
#: repeat costs one short sweep against a well plus an anneal for a fresh cast.
MAX_CONFIRMATION_SWEEPS = 2

#: Reject signatures that are *about the film*. Matched against the gate report's
#: **issues**, not its verdict — see :func:`reject_signature`.
SIGNATURE_OPEN = "open"
SIGNATURE_SHORT = "short"

_SIGNATURE_MARKERS = (
    ("open circuit", SIGNATURE_OPEN),
    ("shorted or dead channel", SIGNATURE_SHORT),
)


def reject_signature(report: Any) -> str | None:
    """``"open"`` / ``"short"`` if this report blames the *film*, else ``None``.

    **Reads the report's issues, never its verdict.** With ``[quality] enabled =
    false`` — which is how the gate ships — a would-be REJECT is returned as
    SUSPECT plus ``"gate disabled"``, so keying on the verdict would make the
    richest label source inert exactly while the gate is being observed. T3.1
    must never itself flip ``[quality] enabled``; reading the report is how it
    accrues labels without being granted authority the gate does not have.

    Everything else returns ``None``: ``"empty"``, ``"unreadable"`` and ``"|Z|
    identical at every frequency"`` are instrument-side and say nothing about the
    film; a short trace and a non-monotonic frequency axis likewise.
    """
    issues: Iterable[Any]
    try:
        issues = list(getattr(report, "issues", None) or ())
    except Exception:
        return None
    for issue in issues:
        text = str(issue).lower()
        for marker, signature in _SIGNATURE_MARKERS:
            if marker in text:
                return signature
    return None


class OutcomeSink(Protocol):
    """Where a trial's outcome + reason are persisted.

    **Interim contract (T3.1b).** Spec §3 puts ``outcome`` and ``failure_reason``
    on ``doe_parameters``, but that migration enters ``core/data_store.py``, which
    is claimed by the parallel session. So the decision is written *through this
    protocol* rather than to a column: T3.1b swaps in an implementation backed by
    the real columns without any caller changing, and until then the labels a
    campaign issues live in the in-memory :class:`FeasibilityModel` for the run
    that issued them.

    What survives a restart today is what already survives one: a NOT NULL
    ``objective_value`` reconstructs a **feasible** label. Infeasible labels do
    not survive, and the model reports its count into the checkpoint so a resume
    can *notice* the disagreement and warn rather than quietly searching
    differently. Undeclared is unknown, never empty.
    """

    def record_outcome(
        self,
        *,
        run_id: str | None,
        channel: int | None,
        params: Mapping[str, Any],
        outcome: str,
        failure_reason: str | None,
    ) -> None:
        """Persist one trial's outcome. Must never raise."""


@dataclass(frozen=True)
class LabelDecision:
    """What one measured (or unmeasured) channel earned.

    ``label`` is ``None`` for every case that is *not proven to be the film* —
    which is most of them. ``train`` separates "recorded" from "learned from":
    a :class:`~softae.errors.FormulationInfeasibleError` is composition-
    attributable and honestly ``infeasible``, but the hard ``feasibility_fn``
    already enforces that boundary exactly, so training on it would teach the
    classifier a region the pool filter has already removed — a wasted degree of
    freedom fit to a place that is never sampled.
    """

    outcome: str
    failure_reason: str | None = None
    label: int | None = None
    train: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.outcome == _FORBIDDEN_OUTCOME:
            raise ValueError(
                f"'{_FORBIDDEN_OUTCOME}' is not an outcome: a channel-level "
                f"suspicion is not a fact about this trial (spec §3)."
            )


@dataclass
class _ChannelHistory:
    """Cross-run reject provenance for one channel — §3.3's raw material."""

    boards: set[Any] = field(default_factory=set)
    compositions: set[str] = field(default_factory=set)
    flagged: bool = False


def _composition_key(params: Mapping[str, Any]) -> str:
    """Stable identity for "was this a *different* composition?"."""
    return repr(sorted((str(k), repr(v)) for k, v in dict(params).items()))


class FailureLabelEngine:
    """Turns gate reports into labels, or (usually) into nothing.

    Holds no manager, opens no connection and drives no hardware. The only
    outward effect beyond a label is §3.3's single ``alerts`` row.
    """

    def __init__(
        self,
        *,
        model: FeasibilityModel | None = None,
        sink: OutcomeSink | None = None,
        emit: Any = None,
        data_store: Any = None,
        run_id: str | None = None,
        max_confirmations: int = MAX_CONFIRMATION_SWEEPS,
    ) -> None:
        self.model = model
        self._sink = sink
        self._emit = emit
        self._data_store = data_store
        self._run_id = run_id
        self.max_confirmations = int(max_confirmations)
        #: Channels that ACCEPTed at least once on a given board in this run —
        #: condition 3's board-level corroborator.
        self._board_accepts: dict[Any, set[int]] = {}
        self._history: dict[int, _ChannelHistory] = {}

    # ── Corroboration bookkeeping ───────────────────────────────────────

    def note_accept(self, *, channel: int, board_id: Any = None) -> None:
        """Record that *channel* produced a usable measurement on *board_id*."""
        self._board_accepts.setdefault(board_id, set()).add(int(channel))

    def board_has_accept(self, board_id: Any = None, *, excluding: int | None = None) -> bool:
        """Condition 3: did anything on this board measure during this run?

        *excluding* drops the channel under judgement, because with one
        measurement per film a failed channel has no ACCEPT of its own and must
        not be allowed to corroborate itself.
        """
        accepts = self._board_accepts.get(board_id, set())
        if excluding is None:
            return bool(accepts)
        return bool(accepts - {int(excluding)})

    def is_flagged(self, channel: int) -> bool:
        """Condition 4: is this channel already the subject of a §3.3 report?"""
        hist = self._history.get(int(channel))
        return bool(hist and hist.flagged)

    # ── The decisions ───────────────────────────────────────────────────

    def record_measured(
        self,
        *,
        params: Mapping[str, Any],
        channel: int | None = None,
        board_id: Any = None,
        objective_value: float | None = None,
    ) -> LabelDecision:
        """A trial that reached ``optimizer.tell`` — positive evidence, label 0.

        The composition mixed, cast, dried and measured. This is the only source
        of the feasible class, and it is also the only label that survives a
        restart today (a NOT NULL ``objective_value`` reconstructs it).
        """
        if channel is not None:
            self.note_accept(channel=channel, board_id=board_id)
        decision = LabelDecision(
            outcome=OUTCOME_MEASURED,
            failure_reason=None,
            label=FEASIBLE,
            train=True,
            evidence="objective recorded",
        )
        self._apply(decision, params=params, channel=channel, board_id=board_id)
        return decision

    def record_rig_failure(
        self,
        *,
        params: Mapping[str, Any],
        what: str,
        channel: int | None = None,
    ) -> LabelDecision:
        """An ``execute:`` / ``analyze:`` failure, a park, a cancel — **no label**.

        Transient rig faults recur independently of what is in the syringe (the
        ESP301 ``VI_ERROR_TMO`` path exists precisely because they do), and an
        extraction exception is ambiguous by construction: an extractor bug and a
        pathological spectrum raise identically.
        """
        decision = LabelDecision(
            outcome=OUTCOME_UNKNOWN,
            failure_reason=str(what),
            label=None,
            train=False,
            evidence="rig- or analysis-side failure; not evidence about the composition",
        )
        self._apply(decision, params=params, channel=channel, board_id=None)
        return decision

    def record_formulation_infeasible(
        self,
        *,
        params: Mapping[str, Any],
        channel: int | None = None,
        detail: str = "",
    ) -> LabelDecision:
        """Over the well's volume budget — labelled, but **excluded from training**."""
        decision = LabelDecision(
            outcome=OUTCOME_INFEASIBLE,
            failure_reason=f"formulation infeasible{': ' + detail if detail else ''}",
            label=INFEASIBLE,
            train=False,
            evidence="already enforced exactly by the hard feasibility_fn; "
                     "recorded for the audit trail only",
        )
        self._apply(decision, params=params, channel=channel, board_id=None)
        return decision

    def record_gate_reject(
        self,
        *,
        params: Mapping[str, Any],
        channel: int,
        primary_report: Any,
        confirmations: Sequence[Any] = (),
        board_id: Any = None,
    ) -> LabelDecision:
        """The four-condition rule (§3.1). Labels only when **all** hold."""
        channel = int(channel)
        signature = reject_signature(primary_report)
        summary = _summarize(primary_report)

        # Provenance is recorded for every open/short reject, labelled or not:
        # §3.3's pattern is built from rejects, and a reject that failed some
        # other condition is still a reject on that channel.
        if signature is not None:
            self._note_reject_history(channel, board_id, params)

        if signature is None:
            return self._withhold(
                params, channel, board_id, summary,
                "reject is not open-circuit or shorted/dead — instrument-side or "
                "inconclusive, and it says nothing about the film",
            )

        confirm_signatures = [reject_signature(r) for r in confirmations]
        self._emit_confirmations(channel, signature, confirmations, confirm_signatures)

        if not confirmations:
            return self._withhold(
                params, channel, board_id, summary,
                f"no confirmation sweep: a single {signature} reading is not "
                f"evidence about the film until it reproduces",
            )
        if any(cs != signature for cs in confirm_signatures):
            got = [cs or "accept/other" for cs in confirm_signatures]
            return self._withhold(
                params, channel, board_id, summary,
                f"confirmation disagreed (primary {signature}, repeats {got}) — "
                f"the reading changed between back-to-back sweeps, so it did not "
                f"reproduce",
            )
        if self.is_flagged(channel):
            return self._withhold(
                params, channel, board_id, summary,
                f"ch{channel} is already the subject of a channel-pattern report "
                f"(§3.3); its rejects track the channel, not the chemistry",
            )
        if not self.board_has_accept(board_id, excluding=channel):
            return self._withhold(
                params, channel, board_id, summary,
                "no other channel on this board ACCEPTed in this run — a board of "
                "rejects is not a board of bad compositions",
            )

        n = len(confirmations)
        evidence = (
            f"{signature} x{n + 1} confirmed on ch{channel} "
            f"(primary + {n} repeat{'s' if n != 1 else ''}, same signature); "
            f"board {board_id!r} has an ACCEPT elsewhere"
        )
        decision = LabelDecision(
            outcome=OUTCOME_INFEASIBLE,
            failure_reason=evidence,
            label=INFEASIBLE,
            train=True,
            evidence=evidence,
        )
        self._apply(decision, params=params, channel=channel, board_id=board_id)
        self._event(
            "infeasible_label_recorded",
            channel=channel, board=board_id, params=dict(params),
            signature=signature, confirmations=n, evidence=evidence,
        )
        return decision

    # ── §3.3 — report and retract, strictly passive ─────────────────────

    def _note_reject_history(
        self, channel: int, board_id: Any, params: Mapping[str, Any]
    ) -> None:
        hist = self._history.setdefault(channel, _ChannelHistory())
        hist.boards.add(board_id)
        hist.compositions.add(_composition_key(params))
        if hist.flagged:
            return
        # The pattern is "this channel, across different boards, on unrelated
        # compositions" — all three, or it is not a channel-tracking pattern.
        if len(hist.boards) >= 2 and len(hist.compositions) >= 2:
            hist.flagged = True
            self._report_channel_pattern(channel, hist)

    def _report_channel_pattern(self, channel: int, hist: _ChannelHistory) -> None:
        """One alert, one retraction. **Nothing else, by decision (vi).**

        The report states the observed pattern and stops there. It names no
        cause, ranks no channels by health, runs no diagnostic or self-test
        sweep, gates no future allocation, and persists no per-channel health
        state. Deciding what the pattern means, and checking the board and the
        connectors, is the operator's — the alert exists so a human can look, and
        the retraction exists so the classifier does not learn chemistry from it.
        """
        retracted = self.model.retract_channel(channel) if self.model is not None else 0
        boards = sorted((str(b) for b in hist.boards))
        # Worded as an observation, not a verdict.
        message = (
            f"ch{channel} rejected on boards {', '.join(boards)} across "
            f"{len(hist.compositions)} unrelated compositions; "
            f"{retracted} label{'s' if retracted != 1 else ''} retracted"
        )
        self._event(
            "channel_reject_pattern",
            channel=channel, boards=boards,
            n_compositions=len(hist.compositions),
            labels_retracted=retracted, message=message,
        )
        self._raise_alert(channel, boards, retracted, message)

    def _raise_alert(
        self, channel: int, boards: list[str], retracted: int, message: str
    ) -> None:
        try:
            from softae.core.alerts import INFO, Alert, raise_alert

            raise_alert(
                Alert(
                    kind="channel_reject_pattern",
                    message=message,
                    # Informational on purpose: this is something to look at, not
                    # a reason to stop the rig. Nothing here parks a run.
                    severity=INFO,
                    run_id=self._run_id,
                    details={
                        "channel": channel,
                        "boards": boards,
                        "labels_retracted": retracted,
                    },
                ),
                data_store=self._data_store,
            )
        except Exception:  # a report about a problem must not become a second one
            logger.warning("channel_pattern_alert_failed", channel=channel,
                           exc_info=True)

    # ── Plumbing ────────────────────────────────────────────────────────

    def _withhold(
        self,
        params: Mapping[str, Any],
        channel: int,
        board_id: Any,
        summary: str,
        why: str,
    ) -> LabelDecision:
        decision = LabelDecision(
            outcome=OUTCOME_UNKNOWN,
            failure_reason=f"{summary} — withheld: {why}",
            label=None,
            train=False,
            evidence=why,
        )
        self._apply(decision, params=params, channel=channel, board_id=board_id)
        return decision

    def _apply(
        self,
        decision: LabelDecision,
        *,
        params: Mapping[str, Any],
        channel: int | None,
        board_id: Any,
    ) -> None:
        if decision.train and decision.label is not None and self.model is not None:
            self.model.add(params, decision.label, channel=channel, board_id=board_id)
        if self._sink is not None:
            try:
                self._sink.record_outcome(
                    run_id=self._run_id, channel=channel, params=dict(params),
                    outcome=decision.outcome,
                    failure_reason=decision.failure_reason,
                )
            except Exception:
                logger.warning("outcome_sink_failed", exc_info=True)

    def _emit_confirmations(
        self,
        channel: int,
        signature: str,
        confirmations: Sequence[Any],
        confirm_signatures: Sequence[str | None],
    ) -> None:
        for i, (report, got) in enumerate(zip(confirmations, confirm_signatures), 1):
            self._event(
                "confirmation_sweep",
                channel=channel, attempt=i, of=len(confirmations),
                verdict=_summarize(report),
                signature=got, matched=(got == signature),
            )

    def _event(self, name: str, **payload: Any) -> None:
        logger.info(name, **payload)
        if self._emit is not None:
            try:
                self._emit(name, **payload)
            except Exception:
                logger.debug("failure_label_emit_failed", event=name, exc_info=True)


def _summarize(report: Any) -> str:
    try:
        return str(report.summary())
    except Exception:
        return str(report)
