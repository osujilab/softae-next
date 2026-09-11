"""Tests for `SpectrumReport.gate_summary` (report.py) — the Analysis-tab-independent
surface of the Gate column.

`_gate_item` (tab_analysis.py) was fixed first ([p136] section 3) to name a failed
`flag`-severity gate instead of letting it fall through to a bare "pass". That mail
post's own "found, not fixed" note flagged the identical defect here, six lines away in
`gate_summary`, which the docstring says is adopted by `_gate_item` "token-for-token so
the two surfaces cannot disagree" — a claim that was false for exactly this branch until
now.
"""
from __future__ import annotations

import numpy as np

from softae.analysis.eis.gates import FLAG, GateResult
from softae.analysis.eis.report import SpectrumReport

_MASK_OK = np.ones(5, dtype=bool)


def _flag_entry(passed: bool, name: str = "arc_closure") -> dict:
    """A `flag`-severity gate log entry in the real `GateResult.as_log_entry()` shape.

    `passed=False` is `gate_arc_closure` on an OPEN arc: it ran (`checked=True`),
    found the arc did not close, and refuses nothing — no `n_dropped`, no rejection.
    That is the shape with no counter of its own, so it has to be named or it is
    invisible.
    """
    detail = "within tolerance" if passed else "apex not bracketed"
    return GateResult(name, FLAG, passed, detail, _MASK_OK).as_log_entry()


def test_gate_summary_failed_flag_renders_flagged_passed_flag_renders_pass():
    failed = SpectrumReport(engine="gated", gate_log=(_flag_entry(passed=False),))
    passed = SpectrumReport(engine="gated", gate_log=(_flag_entry(passed=True),))

    assert failed.gate_summary() == "arc_closure flagged"
    assert passed.gate_summary() == "pass"
