"""The compiled circuit expression must agree with ``impedance`` **bitwise**.

:mod:`softae.analysis.eis.circuit_expr` exists purely to remove a cost, so the only
interesting question about it is whether it changed a number. Every assertion here is an
exact-bytes comparison against ``impedance.models.circuits.fitting.wrapCircuit``, not a
tolerance.

The file also carries its own **positive control** (SUBAGENT_RULES §3.1): a check that
cannot fail is worth nothing, so ``test_wrap_circuit_corrupted_template_falls_back``
deliberately breaks the compiled template and asserts the guard notices.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.circuit_fitting import CIRCUIT_MODELS
from softae.analysis.eis import circuit_expr
from softae.analysis.eis.circuit_expr import circuit_template, wrap_circuit
from softae.analysis.eis.models import EIS_CIRCUITS

pytest.importorskip("impedance")

FREQ = np.logspace(6, -1, 41)

#: Every topology the two engines can actually fit, with a plausible parameter vector.
#: Sourced from the registries rather than retyped, so a new circuit joins this test by
#: being added to the registry.
CASES: list[tuple[str, str, dict]] = [
    (f"eis:{name}", model.circuit, dict(model.constants))
    for name, model in EIS_CIRCUITS.items()
] + [
    (f"legacy:{name}", cfg["circuit"], dict(cfg.get("constants") or {}))
    for name, cfg in CIRCUIT_MODELS.items()
] + [
    # Not in either registry, but fitted directly by ``TestConvergenceTolerance`` in
    # ``test_eis_fitter.py`` — underscored element names parse differently enough to be
    # worth pinning here rather than only reaching the compiler through that suite.
    ("adhoc:underscored", "R_0-p(R_1,C_1)-CPE_1", {}),
]

SEED = {"R": 5.0e4, "C": 3.0e-10, "L": 1.0e-6, "CPE_0": 1.0e-7, "CPE_1": 0.83}


def _guess(circuit: str, constants: dict) -> list[float]:
    from softae.analysis.eis.models import parameter_names

    out = []
    for i, name in enumerate(parameter_names(circuit, constants)):
        base = name.split("_")[0].rstrip("0123456789")
        key = f"{base}_{name.split('_')[-1]}" if base == "CPE" else base
        out.append(SEED.get(key, 1.0) * (1.0 + 0.0137 * i))
    return out


def _reference(circuit: str, constants: dict):
    from impedance.models.circuits.fitting import wrapCircuit

    return wrapCircuit(circuit, dict(constants))


@pytest.mark.parametrize("label,circuit,constants", CASES, ids=[c[0] for c in CASES])
def test_wrap_circuit_matches_reference_bitwise(label, circuit, constants):
    guess = _guess(circuit, constants)
    expected = _reference(circuit, constants)(FREQ, *guess)
    actual = wrap_circuit(circuit, constants)(FREQ, *guess)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("label,circuit,constants", CASES, ids=[c[0] for c in CASES])
def test_wrap_circuit_matches_reference_bitwise_off_seed(label, circuit, constants):
    """The first call is the verified one; later calls must agree too."""
    fast = wrap_circuit(circuit, constants)
    reference = _reference(circuit, constants)
    rng = np.random.default_rng(20260909)
    base = np.asarray(_guess(circuit, constants))
    for _ in range(5):
        params = tuple(np.float64(v) for v in base * rng.uniform(0.3, 3.0, base.size))
        assert fast(FREQ, *params).tobytes() == reference(FREQ, *params).tobytes()


def test_wrap_circuit_template_binds_parameters_by_name_not_literal():
    """The whole saving is that no number is spelled into the source."""
    source = circuit_template("R0-CPE0-p(R1,C0)")
    assert "_f_" in source and "_p_[0]" in source
    assert "e+" not in source and "e-" not in source


def test_wrap_circuit_template_keeps_held_constants_literal():
    source = circuit_template("R0-CPE0-p(R1,C0)", {"C0": 2e-10})
    assert "C([2e-10]" in source
    # ...and the held element consumes no parameter slot.
    assert "_p_[4]" not in source


def test_wrap_circuit_non_finite_parameter_raises_like_reference():
    """``buildCircuit`` spells NaN as the bare token ``nan``; the reference raises."""
    circuit, guess = "R0-CPE0-p(R1,C0)", _guess("R0-CPE0-p(R1,C0)", {})
    bad = [np.nan] + guess[1:]
    with pytest.raises(NameError):
        _reference(circuit, {})(FREQ, *bad)
    with pytest.raises(NameError):
        wrap_circuit(circuit, {})(FREQ, *bad)


def test_wrap_circuit_unbuildable_circuit_falls_back_to_reference():
    """A topology the symbolic build cannot handle must degrade, not raise."""
    calls: list[str] = []

    def explode(*a, **k):
        calls.append("built")
        raise RuntimeError("no template for you")

    circuit_expr._compiled.cache_clear()
    original = circuit_expr._compiled
    circuit_expr._compiled = explode  # type: ignore[assignment]
    try:
        fn = wrap_circuit("R0-CPE0-p(R1,C0)", {})
        guess = _guess("R0-CPE0-p(R1,C0)", {})
        assert fn(FREQ, *guess).tobytes() == _reference("R0-CPE0-p(R1,C0)", {})(
            FREQ, *guess).tobytes()
    finally:
        circuit_expr._compiled = original  # type: ignore[assignment]
    assert calls == ["built"]


def test_wrap_circuit_corrupted_template_falls_back(monkeypatch):
    """POSITIVE CONTROL: the first-call check must be able to fail.

    Without this, a guard that silently never fires is indistinguishable from a guard
    that fires correctly — SUBAGENT_RULES §3.1(e). The template is replaced with one for a
    *different* topology, so the fast path returns wrong numbers on the first call; the
    guard must catch it and serve the reference result instead.
    """
    circuit, constants = "R0-CPE0-p(R1,C0)", {}
    guess = _guess(circuit, constants)
    expected = _reference(circuit, constants)(FREQ, *guess)

    good = circuit_expr._compiled(circuit, ())
    wrong_source = good[3].replace("R([_p_[0]]", "R([_p_[0]*7.0]")
    assert wrong_source != good[3]
    corrupt = (compile(wrong_source, "<corrupt>", "eval"), good[1], good[2], wrong_source)
    monkeypatch.setattr(circuit_expr, "_compiled", lambda *a, **k: corrupt)

    fn = wrap_circuit(circuit, constants)
    first = fn(FREQ, *guess)
    assert first.tobytes() == expected.tobytes(), "guard failed to catch a bad template"
    # ...and it stays demoted rather than re-checking every call.
    assert fn(FREQ, *guess).tobytes() == expected.tobytes()


def test_wrap_circuit_compiles_once_per_topology():
    circuit_expr._compiled.cache_clear()
    for _ in range(4):
        wrap_circuit("R0-CPE0-p(R1,C0)", {})
    info = circuit_expr._compiled.cache_info()
    assert info.misses == 1 and info.hits == 3
