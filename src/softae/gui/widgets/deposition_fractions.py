"""Qt-free fraction/sanity state machine for the deposition panel's Σ indicator.

Pure display logic: given per-row (checked, auto, value, dep_fraction) plus the
target µL, classify the effective Σ of dep-share fractions into an ok/warn/error
severity with the exact message text from the fraction-redesign spec §2.3.  No
PySide6 import — unit-testable without a Qt event loop.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FractionRow:
    """One stock row's fraction-relevant state (built by the panel from its table)."""

    name: str
    checked: bool
    auto: bool
    value: float
    dep_fraction: float  # from Solution.dep_fraction(catalog)


@dataclass(frozen=True)
class FractionSumState:
    """Resolved indicator state: severity + exact message + supporting tallies."""

    severity: str  # "ok" | "warn" | "error"
    message: str  # exact text per the §2.3 table (+ override E==0 sub-case)
    explicit_sum: float  # E over dep-bearing checked rows
    n_auto: int  # A
    carrier_bulk_count: int  # k (carrier-only checked rows with explicit frac>0)


def resolve_sum_state(
    rows: list[FractionRow], target_uL: float, tol: float = 1e-6
) -> FractionSumState:
    """Pure §2.3 state machine (+ the explicit-first E==0 amber sub-case).

    No Qt, no core call — display logic only.  ``E`` is the sum of Auto-off
    (explicit) fractions over dep-bearing checked rows; ``A`` is the count of
    Auto-on dep-bearing checked rows; ``k`` is the count of carrier-only checked
    rows carrying an explicit non-zero fraction (a direct bulk-volume share).
    """
    dep_rows = [r for r in rows if r.checked and r.dep_fraction > 0.0]
    explicit_sum = sum(r.value for r in dep_rows if not r.auto)
    n_auto = sum(1 for r in dep_rows if r.auto)
    carrier_bulk_count = sum(
        1
        for r in rows
        if r.checked and r.dep_fraction <= 0.0 and not r.auto and r.value > 0.0
    )

    e = explicit_sum
    t = target_uL

    if not dep_rows:
        severity = "warn"
        message = "No dep-bearing stock selected — eluted 0 µL from dep-share"
    elif n_auto > 0:
        if e <= 1.0 + tol:
            severity = "ok"
            message = f"Σ = 1.00 ({n_auto} auto-balanced)"
        else:
            severity = "error"
            message = (
                f"explicit fractions exceed 1.00 (Σ_explicit = {e:.2f}) — "
                "reduce values or clear Auto"
            )
    elif abs(e - 1.0) <= tol:
        severity = "ok"
        message = "Σ = 1.00"
    elif e <= tol:
        # OVERRIDE (explicit-first default): all-zero explicit, no Auto rows.
        severity = "warn"
        message = "Σ = 0.00 — set fractions or enable Auto"
    elif e < 1.0 - tol:
        severity = "warn"
        message = (
            f"Σ = {e:.2f} < 1 — deposited dep = {e:.2f}×target "
            f"(short by {(1.0 - e) * t:.2f} µL)"
        )
    else:  # e > 1 + tol
        severity = "warn"
        message = f"Σ = {e:.2f} > 1 — overshoots target by {(e - 1.0) * t:.2f} µL"

    if carrier_bulk_count > 0:
        message += f" (+{carrier_bulk_count} carrier-only bulk share)"

    return FractionSumState(
        severity=severity,
        message=message,
        explicit_sum=explicit_sum,
        n_auto=n_auto,
        carrier_bulk_count=carrier_bulk_count,
    )


def normalize_to_one(
    values: list[float], decimals: int = 2, tol: float = 1e-6
) -> list[float]:
    """Scale ``values`` so they sum to exactly 1.0 at ``decimals`` precision.

    Divides by the current total (handles both Σ<1 and Σ>1), rounds each to
    ``decimals``, then distributes the rounding residual (1.0 − Σ_rounded) onto
    the largest row so the returned list sums to exactly 1.00 at that precision.
    Returns the input unchanged when the total is ≤ ``tol`` (nothing to scale).
    """
    total = sum(values)
    if total <= tol:
        return list(values)  # defensive no-op: caller disables the button here
    scaled = [v / total for v in values]  # handles both Σ<1 (up) and Σ>1 (down)
    rounded = [round(v, decimals) for v in scaled]
    residual = round(1.0 - sum(rounded), decimals)
    i = max(range(len(rounded)), key=rounded.__getitem__)  # largest row
    rounded[i] = round(rounded[i] + residual, decimals)
    return rounded
