"""Is this campaign *searching* a composition, or casting one recipe repeatedly?

A pinned recipe — the same fully-specified formulation cast ``budget`` times —
needs no new machinery. :class:`~softae.core.composition_axes.CompositionAxis`
already says so in its own docstring: ``low == high`` pins the target,
:func:`~softae.core.composition_axes.axes_parameter_space` omits it, and
:func:`~softae.core.composition_axes.build_targets_from_axes` substitutes the
constant. Pair a fully-pinned ``[general_formulation]`` with one ``int``
``replicate`` axis ``1..budget`` under ``optimizer = "grid"`` and the campaign
casts ``budget`` identical wells today, with no source change at all.

**What was missing is honesty, not capability.** The very substitution that makes
pinning free makes a *forgotten* axis silently constant: a target declared
searched (``low != high``) whose name never appears in ``parameter_space`` falls
back to its ``low`` on a ``logger.warning`` and casts anyway. Four trials at the
corner of the declared box, a DOE that looks like a search, and nothing an
operator would ever see — ``SUBAGENT_RULES`` §3.1(a), and the evil twin of the
route above. In the DataStore the two are indistinguishable: "four exact
recipes" and "one axis I forgot to list" write the same rows.

So this module makes the pinning **declared and checked**. It changes no
execution path — :func:`check_pinning` is a refusal at spec load, sitting beside
the unknown-field and ``explicit_none`` refusals in
:mod:`softae.core.campaign_spec_io` and reaching the file-loaded campaigns that
:func:`~softae.core.composition_axes.validate_axes` never sees (its only caller
in ``src/`` is the GUI's axes editor).

:func:`validate_axes` is deliberately *not* reused: it refuses a fully-pinned
axis set outright — *"Every target is pinned … so there is nothing to search"* —
which is exactly the shape a pinned recipe is, and it knows nothing about
``parameter_space``, which is the disagreement actually worth catching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from softae.core.composition_axes import CompositionAxis


class PinnedRecipeError(ValueError):
    """A spec's composition axes and its ``parameter_space`` disagree.

    A :class:`ValueError` so the loaders that already funnel ``TypeError`` /
    ``ValueError`` into their own error type keep working unchanged, and its own
    class so a caller that wants *this* refusal can name it.
    """


def composition_axes(spec: Any) -> "tuple[CompositionAxis, ...]":
    """The declared composition targets of *spec*, or ``()``.

    Read off whichever formulation context declares them rather than off
    ``general_formulation`` by name: the ternary
    :class:`~softae.core.formulation.FormulationContext` carries no axes today,
    and asking for the attribute is what keeps this correct if it gains them.
    """
    for ctx in (getattr(spec, "general_formulation", None),
                getattr(spec, "formulation", None)):
        axes = tuple(getattr(ctx, "axes", ()) or ())
        if axes:
            return axes
    return ()


def pinned_axes(spec: Any) -> "tuple[CompositionAxis, ...]":
    """Targets held constant — ``low == high``."""
    return tuple(a for a in composition_axes(spec) if a.is_fixed)


def searched_axes(spec: Any) -> "tuple[CompositionAxis, ...]":
    """Targets the optimizer is meant to vary."""
    return tuple(a for a in composition_axes(spec) if not a.is_fixed)


def is_pinned_recipe(spec: Any) -> bool:
    """Whether every composition target is pinned — one recipe, cast repeatedly.

    False for a spec with no composition context at all: a legacy volume-mode
    campaign has no recipe to pin, and answering ``True`` for it would spell
    "nothing to pin" with the same token as "pinned deliberately".
    """
    axes = composition_axes(spec)
    return bool(axes) and not any(not a.is_fixed for a in axes)


def describe_pinning(spec: Any) -> str:
    """One operator-readable line: what is pinned, what is searched.

    A pure function returning a string, so a CLI (``softae-campaign check``) or
    a log line can use it without this module knowing either exists.
    """
    axes = composition_axes(spec)
    if not axes:
        return ("no composition targets — this campaign searches its parameters "
                "directly")

    pinned = pinned_axes(spec)
    searched = searched_axes(spec)
    head = "recipe PINNED" if not searched else "composition SEARCHED"
    parts = [f"{head}: {len(searched)} of {len(axes)} composition target(s) "
             f"searched"]
    if pinned:
        parts.append("pinned: " + "; ".join(a.describe() for a in pinned))
    if searched:
        parts.append("searched: " + "; ".join(a.describe() for a in searched))

    axis_names = {a.name for a in axes}
    others = sorted(n for n in (getattr(spec, "parameter_space", None) or {})
                    if n not in axis_names)
    if others:
        casts = "identical cast(s)" if not searched else "trial(s)"
        parts.append(f"{getattr(spec, 'budget', '?')} {casts} over {others}")
    return " — ".join(parts)


def check_pinning(spec: Any) -> None:
    """Raise when a *searched* axis is absent from ``parameter_space``.

    One shape only, and it is the one nothing else catches: an axis the file
    declares searchable that the optimizer is never asked to suggest. Pinning
    itself is never refused here — a pinned recipe is a legitimate campaign, and
    refusing it is what makes :func:`validate_axes` the wrong tool for this job.
    """
    space = set(getattr(spec, "parameter_space", None) or {})
    missing = [a for a in searched_axes(spec) if a.name not in space]
    if not missing:
        return

    named = "; ".join(f"{a.name!r} ({a.describe()})" for a in missing)
    raise PinnedRecipeError(
        f"composition axis {named} is declared searched (low != high) but is "
        f"absent from 'parameter_space', so no suggestion ever carries a value "
        f"for it. Unchecked, this does not fail — it would cast every trial at "
        f"the lower bound "
        f"({', '.join(f'{a.name}={a.low:g}' for a in missing)}), a fixed recipe "
        f"at the corner of the declared box reported as a search. Add the axis "
        f"to 'parameter_space' (axes_parameter_space() writes exactly this), or "
        f"pin it deliberately with low == high."
    )
