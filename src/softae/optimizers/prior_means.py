"""Prior mean models a campaign spec can name rather than carry.

``CampaignSpec.prior_mean`` is a callable ``m(params) -> objective`` the GP models
the residual from. A callable cannot be written to a file, which is why the field
was refused outright by :mod:`softae.core.campaign_spec_io` — and that refusal made
the Live BO tab's Prior-informed group box unlaunchable the moment the campaign
moved into a detached child, because the child is started *from* a file.

**The observation that resolves it:** the panel does not offer arbitrary callables.
It offers a fixed combo of built-ins, so what the operator chose is a *name*. A
name round-trips. So this module is the registry that makes the name meaningful in
both directions — :func:`prior_mean_name` for the writer, :func:`resolve_prior_mean`
for the loader — and a callable that is *not* in it stays unrepresentable, reported
by :func:`~softae.core.campaign_spec_io.spec_toml_completeness` rather than silently
dropped.

It lives beside the optimizer rather than in the GUI because **the child resolves
the name**, and a headless process must not import ``softae.gui`` (and therefore
PySide6) to find out what its prior mean is.
"""

from __future__ import annotations

from typing import Any, Callable

#: ``params -> prior objective value``.
PriorMeanFn = Callable[[dict[str, Any]], float]


def linear_demo(params: dict[str, Any]) -> float:
    """A stand-in physics model: a deterministic weighted sum of the params.

    Weights ascend with the sorted parameter name so the value is reproducible
    regardless of dict ordering. This demonstrates the residual-GP path (the GP
    learns only the correction to this trend) without pretending to be physics.
    """
    return float(
        sum((i + 1) * float(v) for i, (_, v) in enumerate(sorted(params.items())))
    )


#: Registry key → model. A spec file names the **key**; nothing else is loadable.
PRIOR_MEANS: dict[str, PriorMeanFn] = {
    "linear_demo": linear_demo,
}

#: ``(operator-facing label, registry key)``, in the order a picker should show
#: them. Empty key means "no prior mean". Held here rather than in the tab so the
#: labels an operator picks from and the names a file can carry cannot drift: a
#: label with no key behind it would be a choice that refuses its own launch.
PRIOR_MEAN_CHOICES: tuple[tuple[str, str], ...] = (
    ("none", ""),
    ("linear (demo)", "linear_demo"),
)


def resolve_prior_mean(name: str) -> PriorMeanFn:
    """The model *name* refers to. Raises :class:`ValueError` if there is none."""
    try:
        return PRIOR_MEANS[name]
    except KeyError:
        raise ValueError(
            f"unknown prior mean {name!r}; a spec file may only name a built-in "
            f"model, and the built-ins are {sorted(PRIOR_MEANS)}"
        ) from None


def prior_mean_name(fn: Any) -> str | None:
    """The registry key for *fn*, or ``None`` if it is not a built-in.

    Identity rather than equality: two closures over different data compare
    unequal but a re-imported module function is the same object, which is the
    case that has to work.
    """
    for name, known in PRIOR_MEANS.items():
        if fn is known:
            return name
    return None


def label_for_key(key: str) -> str:
    """The picker label for a registry key (``""`` → "none")."""
    for label, k in PRIOR_MEAN_CHOICES:
        if k == key:
            return label
    return key
