"""The modality registry — Tier 2's single plug-in point (T2.5).

A *modality* is a kind of measurement: EIS today, a camera tomorrow (T2.7). Until
now the campaign path knew only one, and said so in five different places — the
step builder in ``_build_deposition_workflow``, the router default in the
executor, the objective extractors and their directions, and the unconditional
``.mscr``-writing block in ``run_autonomous_campaign``. Adding a second modality
meant finding all five. This module is the one place a new data stream registers
into, and :func:`get_modality` is the one lookup the campaign path performs.

**Explicit registration, never a filesystem scan** (ATLAS addendum §3, and
decision (c) in the restructuring spec). A scan makes the set of available
modalities depend on what happens to be importable at the moment somebody looks,
so a half-written module becomes a silently *missing capability* rather than an
error. An explicit dict fails at the lookup instead, and names what it does have.

**What is deliberately NOT here: the deposition twin.** ``simulate_cast`` /
``simulate_trial`` are entered from outside the campaign path too — the HT tab
calls the twin directly since P.12 — and nothing in them reads
``spec.measurement``. Casting a film is the same physical act whatever you
measure afterwards, so coupling the twin to the modality system would make a
second surface depend on a campaign concept it has no use for (see MAIL [p16]).
The registry wraps the *campaign* path only.

Import discipline
-----------------
``core.autonomous_wiring`` imports this module (inside functions), so this module
must not import it at module scope. The built-in EIS modality is therefore
assembled by :func:`_ensure_builtins` on first lookup, not at import time — the
one edge that would close the cycle is deferred past it. This is the same latent
cycle T2.2 found and fixed between the router and the executor; here it is
designed out rather than discovered later.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import structlog

if TYPE_CHECKING:  # annotation-only, to keep this module import-light
    from softae.analysis.eis.router import ResultRouter
    from softae.core.measurement_spec import MeasurementSpec
    from softae.workflows.workflow_model import WorkflowStep

logger = structlog.get_logger(__name__)

__all__ = [
    "Modality",
    "ModalityDisplay",
    "ObjectiveSpec",
    "UnknownModalityError",
    "get_modality",
    "list_modalities",
    "register_modality",
]


class UnknownModalityError(NotImplementedError):
    """A campaign named a modality nothing has registered.

    Subclasses :class:`NotImplementedError` on purpose, for two reasons. It is
    the literal truth — the modality is not implemented here — and it preserves
    T2.4's refusal contract exactly: ``run_autonomous_campaign`` raised a bare
    ``NotImplementedError`` for a non-EIS modality, and every caller and test
    that catches that type keeps working now that the registry lookup has
    replaced the hand-written guard.
    """


# ── The contract a modality registers ────────────────────────────────────────

@dataclass(frozen=True)
class ObjectiveSpec:
    """One objective a modality offers, and the direction it must be optimised in.

    **The direction is not a preference — it is fixed by the metric.** Minimising
    mean |Z| and maximising σ are the same physical goal, so a modality whose
    objectives carried the wrong sign would spend an entire campaign hunting the
    *worst* material on the board while every step reported success. EIS's
    entries are derived from ``autonomous_wiring.OBJECTIVE_DIRECTION`` rather
    than re-spelled here, so the two cannot fork.

    ``extractor`` and ``channel_extractor`` are the objective functions
    *themselves*, not factories that build them. The campaign supplies the
    per-run context (thickness lookup, resolved metric, the trial's tag index) as
    keyword arguments at the call site, because that context is per-trial rather
    than per-modality — and holding the real function means an identity check can
    prove the registry composes the existing extractor instead of quietly
    growing a second implementation of it.

    ``channel_extractor`` is optional: it scores a *single* electrode, which the
    batched (q-BO) and board-placement paths need and an aggregate-only modality
    may not offer.
    """

    name: str
    direction: str
    extractor: Callable[..., float | None]
    channel_extractor: Callable[..., float | None] | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(
                f"ObjectiveSpec '{self.name}': direction must be 'minimize' or "
                f"'maximize' (got {self.direction!r})"
            )


@dataclass(frozen=True)
class ModalityDisplay:
    """GUI-facing metadata, per user decision (c).

    The Autonomous tab is scaffolding today and is intended to become the prime
    AE interface; when it is built it must render *registered* modalities
    generically rather than hard-coding EIS a sixth time. This is the static data
    that lets it: a human-readable name, the presets a user may pick, and the
    unit each objective is reported in.

    **Static data only — no Qt, no widgets, no imports from ``softae.gui``.** A
    registry that reached into the GUI could not be used by the headless CLI, and
    the campaign path must stay importable on a machine with no display.
    """

    display_name: str
    preset_names: tuple[str, ...] = ()
    objective_units: Mapping[str, str] = field(default_factory=dict)
    objective_labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Modality:
    """Everything the campaign path needs to know about one kind of measurement.

    ``build_measure_step(channel, spec)`` returns the per-channel measurement
    step, or ``None`` for a modality that does not measure per electrode (an
    analysis-only or whole-board stream). ``router_factory()`` builds the result
    router that persists what comes back. ``prepare_run(spec, channels)`` is the
    lifecycle hook that runs once before any measurement step does — for EIS it
    writes the ``.mscr`` scripts the steps will read.
    """

    name: str
    build_measure_step: Callable[[int, "MeasurementSpec"], "WorkflowStep | None"]
    router_factory: Callable[[], "ResultRouter | None"]
    objectives: Mapping[str, ObjectiveSpec]
    prepare_run: Callable[..., None]
    display: ModalityDisplay

    def objective(self, kind: str) -> ObjectiveSpec:
        """The named objective, or a :class:`KeyError` naming the ones that exist."""
        try:
            return self.objectives[kind]
        except KeyError:
            raise KeyError(
                f"modality '{self.name}' offers no objective '{kind}'; it offers "
                f"{sorted(self.objectives)}"
            ) from None


# ── The registry ─────────────────────────────────────────────────────────────

#: The registry itself. Explicit, module-level, and written only by
#: :func:`register_modality` — see the module docstring on why this is not a scan.
_MODALITIES: dict[str, Modality] = {}

_bootstrapped = False


def register_modality(modality: Modality) -> Modality:
    """Add *modality* to the registry. Returns it, so it composes in a definition.

    Refuses to overwrite an existing name. Two modalities registered under one
    name would make which of them runs depend on import order — a campaign could
    measure something other than what its spec says, with nothing in the data
    recording the substitution.
    """
    if not isinstance(modality, Modality):
        raise TypeError(f"expected a Modality (got {type(modality).__name__})")
    existing = _MODALITIES.get(modality.name)
    if existing is not None and existing is not modality:
        raise ValueError(
            f"modality '{modality.name}' is already registered. Registering a "
            f"second one under the same name would make the active modality "
            f"depend on import order; pick a distinct name."
        )
    _MODALITIES[modality.name] = modality
    logger.debug("modality_registered", modality=modality.name)
    return modality


def get_modality(name: str) -> Modality:
    """Look up a modality by name, or raise :class:`UnknownModalityError`.

    This is the campaign path's single dispatch point. It is called *early* — in
    ``run_autonomous_campaign``, before a manager connects or a run row is
    written — so a spec naming something unbuildable is refused before it touches
    the rig, rather than silently receiving another modality's steps and
    recording one measurement as another.
    """
    _ensure_builtins()
    modality = _MODALITIES.get(name)
    if modality is None:
        raise UnknownModalityError(
            f"no measurement modality '{name}' is registered; registered "
            f"modalities are {sorted(_MODALITIES)}. A modality supplies the step "
            f"builder, result router and objective extractors for a kind of "
            f"measurement — register one with "
            f"softae.core.modality_registry.register_modality()."
        )
    return modality


def list_modalities() -> tuple[str, ...]:
    """Every registered modality name, sorted. The GUI's enumeration point."""
    _ensure_builtins()
    return tuple(sorted(_MODALITIES))


# ── The built-in EIS modality, composed from the existing pieces ─────────────

def _eis_build_measure_step(channel: int, spec: "MeasurementSpec") -> "WorkflowStep":
    """The existing ``eis_measure_step``, named the way the campaign names it.

    Composed, not reimplemented: the step carries the T1.5 loop-closure tags
    (``channel`` + ``measurement=primary``) that the objective extractors select
    on, and a second builder here would be a second place for those to drift.
    """
    from softae.core.autonomous_wiring import measure_step_name
    from softae.core.deposition_steps import eis_measure_step

    return eis_measure_step(channel, name=measure_step_name(channel))


def _eis_router_factory() -> "ResultRouter":
    """The existing T1.4/T2.2 router, which returns a ``MeasurementResult``."""
    from softae.analysis.eis.router import EISResultRouter

    return EISResultRouter()


def _eis_prepare_run(
    spec: "MeasurementSpec",
    channels: Sequence[int],
    *,
    temp_dir: str | None = None,
    emit: Callable[..., None] | None = None,
) -> None:
    """Write this campaign's ``.mscr`` scripts before any measurement step runs.

    Moved verbatim out of ``run_autonomous_campaign`` (T2.5). The behaviour it
    exists to guarantee is unchanged and worth restating, because it is subtle:
    an ``eis_measure_step`` carries only a *path*, so before this block existed a
    campaign measured with whatever parameters some earlier HT or manual session
    happened to leave in the temp directory — while recording its own preset as
    provenance. Always overwritten, so stale parameters cannot survive into a run.

    ``enabled=False`` (*formulate and cast, but do not measure*) writes nothing
    and emits nothing, exactly as the inline guard did.

    *temp_dir* is accepted for the lifecycle contract and **deliberately ignored
    by EIS**: the script path comes from ``eis_scripts.mscr_path_for_channel``,
    which the measurement *step* reads too. Relocating the files from this side
    alone would point the writer and the reader at different paths. A modality
    whose payload location is genuinely its own — a camera writing frames — is
    what the parameter is for.
    """
    if not spec.enabled:
        return

    from softae.core.eis_scripts import EISParams, build_eis_scripts

    channels = list(channels)
    eis_params = EISParams.from_preset(spec.preset, **spec.overrides)
    build_eis_scripts(channels, eis_params)
    if emit is not None:
        # ``n`` counts the channels *asked for*, not the files successfully
        # written — build_eis_scripts is best-effort per channel and logs its own
        # failures, and the event has always reported the intent.
        emit("eis_scripts_built", n=len(channels), **eis_params.as_metadata())


def _late_bound(attr: str) -> Callable[..., float | None]:
    """A reference to one of ``autonomous_wiring``'s extractors, resolved on call.

    **Late binding is the behaviour being preserved, not an implementation
    detail.** Before T2.5 the campaign called ``eis_impedance_objective(...)`` as
    a module global, so the name resolved afresh on every trial. Capturing the
    function object at registration instead would silently break every caller
    that rebinds the module attribute — including the P1.2/P1.3 park guards,
    which force an unmeasured trial by monkeypatching exactly that name, and
    which would have gone on passing a campaign that no longer parked.

    ``functools.wraps`` gives the wrapper the target's identity for
    introspection (``__wrapped__``, ``__name__``, docstring), so the registry can
    still be *proved* to compose the real extractor rather than a copy.
    """
    from softae.core import autonomous_wiring

    target = getattr(autonomous_wiring, attr)

    @functools.wraps(target)
    def _resolve_and_call(*args: object, **kwargs: object) -> float | None:
        from softae.core import autonomous_wiring as wiring

        return getattr(wiring, attr)(*args, **kwargs)

    return _resolve_and_call


def _eis_objectives() -> dict[str, ObjectiveSpec]:
    """EIS's objectives, with directions *derived* from the existing map.

    ``OBJECTIVE_DIRECTION`` stays the single authority on which way each metric
    is optimised. Re-spelling ``"maximize"`` here would create a second table
    that looks authoritative and can silently disagree with the first.
    """
    from softae.core.autonomous_wiring import OBJECTIVE_DIRECTION

    return {
        kind: ObjectiveSpec(
            name=kind,
            direction=direction,
            extractor=_late_bound("eis_impedance_objective"),
            channel_extractor=_late_bound("eis_impedance_objective_for_channel"),
        )
        for kind, direction in OBJECTIVE_DIRECTION.items()
    }


def _eis_preset_names() -> tuple[str, ...]:
    """The ``[eis_presets.*]`` section names, for a GUI to offer.

    Read from config rather than hard-coded: a shipped tuple would go stale the
    moment somebody adds a preset, and a picker offering a preset that no longer
    exists is worse than one offering none. Degrades to empty if config cannot be
    read — a GUI showing nothing is recoverable; a GUI asserting four presets
    that may not exist is not.
    """
    try:
        from softae.config.loader import eis_presets

        return tuple(sorted(eis_presets() or {}))
    except Exception:
        logger.warning("eis_preset_names_unavailable", exc_info=True)
        return ()


def _ensure_builtins() -> None:
    """Register the built-in modalities on first lookup.

    Deferred rather than done at import time because assembling the EIS modality
    imports ``core.autonomous_wiring``, which imports *this* module — see the
    module docstring. Doing it lazily means the cycle is never closed during
    either module's execution, whichever is imported first.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True  # set first: the assembly below imports modules that
    # may themselves reach the registry, and a second bootstrap would raise on
    # the duplicate-name guard.
    register_modality(
        Modality(
            name="eis",
            build_measure_step=_eis_build_measure_step,
            router_factory=_eis_router_factory,
            objectives=_eis_objectives(),
            prepare_run=_eis_prepare_run,
            display=ModalityDisplay(
                display_name="Electrochemical impedance spectroscopy",
                preset_names=_eis_preset_names(),
                objective_units={"mean_abs_z": "Ω", "sigma": "S/cm"},
                objective_labels={
                    "mean_abs_z": "Mean |Z|",
                    "sigma": "Conductivity (σ)",
                },
            ),
        )
    )
