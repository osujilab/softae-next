"""Custom exception hierarchy for SoftAE.

All SoftAE‑specific exceptions descend from :class:`SoftAEError` so that
callers can catch the entire family with a single ``except SoftAEError:``
clause when appropriate.

Tree::

    Exception
    └── SoftAEError
        ├── InstrumentError
        │   ├── ConnectionError_      — failed to open port / resource
        │   ├── CommunicationError    — timeout, no response, garbled reply
        │   └── SafetyError           — out‑of‑bounds setpoint or interlock
        ├── WorkflowError
        │   ├── StepTimeoutError      — step exceeded its ``timeout_s``
        │   ├── AbortedError          — operator or agent aborted execution
        │   └── ValidationError_      — bad YAML schema / missing param
        └── AnalysisError             — fitting failure, data shape mismatch
        └── OptimizerError            — optimizer config or execution failure
        └── CampaignError             — simulated BO-campaign config or data error
"""

from __future__ import annotations


# ── Base ────────────────────────────────────────────────────────────────────
class SoftAEError(Exception):
    """Root exception for every SoftAE‑specific error."""


# ── Instrument Errors ───────────────────────────────────────────────────────
class InstrumentError(SoftAEError):
    """An error related to a hardware instrument."""

    def __init__(self, message: str, instrument: str | None = None):
        self.instrument = instrument
        super().__init__(f"[{instrument}] {message}" if instrument else message)


class ConnectionError_(InstrumentError):
    """Failed to open or establish a connection to an instrument.

    Named with a trailing underscore to avoid shadowing the built‑in
    :class:`ConnectionError`.
    """


class CommunicationError(InstrumentError):
    """The instrument did not respond, timed out, or returned garbled data."""


class SafetyError(InstrumentError):
    """A commanded value exceeds the configured safety limits.

    Examples: temperature setpoint above ``safety.max_temp``, stage position
    outside the allowed travel range.
    """

    def __init__(
        self,
        message: str,
        *,
        instrument: str | None = None,
        requested: float | None = None,
        limit: float | None = None,
    ):
        self.requested = requested
        self.limit = limit
        super().__init__(message, instrument=instrument)


# ── Workflow Errors ─────────────────────────────────────────────────────────
class WorkflowError(SoftAEError):
    """An error raised during workflow parsing or execution."""


class StepTimeoutError(WorkflowError):
    """A workflow step exceeded its configured ``timeout_s``."""

    def __init__(self, step_name: str, timeout_s: float):
        self.step_name = step_name
        self.timeout_s = timeout_s
        super().__init__(
            f"Step '{step_name}' timed out after {timeout_s:.1f} s"
        )


class AbortedError(WorkflowError):
    """The workflow was explicitly aborted by the operator or autonomous agent."""


class ValidationError_(WorkflowError):
    """Workflow definition failed schema or parameter validation.

    Named with trailing underscore to avoid shadowing ``pydantic.ValidationError``
    or similar third‑party names.
    """


# ── Analysis Errors ─────────────────────────────────────────────────────────
class AnalysisError(SoftAEError):
    """An error in data analysis or circuit‑model fitting."""


# ── Optimizer Errors ────────────────────────────────────────────────────────
class OptimizerError(SoftAEError):
    """An error in optimizer configuration or execution."""


# ── Campaign Errors ─────────────────────────────────────────────────────────
class CampaignError(SoftAEError):
    """An error in a simulated BO-campaign configuration, dataset, or run.

    Raised by the :mod:`softae.campaigns` subsystem for malformed datasets,
    invalid campaign configs, missing optional backends, and pool/oracle
    contract violations.
    """
