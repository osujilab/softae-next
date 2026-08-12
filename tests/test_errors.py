"""Tests for the custom exception hierarchy."""

import pytest

from softae.errors import (
    AbortedError,
    AnalysisError,
    CommunicationError,
    ConnectionError_,
    InstrumentError,
    SafetyError,
    SoftAEError,
    StepTimeoutError,
    ValidationError_,
    WorkflowError,
)


class TestExceptionHierarchy:
    """Verify that every custom exception is catchable via its parent."""

    def test_base_catches_all(self):
        for exc_cls in (
            InstrumentError,
            ConnectionError_,
            CommunicationError,
            SafetyError,
            WorkflowError,
            StepTimeoutError,
            AbortedError,
            ValidationError_,
            AnalysisError,
        ):
            with pytest.raises(SoftAEError):
                if exc_cls is StepTimeoutError:
                    raise exc_cls("test_step", 10.0)
                elif exc_cls is SafetyError:
                    raise exc_cls("test", instrument="x")
                else:
                    raise exc_cls("test")

    def test_instrument_error_has_instrument_attr(self):
        exc = InstrumentError("timeout", instrument="stage")
        assert exc.instrument == "stage"
        assert "stage" in str(exc)

    def test_safety_error_carries_limits(self):
        exc = SafetyError(
            "too hot",
            instrument="temp_controller",
            requested=350.0,
            limit=300.0,
        )
        assert exc.requested == 350.0
        assert exc.limit == 300.0

    def test_step_timeout_error(self):
        exc = StepTimeoutError("measure_eis", 60.0)
        assert exc.step_name == "measure_eis"
        assert exc.timeout_s == 60.0
        assert "60.0" in str(exc)

    def test_workflow_error_catches_subtypes(self):
        for cls in (StepTimeoutError, AbortedError, ValidationError_):
            with pytest.raises(WorkflowError):
                if cls is StepTimeoutError:
                    raise cls("step", 10)
                else:
                    raise cls("test")
