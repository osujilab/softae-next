"""Every headless entrypoint filters its own log stream before it runs anything.

Only the GUI (``gui/app.py``) and a handful of tools ever called
``structlog.configure``. Everything else inherited structlog's default
``PrintLogger``, which emits **every** level — so a CLI that brings the chamber
to condition printed ``rh_duty_sent`` once per RH PID update and buried the
``[settle]`` round lines the operator was actually reading. The emit site is
correct (``drivers/async_rh_controller.py`` logs a per-update duty at ``debug``);
the defect was on the consumer side, in entrypoints that configured nothing.

Three kinds of test, proving different things:

* the **adoption** test reads source and cannot prove the call runs. It pins
  that no entrypoint we own can be rewritten without the rule being noticed —
  the failure mode that actually happened, one tool having the guard and its
  neighbours silently not.
* the **behavioural** tests prove ``configure_logging`` drops DEBUG and keeps
  the run's own ``info`` reporting, against captured output rather than against
  a driver — no RH controller is instantiated here, and none should be.
* the **flag** tests pin ``-v`` on both sides of the subcommand, the spelling
  ``default=argparse.SUPPRESS`` exists to protect.
"""

from __future__ import annotations

import importlib
import inspect
import logging

import pytest
import structlog

from softae.tools import add_verbosity_flag, configure_logging

#: ``module:function`` for the headless entrypoints this session owns. Tools
#: owned by other sessions (``env_hold``, ``eis_timing``, ``campaign``) are
#: deliberately absent rather than asserted-about.
WIRED_ENTRYPOINTS = [
    "softae.tools.commission:main",
    "softae.tools.eis_validate:main",
    "softae.tools.equilibration:main",
]

RH_DUTY_EVENT = "rh_duty_sent"
RH_LOGGER = "softae.drivers.async_rh_controller"


@pytest.fixture()
def structlog_state():
    """Save and restore the process-wide log configuration.

    ``configure_logging`` deliberately mutates global state, so a test that
    exercises it would otherwise leave every later test running at whatever
    level it chose.
    """
    saved = structlog.get_config()
    root_level = logging.getLogger().level
    try:
        yield
    finally:
        structlog.configure(**saved)
        logging.getLogger().setLevel(root_level)


@pytest.fixture()
def level_is_info(monkeypatch):
    """Pin the configured level, so these tests read the helper and not the rig's
    ``softae_config.toml`` (which ships ``[logging] level = "INFO"``)."""
    from softae.config import loader

    monkeypatch.setattr(loader, "log_level", lambda: "INFO")


@pytest.mark.parametrize("target", WIRED_ENTRYPOINTS)
def test_entrypoint_main_calls_configure_logging(target):
    module_path, _, func_name = target.partition(":")
    func = getattr(importlib.import_module(module_path), func_name)
    assert "configure_logging" in inspect.getsource(func), (
        f"{target} runs without filtering its log stream: it will print "
        f"{RH_DUTY_EVENT} on every RH update"
    )


@pytest.mark.parametrize("target", WIRED_ENTRYPOINTS)
def test_entrypoint_parser_accepts_verbose_before_and_after_the_subcommand(target):
    # A subparser copies its own defaults over the outer namespace after it
    # parses, so `-v` *before* the subcommand is the spelling that silently
    # breaks when the flag is declared without argparse.SUPPRESS.
    module_path, _, _ = target.partition(":")
    module = importlib.import_module(module_path)
    subcommand = {
        "softae.tools.commission": ["status"],
        "softae.tools.eis_validate": ["report", "--validation-name", "x"],
        "softae.tools.equilibration": ["plan"],
    }[module_path]

    for argv in (["-v", *subcommand], [*subcommand, "-v"]):
        args = module.build_parser().parse_args(argv)
        assert getattr(args, "verbose", False) is True, argv


def test_add_verbosity_flag_omits_the_attribute_when_the_flag_is_absent():
    # SUPPRESS means "absent", not "False" -- which is why every caller reads it
    # as getattr(args, "verbose", False).
    import argparse

    parser = argparse.ArgumentParser()
    add_verbosity_flag(parser)
    assert not hasattr(parser.parse_args([]), "verbose")


def test_configure_logging_default_honours_the_configured_level(
        structlog_state, monkeypatch):
    from softae.config import loader

    monkeypatch.setattr(loader, "log_level", lambda: "WARNING")
    assert configure_logging() == logging.WARNING
    monkeypatch.setattr(loader, "log_level", lambda: "INFO")
    assert configure_logging() == logging.INFO


def test_configure_logging_default_drops_rh_duty_but_keeps_the_tools_own_info(
        structlog_state, level_is_info, capsys):
    from softae.tools import eis_validate

    configure_logging()
    capsys.readouterr()

    # The exact line the operator drowned in, from the module that emits it --
    # named, not instantiated: this test opens no instrument session.
    structlog.get_logger(RH_LOGGER).debug(RH_DUTY_EVENT, duty=0.42)
    # The tool's own reporting, through its real module logger.
    eis_validate.logger.info("eis_validation_start", validation_name="V1")
    out = capsys.readouterr().out

    assert RH_DUTY_EVENT not in out
    assert "eis_validation_start" in out, "the run's own reporting was silenced"


def test_configure_logging_verbose_restores_debug(
        structlog_state, level_is_info, capsys):
    assert configure_logging(verbose=True) == logging.DEBUG
    capsys.readouterr()

    structlog.get_logger(RH_LOGGER).debug(RH_DUTY_EVENT, duty=0.42)
    assert RH_DUTY_EVENT in capsys.readouterr().out


def test_configure_logging_twice_applies_the_second_level(
        structlog_state, level_is_info, capsys):
    # Idempotence is not enough: logging.basicConfig returns early once the root
    # logger has a handler, so a second call has to set the level itself.
    configure_logging(verbose=True)
    assert configure_logging() == logging.INFO
    assert logging.getLogger().level == logging.INFO
    capsys.readouterr()

    structlog.get_logger(RH_LOGGER).debug(RH_DUTY_EVENT, duty=0.42)
    assert RH_DUTY_EVENT not in capsys.readouterr().out


def test_configure_logging_after_an_existing_configuration_is_safe(
        structlog_state, level_is_info, capsys):
    # The GUI configures structlog itself before anything else runs. Calling the
    # shared helper afterwards must replace that wrapper, not stack on it.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG))
    assert configure_logging() == logging.INFO
    capsys.readouterr()

    structlog.get_logger(RH_LOGGER).debug(RH_DUTY_EVENT, duty=0.42)
    structlog.get_logger(RH_LOGGER).info("rh_setpoint_reached", rh=30.0)
    out = capsys.readouterr().out

    assert RH_DUTY_EVENT not in out
    assert out.count("rh_setpoint_reached") == 1, "output was duplicated or lost"
