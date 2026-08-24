from __future__ import annotations

import logging
from pathlib import Path

import pytest

from bps_agent.cli import (
    _apply_bps_overrides,
    _configure_logging,
    _parser,
    _verdict_console_fields,
)
from bps_agent.models import AppConfig, AttemptRecord, VerdictDocument, VerdictValue


@pytest.mark.parametrize(
    "arguments",
    [
        ["run"],
        ["replay", "--evidence", "artifacts/evaluation/attempt-01/evidence.json"],
    ],
)
def test_commands_default_to_demo_configuration(arguments: list[str]) -> None:
    parsed = _parser().parse_args(arguments)

    assert parsed.config == Path("config/demo.yaml")


def test_run_parser_accepts_template_and_port_overrides() -> None:
    arguments = _parser().parse_args(
        [
            "run",
            "--config",
            "config/demo.yaml",
            "--template",
            "other-performance-template",
            "--ports",
            "6",
            "7",
            "--stop-before-llm",
        ]
    )

    assert arguments.template == "other-performance-template"
    assert arguments.ports == [6, 7]
    assert arguments.stop_before_llm is True


def test_bps_overrides_are_validated_without_mutating_base_config(
    app_config: AppConfig,
) -> None:
    overridden = _apply_bps_overrides(
        app_config,
        template=" other-performance-template ",
        ports=(6, 7),
        resume_id=None,
    )

    assert overridden.bps.template == "other-performance-template"
    assert overridden.bps.ports == (6, 7)
    assert app_config.bps.template == "performance-demo"
    assert app_config.bps.ports == (4, 5)


@pytest.mark.parametrize("ports", [(), (-1,), (6, 6)])
def test_invalid_port_overrides_are_rejected(app_config: AppConfig, ports: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        _apply_bps_overrides(
            app_config,
            template=None,
            ports=ports,
            resume_id=None,
        )


def test_resume_rejects_bps_overrides(app_config: AppConfig) -> None:
    with pytest.raises(ValueError, match="cannot be used with --resume"):
        _apply_bps_overrides(
            app_config,
            template="other-performance-template",
            ports=None,
            resume_id="evaluation-id",
        )


def test_console_verdict_contains_summary_and_observations_only() -> None:
    attempt = AttemptRecord(
        number=1,
        started_at="2026-08-24T00:00:00+00:00",
        verdict=VerdictDocument(
            verdict=VerdictValue.PASS,
            summary="test passed",
            observations=["BPS criteria passed", "DUT remained healthy"],
            confidence=0.95,
        ),
    )

    selected = _verdict_console_fields({"attempts": [attempt.model_dump(mode="json")]})

    assert selected == {
        "summary": "test passed",
        "observations": ["BPS criteria passed", "DUT remained healthy"],
    }


def test_cli_suppresses_http_client_info_logs() -> None:
    _configure_logging(verbose=True)

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
