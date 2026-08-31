from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from bps_agent.cli import (
    EXIT_CODES,
    _configure_logging,
    _parser,
    _verdict_console_fields,
)
from bps_agent.config import apply_overrides
from bps_agent.models.common import (
    DutCollectionMethod,
    EvaluationMode,
    EvaluationOutcome,
    VerdictValue,
)
from bps_agent.models.config import AppConfig, RunOverrides
from bps_agent.models.evaluation import (
    CHECKPOINT_SCHEMA_VERSION,
    AttemptRecord,
    VerdictDocument,
)
from bps_agent.resume import load_resume_config, validate_resume_request
from bps_agent.runtime import build_judge


class _CheckpointConfigState(TypedDict):
    schema_version: str
    config: dict[str, Any]


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
            "--total-bandwidth-mbps",
            "300",
            "--stop-before-llm",
            "--bps-only",
        ]
    )

    assert arguments.template == "other-performance-template"
    assert arguments.ports == [6, 7]
    assert arguments.total_bandwidth_mbps == 300.0
    assert arguments.stop_before_llm is True
    assert arguments.bps_only is True


def test_bps_overrides_are_validated_without_mutating_base_config(
    app_config: AppConfig,
) -> None:
    overridden = apply_overrides(
        app_config,
        RunOverrides(
            template=" other-performance-template ",
            ports=(6, 7),
            total_bandwidth_mbps=300.0,
        ),
    )

    assert overridden.bps.template == "other-performance-template"
    assert overridden.bps.ports == (6, 7)
    assert overridden.bps.total_bandwidth_mbps == 300.0
    assert app_config.bps.template == "performance-demo"
    assert app_config.bps.ports == (4, 5)
    assert app_config.bps.total_bandwidth_mbps == 400.0


@pytest.mark.parametrize("ports", [(), (-1,), (6, 6)])
def test_invalid_port_overrides_are_rejected(app_config: AppConfig, ports: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        apply_overrides(
            app_config,
            RunOverrides(ports=ports),
        )


@pytest.mark.parametrize(
    "override_values",
    [
        {"template": "other-performance-template"},
        {"dut_host": "10.66.246.156"},
        {"total_bandwidth_mbps": 300.0},
        {"evaluation_mode": EvaluationMode.BPS_ONLY},
    ],
)
def test_resume_rejects_run_overrides(override_values: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="cannot be used with --resume"):
        validate_resume_request("evaluation-id", RunOverrides.model_validate(override_values))


@pytest.mark.parametrize("bandwidth", [0.0, -1.0])
def test_invalid_bandwidth_overrides_are_rejected(app_config: AppConfig, bandwidth: float) -> None:
    with pytest.raises(ValueError):
        apply_overrides(
            app_config,
            RunOverrides(total_bandwidth_mbps=bandwidth),
        )


def test_cli_allows_target_above_400_for_larger_templates(app_config: AppConfig) -> None:
    overridden = apply_overrides(
        app_config,
        RunOverrides(total_bandwidth_mbps=800.0),
    )

    assert overridden.bps.total_bandwidth_mbps == 800.0


def test_cli_can_override_the_evaluation_to_bps_only(app_config: AppConfig) -> None:
    overridden = apply_overrides(
        app_config,
        RunOverrides(evaluation_mode=EvaluationMode.BPS_ONLY),
    )

    assert overridden.evaluation.mode == EvaluationMode.BPS_ONLY
    assert app_config.evaluation.mode == EvaluationMode.BPS_AND_DUT


def test_cli_can_override_backend_dut_parameters(app_config: AppConfig) -> None:
    overridden = apply_overrides(
        app_config,
        RunOverrides(
            dut_collection_method=DutCollectionMethod.BACKEND_SSH,
            dut_host="10.66.246.156",
            dut_port=50023,
            dut_interfaces=("T1/1", "T1/2"),
            dut_interval_seconds=10,
        ),
    )

    assert overridden.dut.collection_method == DutCollectionMethod.BACKEND_SSH
    assert overridden.dut.backend.host == "10.66.246.156"
    assert overridden.dut.backend.port == 50023
    assert overridden.dut.interfaces == ("T1/1", "T1/2")
    assert overridden.dut.backend.interval_seconds == 10


def test_parser_accepts_repeatable_dut_backend_overrides() -> None:
    arguments = _parser().parse_args(
        [
            "run",
            "--dut-collection-method",
            "backend_ssh",
            "--dut-host",
            "10.66.246.156",
            "--dut-port",
            "50023",
            "--dut-interface",
            "T1/1",
            "--dut-interface",
            "T1/2",
            "--dut-interval-seconds",
            "10",
        ]
    )

    assert arguments.dut_collection_method == DutCollectionMethod.BACKEND_SSH
    assert arguments.dut_interfaces == ["T1/1", "T1/2"]


def test_resume_loads_runtime_configuration_from_checkpoint(app_config: AppConfig) -> None:
    builder = StateGraph(_CheckpointConfigState)
    builder.add_node("finish", lambda _state: {})
    builder.add_edge(START, "finish")
    builder.add_edge("finish", END)
    invocation = {"configurable": {"thread_id": "resume-config"}}
    checkpoint_config = app_config.model_dump(mode="json")
    with SqliteSaver.from_conn_string(str(app_config.storage.checkpoint_db)) as saver:
        builder.compile(checkpointer=saver).invoke(
            {"schema_version": CHECKPOINT_SCHEMA_VERSION, "config": checkpoint_config},
            config=invocation,
        )

    changed = app_config.model_copy(
        update={
            "bps": app_config.bps.model_copy(
                update={"endpoint": "https://new-bps.example.test", "ports": (8, 9)}
            ),
            "dut": app_config.dut.model_copy(
                update={
                    "frontend": app_config.dut.frontend.model_copy(
                        update={"endpoint": "https://new-dut.example.test"}
                    )
                }
            ),
        }
    )

    restored = load_resume_config(changed, "resume-config")

    assert restored == app_config


@pytest.mark.parametrize("checkpoint_version", [None, "0"])
def test_resume_rejects_unsupported_checkpoint_schema(
    app_config: AppConfig,
    checkpoint_version: str | None,
) -> None:
    builder = StateGraph(_CheckpointConfigState)
    builder.add_node("finish", lambda _state: {})
    builder.add_edge(START, "finish")
    builder.add_edge("finish", END)
    thread_id = f"unsupported-schema-{checkpoint_version}"
    state: dict[str, Any] = {"config": app_config.model_dump(mode="json")}
    if checkpoint_version is not None:
        state["schema_version"] = checkpoint_version
    with SqliteSaver.from_conn_string(str(app_config.storage.checkpoint_db)) as saver:
        builder.compile(checkpointer=saver).invoke(
            state,  # type: ignore[arg-type]
            config={"configurable": {"thread_id": thread_id}},
        )

    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_resume_config(app_config, thread_id)


def test_console_verdict_contains_the_complete_parsed_document() -> None:
    attempt = AttemptRecord(
        number=1,
        started_at="2026-08-24T00:00:00+00:00",
        verdict=VerdictDocument(
            verdict=VerdictValue.PASS,
            summary="test passed",
            observations=["BPS criteria passed", "DUT remained healthy"],
            risks=["fixture risk"],
            confidence=0.95,
        ),
    )

    selected = _verdict_console_fields({"attempts": [attempt.model_dump(mode="json")]})

    assert selected == {
        "parsed": {
            "verdict": "pass",
            "schema_version": "dev-1",
            "summary": "test passed",
            "observations": ["BPS criteria passed", "DUT remained healthy"],
            "risks": ["fixture risk"],
            "confidence": 0.95,
        }
    }


def test_cli_suppresses_http_client_info_logs() -> None:
    _configure_logging(verbose=True)

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_live_and_replay_share_the_complete_llm_configuration(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class CapturingJudge:
        def __init__(
            self,
            provider_name: str,
            provider_config: object,
            *,
            token: str,
            reasoning_effort: str,
        ) -> None:
            calls.append(
                {
                    "provider_name": provider_name,
                    "provider_config": provider_config,
                    "token": token,
                    "reasoning_effort": reasoning_effort,
                }
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("bps_agent.runtime.DeepSeekJudge", CapturingJudge)
    config = app_config.model_copy(
        update={"llm": app_config.llm.model_copy(update={"reasoning_effort": "high"})}
    )
    credentials = {config.llm.selected.token_env: "token"}

    live_judge = build_judge(config, credentials)
    replay_judge = build_judge(config, credentials)

    assert live_judge is not replay_judge
    assert calls[0] == calls[1]
    assert calls[0]["reasoning_effort"] == "high"


def test_degraded_pass_has_distinct_nonzero_exit_code() -> None:
    assert EXIT_CODES[EvaluationOutcome.PASSED.value] == 0
    assert EXIT_CODES[EvaluationOutcome.DEGRADED_PASS.value] == 1
