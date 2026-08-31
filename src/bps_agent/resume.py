"""Checkpoint validation and interrupted Evaluation Run recovery."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from bps_agent.models.bps import PortReservationState
from bps_agent.models.config import AppConfig, RunOverrides
from bps_agent.models.evaluation import CHECKPOINT_SCHEMA_VERSION, AttemptRecord
from bps_agent.ports import BpsPort

LOGGER = logging.getLogger(__name__)


def validate_resume_request(resume_id: str | None, overrides: RunOverrides) -> None:
    if resume_id and overrides.has_values:
        raise ValueError("run configuration overrides cannot be used with --resume")


def load_resume_config(current_config: AppConfig, resume_id: str) -> AppConfig:
    checkpoint_db = current_config.storage.checkpoint_db
    invocation_config: RunnableConfig = {"configurable": {"thread_id": resume_id}}
    with SqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        checkpoint_tuple = saver.get_tuple(invocation_config)
    if checkpoint_tuple is None:
        raise ValueError(f"no checkpoint exists for Evaluation Run {resume_id}")
    channel_values = checkpoint_tuple.checkpoint.get("channel_values")
    if not isinstance(channel_values, dict):
        raise ValueError(f"checkpoint for Evaluation Run {resume_id} has invalid state")
    version = channel_values.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {version!r}")
    checkpoint_config = channel_values.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"checkpoint for Evaluation Run {resume_id} omitted its configuration")
    try:
        restored = AppConfig.model_validate(checkpoint_config)
    except ValueError as exc:
        raise ValueError(
            f"checkpoint for Evaluation Run {resume_id} contains an invalid configuration"
        ) from exc
    if restored.storage.checkpoint_db.resolve() != checkpoint_db.resolve():
        raise ValueError(
            f"checkpoint for Evaluation Run {resume_id} refers to a different checkpoint database"
        )
    return restored


def invoke_evaluation(
    graph: Any,
    evaluation_id: str,
    config: AppConfig,
    *,
    resume: bool,
    bps: BpsPort,
) -> dict[str, Any]:
    invocation_config: RunnableConfig = {"configurable": {"thread_id": evaluation_id}}
    if resume:
        snapshot = graph.get_state(invocation_config)
        if not snapshot.values:
            raise ValueError(f"no checkpoint exists for Evaluation Run {evaluation_id}")
    try:
        if resume:
            result = graph.invoke(None, config=invocation_config)
        else:
            from bps_agent.graph import initial_state

            result = graph.invoke(
                initial_state(evaluation_id, config), config=invocation_config
            )
    except BaseException:
        _request_interrupted_run_stop(graph, invocation_config, bps)
        raise
    if not isinstance(result, dict):
        raise RuntimeError("Evaluation graph returned an invalid result")
    return result


def _request_interrupted_run_stop(
    graph: Any,
    invocation_config: RunnableConfig,
    bps: BpsPort,
) -> None:
    with suppress(Exception):
        snapshot = graph.get_state(invocation_config)
        attempts = snapshot.values.get("attempts", []) if snapshot.values else []
        if not attempts:
            return
        active = AttemptRecord.model_validate(attempts[-1])
        if active.bps_run_id and not active.terminal_confirmed:
            with suppress(Exception):
                bps.stop_run(active.bps_run_id)
            LOGGER.error(
                "Stop requested for BPS run %s; terminal state remains unconfirmed",
                active.bps_run_id,
            )


def has_unsafe_reservation(result: dict[str, Any]) -> bool:
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    try:
        attempt = AttemptRecord.model_validate(attempts[-1])
    except ValueError:
        return True
    return attempt.port_reservation_state != PortReservationState.NONE
