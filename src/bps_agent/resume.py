"""Checkpoint validation and interrupted Evaluation Run recovery."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from bps_agent.errors import CheckpointError, ErrorCode
from bps_agent.models.bps import PortReservationState
from bps_agent.models.config import AppConfig, RunOverrides
from bps_agent.models.evaluation import AttemptRecord
from bps_agent.ports import BpsPort

LOGGER = logging.getLogger(__name__)


@contextmanager
def checkpoint_store(checkpoint_db: Any) -> Iterator[Any]:
    """Open the LangGraph checkpoint store with a stable Agent error boundary."""

    try:
        manager = SqliteSaver.from_conn_string(str(checkpoint_db))
        saver = manager.__enter__()
    except (sqlite3.Error, OSError) as exc:
        raise CheckpointError(
            "could not open the checkpoint database",
            code=ErrorCode.CHECKPOINT_IO_ERROR,
            hint="Check the checkpoint database path, permissions, and lock state.",
        ) from exc
    try:
        yield saver
    except BaseException as exc:
        try:
            manager.__exit__(type(exc), exc, exc.__traceback__)
        except (sqlite3.Error, OSError) as close_exc:
            raise CheckpointError(
                "could not close the checkpoint database",
                code=ErrorCode.CHECKPOINT_IO_ERROR,
                hint="Check the checkpoint database path and filesystem health.",
            ) from close_exc
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except (sqlite3.Error, OSError) as exc:
            raise CheckpointError(
                "could not close the checkpoint database",
                code=ErrorCode.CHECKPOINT_IO_ERROR,
                hint="Check the checkpoint database path and filesystem health.",
            ) from exc


def validate_resume_request(resume_id: str | None, overrides: RunOverrides) -> None:
    if resume_id and overrides.has_values:
        raise CheckpointError(
            "run configuration overrides cannot be used with --resume",
            code=ErrorCode.RESUME_CONFLICT,
            hint="Remove run overrides or start a new Evaluation Run.",
        )


def load_resume_config(current_config: AppConfig, resume_id: str) -> AppConfig:
    checkpoint_db = current_config.storage.checkpoint_db
    invocation_config: RunnableConfig = {"configurable": {"thread_id": resume_id}}
    try:
        with checkpoint_store(checkpoint_db) as saver:
            checkpoint_tuple = saver.get_tuple(invocation_config)
    except CheckpointError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise CheckpointError(
            f"could not read checkpoint for Evaluation Run {resume_id}",
            code=ErrorCode.CHECKPOINT_IO_ERROR,
            hint="Check the checkpoint database path, permissions, and lock state.",
        ) from exc
    except (KeyError, TypeError, ValueError, EOFError) as exc:
        raise CheckpointError(
            f"invalid checkpoint for Evaluation Run {resume_id}",
            code=ErrorCode.CHECKPOINT_INVALID,
        ) from exc
    if checkpoint_tuple is None:
        raise CheckpointError(
            f"no checkpoint exists for Evaluation Run {resume_id}",
            code=ErrorCode.CHECKPOINT_NOT_FOUND,
            hint="Check the Evaluation Run ID or start a new run.",
        )
    try:
        checkpoint = checkpoint_tuple.checkpoint
    except (AttributeError, TypeError, ValueError) as exc:
        raise CheckpointError(
            f"invalid checkpoint for Evaluation Run {resume_id}",
            code=ErrorCode.CHECKPOINT_INVALID,
        ) from exc
    if not isinstance(checkpoint, dict):
        raise CheckpointError(
            f"invalid checkpoint for Evaluation Run {resume_id}",
            code=ErrorCode.CHECKPOINT_INVALID,
        )
    channel_values = checkpoint.get("channel_values")
    if not isinstance(channel_values, dict):
        raise CheckpointError(
            f"invalid checkpoint for Evaluation Run {resume_id}",
            code=ErrorCode.CHECKPOINT_INVALID,
        )
    checkpoint_config = channel_values.get("config")
    if not isinstance(checkpoint_config, dict):
        raise CheckpointError(
            f"invalid checkpoint for Evaluation Run {resume_id}: missing configuration",
            code=ErrorCode.CHECKPOINT_INVALID,
        )
    try:
        restored = AppConfig.model_validate(checkpoint_config)
    except ValueError as exc:
        raise CheckpointError(
            f"invalid checkpoint for Evaluation Run {resume_id}: invalid configuration",
            code=ErrorCode.CHECKPOINT_INVALID,
        ) from exc
    if restored.storage.checkpoint_db.resolve() != checkpoint_db.resolve():
        raise CheckpointError(
            f"checkpoint for Evaluation Run {resume_id} refers to a different checkpoint database",
            code=ErrorCode.CHECKPOINT_INVALID,
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
        try:
            snapshot = graph.get_state(invocation_config)
        except (sqlite3.Error, OSError) as exc:
            raise CheckpointError(
                f"could not read checkpoint for Evaluation Run {evaluation_id}",
                code=ErrorCode.CHECKPOINT_IO_ERROR,
                hint="Check the checkpoint database path, permissions, and lock state.",
            ) from exc
        except (ValidationError, KeyError, TypeError) as exc:
            raise CheckpointError(
                f"invalid checkpoint for Evaluation Run {evaluation_id}",
                code=ErrorCode.CHECKPOINT_INVALID,
            ) from exc
        values = getattr(snapshot, "values", None)
        if values is None:
            raise CheckpointError(
                f"no checkpoint exists for Evaluation Run {evaluation_id}",
                code=ErrorCode.CHECKPOINT_NOT_FOUND,
                hint="Check the Evaluation Run ID or start a new run.",
            )
        if not isinstance(values, dict):
            raise CheckpointError(
                f"invalid checkpoint for Evaluation Run {evaluation_id}",
                code=ErrorCode.CHECKPOINT_INVALID,
            )
        if not values:
            raise CheckpointError(
                f"no checkpoint exists for Evaluation Run {evaluation_id}",
                code=ErrorCode.CHECKPOINT_NOT_FOUND,
                hint="Check the Evaluation Run ID or start a new run.",
            )
    try:
        if resume:
            result = graph.invoke(None, config=invocation_config)
        else:
            from bps_agent.graph import initial_state

            result = graph.invoke(initial_state(evaluation_id, config), config=invocation_config)
    except (sqlite3.Error, OSError) as exc:
        _request_interrupted_run_stop(graph, invocation_config, bps)
        raise CheckpointError(
            f"could not update checkpoint for Evaluation Run {evaluation_id}",
            code=ErrorCode.CHECKPOINT_IO_ERROR,
            hint="Check the checkpoint database path, permissions, and lock state.",
        ) from exc
    except ValidationError as exc:
        _request_interrupted_run_stop(graph, invocation_config, bps)
        raise CheckpointError(
            f"invalid checkpoint for Evaluation Run {evaluation_id}",
            code=ErrorCode.CHECKPOINT_INVALID,
        ) from exc
    except (KeyError, TypeError) as exc:
        _request_interrupted_run_stop(graph, invocation_config, bps)
        raise CheckpointError(
            f"invalid checkpoint for Evaluation Run {evaluation_id}",
            code=ErrorCode.CHECKPOINT_INVALID,
        ) from exc
    except BaseException:
        _request_interrupted_run_stop(graph, invocation_config, bps)
        raise
    if not isinstance(result, dict):
        raise CheckpointError(
            "Evaluation graph returned an invalid result",
            code=ErrorCode.CHECKPOINT_INVALID,
        )
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
