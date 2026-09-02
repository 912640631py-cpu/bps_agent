"""Durable coordination for externally launched BPS runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from bps_agent.artifacts import ArtifactStore
from bps_agent.errors import BpsError, ErrorCode
from bps_agent.models.common import utc_now
from bps_agent.ports import BpsPort


class LaunchReconciliationError(BpsError):
    """The external launch result cannot be identified without risking a duplicate run."""

    default_code = ErrorCode.BPS_LAUNCH_ERROR.value


class LaunchIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    attempt_number: int
    template: str
    group: int
    status: Literal["prepared", "requesting", "started", "terminal", "released"]
    run_id: str | None = None
    prepared_at: str
    request_started_at: str | None = None
    launched_at: str | None = None
    updated_at: str


class LaunchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    launched_at: str


class RunLaunchCoordinator:
    """Hide durable launch/reconciliation mechanics behind a small interface."""

    def __init__(self, bps: BpsPort, artifacts: ArtifactStore) -> None:
        self._bps = bps
        self._artifacts = artifacts

    def _path(self, evaluation_id: str, attempt_number: int) -> Path:
        return self._artifacts.attempt_dir(evaluation_id, attempt_number) / "bps-launch.json"

    def _read(self, evaluation_id: str, attempt_number: int) -> LaunchIntent | None:
        path = self._path(evaluation_id, attempt_number)
        if not path.is_file():
            return None
        try:
            intent = LaunchIntent.model_validate(self._artifacts.read_json(path))
        except (
            ValidationError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            OSError,
        ) as exc:
            raise LaunchReconciliationError("invalid BPS launch artifact") from exc
        if intent.evaluation_id != evaluation_id or intent.attempt_number != attempt_number:
            raise LaunchReconciliationError(
                "BPS launch journal identity does not match the Attempt"
            )
        return intent

    def _write(self, intent: LaunchIntent) -> None:
        self._artifacts.write_attempt_json(
            intent.evaluation_id,
            intent.attempt_number,
            "bps-launch.json",
            intent,
        )

    def _record_started(self, intent: LaunchIntent, run_id: str) -> LaunchResult:
        launched_at = intent.request_started_at or utc_now()
        started = intent.model_copy(
            update={
                "status": "started",
                "run_id": run_id,
                "launched_at": launched_at,
                "updated_at": utc_now(),
            }
        )
        self._write(started)
        return LaunchResult(run_id=run_id, launched_at=launched_at)

    def _reconcile(self, intent: LaunchIntent) -> LaunchResult:
        try:
            candidates = self._bps.find_running_runs(
                template=intent.template,
                group=intent.group,
            )
        except Exception as exc:
            raise LaunchReconciliationError(
                "could not query BPS running tasks for an ambiguous launch"
            ) from exc
        if len(candidates) != 1:
            raise LaunchReconciliationError(
                "ambiguous BPS launch matched "
                f"{len(candidates)} running tasks; refusing to start duplicate traffic"
            )
        return self._record_started(intent, candidates[0])

    def recover(
        self,
        evaluation_id: str,
        attempt_number: int,
        *,
        template: str,
        group: int,
    ) -> LaunchResult | None:
        intent = self._read(evaluation_id, attempt_number)
        if intent is None:
            return None
        if intent.template != template or intent.group != group:
            raise LaunchReconciliationError("BPS launch journal target differs from configuration")
        if intent.run_id is not None and intent.launched_at is not None:
            return LaunchResult(run_id=intent.run_id, launched_at=intent.launched_at)
        if intent.status == "prepared":
            return self._request_start(intent)
        return self._reconcile(intent)

    def recovery_requires_reservation(
        self,
        evaluation_id: str,
        attempt_number: int,
    ) -> bool:
        """Whether a durable launch is still prepared and has not sent its run request."""

        intent = self._read(evaluation_id, attempt_number)
        return intent is not None and intent.status == "prepared"

    def start(
        self,
        evaluation_id: str,
        attempt_number: int,
        *,
        template: str,
        group: int,
    ) -> LaunchResult:
        if self._read(evaluation_id, attempt_number) is not None:
            raise LaunchReconciliationError("BPS launch journal already exists")
        now = utc_now()
        intent = LaunchIntent(
            evaluation_id=evaluation_id,
            attempt_number=attempt_number,
            template=template,
            group=group,
            status="prepared",
            prepared_at=now,
            updated_at=now,
        )
        self._write(intent)
        return self._request_start(intent)

    def _request_start(self, intent: LaunchIntent) -> LaunchResult:
        requesting = intent.model_copy(
            update={
                "status": "requesting",
                "request_started_at": intent.request_started_at or utc_now(),
                "updated_at": utc_now(),
            }
        )
        self._write(requesting)
        try:
            run_id = self._bps.start_run()
        except Exception as exc:
            try:
                return self._reconcile(requesting)
            except LaunchReconciliationError as reconciliation_exc:
                raise LaunchReconciliationError(
                    f"BPS start result is ambiguous after request failure: {exc}"
                ) from reconciliation_exc
        return self._record_started(requesting, run_id)

    def mark_terminal(self, evaluation_id: str, attempt_number: int, run_id: str) -> None:
        self._advance(evaluation_id, attempt_number, run_id, "terminal")

    def mark_released(self, evaluation_id: str, attempt_number: int, run_id: str) -> None:
        self._advance(evaluation_id, attempt_number, run_id, "released")

    def _advance(
        self,
        evaluation_id: str,
        attempt_number: int,
        run_id: str,
        status: Literal["terminal", "released"],
    ) -> None:
        intent = self._read(evaluation_id, attempt_number)
        if intent is None or intent.run_id != run_id:
            raise LaunchReconciliationError("BPS launch journal omitted the active Run")
        self._write(intent.model_copy(update={"status": status, "updated_at": utc_now()}))
