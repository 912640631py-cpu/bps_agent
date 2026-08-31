"""High-level ports used by the Evaluation Run orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from bps_agent.models.bps import PortReservationStatus, RunCompletion
from bps_agent.models.dut import DutCaptureResult
from bps_agent.models.evaluation import EvidenceBundle, VerdictDocument


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class BpsPort(Protocol):
    def find_template(self, name: str) -> dict[str, Any]: ...

    def reserve_ports(self) -> None: ...

    def port_reservation_status(self) -> PortReservationStatus: ...

    def find_active_runs_for_ports(self) -> tuple[str, ...]: ...

    def set_total_bandwidth(self, percentage: float) -> None: ...

    def start_run(self) -> str: ...

    def find_running_runs(self, *, template: str, group: int) -> tuple[str, ...]: ...

    def wait_for_completion(self, run_id: str, on_poll: Callable[[], None]) -> RunCompletion: ...

    def wait_for_report(self, run_id: str) -> Any: ...

    def export_report(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> Path: ...

    def schedule_full_report_pdf(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> None: ...

    def release_ports(self, ports: tuple[int, ...] | None = None) -> None: ...

    def stop_run(self, run_id: str) -> None: ...

    def close(self) -> None: ...


class DutPort(Protocol):
    def prepare_attempt(self, attempt_dir: Path) -> None: ...

    def traffic_started(self, started_at: str) -> None: ...

    def restore_attempt(
        self,
        attempt_dir: Path,
        started_at: str,
        finished_at: str | None,
    ) -> None: ...

    def traffic_finished(self, finished_at: str) -> None: ...

    def finalize_attempt(self) -> DutCaptureResult: ...

    def close(self) -> None: ...


class JudgePort(Protocol):
    provider_name: str
    model_name: str

    def adjudicate(self, evidence: EvidenceBundle) -> tuple[VerdictDocument, dict[str, Any]]: ...

    def close(self) -> None: ...
