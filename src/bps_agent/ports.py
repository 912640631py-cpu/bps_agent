"""High-level ports used by the Evaluation Run orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from bps_agent.models import (
    EvidenceBundle,
    ResourceObservation,
    RunCompletion,
    SupplementalSnapshot,
    VerdictDocument,
)


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class BpsPort(Protocol):
    def find_template(self, name: str) -> dict[str, Any]: ...

    def reserve_ports(self) -> None: ...

    def start_run(self) -> str: ...

    def wait_for_completion(self, run_id: str, on_poll: Callable[[], None]) -> RunCompletion: ...

    def wait_for_report(self, run_id: str) -> Any: ...

    def export_report(self, run_id: str, destination: Path) -> Path: ...

    def release_ports(self) -> None: ...

    def stop_run(self, run_id: str) -> None: ...


class DutPort(Protocol):
    def collect_resources(self, phase: str) -> ResourceObservation: ...

    def collect_supplemental(self) -> SupplementalSnapshot: ...


class JudgePort(Protocol):
    provider_name: str
    model_name: str

    def adjudicate(self, evidence: EvidenceBundle) -> tuple[VerdictDocument, dict[str, Any]]: ...
