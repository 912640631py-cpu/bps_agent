"""DUT observations, backend snapshots, and Evidence models."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bps_agent.models.common import DutCollectionMethod, ObservationPhase, StrictModel


class ExternalDutPayloadModel(BaseModel):
    """Typed fields consumed by the Agent; unknown device fields are ignored."""

    model_config = ConfigDict(extra="ignore")


def _document_has_points(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    data = value.get("data")
    if isinstance(data, list):
        return bool(data)
    return isinstance(data, dict) and isinstance(data.get("data"), list) and bool(data["data"])


def _document_parts(value: Any) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not isinstance(value, dict):
        return {}, ()
    metadata = deepcopy(value)
    data = metadata.get("data")
    raw_points: Any = ()
    if isinstance(data, list):
        raw_points = metadata.pop("data")
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        raw_points = data.pop("data")
    points = tuple(item for item in raw_points if isinstance(item, dict))
    return metadata, points


class ResourceObservation(StrictModel):
    phase: ObservationPhase
    started_at: str
    finished_at: str
    window_started_at: str | None = None
    window_finished_at: str | None = None
    dut_clock_offset_seconds: float | None = None
    resources: dict[str, Any] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    def is_complete(self, interfaces: tuple[str, ...]) -> bool:
        required = {"cpu", "memory", "new_sessions", "concurrent_sessions", "traffic"}
        if not required.issubset(self.resources):
            return False
        traffic = self.resources.get("traffic")
        return isinstance(traffic, dict) and all(name in traffic for name in interfaces)

    def populated_series(self) -> frozenset[str]:
        populated: set[str] = set()
        for name in ("cpu", "memory", "new_sessions", "concurrent_sessions"):
            if _document_has_points(self.resources.get(name)):
                populated.add(name)
        traffic = self.resources.get("traffic")
        if isinstance(traffic, dict):
            for interface, document in traffic.items():
                if _document_has_points(document):
                    populated.add(f"traffic.{interface}")
        return frozenset(populated)


class ObservationWindow(StrictModel):
    started_at: str | None = None
    finished_at: str | None = None
    errors: dict[str, str] = Field(default_factory=dict)


class ObservedSeries(StrictModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    points: dict[ObservationPhase, tuple[dict[str, Any], ...]] = Field(default_factory=dict)


class DutObservations(StrictModel):
    collection_started_at: str
    collection_finished_at: str
    dut_clock_offset_seconds: float | None = None
    windows: dict[ObservationPhase, ObservationWindow]
    resources: dict[str, ObservedSeries] = Field(default_factory=dict)
    traffic: dict[str, ObservedSeries] = Field(default_factory=dict)

    @classmethod
    def from_resource_observations(
        cls, observations: tuple[ResourceObservation, ...]
    ) -> DutObservations:
        if not observations:
            raise ValueError("at least one DUT resource observation is required")
        phases = [item.phase for item in observations]
        duplicate_phases = sorted(
            {phase.value for phase in phases if phases.count(phase) > 1}
        )
        if duplicate_phases:
            raise ValueError(
                "duplicate DUT resource observation phases: " + ", ".join(duplicate_phases)
            )

        def build_series(documents: dict[ObservationPhase, Any]) -> ObservedSeries:
            metadata: dict[str, Any] = {}
            points: dict[ObservationPhase, tuple[dict[str, Any], ...]] = {}
            for phase in ObservationPhase:
                document_metadata, document_points = _document_parts(documents.get(phase))
                if not metadata and document_metadata:
                    metadata = document_metadata
                points[phase] = document_points
            return ObservedSeries(metadata=metadata, points=points)

        by_phase = {item.phase: item for item in observations}
        windows = {
            phase: ObservationWindow(
                started_at=observation.window_started_at,
                finished_at=observation.window_finished_at,
                errors=observation.errors,
            )
            for phase, observation in by_phase.items()
        }
        resources: dict[str, ObservedSeries] = {}
        for name in ("cpu", "memory", "new_sessions", "concurrent_sessions"):
            documents = {
                phase: observation.resources.get(name) for phase, observation in by_phase.items()
            }
            if any(isinstance(value, dict) for value in documents.values()):
                resources[name] = build_series(documents)

        interface_names: set[str] = set()
        for observation in observations:
            traffic = observation.resources.get("traffic")
            if isinstance(traffic, dict):
                interface_names.update(str(name) for name in traffic)
        traffic_series: dict[str, ObservedSeries] = {}
        for interface in sorted(interface_names):
            documents = {}
            for phase, observation in by_phase.items():
                traffic = observation.resources.get("traffic")
                documents[phase] = traffic.get(interface) if isinstance(traffic, dict) else None
            traffic_series[interface] = build_series(documents)

        def instant(value: str) -> datetime:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                raise ValueError("DUT observation timestamps must include a timezone")
            return parsed

        first = observations[0]
        return cls(
            collection_started_at=min(
                observations, key=lambda item: instant(item.started_at)
            ).started_at,
            collection_finished_at=max(
                observations, key=lambda item: instant(item.finished_at)
            ).finished_at,
            dut_clock_offset_seconds=first.dut_clock_offset_seconds,
            windows=windows,
            resources=resources,
            traffic=traffic_series,
        )

    def populated_series(self, phase: ObservationPhase) -> frozenset[str]:
        populated = {name for name, series in self.resources.items() if series.points.get(phase)}
        populated.update(
            f"traffic.{interface}"
            for interface, series in self.traffic.items()
            if series.points.get(phase)
        )
        return frozenset(populated)


class SupplementalSnapshot(StrictModel):
    captured_at: str
    values: dict[str, Any] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return {"interfaces", "hardware", "system"}.issubset(self.values)


class CpuReading(ExternalDutPayloadModel):
    timestamp: str | int | None = None
    time: str | None = None
    mgt: int | float | None = None
    data: int | float | None = None


class CpuThresholds(ExternalDutPayloadModel):
    mgt: int | float | None = None
    data: int | float | None = None


class CpuMetrics(ExternalDutPayloadModel):
    latest: CpuReading | None = None
    threshold: CpuThresholds | None = None


class MemoryReading(ExternalDutPayloadModel):
    time: str | None = None
    percent: int | float | None = None


class MemoryMetrics(ExternalDutPayloadModel):
    latest: MemoryReading | None = None
    threshold: int | float | None = None


class SessionMetrics(ExternalDutPayloadModel):
    time: str | None = None
    count: int | float | None = None


class InterfaceTraffic(ExternalDutPayloadModel):
    time: str | None = None
    ibps: int | float | None = None
    obps: int | float | None = None


class BackendDutSnapshot(ExternalDutPayloadModel):
    collected_at: str
    cpu: CpuMetrics
    memory: MemoryMetrics
    new_sessions: SessionMetrics | None
    concurrent_sessions: SessionMetrics | None
    traffic: dict[str, InterfaceTraffic | None]

    def require_interfaces(self, interfaces: tuple[str, ...]) -> BackendDutSnapshot:
        missing = tuple(interface for interface in interfaces if interface not in self.traffic)
        if missing:
            raise ValueError("DUT backend snapshot omitted interfaces: " + ", ".join(missing))
        return self


class BackendDutSample(StrictModel):
    started_at: str
    finished_at: str
    resources: BackendDutSnapshot


class BackendDutSampleError(StrictModel):
    scheduled_at: str
    started_at: str
    finished_at: str
    error: str


class BackendDutTarget(StrictModel):
    host: str
    port: int
    transport: Literal["ssh"] = "ssh"
    backend: Literal["PHP Dashboard/SystemChart.php"] = "PHP Dashboard/SystemChart.php"


class BackendDutCaptureArtifact(StrictModel):
    target: BackendDutTarget
    interfaces: tuple[str, ...]
    traffic_started_at: str | None = None
    traffic_finished_at: str | None = None
    interval_seconds: float
    missed_sample_count: int = Field(default=0, ge=0)
    worker_failure: str | None = None
    samples: tuple[BackendDutSample, ...] = ()
    errors: tuple[BackendDutSampleError, ...] = ()


class BackendDutEvidence(StrictModel):
    collection_method: Literal[DutCollectionMethod.BACKEND_SSH] = DutCollectionMethod.BACKEND_SSH
    target: BackendDutTarget
    interfaces: tuple[str, ...]
    traffic_started_at: str
    traffic_finished_at: str
    interval_seconds: float
    successful_sample_count: int = Field(ge=0)
    failed_sample_count: int = Field(ge=0)
    missed_sample_count: int = Field(default=0, ge=0)
    errors: tuple[BackendDutSampleError, ...] = ()
    metrics_csv: str


class FrontendDutEvidence(StrictModel):
    collection_method: Literal[DutCollectionMethod.FRONTEND_API] = DutCollectionMethod.FRONTEND_API
    endpoint: str
    interfaces: tuple[str, ...]
    traffic_started_at: str
    traffic_finished_at: str
    observations: DutObservations
    before: SupplementalSnapshot
    after: SupplementalSnapshot
    warnings: tuple[str, ...] = ()

DutEvidence = Annotated[
    BackendDutEvidence | FrontendDutEvidence,
    Field(discriminator="collection_method"),
]


class DutCaptureResult(StrictModel):
    evidence: DutEvidence
    raw_artifact_path: str | None = None
    csv_artifact_path: str | None = None
    warnings: tuple[str, ...] = ()
