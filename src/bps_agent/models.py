"""Domain and configuration models for BPS test evaluation."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled configuration and evidence fields."""

    model_config = ConfigDict(extra="forbid")


def _validated_https_url(value: str, label: str) -> str:
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be a credential-free HTTPS URL")
    return value


class EvaluationOutcome(StrEnum):
    PASSED = "PASSED"
    NOT_PASSED = "NOT_PASSED"
    INCONCLUSIVE = "INCONCLUSIVE"


class VerdictValue(StrEnum):
    PASS = "pass"
    RETRY = "retry"


class ObservationPhase(StrEnum):
    BASELINE = "baseline"
    DURING = "during"
    RECOVERY = "recovery"


class BpsConfig(StrictModel):
    endpoint: str
    template: str
    slot: int = Field(ge=0)
    ports: tuple[int, ...]
    group: int = Field(ge=0)
    allow_malware: bool = False
    verify_tls: bool = False
    poll_interval_seconds: float = Field(default=5.0, gt=0)
    run_timeout_seconds: float = Field(default=7200.0, gt=0)
    registration_grace_seconds: float = Field(default=30.0, ge=0)
    report_poll_interval_seconds: float = Field(default=10.0, gt=0)
    report_attempts: int = Field(default=30, ge=1)
    report_type: Literal["CSV"] = "CSV"
    report_data_type: str = "ALL"
    report_section_ids: tuple[str, ...] = ()
    max_report_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)

    @field_validator("endpoint")
    @classmethod
    def validate_https_endpoint(cls, value: str) -> str:
        return _validated_https_url(value, "BPS endpoint")

    @field_validator("ports")
    @classmethod
    def validate_ports(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("at least one BPS port is required")
        if any(port < 0 for port in value):
            raise ValueError("BPS ports must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("BPS ports must be unique")
        return value


class DutConfig(StrictModel):
    endpoint: str
    interfaces: tuple[str, ...]
    verify_tls: bool = False
    cooldown_seconds: float = Field(default=10.0, ge=0)
    keepalive_interval_seconds: float = Field(default=60.0, gt=0)
    baseline_seconds: float = Field(default=600.0, gt=0)
    read_attempts: int = Field(default=3, ge=1)
    read_retry_backoff_seconds: float = Field(default=0.5, ge=0)
    period: str | None = None

    @field_validator("endpoint")
    @classmethod
    def validate_https_endpoint(cls, value: str) -> str:
        return _validated_https_url(value, "DUT endpoint")

    @field_validator("interfaces")
    @classmethod
    def validate_interfaces(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("at least one non-empty DUT interface is required")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("DUT interfaces must be unique")
        return cleaned


class AssessmentConfig(StrictModel):
    goal: str = "验证 DUT 在指定性能流量下的资源与转发表现"
    expectations: tuple[str, ...] = (
        "结合 BPS 自带 Test Criteria 判断测试目标是否达成",
        "比较 DUT 打流前基线、打流中状态和冷却后的恢复情况",
        "只依据给定证据判断，不臆造缺失指标",
    )


class ProviderConfig(StrictModel):
    base_url: str
    model: str
    token_env: str
    timeout_seconds: float = Field(default=300.0, gt=0)
    attempts: int = Field(default=3, ge=1)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _validated_https_url(value, "LLM base URL")


class LlmConfig(StrictModel):
    provider: Literal["company", "official"] = "company"
    company: ProviderConfig = ProviderConfig(
        base_url="https://aigw.inone.nsfocus.com/deepseek/v1",
        model="deepseek-v4-flash-0731",
        token_env="COMPANY_DEEPSEEK_API_KEY",
    )
    official: ProviderConfig = ProviderConfig(
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        token_env="DEEPSEEK_API_KEY",
    )

    @property
    def selected(self) -> ProviderConfig:
        return self.company if self.provider == "company" else self.official


class StorageConfig(StrictModel):
    artifact_dir: Path = Path("artifacts")
    checkpoint_db: Path = Path(".state/checkpoints.sqlite3")
    lock_dir: Path = Path(".state/locks")


class EvaluationConfig(StrictModel):
    max_attempts: int = Field(default=3, ge=1, le=3)

    @field_validator("max_attempts")
    @classmethod
    def require_three_attempt_capacity(cls, value: int) -> int:
        if value != 3:
            raise ValueError("the demo requires max_attempts to be exactly 3")
        return value


class AppConfig(StrictModel):
    bps: BpsConfig
    dut: DutConfig
    assessment: AssessmentConfig = AssessmentConfig()
    llm: LlmConfig = LlmConfig()
    storage: StorageConfig = StorageConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


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

        first = observations[0]
        return cls(
            collection_started_at=first.started_at,
            collection_finished_at=first.finished_at,
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


class RunCompletion(StrictModel):
    terminal: bool
    details: dict[str, Any] = Field(default_factory=dict)


class VerdictDocument(BaseModel):
    """Only the routing field is stable; all explanatory fields remain flexible."""

    model_config = ConfigDict(extra="allow")
    verdict: VerdictValue
    schema_version: str = "dev-1"


class EvidenceBundle(StrictModel):
    evaluation_id: str
    attempt_number: int
    bps_run_id: str
    bps_template: str
    bps_template_metadata: dict[str, Any]
    bps_run_details: dict[str, Any]
    bps_report: str
    bps_report_toc: Any | None = Field(default=None, exclude=True)
    assessment: AssessmentConfig
    dut_endpoint: str
    dut_interfaces: tuple[str, ...]
    traffic_started_at: str
    traffic_finished_at: str
    dut_observations: DutObservations
    dut_before: SupplementalSnapshot
    dut_after: SupplementalSnapshot
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator("dut_observations", mode="before")
    @classmethod
    def upgrade_legacy_dut_observations(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            observations = tuple(ResourceObservation.model_validate(item) for item in value)
            return DutObservations.from_resource_observations(observations)
        return value


class AttemptRecord(StrictModel):
    number: int
    started_at: str
    bps_run_id: str | None = None
    traffic_started_at: str | None = None
    traffic_finished_at: str | None = None
    bps_template_metadata: dict[str, Any] = Field(default_factory=dict)
    bps_run_details: dict[str, Any] = Field(default_factory=dict)
    dut_observations: tuple[ResourceObservation, ...] = ()
    dut_before: SupplementalSnapshot | None = None
    dut_after: SupplementalSnapshot | None = None
    report_path: str | None = None
    report_toc_path: str | None = None
    evidence_path: str | None = None
    verdict_path: str | None = None
    verdict: VerdictDocument | None = None
    evidence_complete: bool = False
    ports_reserved: bool = False
    terminal_confirmed: bool = False
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_number(self) -> AttemptRecord:
        if self.number < 1 or self.number > 3:
            raise ValueError("Attempt number must be between 1 and 3")
        return self


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
