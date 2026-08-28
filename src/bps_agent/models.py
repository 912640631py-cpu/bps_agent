"""Domain and configuration models for BPS test evaluation."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal
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
    DEGRADED_PASS = "DEGRADED_PASS"
    NOT_PASSED = "NOT_PASSED"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvaluationMode(StrEnum):
    BPS_AND_DUT = "bps_and_dut"
    BPS_ONLY = "bps_only"


class DutCollectionMethod(StrEnum):
    BACKEND_SSH = "backend_ssh"
    FRONTEND_API = "frontend_api"


class VerdictValue(StrEnum):
    PASS = "pass"
    RETRY = "retry"


class ObservationPhase(StrEnum):
    BASELINE = "baseline"
    DURING = "during"
    RECOVERY = "recovery"


class PerformancePhase(StrEnum):
    RAMP_UP = "ramp_up"
    STABLE = "stable"
    RAMP_DOWN = "ramp_down"


class PerformanceAssessment(StrEnum):
    NORMAL = "normal"
    SHORT_FLUCTUATION = "short_fluctuation"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    SEVERE_PERFORMANCE_ANOMALY = "severe_performance_anomaly"
    NORMAL_LOAD_CHANGE = "normal_load_change"


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]


class BpsConfig(StrictModel):
    endpoint: str
    template: str
    slot: int = Field(ge=0)
    ports: tuple[int, ...]
    group: int = Field(ge=0)
    total_bandwidth_mbps: float = Field(default=400.0, gt=0)
    allow_malware: bool = False
    verify_tls: bool = False
    poll_interval_seconds: float = Field(default=5.0, gt=0)
    run_timeout_seconds: float = Field(default=7200.0, gt=0)
    registration_grace_seconds: float = Field(default=30.0, ge=0)
    port_release_attempts: int = Field(default=6, ge=1)
    port_release_retry_backoff_seconds: float = Field(default=5.0, ge=0)
    report_poll_interval_seconds: float = Field(default=10.0, gt=0)
    report_attempts: int = Field(default=30, ge=1)
    report_type: Literal["CSV"] = "CSV"
    report_data_type: str = "ALL"
    max_report_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    pdf_report_timeout_seconds: float = Field(default=1200.0, gt=0)
    max_pdf_report_bytes: int = Field(default=512 * 1024 * 1024, ge=1024)

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


class DutBackendConfig(StrictModel):
    host: str
    port: int = Field(default=50023, ge=1, le=65535)
    interval_seconds: float = Field(default=10.0, gt=0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    command_timeout_seconds: float = Field(default=30.0, gt=0)
    read_attempts: int = Field(default=3, ge=1)
    read_retry_backoff_seconds: float = Field(default=0.5, ge=0)

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("DUT backend host must not be empty")
        return value


class DutFrontendConfig(StrictModel):
    endpoint: str
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


class DutConfig(StrictModel):
    collection_method: DutCollectionMethod = DutCollectionMethod.BACKEND_SSH
    interfaces: tuple[str, ...]
    backend: DutBackendConfig | None = None
    frontend: DutFrontendConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_frontend_config(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "endpoint" not in value:
            return value
        document = dict(value)
        legacy_names = {
            "endpoint",
            "verify_tls",
            "cooldown_seconds",
            "keepalive_interval_seconds",
            "baseline_seconds",
            "read_attempts",
            "read_retry_backoff_seconds",
            "period",
        }
        frontend = {name: document.pop(name) for name in legacy_names if name in document}
        document.setdefault("collection_method", DutCollectionMethod.FRONTEND_API.value)
        document.setdefault("frontend", frontend)
        return document

    @field_validator("interfaces")
    @classmethod
    def validate_interfaces(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("at least one non-empty DUT interface is required")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("DUT interfaces must be unique")
        return cleaned

    @model_validator(mode="after")
    def require_selected_configuration(self) -> DutConfig:
        if self.collection_method == DutCollectionMethod.BACKEND_SSH and self.backend is None:
            raise ValueError("backend_ssh DUT collection requires dut.backend")
        if self.collection_method == DutCollectionMethod.FRONTEND_API and self.frontend is None:
            raise ValueError("frontend_api DUT collection requires dut.frontend")
        return self


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
    reasoning_effort: ReasoningEffort = "max"
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


class EvaluationConfig(StrictModel):
    mode: EvaluationMode = EvaluationMode.BPS_AND_DUT
    max_attempts: int = Field(default=6, ge=1, le=6)

    @field_validator("max_attempts")
    @classmethod
    def require_full_attempt_capacity(cls, value: int) -> int:
        if value != 6:
            raise ValueError("the bandwidth fallback policy requires max_attempts to be exactly 6")
        return value


class AppConfig(StrictModel):
    bps: BpsConfig
    dut: DutConfig
    assessment: AssessmentConfig = AssessmentConfig()
    bps_only_assessment: AssessmentConfig = AssessmentConfig(
        goal="仅依据 BPS 证据验证指定流量目标与性能稳定性",
        expectations=(
            "结合 BPS 自带 Test Criteria 判断测试目标是否达成",
            "结合吞吐、Concurrent Flows 和 Flow Rate 时序分析判断性能是否稳定",
            "BPS-only 模式不提供 DUT 观测，不得臆造或要求 DUT 指标",
        ),
    )
    llm: LlmConfig = LlmConfig()
    storage: StorageConfig = StorageConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @property
    def selected_assessment(self) -> AssessmentConfig:
        if self.evaluation.mode == EvaluationMode.BPS_ONLY:
            return self.bps_only_assessment
        return self.assessment


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
        collection_started_at = min(
            observations, key=lambda item: instant(item.started_at)
        ).started_at
        collection_finished_at = max(
            observations, key=lambda item: instant(item.finished_at)
        ).finished_at
        return cls(
            collection_started_at=collection_started_at,
            collection_finished_at=collection_finished_at,
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


class PerformanceAnalysisThresholds(StrictModel):
    resample_interval_seconds: float = 1.0
    nearest_alignment_tolerance_seconds: float = 0.6
    minimum_alignment_coverage_ratio: float = 0.8
    stable_flow_floor_ratio_to_peak: float = 0.9
    stable_flow_tolerance_ratio: float = 0.1
    minimum_stable_samples: int = 5
    stable_baseline_sample_count: int = Field(default=5, ge=1)
    throughput_drop_ratio: float = 0.1
    throughput_drop_minimum_samples: int = 3
    severe_throughput_drop_ratio: float = 0.2
    severe_throughput_drop_minimum_samples: int = 2
    expanding_drop_minimum_increase_ratio: float = 0.05
    flow_rate_auxiliary_change_ratio: float = 0.2
    load_change_flow_drop_ratio: float = 0.1
    load_change_minimum_samples: int = 2


class PerformancePhaseWindow(StrictModel):
    phase: PerformancePhase
    started_at_second: int
    finished_at_second: int
    sample_count: int
    minimum_concurrent_flows: float
    median_concurrent_flows: float
    maximum_concurrent_flows: float


class PerformanceStableBaseline(StrictModel):
    started_at_second: int
    finished_at_second: int
    sample_count: int
    tx_median_mbps: float
    rx_median_mbps: float
    concurrent_flows_median: float
    flow_rate_median_per_second: float


class PerformanceDiagnosticEvent(StrictModel):
    assessment: PerformanceAssessment
    started_at_second: int
    finished_at_second: int
    sample_count: int
    maximum_tx_drop_ratio: float
    maximum_rx_drop_ratio: float
    concurrent_flow_change_ratio: float
    flow_rate_change_ratio: float | None
    flow_rate_auxiliary_change: bool
    recovered: bool
    rationale: str


class PerformanceTimeseriesAnalysis(StrictModel):
    schema_version: str = "1"
    assessment: PerformanceAssessment
    summary: str
    source_tables: tuple[str, ...]
    expected_resampled_points: int
    aligned_sample_count: int
    alignment_coverage_ratio: float
    phases: tuple[PerformancePhaseWindow, ...]
    stable_baseline: PerformanceStableBaseline
    events: tuple[PerformanceDiagnosticEvent, ...]
    thresholds: PerformanceAnalysisThresholds


class VerdictDocument(BaseModel):
    """Only the routing field is stable; all explanatory fields remain flexible."""

    model_config = ConfigDict(extra="allow")
    verdict: VerdictValue
    schema_version: str = "dev-1"


class BackendDutTarget(StrictModel):
    host: str
    port: int
    transport: Literal["ssh"] = "ssh"
    backend: Literal["PHP Dashboard/SystemChart.php"] = "PHP Dashboard/SystemChart.php"


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
    errors: tuple[dict[str, Any], ...] = ()
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

    @field_validator("observations", mode="before")
    @classmethod
    def upgrade_legacy_observations(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            observations = tuple(ResourceObservation.model_validate(item) for item in value)
            return DutObservations.from_resource_observations(observations)
        return value


DutEvidence = Annotated[
    BackendDutEvidence | FrontendDutEvidence,
    Field(discriminator="collection_method"),
]


class DutCaptureResult(StrictModel):
    evidence: DutEvidence
    raw_artifact_path: str | None = None
    csv_artifact_path: str | None = None
    warnings: tuple[str, ...] = ()


class EvidenceBundle(StrictModel):
    evaluation_mode: EvaluationMode = EvaluationMode.BPS_AND_DUT
    evaluation_id: str
    attempt_number: int
    bps_run_id: str
    bps_template: str
    bps_template_total_bandwidth_mbps: float | None = None
    bps_total_bandwidth_mbps: float | None = None
    bps_total_bandwidth_percent: float | None = None
    bps_template_metadata: dict[str, Any]
    bps_run_details: dict[str, Any]
    bps_report: str
    bps_performance_analysis: PerformanceTimeseriesAnalysis | None = None
    bps_report_toc: Any | None = Field(default=None, exclude=True)
    assessment: AssessmentConfig
    traffic_started_at: str
    traffic_finished_at: str
    dut_evidence: DutEvidence | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_dut_evidence(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        document = dict(value)
        legacy_names = (
            "dut_endpoint",
            "dut_interfaces",
            "dut_observations",
            "dut_before",
            "dut_after",
        )
        has_legacy_evidence = all(document.get(name) is not None for name in legacy_names)
        if "dut_evidence" not in document and has_legacy_evidence:
            document["dut_evidence"] = {
                "collection_method": DutCollectionMethod.FRONTEND_API.value,
                "endpoint": document["dut_endpoint"],
                "interfaces": document["dut_interfaces"],
                "traffic_started_at": document.get("traffic_started_at"),
                "traffic_finished_at": document.get("traffic_finished_at"),
                "observations": document["dut_observations"],
                "before": document["dut_before"],
                "after": document["dut_after"],
            }
        for name in legacy_names:
            document.pop(name, None)
        return document

    @model_validator(mode="after")
    def validate_evidence_for_mode(self) -> EvidenceBundle:
        if self.evaluation_mode == EvaluationMode.BPS_AND_DUT and self.dut_evidence is None:
            raise ValueError("bps_and_dut Evidence requires DUT evidence")
        if self.evaluation_mode == EvaluationMode.BPS_ONLY and self.dut_evidence is not None:
            raise ValueError("bps_only Evidence must omit DUT evidence")
        return self

    def as_document(self) -> dict[str, Any]:
        document = self.model_dump(mode="json")
        if self.evaluation_mode == EvaluationMode.BPS_ONLY:
            document.pop("dut_evidence", None)
        return document


class AttemptRecord(StrictModel):
    number: int
    started_at: str
    bps_run_id: str | None = None
    bps_template_total_bandwidth_mbps: float | None = None
    bps_total_bandwidth_mbps: float | None = None
    bps_total_bandwidth_percent: float | None = None
    traffic_started_at: str | None = None
    traffic_finished_at: str | None = None
    bps_template_metadata: dict[str, Any] = Field(default_factory=dict)
    bps_run_details: dict[str, Any] = Field(default_factory=dict)
    # Retained only so checkpoints created by the previous frontend collector remain readable.
    dut_observations: tuple[ResourceObservation, ...] = ()
    dut_before: SupplementalSnapshot | None = None
    dut_after: SupplementalSnapshot | None = None
    dut_collection_method: DutCollectionMethod | None = None
    dut_raw_artifact_path: str | None = None
    dut_csv_artifact_path: str | None = None
    dut_successful_sample_count: int | None = None
    dut_failed_sample_count: int | None = None
    dut_missed_sample_count: int | None = None
    report_path: str | None = None
    pdf_report_path: str | None = None
    performance_timeseries_path: str | None = None
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
        if self.number < 1 or self.number > 6:
            raise ValueError("Attempt number must be between 1 and 6")
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
