"""Domain and configuration models for BPS test evaluation."""

from __future__ import annotations

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
    sample_interval_seconds: float = Field(default=10.0, gt=0)
    cooldown_seconds: float = Field(default=10.0, ge=0)
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
    resources: dict[str, Any] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    def is_complete(self, interfaces: tuple[str, ...]) -> bool:
        required = {"cpu", "memory", "new_sessions", "concurrent_sessions", "traffic"}
        if not required.issubset(self.resources):
            return False
        traffic = self.resources.get("traffic")
        return isinstance(traffic, dict) and all(name in traffic for name in interfaces)


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
    bps_report_toc: Any
    assessment: AssessmentConfig
    dut_endpoint: str
    dut_interfaces: tuple[str, ...]
    dut_observations: tuple[ResourceObservation, ...]
    dut_before: SupplementalSnapshot
    dut_after: SupplementalSnapshot
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class AttemptRecord(StrictModel):
    number: int
    started_at: str
    bps_run_id: str | None = None
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
