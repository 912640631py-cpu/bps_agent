"""Validated configuration and CLI Run Override models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from bps_agent.models.common import (
    DutCollectionMethod,
    EvaluationMode,
    ReasoningEffort,
    StrictModel,
    validated_https_url,
)


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
        return validated_https_url(value, "BPS endpoint")

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
    worker_stop_timeout_seconds: float = Field(default=5.0, gt=0)
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
        return validated_https_url(value, "DUT endpoint")


class DutConfig(StrictModel):
    collection_method: DutCollectionMethod = DutCollectionMethod.BACKEND_SSH
    interfaces: tuple[str, ...]
    backend: DutBackendConfig | None = None
    frontend: DutFrontendConfig | None = None

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
        return validated_https_url(value, "LLM base URL")


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

    def resolved_relative_to(self, base_dir: Path) -> StorageConfig:
        def resolved(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (base_dir / path).resolve()

        return self.model_copy(
            update={
                "artifact_dir": resolved(self.artifact_dir),
                "checkpoint_db": resolved(self.checkpoint_db),
            }
        )


class EvaluationConfig(StrictModel):
    mode: EvaluationMode = EvaluationMode.BPS_AND_DUT


class AppConfig(StrictModel):
    bps: BpsConfig
    dut: DutConfig | None = None
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

    @model_validator(mode="after")
    def require_dut_for_selected_mode(self) -> AppConfig:
        if self.evaluation.mode == EvaluationMode.BPS_AND_DUT and self.dut is None:
            raise ValueError("bps_and_dut evaluation requires dut configuration")
        return self

    @property
    def selected_assessment(self) -> AssessmentConfig:
        if self.evaluation.mode == EvaluationMode.BPS_ONLY:
            return self.bps_only_assessment
        return self.assessment


class RunOverrides(StrictModel):
    template: str | None = None
    ports: tuple[int, ...] | None = None
    total_bandwidth_mbps: float | None = None
    evaluation_mode: EvaluationMode | None = None
    dut_collection_method: DutCollectionMethod | None = None
    dut_host: str | None = None
    dut_port: int | None = None
    dut_interfaces: tuple[str, ...] | None = None
    dut_interval_seconds: float | None = None

    @property
    def has_values(self) -> bool:
        return any(value is not None for value in self.model_dump(mode="python").values())
