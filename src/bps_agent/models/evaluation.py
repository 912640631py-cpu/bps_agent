"""Attempt, Evidence Bundle, Verdict, and Evaluation Outcome records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bps_agent.models.bps import PortReservationState
from bps_agent.models.common import (
    ATTEMPT_BANDWIDTH_FACTORS,
    DutCollectionMethod,
    EvaluationMode,
    StrictModel,
    VerdictValue,
)
from bps_agent.models.config import AssessmentConfig
from bps_agent.models.dut import DutEvidence
from bps_agent.models.performance import PerformanceTimeseriesAnalysis


class VerdictDocument(BaseModel):
    model_config = ConfigDict(extra="allow")
    verdict: VerdictValue


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
    port_reservation_state: PortReservationState = PortReservationState.NONE
    terminal_confirmed: bool = False
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_number(self) -> AttemptRecord:
        if self.number < 1 or self.number > len(ATTEMPT_BANDWIDTH_FACTORS):
            raise ValueError(
                f"Attempt number must be between 1 and {len(ATTEMPT_BANDWIDTH_FACTORS)}"
            )
        return self
