"""Shared primitives used across Evaluation Run domains."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def validated_https_url(value: str, label: str) -> str:
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
ATTEMPT_BANDWIDTH_FACTORS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
