"""Performance time-series analysis models."""

from __future__ import annotations

from pydantic import Field

from bps_agent.models.common import (
    PerformanceAssessment,
    PerformancePhase,
    StrictModel,
)


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
