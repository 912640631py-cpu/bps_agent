from __future__ import annotations

from pathlib import Path

import pytest

from bps_agent.models import PerformanceAnalysisThresholds, PerformanceAssessment
from bps_agent.performance_timeseries import _align, analyze_performance_timeseries


def _report(
    tmp_path: Path,
    *,
    throughput_overrides: dict[int, float] | None = None,
    flow_rate_overrides: dict[int, float] | None = None,
    stable_sample_count: int = 9,
) -> Path:
    throughput_overrides = throughput_overrides or {}
    flow_rate_overrides = flow_rate_overrides or {}
    flows = (0, 250, 500, 750, *([1000] * stable_sample_count), 700, 300, 0)
    lines = [
        "Test Results for fixture",
        "2. Aggregate Stats",
        "2.1. Detail",
        "2.1.1. Ethernet Data Rates",
        "Timestamp,Transmit rate,Receive rate",
        "Seconds,Megabits/s,",
    ]
    for second in range(len(flows)):
        throughput = throughput_overrides.get(second, 100.0)
        lines.append(f"{second + 0.45:.2f},{throughput:.2f},{throughput:.2f}")
    lines.extend(
        [
            "2.1.1.1. Ethernet Data Rates: 1",
            "Timestamp,Transmit rate,Receive rate",
            "Seconds,Megabits/s,",
            "0.45,999,999",
            "2.1.2. Concurrent Flows",
            "Timestamp,TCP,UDP,SCTP,Total,Super Flow",
            "Seconds,Flows,,,,",
        ]
    )
    for second, total in enumerate(flows):
        lines.append(f"{second + 0.10:.2f},0,0,0,{total},0")
    lines.extend(
        [
            "2.1.3. Flow Rates",
            "Timestamp,TCP rate,UDP rate,SCTP rate,Total rate,Super Flow rate",
            "Seconds,Flows/s,,,,",
        ]
    )
    for second in range(len(flows)):
        rate = flow_rate_overrides.get(second, 10.0)
        lines.append(f"{second + 0.30:.2f},0,0,0,{rate:.2f},0")
    path = tmp_path / "bps-performance-timeseries.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_aligns_nearest_points_without_equal_timestamps(tmp_path: Path) -> None:
    analysis = analyze_performance_timeseries(_report(tmp_path))

    assert analysis.assessment == PerformanceAssessment.NORMAL
    assert analysis.aligned_sample_count == analysis.expected_resampled_points == 15
    assert analysis.alignment_coverage_ratio == pytest.approx(1.0)
    assert analysis.stable_baseline.tx_median_mbps == pytest.approx(100.0)
    assert analysis.stable_baseline.concurrent_flows_median == pytest.approx(1000.0)
    assert analysis.stable_baseline.sample_count == 5
    assert [phase.phase.value for phase in analysis.phases] == [
        "ramp_up",
        "stable",
        "ramp_down",
    ]
    assert "aligned_samples" not in analysis.model_dump(mode="json")


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({9: 85, 10: 85}, PerformanceAssessment.SHORT_FLUCTUATION),
        ({9: 85, 10: 85, 11: 85}, PerformanceAssessment.PERFORMANCE_ANOMALY),
        ({9: 70, 10: 70}, PerformanceAssessment.SEVERE_PERFORMANCE_ANOMALY),
        ({13: 70, 14: 30, 15: 0}, PerformanceAssessment.NORMAL_LOAD_CHANGE),
    ],
)
def test_classifies_deterministic_throughput_patterns(
    tmp_path: Path,
    overrides: dict[int, float],
    expected: PerformanceAssessment,
) -> None:
    analysis = analyze_performance_timeseries(_report(tmp_path, throughput_overrides=overrides))

    assert analysis.assessment == expected
    assert analysis.events


def test_flow_rate_only_corroborates_a_throughput_event(tmp_path: Path) -> None:
    analysis = analyze_performance_timeseries(
        _report(
            tmp_path,
            throughput_overrides={9: 85, 10: 85, 11: 85},
            flow_rate_overrides={9: 5, 10: 5, 11: 5},
        )
    )

    assert analysis.assessment == PerformanceAssessment.PERFORMANCE_ANOMALY
    assert analysis.events[0].flow_rate_auxiliary_change is True


def test_flow_rate_change_alone_is_not_a_performance_anomaly(tmp_path: Path) -> None:
    analysis = analyze_performance_timeseries(
        _report(tmp_path, flow_rate_overrides={9: 2, 10: 2, 11: 2})
    )

    assert analysis.assessment == PerformanceAssessment.NORMAL
    assert analysis.events == ()


def test_early_stable_baseline_is_not_polluted_by_a_long_degradation(
    tmp_path: Path,
) -> None:
    analysis = analyze_performance_timeseries(
        _report(
            tmp_path,
            stable_sample_count=18,
            throughput_overrides={second: 60 for second in range(9, 22)},
        )
    )

    assert analysis.stable_baseline.sample_count == 5
    assert analysis.stable_baseline.tx_median_mbps == pytest.approx(100.0)
    assert analysis.stable_baseline.rx_median_mbps == pytest.approx(100.0)
    assert analysis.assessment == PerformanceAssessment.SEVERE_PERFORMANCE_ANOMALY


def test_alignment_does_not_rescan_every_timestamp_for_each_second() -> None:
    timestamp_accesses = 0

    class CountingPoint:
        def __init__(self, timestamp: float, values: tuple[float, ...]) -> None:
            self._timestamp = timestamp
            self.values = values

        @property
        def timestamp(self) -> float:
            nonlocal timestamp_accesses
            timestamp_accesses += 1
            return self._timestamp

    count = 200
    tables = {
        "Ethernet Data Rates": tuple(
            CountingPoint(float(second), (100.0, 100.0)) for second in range(count)
        ),
        "Concurrent Flows": tuple(
            CountingPoint(float(second), (1000.0,)) for second in range(count)
        ),
        "Flow Rates": tuple(CountingPoint(float(second), (10.0,)) for second in range(count)),
    }

    aligned, expected, coverage = _align(tables, PerformanceAnalysisThresholds())  # type: ignore[arg-type]

    assert len(aligned) == expected == count
    assert coverage == pytest.approx(1.0)
    assert timestamp_accesses < count * 25
