"""Deterministic analysis for the separately exported BPS performance time series."""

from __future__ import annotations

import bisect
import csv
import math
import re
import statistics
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from bps_agent.models import (
    PerformanceAnalysisThresholds,
    PerformanceAssessment,
    PerformanceDiagnosticEvent,
    PerformancePhase,
    PerformancePhaseWindow,
    PerformanceStableBaseline,
    PerformanceTimeseriesAnalysis,
)

_SECTION_HEADING = re.compile(r"^\d+(?:\.\d+)*\.\s+(.+?)\s*$")
_TABLE_COLUMNS = {
    "Ethernet Data Rates": ("Transmit rate", "Receive rate"),
    "Concurrent Flows": ("Total",),
    "Flow Rates": ("Total rate",),
}
_SOURCE_TABLES = tuple(_TABLE_COLUMNS)


class PerformanceTimeseriesError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Point:
    timestamp: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class _AlignedSample:
    second: int
    tx_mbps: float
    rx_mbps: float
    concurrent_flows: float
    flow_rate_per_second: float


def _number(value: str) -> float:
    cleaned = value.strip().replace(",", "")
    number = float(cleaned)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def _parse_tables(path: Path) -> dict[str, tuple[_Point, ...]]:
    collected: dict[str, list[_Point]] = {title: [] for title in _SOURCE_TABLES}
    seen_sections: set[str] = set()
    current_title: str | None = None
    header: dict[str, int] | None = None

    try:
        handle = path.open(mode="r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise PerformanceTimeseriesError(f"cannot read performance time-series CSV: {exc}") from exc

    with handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            first = row[0].strip()
            heading = _SECTION_HEADING.fullmatch(first)
            if heading:
                current_title = heading.group(1)
                header = None
                if current_title in _TABLE_COLUMNS:
                    if current_title in seen_sections:
                        raise PerformanceTimeseriesError(
                            f"duplicate top-level performance table: {current_title}"
                        )
                    seen_sections.add(current_title)
                continue
            if current_title not in _TABLE_COLUMNS:
                continue
            if first.casefold() == "timestamp":
                header = {name.strip(): index for index, name in enumerate(row)}
                required = ("Timestamp", *_TABLE_COLUMNS[current_title])
                missing = [name for name in required if name not in header]
                if missing:
                    raise PerformanceTimeseriesError(
                        f"{current_title} table is missing columns: {', '.join(missing)}"
                    )
                continue
            if header is None or first.casefold() == "seconds":
                continue
            try:
                timestamp = _number(row[header["Timestamp"]])
                values = tuple(_number(row[header[name]]) for name in _TABLE_COLUMNS[current_title])
            except (IndexError, ValueError) as exc:
                raise PerformanceTimeseriesError(
                    f"invalid {current_title} data at CSV row {row_number}"
                ) from exc
            collected[current_title].append(_Point(timestamp=timestamp, values=values))

    missing_tables = [title for title, points in collected.items() if not points]
    if missing_tables:
        raise PerformanceTimeseriesError(
            "performance time-series CSV is missing data for: " + ", ".join(missing_tables)
        )

    parsed: dict[str, tuple[_Point, ...]] = {}
    for title, points in collected.items():
        ordered = sorted(points, key=lambda point: point.timestamp)
        if any(current.timestamp <= previous.timestamp for previous, current in pairwise(ordered)):
            raise PerformanceTimeseriesError(f"{title} timestamps must be unique")
        parsed[title] = tuple(ordered)
    return parsed


def _nearest(points: tuple[_Point, ...], timestamp: float) -> tuple[_Point, float]:
    timestamps = [point.timestamp for point in points]
    position = bisect.bisect_left(timestamps, timestamp)
    candidates = points[max(0, position - 1) : min(len(points), position + 1)]
    point = min(candidates, key=lambda item: (abs(item.timestamp - timestamp), item.timestamp))
    return point, abs(point.timestamp - timestamp)


def _align(
    tables: dict[str, tuple[_Point, ...]], thresholds: PerformanceAnalysisThresholds
) -> tuple[tuple[_AlignedSample, ...], int, float]:
    start = math.ceil(max(points[0].timestamp for points in tables.values()))
    finish = math.floor(min(points[-1].timestamp for points in tables.values()))
    if finish < start:
        raise PerformanceTimeseriesError("performance tables have no overlapping time range")

    expected = finish - start + 1
    aligned: list[_AlignedSample] = []
    for second in range(start, finish + 1):
        data_rate, data_distance = _nearest(tables["Ethernet Data Rates"], second)
        concurrent, concurrent_distance = _nearest(tables["Concurrent Flows"], second)
        flow_rate, flow_rate_distance = _nearest(tables["Flow Rates"], second)
        if max(data_distance, concurrent_distance, flow_rate_distance) > (
            thresholds.nearest_alignment_tolerance_seconds
        ):
            continue
        aligned.append(
            _AlignedSample(
                second=second,
                tx_mbps=data_rate.values[0],
                rx_mbps=data_rate.values[1],
                concurrent_flows=concurrent.values[0],
                flow_rate_per_second=flow_rate.values[0],
            )
        )

    coverage = len(aligned) / expected
    if coverage < thresholds.minimum_alignment_coverage_ratio:
        raise PerformanceTimeseriesError(
            "performance time-series alignment coverage is too low: "
            f"{coverage:.3f} < {thresholds.minimum_alignment_coverage_ratio:.3f}"
        )
    return tuple(aligned), expected, coverage


def _median_window(values: tuple[float, ...], index: int, radius: int = 2) -> float:
    return statistics.median(values[max(0, index - radius) : index + radius + 1])


def _phase_window(
    phase: PerformancePhase, samples: tuple[_AlignedSample, ...]
) -> PerformancePhaseWindow:
    flows = tuple(sample.concurrent_flows for sample in samples)
    return PerformancePhaseWindow(
        phase=phase,
        started_at_second=samples[0].second,
        finished_at_second=samples[-1].second,
        sample_count=len(samples),
        minimum_concurrent_flows=min(flows),
        median_concurrent_flows=statistics.median(flows),
        maximum_concurrent_flows=max(flows),
    )


def _phases(
    samples: tuple[_AlignedSample, ...], thresholds: PerformanceAnalysisThresholds
) -> tuple[tuple[PerformancePhaseWindow, ...], int, int]:
    flows = tuple(sample.concurrent_flows for sample in samples)
    smoothed = tuple(_median_window(flows, index) for index in range(len(flows)))
    peak_index = max(range(len(smoothed)), key=smoothed.__getitem__)
    stable_floor = smoothed[peak_index] * thresholds.stable_flow_floor_ratio_to_peak
    stable_start = peak_index
    stable_finish = peak_index
    while (
        stable_start > 0
        and smoothed[stable_start - 1] >= stable_floor
        and flows[stable_start - 1] >= stable_floor
    ):
        stable_start -= 1
    while (
        stable_finish + 1 < len(samples)
        and smoothed[stable_finish + 1] >= stable_floor
        and flows[stable_finish + 1] >= stable_floor
    ):
        stable_finish += 1
    stable_count = stable_finish - stable_start + 1
    if stable_count < thresholds.minimum_stable_samples:
        raise PerformanceTimeseriesError(
            "cannot identify a sufficiently long Stable phase from Concurrent Flows"
        )

    windows: list[PerformancePhaseWindow] = []
    if stable_start:
        windows.append(_phase_window(PerformancePhase.RAMP_UP, samples[:stable_start]))
    windows.append(
        _phase_window(PerformancePhase.STABLE, samples[stable_start : stable_finish + 1])
    )
    if stable_finish + 1 < len(samples):
        windows.append(_phase_window(PerformancePhase.RAMP_DOWN, samples[stable_finish + 1 :]))
    return tuple(windows), stable_start, stable_finish


def _ratio_change(current: float, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    return (current - baseline) / baseline


def _drop(current: float, baseline: float) -> float:
    change = _ratio_change(current, baseline)
    if change is None:
        raise PerformanceTimeseriesError("Stable throughput baseline must be positive")
    return max(0.0, -change)


def _groups(indices: list[int], samples: tuple[_AlignedSample, ...]) -> tuple[tuple[int, ...], ...]:
    if not indices:
        return ()
    groups: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        previous = groups[-1][-1]
        if index == previous + 1 and samples[index].second == samples[previous].second + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _event(
    assessment: PerformanceAssessment,
    group: tuple[int, ...],
    samples: tuple[_AlignedSample, ...],
    baseline: PerformanceStableBaseline,
    thresholds: PerformanceAnalysisThresholds,
    *,
    recovered: bool,
    rationale: str,
) -> PerformanceDiagnosticEvent:
    tx_drops = tuple(_drop(samples[index].tx_mbps, baseline.tx_median_mbps) for index in group)
    rx_drops = tuple(_drop(samples[index].rx_mbps, baseline.rx_median_mbps) for index in group)
    worst_index = group[
        max(range(len(group)), key=lambda position: max(tx_drops[position], rx_drops[position]))
    ]
    flow_change = _ratio_change(
        samples[worst_index].concurrent_flows, baseline.concurrent_flows_median
    )
    rate_change = _ratio_change(
        samples[worst_index].flow_rate_per_second, baseline.flow_rate_median_per_second
    )
    return PerformanceDiagnosticEvent(
        assessment=assessment,
        started_at_second=samples[group[0]].second,
        finished_at_second=samples[group[-1]].second,
        sample_count=len(group),
        maximum_tx_drop_ratio=max(tx_drops),
        maximum_rx_drop_ratio=max(rx_drops),
        concurrent_flow_change_ratio=flow_change or 0.0,
        flow_rate_change_ratio=rate_change,
        flow_rate_auxiliary_change=(
            rate_change is not None
            and abs(rate_change) >= thresholds.flow_rate_auxiliary_change_ratio
        ),
        recovered=recovered,
        rationale=rationale,
    )


def analyze_performance_timeseries(path: Path) -> PerformanceTimeseriesAnalysis:
    """Parse, align, phase, and classify one BPS performance time-series CSV."""

    thresholds = PerformanceAnalysisThresholds()
    tables = _parse_tables(path)
    samples, expected, coverage = _align(tables, thresholds)
    phases, stable_start, stable_finish = _phases(samples, thresholds)
    stable_samples = samples[stable_start : stable_finish + 1]
    baseline = PerformanceStableBaseline(
        started_at_second=stable_samples[0].second,
        finished_at_second=stable_samples[-1].second,
        sample_count=len(stable_samples),
        tx_median_mbps=statistics.median(sample.tx_mbps for sample in stable_samples),
        rx_median_mbps=statistics.median(sample.rx_mbps for sample in stable_samples),
        concurrent_flows_median=statistics.median(
            sample.concurrent_flows for sample in stable_samples
        ),
        flow_rate_median_per_second=statistics.median(
            sample.flow_rate_per_second for sample in stable_samples
        ),
    )
    if baseline.concurrent_flows_median <= 0:
        raise PerformanceTimeseriesError("Stable Concurrent Flows baseline must be positive")

    stable_candidates: list[int] = []
    for index in range(stable_start, stable_finish + 1):
        sample = samples[index]
        throughput_drop = max(
            _drop(sample.tx_mbps, baseline.tx_median_mbps),
            _drop(sample.rx_mbps, baseline.rx_median_mbps),
        )
        flow_change = _ratio_change(sample.concurrent_flows, baseline.concurrent_flows_median)
        if (
            throughput_drop >= thresholds.throughput_drop_ratio
            and flow_change is not None
            and abs(flow_change) <= thresholds.stable_flow_tolerance_ratio
        ):
            stable_candidates.append(index)

    events: list[PerformanceDiagnosticEvent] = []
    for group in _groups(stable_candidates, samples):
        drops = tuple(
            max(
                _drop(samples[index].tx_mbps, baseline.tx_median_mbps),
                _drop(samples[index].rx_mbps, baseline.rx_median_mbps),
            )
            for index in group
        )
        expanding = (
            len(group) >= thresholds.throughput_drop_minimum_samples
            and all(current >= previous for previous, current in pairwise(drops))
            and drops[-1] - drops[0] >= thresholds.expanding_drop_minimum_increase_ratio
        )
        severe = (
            len(group) >= thresholds.severe_throughput_drop_minimum_samples
            and max(drops) >= thresholds.severe_throughput_drop_ratio
        ) or expanding
        if severe:
            assessment = PerformanceAssessment.SEVERE_PERFORMANCE_ANOMALY
            rationale = "Stable load with a severe or continuously expanding throughput drop"
        elif len(group) >= thresholds.throughput_drop_minimum_samples:
            assessment = PerformanceAssessment.PERFORMANCE_ANOMALY
            rationale = "Stable load with a sustained throughput drop"
        else:
            assessment = PerformanceAssessment.SHORT_FLUCTUATION
            rationale = "Stable load with a throughput drop lasting fewer than three samples"
        next_index = group[-1] + 1
        recovered = (
            next_index <= stable_finish
            and max(
                _drop(samples[next_index].tx_mbps, baseline.tx_median_mbps),
                _drop(samples[next_index].rx_mbps, baseline.rx_median_mbps),
            )
            < thresholds.throughput_drop_ratio
        )
        events.append(
            _event(
                assessment,
                group,
                samples,
                baseline,
                thresholds,
                recovered=recovered,
                rationale=rationale,
            )
        )

    ramp_down_candidates: list[int] = []
    for index in range(stable_finish + 1, len(samples)):
        sample = samples[index]
        previous = samples[index - 1]
        throughput_drop = max(
            _drop(sample.tx_mbps, baseline.tx_median_mbps),
            _drop(sample.rx_mbps, baseline.rx_median_mbps),
        )
        flow_change = _ratio_change(sample.concurrent_flows, baseline.concurrent_flows_median)
        if (
            throughput_drop >= thresholds.throughput_drop_ratio
            and flow_change is not None
            and flow_change <= -thresholds.load_change_flow_drop_ratio
            and sample.concurrent_flows < previous.concurrent_flows
            and (sample.tx_mbps < previous.tx_mbps or sample.rx_mbps < previous.rx_mbps)
        ):
            ramp_down_candidates.append(index)
    for group in _groups(ramp_down_candidates, samples):
        if len(group) < thresholds.load_change_minimum_samples:
            continue
        events.append(
            _event(
                PerformanceAssessment.NORMAL_LOAD_CHANGE,
                group,
                samples,
                baseline,
                thresholds,
                recovered=False,
                rationale="Throughput and Concurrent Flows declined together during Ramp-down",
            )
        )

    assessments = {event.assessment for event in events}
    if PerformanceAssessment.SEVERE_PERFORMANCE_ANOMALY in assessments:
        assessment = PerformanceAssessment.SEVERE_PERFORMANCE_ANOMALY
        summary = "Stable load contains a severe sustained throughput degradation."
    elif PerformanceAssessment.PERFORMANCE_ANOMALY in assessments:
        assessment = PerformanceAssessment.PERFORMANCE_ANOMALY
        summary = "Stable load contains a sustained throughput degradation."
    elif PerformanceAssessment.SHORT_FLUCTUATION in assessments:
        assessment = PerformanceAssessment.SHORT_FLUCTUATION
        summary = "Stable load contains only short throughput fluctuations."
    elif PerformanceAssessment.NORMAL_LOAD_CHANGE in assessments:
        assessment = PerformanceAssessment.NORMAL_LOAD_CHANGE
        summary = "Observed throughput decline follows the Ramp-down load reduction."
    else:
        assessment = PerformanceAssessment.NORMAL
        summary = "Stable-phase Tx/Rx throughput remains near its test-specific median baseline."

    events.sort(key=lambda event: (event.started_at_second, event.finished_at_second))
    return PerformanceTimeseriesAnalysis(
        assessment=assessment,
        summary=summary,
        source_tables=_SOURCE_TABLES,
        expected_resampled_points=expected,
        aligned_sample_count=len(samples),
        alignment_coverage_ratio=coverage,
        phases=phases,
        stable_baseline=baseline,
        events=tuple(events),
        thresholds=thresholds,
    )
