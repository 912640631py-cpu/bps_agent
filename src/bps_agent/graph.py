"""LangGraph orchestration for an Evaluation Run."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from bps_agent.artifacts import ArtifactStore
from bps_agent.models import (
    AppConfig,
    AttemptRecord,
    DutObservations,
    EvaluationMode,
    EvaluationOutcome,
    EvidenceBundle,
    ObservationPhase,
    PerformanceTimeseriesAnalysis,
    SupplementalSnapshot,
    VerdictValue,
    utc_now,
)
from bps_agent.performance_timeseries import analyze_performance_timeseries
from bps_agent.ports import BpsPort, Clock, DutPort, JudgePort
from bps_agent.report_sections import (
    extract_report_sections,
    resolve_minimal_analysis_sections,
    resolve_performance_timeseries_sections,
)

LOGGER = logging.getLogger(__name__)

_ATTEMPT_BANDWIDTH_FACTORS = (1.0, 0.8, 0.6, 0.4, 0.2)


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class EvaluationState(TypedDict):
    evaluation_id: str
    config: dict[str, Any]
    template_metadata: dict[str, Any]
    attempts: list[dict[str, Any]]
    outcome: str | None
    error: str | None
    final_artifact: str | None


@dataclass(frozen=True)
class EvaluationServices:
    config: AppConfig
    bps: BpsPort
    dut: DutPort | None
    judge: JudgePort
    artifacts: ArtifactStore
    clock: Clock


def initial_state(evaluation_id: str, config: AppConfig) -> EvaluationState:
    return {
        "evaluation_id": evaluation_id,
        "config": config.model_dump(mode="json"),
        "template_metadata": {},
        "attempts": [],
        "outcome": None,
        "error": None,
        "final_artifact": None,
    }


def _attempts(state: EvaluationState) -> list[AttemptRecord]:
    return [AttemptRecord.model_validate(item) for item in state["attempts"]]


def _replace_last(state: EvaluationState, attempt: AttemptRecord) -> list[dict[str, Any]]:
    attempts = list(state["attempts"])
    attempts[-1] = attempt.model_dump(mode="json")
    return attempts


def _append_error(attempt: AttemptRecord, message: str) -> AttemptRecord:
    return attempt.model_copy(update={"errors": (*attempt.errors, message)})


def _has_minimum_monitoring(observations: DutObservations) -> bool:
    return bool(
        observations.populated_series(ObservationPhase.BASELINE).intersection(
            observations.populated_series(ObservationPhase.DURING)
        )
    )


def _template_total_bandwidth_mbps(metadata: dict[str, Any]) -> float:
    value = metadata.get("totalBandwidthMbps")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("BPS template metadata omitted a valid totalBandwidthMbps")
    return float(value)


def _attempt_bandwidth_target(
    configured_mbps: float,
    template_total_bandwidth_mbps: float,
    attempt_number: int,
) -> tuple[float, float]:
    if not 1 <= attempt_number <= len(_ATTEMPT_BANDWIDTH_FACTORS):
        raise ValueError(f"unsupported Attempt number: {attempt_number}")
    factor = _ATTEMPT_BANDWIDTH_FACTORS[attempt_number - 1]
    target_mbps = round(configured_mbps * factor, 6)
    percentage = round(target_mbps / template_total_bandwidth_mbps * 100, 6)
    return target_mbps, percentage


def build_graph(
    services: EvaluationServices,
    checkpointer: Any | None = None,
    *,
    interrupt_before: list[str] | None = None,
) -> Any:
    cfg = services.config
    dut_enabled = cfg.evaluation.mode == EvaluationMode.BPS_AND_DUT
    if dut_enabled and services.dut is None:
        raise ValueError("bps_and_dut mode requires a DUT adapter")

    def initialize(state: EvaluationState) -> dict[str, Any]:
        evaluation_id = state["evaluation_id"]
        services.artifacts.ensure_evaluation(evaluation_id)
        services.artifacts.write_evaluation_json(evaluation_id, "config.json", state["config"])
        try:
            metadata = services.bps.find_template(cfg.bps.template)
            template_bandwidth_mbps = _template_total_bandwidth_mbps(metadata)
            if cfg.bps.total_bandwidth_mbps > template_bandwidth_mbps:
                raise ValueError(
                    f"configured Total Bandwidth {cfg.bps.total_bandwidth_mbps:g} Mbps "
                    f"exceeds template original value {template_bandwidth_mbps:g} Mbps"
                )
        except Exception as exc:
            return {
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": f"BPS template preflight failed: {exc}",
            }
        services.artifacts.write_evaluation_json(evaluation_id, "template.json", metadata)
        return {"template_metadata": metadata}

    def start_attempt(state: EvaluationState) -> dict[str, Any]:
        attempt_number = len(state["attempts"]) + 1
        template_bandwidth_mbps = _template_total_bandwidth_mbps(state["template_metadata"])
        target_mbps, bandwidth_percent = _attempt_bandwidth_target(
            cfg.bps.total_bandwidth_mbps,
            template_bandwidth_mbps,
            attempt_number,
        )
        attempt = AttemptRecord(
            number=attempt_number,
            started_at=utc_now(),
            bps_template_total_bandwidth_mbps=template_bandwidth_mbps,
            bps_total_bandwidth_mbps=target_mbps,
            bps_total_bandwidth_percent=bandwidth_percent,
        )
        reserved = False
        try:
            before: SupplementalSnapshot | None = None
            if dut_enabled:
                assert services.dut is not None
                before = services.dut.collect_supplemental()
                if not before.is_complete:
                    raise RuntimeError("required pre-traffic DUT evidence is incomplete")
            services.bps.reserve_ports()
            reserved = True
            services.bps.set_total_bandwidth(bandwidth_percent)
            run_id = services.bps.start_run()
            traffic_started_at = utc_now()
            attempt = attempt.model_copy(
                update={
                    "bps_run_id": run_id,
                    "traffic_started_at": traffic_started_at,
                    "bps_template_metadata": state["template_metadata"],
                    "dut_before": before,
                    "ports_reserved": True,
                }
            )
            services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt_number, "attempt.json", attempt
            )
            LOGGER.info(
                "Attempt %s started as BPS run %s at %.6g Mbps (%.6g%% total bandwidth)",
                attempt_number,
                run_id,
                target_mbps,
                bandwidth_percent,
            )
            return {"attempts": [*state["attempts"], attempt.model_dump(mode="json")]}
        except Exception as exc:
            if reserved:
                try:
                    services.bps.release_ports()
                except Exception as cleanup_exc:
                    exc = RuntimeError(f"{exc}; pre-run port cleanup failed: {cleanup_exc}")
            return {
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": f"Attempt preflight failed before BPS run creation: {exc}",
            }

    def monitor_attempt(state: EvaluationState) -> dict[str, Any]:
        attempt = _attempts(state)[-1]
        assert attempt.bps_run_id is not None
        next_keepalive = services.clock.monotonic() + cfg.dut.keepalive_interval_seconds
        keepalive_errors: list[str] = []

        def keepalive_if_due() -> None:
            nonlocal next_keepalive
            if not dut_enabled or services.dut is None:
                return
            if services.clock.monotonic() < next_keepalive:
                return
            try:
                services.dut.keepalive()
            except Exception as exc:
                message = f"DUT keepalive failed: {exc}"
                keepalive_errors.append(message)
                LOGGER.warning(
                    "Attempt %s DUT keepalive failed while BPS run %s remained active: %s",
                    attempt.number,
                    attempt.bps_run_id,
                    exc,
                )
            finally:
                next_keepalive = services.clock.monotonic() + cfg.dut.keepalive_interval_seconds

        try:
            completion = services.bps.wait_for_completion(attempt.bps_run_id, keepalive_if_due)
            if not completion.terminal:
                raise RuntimeError("BPS adapter did not confirm a terminal run state")
        except Exception as exc:
            attempt = _append_error(attempt, f"BPS monitoring failed: {exc}")
            run_id = attempt.bps_run_id
            assert run_id is not None
            try:
                services.bps.stop_run(run_id)
                completion = services.bps.wait_for_completion(run_id, lambda: None)
                if not completion.terminal:
                    raise RuntimeError("stopped run did not reach a confirmed terminal state")
            except Exception as recovery_exc:
                recovery_error = f"manual recovery required: {recovery_exc}"
                attempt = _append_error(attempt, recovery_error)
                for message in keepalive_errors:
                    attempt = _append_error(attempt, message)
                attempts = _replace_last(state, attempt)
                services.artifacts.write_attempt_json(
                    state["evaluation_id"], attempt.number, "attempt.json", attempt
                )
                return {
                    "attempts": attempts,
                    "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                    "error": recovery_error,
                }

        attempt = attempt.model_copy(
            update={
                "traffic_finished_at": utc_now(),
                "bps_run_details": completion.details,
                "terminal_confirmed": True,
            }
        )
        try:
            services.bps.release_ports()
        except Exception as exc:
            release_error = f"BPS port release failed after confirmed terminal state: {exc}"
            attempt = _append_error(attempt, release_error)
            for message in keepalive_errors:
                attempt = _append_error(attempt, message)
            attempts = _replace_last(state, attempt)
            services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "attempt.json", attempt
            )
            return {
                "attempts": attempts,
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": release_error,
            }
        attempt = attempt.model_copy(update={"ports_reserved": False})

        for message in keepalive_errors:
            attempt = _append_error(attempt, message)
        services.artifacts.write_attempt_json(
            state["evaluation_id"], attempt.number, "attempt.json", attempt
        )
        return {"attempts": _replace_last(state, attempt)}

    def collect_evidence(state: EvaluationState) -> dict[str, Any]:
        attempt = _attempts(state)[-1]
        assert attempt.bps_run_id is not None
        observations = list(attempt.dut_observations)
        compact_observations: DutObservations | None = None
        report_text = ""
        report_toc: Any = None
        performance_analysis: PerformanceTimeseriesAnalysis | None = None
        after: SupplementalSnapshot | None = None
        try:
            if not attempt.traffic_started_at or not attempt.traffic_finished_at:
                raise RuntimeError("BPS traffic time window is incomplete")
            if dut_enabled:
                assert services.dut is not None
                services.clock.sleep(cfg.dut.cooldown_seconds)
                before = attempt.dut_before
                if before is None or not before.is_complete:
                    raise RuntimeError("required pre-traffic DUT evidence is incomplete")
                after = services.dut.collect_supplemental()
                if not after.is_complete:
                    raise RuntimeError("required post-traffic DUT evidence is incomplete")
                observations = list(
                    services.dut.collect_monitoring_window(
                        attempt.traffic_started_at,
                        attempt.traffic_finished_at,
                        before,
                        after,
                    )
                )
                compact_observations = DutObservations.from_resource_observations(
                    tuple(observations)
                )
                services.artifacts.write_attempt_json(
                    state["evaluation_id"],
                    attempt.number,
                    "dut-observations.json",
                    compact_observations,
                )
            report_toc = services.bps.wait_for_report(attempt.bps_run_id)
            run_id = attempt.bps_run_id
            assert run_id is not None
            toc_path = services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "bps-report-toc.json", report_toc
            )
            attempt = attempt.model_copy(update={"report_toc_path": str(toc_path)})
            report_sections = extract_report_sections(report_toc)
            if not report_sections:
                raise RuntimeError("current BPS Run TOC contains no recognizable titled sections")
            selection = resolve_minimal_analysis_sections(report_sections)
            performance_selection = resolve_performance_timeseries_sections(report_sections)
            services.artifacts.write_attempt_json(
                state["evaluation_id"],
                attempt.number,
                "bps-report-sections.json",
                selection.as_dict(toc_section_count=len(report_sections)),
            )
            services.artifacts.write_attempt_json(
                state["evaluation_id"],
                attempt.number,
                "bps-performance-timeseries-sections.json",
                performance_selection.as_dict(
                    toc_section_count=len(report_sections),
                    mode="performance-timeseries-by-title-and-parent-path",
                ),
            )
            if selection.required_missing:
                raise RuntimeError(
                    "current BPS Run TOC is missing required report sections: "
                    + "; ".join(selection.required_missing)
                )
            if performance_selection.required_missing:
                raise RuntimeError(
                    "current BPS Run TOC is missing required performance time-series sections: "
                    + "; ".join(performance_selection.required_missing)
                )
            if performance_selection.ambiguous_required:
                raise RuntimeError(
                    "current BPS Run TOC has ambiguous performance time-series sections: "
                    f"{performance_selection.ambiguous_required}"
                )
            if selection.ambiguous_required:
                LOGGER.warning(
                    "Required BPS report section paths were ambiguous: %s",
                    selection.ambiguous_required,
                )
            destination = (
                services.artifacts.attempt_dir(state["evaluation_id"], attempt.number)
                / "bps-report.csv"
            )
            report_path = services.bps.export_report(
                run_id,
                destination,
                selection.section_ids,
            )
            performance_destination = destination.with_name("bps-performance-timeseries.csv")
            performance_path = services.bps.export_report(
                run_id,
                performance_destination,
                performance_selection.section_ids,
            )
            attempt = attempt.model_copy(
                update={
                    "report_path": str(report_path),
                    "report_toc_path": str(toc_path),
                    "performance_timeseries_path": str(performance_path),
                }
            )
            performance_analysis = analyze_performance_timeseries(performance_path)
            report_text = report_path.read_text(encoding="utf-8-sig", errors="replace")
            attempt = attempt.model_copy(
                update={
                    "dut_observations": tuple(observations),
                    "dut_after": after,
                }
            )
        except Exception as exc:
            attempt = _append_error(attempt, f"evidence collection failed: {exc}")

        attempt = attempt.model_copy(
            update={
                "dut_observations": tuple(observations),
                "dut_after": after,
            }
        )

        bps_complete = bool(
            attempt.bps_run_id
            and attempt.traffic_started_at
            and attempt.traffic_finished_at
            and report_text
            and performance_analysis
        )
        dut_complete = not dut_enabled or bool(
            attempt.dut_before
            and attempt.dut_before.is_complete
            and after
            and after.is_complete
            and compact_observations
            and _has_minimum_monitoring(compact_observations)
        )
        complete = bps_complete and dut_complete
        attempt = attempt.model_copy(update={"evidence_complete": complete})
        if complete:
            evidence_run_id = attempt.bps_run_id
            assert evidence_run_id is not None
            traffic_started_at = attempt.traffic_started_at
            traffic_finished_at = attempt.traffic_finished_at
            assert traffic_started_at is not None and traffic_finished_at is not None
            evidence = EvidenceBundle(
                evaluation_mode=cfg.evaluation.mode,
                evaluation_id=state["evaluation_id"],
                attempt_number=attempt.number,
                bps_run_id=evidence_run_id,
                bps_template=cfg.bps.template,
                bps_template_total_bandwidth_mbps=(attempt.bps_template_total_bandwidth_mbps),
                bps_total_bandwidth_mbps=attempt.bps_total_bandwidth_mbps,
                bps_total_bandwidth_percent=attempt.bps_total_bandwidth_percent,
                bps_template_metadata=attempt.bps_template_metadata,
                bps_run_details=attempt.bps_run_details,
                bps_report=report_text,
                bps_performance_analysis=performance_analysis,
                assessment=cfg.selected_assessment,
                dut_endpoint=cfg.dut.endpoint if dut_enabled else None,
                dut_interfaces=cfg.dut.interfaces if dut_enabled else None,
                traffic_started_at=traffic_started_at,
                traffic_finished_at=traffic_finished_at,
                dut_observations=compact_observations if dut_enabled else None,
                dut_before=attempt.dut_before if dut_enabled else None,
                dut_after=after if dut_enabled else None,
            )
            evidence_document = evidence.as_document()
            evidence_path = services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "evidence.json", evidence_document
            )
            attempt = attempt.model_copy(update={"evidence_path": str(evidence_path)})
        elif not attempt.errors:
            attempt = _append_error(attempt, "required Evidence Bundle fields are incomplete")

        services.artifacts.write_attempt_json(
            state["evaluation_id"], attempt.number, "attempt.json", attempt
        )
        update: dict[str, Any] = {"attempts": _replace_last(state, attempt)}
        if not complete:
            update.update(
                {
                    "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                    "error": attempt.errors[-1],
                }
            )
        return update

    def adjudicate(state: EvaluationState) -> dict[str, Any]:
        attempt = _attempts(state)[-1]
        assert attempt.evidence_path is not None
        evidence = EvidenceBundle.model_validate(
            services.artifacts.read_json(Path(attempt.evidence_path))
        )
        try:
            verdict, raw_response = services.judge.adjudicate(evidence)
        except Exception as exc:
            attempt = _append_error(attempt, f"LLM adjudication failed: {exc}")
            services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "attempt.json", attempt
            )
            return {
                "attempts": _replace_last(state, attempt),
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": attempt.errors[-1],
            }
        verdict_path = services.artifacts.write_attempt_json(
            state["evaluation_id"],
            attempt.number,
            "verdict.json",
            {"parsed": verdict.model_dump(mode="json"), "raw_response": raw_response},
        )
        attempt = attempt.model_copy(update={"verdict": verdict, "verdict_path": str(verdict_path)})
        services.artifacts.write_attempt_json(
            state["evaluation_id"], attempt.number, "attempt.json", attempt
        )
        return {"attempts": _replace_last(state, attempt)}

    def route_after_initialize(
        state: EvaluationState,
    ) -> Literal["start_attempt", "finalize"]:
        return "finalize" if state["outcome"] else "start_attempt"

    def route_after_start(
        state: EvaluationState,
    ) -> Literal["monitor_attempt", "finalize"]:
        return "finalize" if state["outcome"] else "monitor_attempt"

    def route_after_monitor(
        state: EvaluationState,
    ) -> Literal["collect_evidence", "finalize"]:
        return "finalize" if state["outcome"] else "collect_evidence"

    def route_after_evidence(
        state: EvaluationState,
    ) -> Literal["adjudicate", "finalize"]:
        attempt = _attempts(state)[-1]
        return "adjudicate" if attempt.evidence_complete else "finalize"

    def route_after_verdict(
        state: EvaluationState,
    ) -> Literal["start_attempt", "finalize"]:
        if state["outcome"]:
            return "finalize"
        attempt = _attempts(state)[-1]
        assert attempt.verdict is not None
        if attempt.verdict.verdict == VerdictValue.PASS:
            return "finalize"
        if len(state["attempts"]) < cfg.evaluation.max_attempts:
            return "start_attempt"
        return "finalize"

    def finalize(state: EvaluationState) -> dict[str, Any]:
        attempts = _attempts(state)
        outcome = state["outcome"]
        if outcome is None:
            last = attempts[-1] if attempts else None
            if last and last.verdict and last.verdict.verdict == VerdictValue.PASS:
                outcome = (
                    EvaluationOutcome.PASSED.value
                    if last.number == 1
                    else EvaluationOutcome.DEGRADED_PASS.value
                )
            elif last and last.verdict and last.verdict.verdict == VerdictValue.RETRY:
                outcome = EvaluationOutcome.NOT_PASSED.value
            else:
                outcome = EvaluationOutcome.INCONCLUSIVE.value
        summary = {
            "evaluation_id": state["evaluation_id"],
            "outcome": outcome,
            "provider": services.judge.provider_name,
            "model": services.judge.model_name,
            "attempts": [item.model_dump(mode="json") for item in attempts],
            "error": state["error"],
            "finished_at": utc_now(),
        }
        path = services.artifacts.write_evaluation_json(
            state["evaluation_id"], "result.json", summary
        )
        return {"outcome": outcome, "final_artifact": str(path)}

    graph = StateGraph(EvaluationState)
    graph.add_node("initialize", initialize)
    graph.add_node("start_attempt", start_attempt)
    graph.add_node("monitor_attempt", monitor_attempt)
    graph.add_node("collect_evidence", collect_evidence)
    graph.add_node("adjudicate", adjudicate)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "initialize")
    graph.add_conditional_edges("initialize", route_after_initialize)
    graph.add_conditional_edges("start_attempt", route_after_start)
    graph.add_conditional_edges("monitor_attempt", route_after_monitor)
    graph.add_conditional_edges("collect_evidence", route_after_evidence)
    graph.add_conditional_edges("adjudicate", route_after_verdict)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
