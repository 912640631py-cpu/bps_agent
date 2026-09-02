"""LangGraph orchestration for an Evaluation Run."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from bps_agent.adjudication import verdict_artifact
from bps_agent.launch import LaunchReconciliationError, RunLaunchCoordinator
from bps_agent.models.bps import PortReservationState, PortReservationStatus
from bps_agent.models.common import (
    ATTEMPT_BANDWIDTH_FACTORS,
    EvaluationMode,
    EvaluationOutcome,
    ObservationPhase,
    VerdictValue,
    utc_now,
)
from bps_agent.models.config import AppConfig
from bps_agent.models.dut import (
    BackendDutEvidence,
    DutEvidence,
    DutObservations,
    FrontendDutEvidence,
)
from bps_agent.models.evaluation import (
    AttemptRecord,
    EvidenceBundle,
)
from bps_agent.models.performance import PerformanceTimeseriesAnalysis
from bps_agent.performance_timeseries import analyze_performance_timeseries
from bps_agent.report_sections import (
    extract_report_sections,
    resolve_minimal_analysis_sections,
    resolve_performance_timeseries_sections,
)
from bps_agent.runtime import RuntimeResources

LOGGER = logging.getLogger(__name__)


class EvaluationState(TypedDict):
    evaluation_id: str
    config: dict[str, Any]
    template_metadata: dict[str, Any]
    attempts: list[dict[str, Any]]
    outcome: str | None
    error: str | None
    final_artifact: str | None


EvaluationServices = RuntimeResources


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


def _dut_evidence_complete(evidence: DutEvidence) -> bool:
    if isinstance(evidence, BackendDutEvidence):
        return evidence.successful_sample_count > 0 and bool(evidence.metrics_csv.strip())
    if isinstance(evidence, FrontendDutEvidence):
        return bool(
            evidence.before.is_complete
            and evidence.after.is_complete
            and _has_minimum_monitoring(evidence.observations)
        )
    return False


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
    if not 1 <= attempt_number <= len(ATTEMPT_BANDWIDTH_FACTORS):
        raise ValueError(f"unsupported Attempt number: {attempt_number}")
    factor = ATTEMPT_BANDWIDTH_FACTORS[attempt_number - 1]
    target_mbps = round(configured_mbps * factor, 6)
    percentage = round(target_mbps / template_total_bandwidth_mbps * 100, 6)
    return target_mbps, percentage


def build_graph(
    services: RuntimeResources,
    checkpointer: Any | None = None,
    *,
    interrupt_before: list[str] | None = None,
) -> Any:
    cfg = services.config
    launcher = RunLaunchCoordinator(services.bps, services.artifacts)
    dut_enabled = cfg.evaluation.mode == EvaluationMode.BPS_AND_DUT
    dut_config = cfg.dut
    if dut_enabled and services.dut is None:
        raise ValueError("bps_and_dut mode requires a DUT adapter")
    if dut_enabled and dut_config is None:
        raise ValueError("bps_and_dut mode requires DUT configuration")

    def actual_reservation_status() -> PortReservationStatus:
        return services.bps.port_reservation_status()

    def reject_foreign_reservation(status: PortReservationStatus) -> None:
        if not status.has_foreign_reservation:
            return
        raise RuntimeError(
            "configured BPS ports are reserved by another account: "
            + ", ".join(status.foreign_owners)
        )

    def attempt_reservation_update(status: PortReservationStatus) -> dict[str, Any]:
        return {"port_reservation_state": status.state}

    def release_agent_reservation_if_inactive(
        status: PortReservationStatus | None = None,
    ) -> PortReservationStatus:
        status = status or actual_reservation_status()
        reject_foreign_reservation(status)
        if status.state == PortReservationState.NONE:
            return status
        active_runs = services.bps.find_active_runs_for_ports()
        if active_runs:
            reservation_label = (
                "partially reserved"
                if status.state == PortReservationState.PARTIAL_AGENT
                else "reserved"
            )
            raise RuntimeError(
                f"refusing to clean {reservation_label} Agent-owned BPS ports while "
                "active Run(s) exist: " + ", ".join(active_runs)
            )
        selected_ports = (
            status.agent_owned_ports if status.state == PortReservationState.PARTIAL_AGENT else None
        )
        try:
            if selected_ports is None:
                services.bps.release_ports()
            else:
                services.bps.release_ports(selected_ports)
        except Exception:
            reconciled = actual_reservation_status()
            reject_foreign_reservation(reconciled)
            if not reconciled.agent_owned_ports:
                return reconciled
            raise
        reconciled = actual_reservation_status()
        reject_foreign_reservation(reconciled)
        if reconciled.state != PortReservationState.NONE:
            raise RuntimeError(
                "BPS ports remained reserved after cleanup: " + reconciled.state.value
            )
        return reconciled

    def reconcile_partial_reservation(
        status: PortReservationStatus,
    ) -> PortReservationStatus:
        reject_foreign_reservation(status)
        if status.state != PortReservationState.PARTIAL_AGENT:
            return status
        return release_agent_reservation_if_inactive(status)

    def reserve_all_ports() -> PortReservationStatus:
        services.bps.reserve_ports()
        status = actual_reservation_status()
        if not status.is_fully_agent_owned:
            if status.state == PortReservationState.PARTIAL_AGENT:
                release_agent_reservation_if_inactive(status)
            raise RuntimeError(
                "BPS reserve did not produce a complete Agent reservation: " + status.state.value
            )
        return status

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
        reservation = PortReservationStatus.classify(())
        run_id: str | None = None
        dut_started = False
        try:
            reservation = reconcile_partial_reservation(actual_reservation_status())
            if launcher.recovery_requires_reservation(state["evaluation_id"], attempt_number):
                if reservation.state == PortReservationState.ALL_AGENT:
                    active_runs = services.bps.find_active_runs_for_ports()
                    if active_runs:
                        raise RuntimeError(
                            "refusing to send a prepared BPS launch while active Run(s) exist: "
                            + ", ".join(active_runs)
                        )
                elif reservation.state == PortReservationState.NONE:
                    reservation = reserve_all_ports()
            launch = launcher.recover(
                state["evaluation_id"],
                attempt_number,
                template=cfg.bps.template,
                group=cfg.bps.group,
            )
            recovered = launch is not None
            if launch is None:
                if reservation.state == PortReservationState.ALL_AGENT:
                    reservation = release_agent_reservation_if_inactive(reservation)
                if dut_enabled:
                    assert services.dut is not None
                    services.dut.prepare_attempt(
                        services.artifacts.attempt_dir(state["evaluation_id"], attempt_number)
                    )
                reservation = reserve_all_ports()
                services.bps.set_total_bandwidth(bandwidth_percent)
                launch = launcher.start(
                    state["evaluation_id"],
                    attempt_number,
                    template=cfg.bps.template,
                    group=cfg.bps.group,
                )
            else:
                reservation = reconcile_partial_reservation(actual_reservation_status())
            run_id = launch.run_id
            traffic_started_at = launch.launched_at
            if dut_enabled:
                assert services.dut is not None
                if recovered:
                    services.dut.restore_attempt(
                        services.artifacts.attempt_dir(state["evaluation_id"], attempt_number),
                        traffic_started_at,
                        None,
                    )
                else:
                    services.dut.traffic_started(traffic_started_at)
                dut_started = True
            attempt = attempt.model_copy(
                update={
                    "bps_run_id": run_id,
                    "traffic_started_at": traffic_started_at,
                    "bps_template_metadata": state["template_metadata"],
                    "dut_collection_method": (
                        dut_config.collection_method
                        if dut_enabled and dut_config is not None
                        else None
                    ),
                    **attempt_reservation_update(reservation),
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
        except LaunchReconciliationError as exc:
            attempt = _append_error(attempt, f"BPS launch reconciliation failed: {exc}")
            try:
                reservation = actual_reservation_status()
            except Exception as reservation_exc:
                attempt = _append_error(
                    attempt, f"BPS reservation reconciliation failed: {reservation_exc}"
                )
            attempt = attempt.model_copy(update=attempt_reservation_update(reservation))
            services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt_number, "attempt.json", attempt
            )
            return {
                "attempts": [*state["attempts"], attempt.model_dump(mode="json")],
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": attempt.errors[-1],
            }
        except Exception as exc:
            if run_id is not None:
                try:
                    services.bps.stop_run(run_id)
                    completion = services.bps.wait_for_completion(run_id, lambda: None)
                    if not completion.terminal:
                        raise RuntimeError("stopped run did not reach a confirmed terminal state")
                    if dut_started and services.dut is not None:
                        services.dut.traffic_finished(utc_now())
                except Exception as cleanup_exc:
                    exc = RuntimeError(f"{exc}; BPS run cleanup failed: {cleanup_exc}")
            if reservation.state in {
                PortReservationState.ALL_AGENT,
                PortReservationState.PARTIAL_AGENT,
            }:
                try:
                    reservation = release_agent_reservation_if_inactive(reservation)
                except Exception as cleanup_exc:
                    exc = RuntimeError(f"{exc}; pre-run port cleanup failed: {cleanup_exc}")
            return {
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": f"Attempt start failed: {exc}",
            }

    def monitor_attempt(state: EvaluationState) -> dict[str, Any]:
        attempt = _attempts(state)[-1]
        monitored_run_id = attempt.bps_run_id
        assert monitored_run_id is not None
        try:
            reservation = actual_reservation_status()
            attempt = attempt.model_copy(update=attempt_reservation_update(reservation))
            reservation = reconcile_partial_reservation(reservation)
        except Exception as exc:
            recovery_error = f"BPS reservation recovery failed: {exc}"
            attempt = _append_error(attempt, recovery_error)
            services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "attempt.json", attempt
            )
            return {
                "attempts": _replace_last(state, attempt),
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": recovery_error,
            }
        attempt = attempt.model_copy(update=attempt_reservation_update(reservation))
        try:
            if dut_enabled:
                assert services.dut is not None
                assert attempt.traffic_started_at is not None
                services.dut.restore_attempt(
                    services.artifacts.attempt_dir(state["evaluation_id"], attempt.number),
                    attempt.traffic_started_at,
                    None,
                )
            completion = services.bps.wait_for_completion(monitored_run_id, lambda: None)
            if not completion.terminal:
                raise RuntimeError("BPS adapter did not confirm a terminal run state")
            launcher.mark_terminal(state["evaluation_id"], attempt.number, monitored_run_id)
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
                attempts = _replace_last(state, attempt)
                services.artifacts.write_attempt_json(
                    state["evaluation_id"], attempt.number, "attempt.json", attempt
                )
                return {
                    "attempts": attempts,
                    "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                    "error": recovery_error,
                }

        traffic_finished_at = utc_now()
        if dut_enabled:
            assert services.dut is not None
            try:
                services.dut.traffic_finished(traffic_finished_at)
            except Exception as exc:
                attempt = _append_error(attempt, f"DUT collection stop failed: {exc}")
        attempt = attempt.model_copy(
            update={
                "traffic_finished_at": traffic_finished_at,
                "bps_run_details": completion.details,
                "terminal_confirmed": True,
            }
        )
        try:
            reservation = release_agent_reservation_if_inactive()
        except Exception as exc:
            release_error = f"BPS port release failed after confirmed terminal state: {exc}"
            attempt = _append_error(attempt, release_error)
            attempts = _replace_last(state, attempt)
            services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "attempt.json", attempt
            )
            return {
                "attempts": attempts,
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": release_error,
            }
        attempt = attempt.model_copy(update=attempt_reservation_update(reservation))
        released_run_id = attempt.bps_run_id
        assert released_run_id is not None
        launcher.mark_released(state["evaluation_id"], attempt.number, released_run_id)
        services.artifacts.write_attempt_json(
            state["evaluation_id"], attempt.number, "attempt.json", attempt
        )
        return {"attempts": _replace_last(state, attempt)}

    def collect_evidence(state: EvaluationState) -> dict[str, Any]:
        attempt = _attempts(state)[-1]
        assert attempt.bps_run_id is not None
        dut_evidence: DutEvidence | None = None
        report_text = ""
        report_toc: Any = None
        performance_analysis: PerformanceTimeseriesAnalysis | None = None
        try:
            if not attempt.traffic_started_at or not attempt.traffic_finished_at:
                raise RuntimeError("BPS traffic time window is incomplete")
            if dut_enabled:
                assert services.dut is not None
                services.dut.restore_attempt(
                    services.artifacts.attempt_dir(state["evaluation_id"], attempt.number),
                    attempt.traffic_started_at,
                    attempt.traffic_finished_at,
                )
                capture = services.dut.finalize_attempt()
                dut_evidence = capture.evidence
                for warning in capture.warnings:
                    attempt = _append_error(attempt, warning)
                services.artifacts.write_attempt_json(
                    state["evaluation_id"],
                    attempt.number,
                    "dut-evidence.json",
                    dut_evidence,
                )
                sample_updates: dict[str, Any] = {
                    "dut_raw_artifact_path": capture.raw_artifact_path,
                    "dut_csv_artifact_path": capture.csv_artifact_path,
                }
                if isinstance(dut_evidence, BackendDutEvidence):
                    sample_updates.update(
                        {
                            "dut_successful_sample_count": (dut_evidence.successful_sample_count),
                            "dut_failed_sample_count": dut_evidence.failed_sample_count,
                            "dut_missed_sample_count": dut_evidence.missed_sample_count,
                        }
                    )
                attempt = attempt.model_copy(update=sample_updates)
            run_id = attempt.bps_run_id
            assert run_id is not None
            report_toc = services.bps.wait_for_report(run_id)
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
            attempt_dir = services.artifacts.attempt_dir(state["evaluation_id"], attempt.number)
            destination = attempt_dir / "bps-report.csv"
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
            pdf_destination = attempt_dir / "bps-report-full.pdf"
            full_section_ids = tuple(
                section.section_id for section in report_sections if section.parent_id is None
            )
            try:
                services.bps.schedule_full_report_pdf(
                    run_id,
                    pdf_destination,
                    full_section_ids,
                )
            except Exception as exc:
                LOGGER.warning("Optional full PDF report export could not be scheduled: %s", exc)
        except Exception as exc:
            attempt = _append_error(attempt, f"evidence collection failed: {exc}")

        bps_complete = bool(
            attempt.bps_run_id
            and attempt.traffic_started_at
            and attempt.traffic_finished_at
            and report_text
            and performance_analysis
        )
        dut_complete = not dut_enabled or bool(
            dut_evidence is not None and _dut_evidence_complete(dut_evidence)
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
                traffic_started_at=traffic_started_at,
                traffic_finished_at=traffic_finished_at,
                dut_evidence=dut_evidence if dut_enabled else None,
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
        evidence_path = Path(attempt.evidence_path)
        evidence = EvidenceBundle.model_validate(services.artifacts.read_json(evidence_path))
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
            verdict_artifact(
                provider=services.judge.provider_name,
                model=services.judge.model_name,
                reasoning_effort=getattr(services.judge, "reasoning_effort", None),
                verdict=verdict,
                provider_exchange=raw_response,
                evidence_path=evidence_path,
            ),
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
        if len(state["attempts"]) < len(ATTEMPT_BANDWIDTH_FACTORS):
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
