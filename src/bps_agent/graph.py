"""LangGraph orchestration for an Evaluation Run."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from bps_agent.artifacts import ArtifactStore
from bps_agent.models import (
    AppConfig,
    AttemptRecord,
    EvaluationOutcome,
    EvidenceBundle,
    ObservationPhase,
    ResourceObservation,
    SupplementalSnapshot,
    VerdictValue,
    utc_now,
)
from bps_agent.ports import BpsPort, Clock, DutPort, JudgePort

LOGGER = logging.getLogger(__name__)


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
    dut: DutPort
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


def _complete_observation(
    observations: tuple[ResourceObservation, ...],
    phase: ObservationPhase,
    interfaces: tuple[str, ...],
) -> bool:
    return any(item.phase == phase and item.is_complete(interfaces) for item in observations)


def build_graph(services: EvaluationServices, checkpointer: Any | None = None) -> Any:
    cfg = services.config

    def initialize(state: EvaluationState) -> dict[str, Any]:
        evaluation_id = state["evaluation_id"]
        services.artifacts.ensure_evaluation(evaluation_id)
        services.artifacts.write_evaluation_json(evaluation_id, "config.json", state["config"])
        try:
            metadata = services.bps.find_template(cfg.bps.template)
        except Exception as exc:
            return {
                "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                "error": f"BPS template preflight failed: {exc}",
            }
        services.artifacts.write_evaluation_json(evaluation_id, "template.json", metadata)
        return {"template_metadata": metadata}

    def start_attempt(state: EvaluationState) -> dict[str, Any]:
        attempt_number = len(state["attempts"]) + 1
        attempt = AttemptRecord(number=attempt_number, started_at=utc_now())
        reserved = False
        try:
            baseline = services.dut.collect_resources(ObservationPhase.BASELINE.value)
            before = services.dut.collect_supplemental()
            if not baseline.is_complete(cfg.dut.interfaces) or not before.is_complete:
                raise RuntimeError("required pre-traffic DUT evidence is incomplete")
            services.bps.reserve_ports()
            reserved = True
            run_id = services.bps.start_run()
            attempt = attempt.model_copy(
                update={
                    "bps_run_id": run_id,
                    "bps_template_metadata": state["template_metadata"],
                    "dut_observations": (baseline,),
                    "dut_before": before,
                    "ports_reserved": True,
                }
            )
            services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt_number, "attempt.json", attempt
            )
            LOGGER.info("Attempt %s started as BPS run %s", attempt_number, run_id)
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
        observations = list(attempt.dut_observations)
        immediate = services.dut.collect_resources(ObservationPhase.DURING.value)
        observations.append(immediate)
        next_sample = services.clock.monotonic() + cfg.dut.sample_interval_seconds

        def sample_when_due() -> None:
            nonlocal next_sample
            now = services.clock.monotonic()
            if now < next_sample:
                return
            observation = services.dut.collect_resources(ObservationPhase.DURING.value)
            observations.append(observation)
            services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "dut-observations.json", observations
            )
            next_sample = now + cfg.dut.sample_interval_seconds

        try:
            completion = services.bps.wait_for_completion(attempt.bps_run_id, sample_when_due)
            if not completion.terminal:
                raise RuntimeError("BPS adapter did not confirm a terminal run state")
            services.bps.release_ports()
            attempt = attempt.model_copy(
                update={
                    "dut_observations": tuple(observations),
                    "bps_run_details": completion.details,
                    "terminal_confirmed": True,
                    "ports_reserved": False,
                }
            )
        except Exception as exc:
            attempt = _append_error(
                attempt.model_copy(update={"dut_observations": tuple(observations)}),
                f"BPS monitoring failed: {exc}",
            )
            run_id = attempt.bps_run_id
            assert run_id is not None
            try:
                services.bps.stop_run(run_id)
                completion = services.bps.wait_for_completion(run_id, lambda: None)
                if not completion.terminal:
                    raise RuntimeError("stopped run did not reach a confirmed terminal state")
                services.bps.release_ports()
                attempt = attempt.model_copy(
                    update={
                        "terminal_confirmed": True,
                        "ports_reserved": False,
                        "bps_run_details": completion.details,
                    }
                )
            except Exception as recovery_exc:
                attempt = _append_error(attempt, f"manual recovery required: {recovery_exc}")
                attempts = _replace_last(state, attempt)
                services.artifacts.write_attempt_json(
                    state["evaluation_id"], attempt.number, "attempt.json", attempt
                )
                return {
                    "attempts": attempts,
                    "outcome": EvaluationOutcome.INCONCLUSIVE.value,
                    "error": attempt.errors[-1],
                }

        services.artifacts.write_attempt_json(
            state["evaluation_id"], attempt.number, "dut-observations.json", observations
        )
        services.artifacts.write_attempt_json(
            state["evaluation_id"], attempt.number, "attempt.json", attempt
        )
        return {"attempts": _replace_last(state, attempt)}

    def collect_evidence(state: EvaluationState) -> dict[str, Any]:
        attempt = _attempts(state)[-1]
        assert attempt.bps_run_id is not None
        observations = list(attempt.dut_observations)
        report_text = ""
        report_toc: Any = None
        after: SupplementalSnapshot | None = None
        try:
            services.clock.sleep(cfg.dut.cooldown_seconds)
            recovery = services.dut.collect_resources(ObservationPhase.RECOVERY.value)
            observations.append(recovery)
            after = services.dut.collect_supplemental()
            report_toc = services.bps.wait_for_report(attempt.bps_run_id)
            destination = (
                services.artifacts.attempt_dir(state["evaluation_id"], attempt.number)
                / "bps-report.csv"
            )
            report_path = services.bps.export_report(attempt.bps_run_id, destination)
            report_text = report_path.read_text(encoding="utf-8-sig", errors="replace")
            toc_path = services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "bps-report-toc.json", report_toc
            )
            attempt = attempt.model_copy(
                update={
                    "dut_observations": tuple(observations),
                    "dut_after": after,
                    "report_path": str(report_path),
                    "report_toc_path": str(toc_path),
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

        complete = bool(
            attempt.bps_run_id
            and report_text
            and attempt.dut_before
            and attempt.dut_before.is_complete
            and after
            and after.is_complete
            and _complete_observation(
                tuple(observations), ObservationPhase.BASELINE, cfg.dut.interfaces
            )
            and _complete_observation(
                tuple(observations), ObservationPhase.DURING, cfg.dut.interfaces
            )
            and _complete_observation(
                tuple(observations), ObservationPhase.RECOVERY, cfg.dut.interfaces
            )
        )
        attempt = attempt.model_copy(update={"evidence_complete": complete})
        if complete:
            assert attempt.dut_before is not None and after is not None
            run_id = attempt.bps_run_id
            assert run_id is not None
            evidence = EvidenceBundle(
                evaluation_id=state["evaluation_id"],
                attempt_number=attempt.number,
                bps_run_id=run_id,
                bps_template=cfg.bps.template,
                bps_template_metadata=attempt.bps_template_metadata,
                bps_run_details=attempt.bps_run_details,
                bps_report=report_text,
                bps_report_toc=report_toc,
                assessment=cfg.assessment,
                dut_endpoint=cfg.dut.endpoint,
                dut_interfaces=cfg.dut.interfaces,
                dut_observations=tuple(observations),
                dut_before=attempt.dut_before,
                dut_after=after,
            )
            evidence_path = services.artifacts.write_attempt_json(
                state["evaluation_id"], attempt.number, "evidence.json", evidence
            )
            attempt = attempt.model_copy(update={"evidence_path": str(evidence_path)})
        elif not attempt.errors:
            attempt = _append_error(attempt, "required Evidence Bundle fields are incomplete")

        services.artifacts.write_attempt_json(
            state["evaluation_id"], attempt.number, "attempt.json", attempt
        )
        return {"attempts": _replace_last(state, attempt)}

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

    def route_after_monitor(
        state: EvaluationState,
    ) -> Literal["collect_evidence", "finalize"]:
        return "finalize" if state["outcome"] else "collect_evidence"

    def route_after_evidence(
        state: EvaluationState,
    ) -> Literal["adjudicate", "start_attempt", "finalize"]:
        attempt = _attempts(state)[-1]
        if attempt.evidence_complete:
            return "adjudicate"
        if len(state["attempts"]) < cfg.evaluation.max_attempts:
            return "start_attempt"
        return "finalize"

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
                outcome = EvaluationOutcome.PASSED.value
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
    graph.add_edge("start_attempt", "monitor_attempt")
    graph.add_conditional_edges("monitor_attempt", route_after_monitor)
    graph.add_conditional_edges("collect_evidence", route_after_evidence)
    graph.add_conditional_edges("adjudicate", route_after_verdict)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
