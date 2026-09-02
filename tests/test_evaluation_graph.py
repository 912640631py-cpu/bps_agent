from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from bps_agent.artifacts import ArtifactStore
from bps_agent.graph import (
    EvaluationServices,
    build_graph,
    initial_state,
)
from bps_agent.models.bps import (
    PortReservation,
    PortReservationState,
    PortReservationStatus,
    RunCompletion,
)
from bps_agent.models.common import (
    DutCollectionMethod,
    EvaluationMode,
    EvaluationOutcome,
    ObservationPhase,
    VerdictValue,
)
from bps_agent.models.config import AppConfig
from bps_agent.models.dut import (
    BackendDutEvidence,
    BackendDutTarget,
    DutCaptureResult,
    DutObservations,
    FrontendDutEvidence,
    ResourceObservation,
    SupplementalSnapshot,
)
from bps_agent.models.evaluation import VerdictDocument


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def performance_report_csv() -> str:
    flows = (0, 250, 500, 750, *([1000] * 9), 700, 300, 0)
    lines = [
        "2.1.1. Ethernet Data Rates",
        "Timestamp,Transmit rate,Receive rate",
        "Seconds,Megabits/s,",
        *(f"{second + 0.45:.2f},100,100" for second in range(16)),
        "2.1.2. Concurrent Flows",
        "Timestamp,TCP,UDP,SCTP,Total,Super Flow",
        "Seconds,Flows,,,,",
        *(f"{second + 0.10:.2f},0,0,0,{total},0" for second, total in enumerate(flows)),
        "2.1.3. Flow Rates",
        "Timestamp,TCP rate,UDP rate,SCTP rate,Total rate,Super Flow rate",
        "Seconds,Flows/s,,,,",
        *(f"{second + 0.30:.2f},0,0,0,10,0" for second in range(16)),
    ]
    return "\n".join(lines) + "\n"


class FakeBps:
    def __init__(
        self,
        clock: FakeClock,
        *,
        template_error: bool = False,
        template_bandwidth_mbps: float = 400.0,
        report_toc: Any = None,
        block_pdf_until_released: bool = False,
    ) -> None:
        self.clock = clock
        self.template_error = template_error
        self.template_bandwidth_mbps = template_bandwidth_mbps
        self.report_toc = (
            report_toc
            if report_toc is not None
            else {
                "sections": [
                    {"sectionId": "10", "sectionName": "Synopsis"},
                    {"sectionId": "10.4", "sectionName": "Test parameters"},
                    {"sectionId": "10.5", "sectionName": "Test Criteria"},
                    {"sectionId": "10.6", "sectionName": "Summary of Results"},
                    {"sectionId": "12", "sectionName": "Test Environment"},
                    {"sectionId": "12.8", "sectionName": "Interfaces"},
                    {"sectionId": "20", "sectionName": "Test Results for AppSim"},
                    {"sectionId": "20.3", "sectionName": "Component Results"},
                    {"sectionId": "20.9", "sectionName": "UDP Summary"},
                    {"sectionId": "30", "sectionName": "Aggregate Stats"},
                    {"sectionId": "30.2", "sectionName": "Ethernet Summary"},
                    {"sectionId": "30.4", "sectionName": "Detail"},
                    {"sectionId": "30.4.5", "sectionName": "Ethernet Data Rates"},
                    {"sectionId": "30.4.7", "sectionName": "Concurrent Flows"},
                    {"sectionId": "30.4.8", "sectionName": "Flow Rates"},
                ]
            }
        )
        self.run_count = 0
        self.reserve_count = 0
        self.release_count = 0
        self.released_port_sets: list[tuple[int, ...]] = []
        self.stop_count = 0
        self.reservation_owners: dict[int, str | None] = {4: None, 5: None}
        self.active_run_ids: set[str] = set()
        self.export_section_ids: list[tuple[str, ...]] = []
        self.bandwidth_percentages: list[float] = []
        self.block_pdf_until_released = block_pdf_until_released
        self.pdf_started = Event()
        self.pdf_release = Event()
        self.pdf_finished = Event()

    def find_template(self, name: str) -> dict[str, Any]:
        if self.template_error:
            raise RuntimeError("missing template")
        return {
            "name": name,
            "version": 1,
            "totalBandwidthMbps": self.template_bandwidth_mbps,
            "sharedComponentSettings": [
                {
                    "name": "totalBandwidth",
                    "originalValue": str(self.template_bandwidth_mbps),
                    "enabled": True,
                }
            ],
        }

    def reserve_ports(self) -> None:
        self.reserve_count += 1
        self.reservation_owner = "agent-user"

    @property
    def reservation_owner(self) -> str | None:
        owners = set(self.reservation_owners.values())
        return owners.pop() if len(owners) == 1 else None

    @reservation_owner.setter
    def reservation_owner(self, owner: str | None) -> None:
        self.reservation_owners = dict.fromkeys((4, 5), owner)

    def port_reservation_status(self) -> PortReservationStatus:
        return PortReservationStatus.classify(
            tuple(
                PortReservation(
                    slot=4,
                    port=port,
                    owner=owner,
                    owned_by_agent=owner == "agent-user",
                )
                for port, owner in self.reservation_owners.items()
            )
        )

    def find_active_runs_for_ports(self) -> tuple[str, ...]:
        return tuple(sorted(self.active_run_ids))

    def set_total_bandwidth(self, percentage: float) -> None:
        self.bandwidth_percentages.append(percentage)

    def start_run(self) -> str:
        self.run_count += 1
        run_id = f"run-{self.run_count}"
        self.active_run_ids.add(run_id)
        return run_id

    def find_running_runs(self, *, template: str, group: int) -> tuple[str, ...]:
        del template, group
        return (f"run-{self.run_count}",) if self.run_count > self.stop_count else ()

    def wait_for_completion(self, run_id: str, on_poll: Callable[[], None]) -> RunCompletion:
        self.clock.sleep(10)
        on_poll()
        self.active_run_ids.discard(run_id)
        return RunCompletion(terminal=True, details={"run_id": run_id, "result": "complete"})

    def wait_for_report(self, run_id: str) -> Any:
        del run_id
        return self.report_toc

    def export_report(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> Path:
        self.export_section_ids.append(section_ids)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "bps-performance-timeseries.csv":
            content = performance_report_csv()
        else:
            content = f"Test Model,performance-demo\nRun ID,{run_id}\nResult,passed\n"
        destination.write_text(content, encoding="utf-8")
        return destination

    def schedule_full_report_pdf(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> None:
        del run_id, destination, section_ids

        def export() -> None:
            self.pdf_started.set()
            if self.block_pdf_until_released:
                assert self.pdf_release.wait(5)
            self.pdf_finished.set()

        Thread(target=export, name="fake-bps-full-pdf", daemon=True).start()

    def release_ports(self, ports: tuple[int, ...] | None = None) -> None:
        self.release_count += 1
        selected_ports = ports or tuple(self.reservation_owners)
        self.released_port_sets.append(selected_ports)
        for port in selected_ports:
            self.reservation_owners[port] = None

    def stop_run(self, run_id: str) -> None:
        self.stop_count += 1
        self.active_run_ids.discard(run_id)

    def close(self) -> None:
        return None


class FakeDut:
    def __init__(
        self,
        *,
        empty_recovery: bool = False,
        missing_monitoring: bool = False,
        keepalive_failure: bool = False,
    ) -> None:
        self.empty_recovery = empty_recovery
        self.missing_monitoring = missing_monitoring
        self.keepalive_failure = keepalive_failure
        self.keepalive_calls = 0
        self.monitoring_calls = 0
        self.supplemental_calls = 0
        self.before: SupplementalSnapshot | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None

    def keepalive(self) -> None:
        self.keepalive_calls += 1
        if self.keepalive_failure:
            raise RuntimeError("fixture keepalive failed")

    def collect_monitoring_window(
        self,
        traffic_started_at: str,
        traffic_finished_at: str,
        before: SupplementalSnapshot,
        after: SupplementalSnapshot,
    ) -> tuple[ResourceObservation, ...]:
        self.monitoring_calls += 1
        assert traffic_started_at < traffic_finished_at
        assert before.is_complete and after.is_complete
        observations: list[ResourceObservation] = []
        for phase in ObservationPhase:
            has_points = not self.missing_monitoring and not (
                self.empty_recovery and phase == ObservationPhase.RECOVERY
            )
            observations.append(
                ResourceObservation(
                    phase=phase,
                    started_at="2026-08-24T00:00:00+00:00",
                    finished_at="2026-08-24T00:00:01+00:00",
                    resources={
                        "cpu": {
                            "code": 0,
                            "data": [{"time": "08-24 00:00:00", "percent": 20}]
                            if has_points
                            else [],
                        }
                    },
                )
            )
        return tuple(observations)

    def collect_supplemental(self) -> SupplementalSnapshot:
        self.supplemental_calls += 1
        return SupplementalSnapshot(
            captured_at="2026-08-24T00:00:00+00:00",
            values={"interfaces": {}, "hardware": {}, "system": {}},
        )

    def prepare_attempt(self, attempt_dir: Path) -> None:
        del attempt_dir
        self.before = self.collect_supplemental()

    def traffic_started(self, started_at: str) -> None:
        self.started_at = started_at
        with suppress(RuntimeError):
            self.keepalive()

    def restore_attempt(
        self,
        attempt_dir: Path,
        started_at: str,
        finished_at: str | None,
    ) -> None:
        del attempt_dir
        assert self.started_at == started_at
        if finished_at is not None:
            self.finished_at = finished_at

    def traffic_finished(self, finished_at: str) -> None:
        self.finished_at = finished_at

    def finalize_attempt(self) -> DutCaptureResult:
        assert self.started_at is not None and self.finished_at is not None
        assert self.before is not None
        after = self.collect_supplemental()
        observations = self.collect_monitoring_window(
            self.started_at,
            self.finished_at,
            self.before,
            after,
        )
        warnings = (
            ("DUT keepalive failed: fixture keepalive failed",) if self.keepalive_failure else ()
        )
        evidence = FrontendDutEvidence(
            endpoint="https://dut.example.test",
            interfaces=("T1/1", "T1/2"),
            traffic_started_at=self.started_at,
            traffic_finished_at=self.finished_at,
            observations=DutObservations.from_resource_observations(observations),
            before=self.before,
            after=after,
            warnings=warnings,
        )
        return DutCaptureResult(evidence=evidence, warnings=warnings)

    def close(self) -> None:
        return None


class FakeBackendDut:
    def __init__(self, successful_samples: int = 1) -> None:
        self.successful_samples = successful_samples
        self.started_at: str | None = None
        self.finished_at: str | None = None

    def prepare_attempt(self, attempt_dir: Path) -> None:
        attempt_dir.mkdir(parents=True, exist_ok=True)

    def traffic_started(self, started_at: str) -> None:
        self.started_at = started_at

    def restore_attempt(
        self,
        attempt_dir: Path,
        started_at: str,
        finished_at: str | None,
    ) -> None:
        del attempt_dir
        assert self.started_at == started_at
        if finished_at is not None:
            self.finished_at = finished_at

    def traffic_finished(self, finished_at: str) -> None:
        self.finished_at = finished_at

    def finalize_attempt(self) -> DutCaptureResult:
        assert self.started_at is not None and self.finished_at is not None
        csv_text = "sample_index,time_origin,elapsed_seconds,cpu_mgt_percent\n"
        if self.successful_samples:
            csv_text += f"1,{self.started_at},0,11\n"
        evidence = BackendDutEvidence(
            target=BackendDutTarget(host="10.66.246.133", port=50023),
            interfaces=("T1/1", "T1/2"),
            traffic_started_at=self.started_at,
            traffic_finished_at=self.finished_at,
            interval_seconds=10,
            successful_sample_count=self.successful_samples,
            failed_sample_count=0,
            metrics_csv=csv_text,
        )
        return DutCaptureResult(evidence=evidence)

    def close(self) -> None:
        return None


class FakeJudge:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, verdicts: list[VerdictValue], *, failure: bool = False) -> None:
        self.verdicts = verdicts
        self.failure = failure
        self.calls = 0

    def adjudicate(self, evidence: Any) -> tuple[VerdictDocument, dict[str, Any]]:
        self.calls += 1
        if self.failure:
            raise RuntimeError("model unavailable")
        verdict = self.verdicts.pop(0)
        return VerdictDocument(verdict=verdict, summary="fixture"), {"fixture": True}

    def close(self) -> None:
        return None


def run_graph(
    config: AppConfig,
    bps: FakeBps,
    dut: FakeDut | None,
    judge: FakeJudge,
    clock: FakeClock,
) -> dict[str, Any]:
    graph = build_graph(
        EvaluationServices(
            config=config,
            bps=bps,
            dut=dut,
            judge=judge,
            artifacts=ArtifactStore(config.storage.artifact_dir),
            clock=clock,
        )
    )
    return graph.invoke(initial_state("evaluation-1", config))


def test_bps_only_mode_never_calls_dut_and_omits_dut_evidence(app_config: AppConfig) -> None:
    config = app_config.model_copy(
        update={
            "evaluation": app_config.evaluation.model_copy(update={"mode": EvaluationMode.BPS_ONLY})
        }
    )
    clock = FakeClock()
    dut = FakeDut()
    bps = FakeBps(clock)

    result = run_graph(
        config,
        bps,
        dut,
        FakeJudge([VerdictValue.PASS]),
        clock,
    )

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert bps.run_count == 1
    attempt = result["attempts"][0]
    assert attempt["evidence_complete"] is True
    evidence_path = Path(attempt["evidence_path"])
    evidence = ArtifactStore.read_json(evidence_path)
    assert evidence["evaluation_mode"] == "bps_only"
    assert evidence["assessment"] == config.bps_only_assessment.model_dump(mode="json")
    assert not any(name.startswith("dut_") for name in evidence)
    assert not (evidence_path.parent / "dut-observations.json").exists()
    assert clock.sleeps == [10]
    assert dut.keepalive_calls == 0
    assert dut.monitoring_calls == 0
    assert dut.supplemental_calls == 0


def test_passes_on_first_complete_attempt(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    dut = FakeDut()
    judge = FakeJudge([VerdictValue.PASS])

    result = run_graph(app_config, bps, dut, judge, clock)

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert bps.run_count == 1
    assert bps.bandwidth_percentages == [100.0]
    assert bps.reserve_count == bps.release_count == 1
    assert dut.monitoring_calls == 1
    assert dut.supplemental_calls == 2
    assert dut.keepalive_calls == 1
    assert judge.calls == 1
    assert Path(result["final_artifact"]).exists()


def test_pdf_export_does_not_block_evaluation(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock, block_pdf_until_released=True)

    try:
        result = run_graph(
            app_config,
            bps,
            FakeDut(),
            FakeJudge([VerdictValue.PASS]),
            clock,
        )

        assert bps.pdf_started.is_set()
        assert not bps.pdf_finished.is_set()
        assert result["outcome"] == EvaluationOutcome.PASSED.value
        assert Path(result["final_artifact"]).exists()
    finally:
        bps.pdf_release.set()
    assert bps.pdf_finished.wait(1)


def test_backend_without_a_successful_sample_is_inconclusive(
    app_config: AppConfig,
) -> None:
    config = app_config.model_copy(
        update={
            "dut": app_config.dut.model_copy(
                update={"collection_method": DutCollectionMethod.BACKEND_SSH}
            )
        }
    )
    clock = FakeClock()
    judge = FakeJudge([VerdictValue.PASS])
    result = run_graph(
        config,
        FakeBps(clock),
        FakeBackendDut(successful_samples=0),
        judge,
        clock,
    )

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert judge.calls == 0


def test_backend_worker_stop_timeout_makes_attempt_inconclusive(
    app_config: AppConfig,
) -> None:
    class TimedOutBackendDut(FakeBackendDut):
        def traffic_finished(self, finished_at: str) -> None:
            super().traffic_finished(finished_at)
            raise RuntimeError("collector worker did not stop")

        def finalize_attempt(self) -> DutCaptureResult:
            raise RuntimeError("collector worker did not stop")

    config = app_config.model_copy(
        update={
            "dut": app_config.dut.model_copy(
                update={"collection_method": DutCollectionMethod.BACKEND_SSH}
            )
        }
    )
    clock = FakeClock()
    judge = FakeJudge([VerdictValue.PASS])

    result = run_graph(
        config,
        FakeBps(clock),
        TimedOutBackendDut(),
        judge,
        clock,
    )

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert judge.calls == 0
    assert any(
        "collector worker did not stop" in error for error in result["attempts"][0]["errors"]
    )


def test_can_stop_after_evidence_without_calling_the_llm(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    dut = FakeDut()
    judge = FakeJudge([VerdictValue.PASS])
    graph = build_graph(
        EvaluationServices(
            config=app_config,
            bps=bps,
            dut=dut,
            judge=judge,
            artifacts=ArtifactStore(app_config.storage.artifact_dir),
            clock=clock,
        ),
        interrupt_before=["adjudicate"],
    )

    result = graph.invoke(initial_state("evidence-only", app_config))

    assert result["outcome"] is None
    assert result["attempts"][0]["evidence_complete"] is True
    evidence_path = Path(result["attempts"][0]["evidence_path"])
    assert evidence_path.exists()
    assert judge.calls == 0


def test_missing_required_dynamic_report_sections_is_inconclusive(
    app_config: AppConfig,
) -> None:
    clock = FakeClock()
    bps = FakeBps(
        clock,
        report_toc={"sections": [{"sectionId": "1", "sectionName": "Appendix"}]},
    )

    result = run_graph(app_config, bps, FakeDut(), FakeJudge([VerdictValue.PASS]), clock)

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert bps.export_section_ids == []
    toc_path = Path(result["attempts"][0]["report_toc_path"])
    assert toc_path.exists()
    assert (toc_path.parent / "bps-report-sections.json").exists()


def test_dut_keepalive_failure_is_a_non_fatal_attempt_warning(
    app_config: AppConfig,
) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    dut = FakeDut(keepalive_failure=True)
    judge = FakeJudge([VerdictValue.PASS])

    result = run_graph(app_config, bps, dut, judge, clock)

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert dut.keepalive_calls == 1
    assert bps.stop_count == 0
    assert any("DUT keepalive failed" in error for error in result["attempts"][0]["errors"])


def test_stale_agent_reservation_is_released_only_when_no_active_run_exists(
    app_config: AppConfig,
) -> None:
    bps = FakeBps(FakeClock())
    bps.reservation_owner = "agent-user"

    result = run_graph(
        app_config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.PASS]),
        bps.clock,
    )

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert bps.reserve_count == 1
    assert bps.release_count == 2  # stale reservation, then the completed BPS Run


def test_partial_agent_reservation_without_active_run_is_cleaned_then_reserved(
    app_config: AppConfig,
) -> None:
    bps = FakeBps(FakeClock())
    bps.reservation_owners = {4: "agent-user", 5: None}

    result = run_graph(
        app_config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.PASS]),
        bps.clock,
    )

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert bps.reserve_count == 1
    assert bps.released_port_sets == [(4,), (4, 5)]
    assert result["attempts"][0]["port_reservation_state"] == PortReservationState.NONE


def test_partial_agent_reservation_with_active_run_is_inconclusive(
    app_config: AppConfig,
) -> None:
    bps = FakeBps(FakeClock())
    bps.reservation_owners = {4: "agent-user", 5: None}
    bps.active_run_ids.add("partial-active-run")

    result = run_graph(
        app_config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.PASS]),
        bps.clock,
    )

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert "partially reserved" in result["error"]
    assert bps.reserve_count == bps.release_count == 0


def test_agent_reservation_with_active_run_is_not_unreserved_without_journal(
    app_config: AppConfig,
) -> None:
    bps = FakeBps(FakeClock())
    bps.reservation_owner = "agent-user"
    bps.active_run_ids.add("external-active-run")

    result = run_graph(
        app_config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.PASS]),
        bps.clock,
    )

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert "active Run(s) exist" in result["error"]
    assert bps.reserve_count == bps.release_count == 0


def test_other_account_reservation_is_never_unreserved(app_config: AppConfig) -> None:
    bps = FakeBps(FakeClock())
    bps.reservation_owner = "another-user"

    result = run_graph(
        app_config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.PASS]),
        bps.clock,
    )

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert "reserved by another account" in result["error"]
    assert bps.reserve_count == bps.release_count == 0


def test_resume_reconciles_unreserved_ports_instead_of_trusting_checkpoint(
    app_config: AppConfig,
) -> None:
    bps = FakeBps(FakeClock())
    with SqliteSaver.from_conn_string(str(app_config.storage.checkpoint_db)) as saver:
        graph = build_graph(
            EvaluationServices(
                config=app_config,
                bps=bps,
                dut=FakeDut(),
                judge=FakeJudge([VerdictValue.PASS]),
                artifacts=ArtifactStore(app_config.storage.artifact_dir),
                clock=bps.clock,
            ),
            checkpointer=saver,
        )
        invocation = {"configurable": {"thread_id": "reservation-recovery"}}
        stream = graph.stream(initial_state("reservation-recovery", app_config), config=invocation)
        for event in stream:
            if "start_attempt" in event:
                break
        stream.close()
        assert (
            graph.get_state(invocation).values["attempts"][-1]["port_reservation_state"]
            == PortReservationState.ALL_AGENT
        )

        bps.reservation_owner = None
        result = graph.invoke(None, config=invocation)

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert result["attempts"][0]["port_reservation_state"] == PortReservationState.NONE
    assert bps.release_count == 0


def test_port_release_failure_does_not_erase_confirmed_run_terminal_state(
    app_config: AppConfig,
) -> None:
    class ReleaseFailingBps(FakeBps):
        def release_ports(self) -> None:
            self.release_count += 1
            raise RuntimeError("release retries exhausted")

    clock = FakeClock()
    bps = ReleaseFailingBps(clock)

    result = run_graph(
        app_config,
        bps,
        FakeDut(keepalive_failure=True),
        FakeJudge([VerdictValue.PASS]),
        clock,
    )

    attempt = result["attempts"][0]
    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert attempt["terminal_confirmed"] is True
    assert attempt["traffic_finished_at"] is not None
    assert attempt["port_reservation_state"] == PortReservationState.ALL_AGENT
    assert bps.stop_count == 0
    assert result["error"].startswith("BPS port release failed after confirmed terminal state:")


def test_six_retry_verdicts_reduce_bandwidth_then_end_not_passed(
    app_config: AppConfig,
) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    judge = FakeJudge([VerdictValue.RETRY] * 6)

    result = run_graph(app_config, bps, FakeDut(), judge, clock)

    assert result["outcome"] == EvaluationOutcome.NOT_PASSED.value
    assert bps.run_count == 6
    assert bps.reserve_count == bps.release_count == 6
    assert bps.bandwidth_percentages == [100.0, 80.0, 60.0, 40.0, 20.0, 10.0]
    assert judge.calls == 6
    assert len(result["attempts"]) == 6
    assert [attempt["bps_total_bandwidth_mbps"] for attempt in result["attempts"]] == [
        400.0,
        320.0,
        240.0,
        160.0,
        80.0,
        40.0,
    ]


def test_target_above_template_bandwidth_fails_before_reserving_ports(
    app_config: AppConfig,
) -> None:
    config = app_config.model_copy(
        update={"bps": app_config.bps.model_copy(update={"total_bandwidth_mbps": 500.0})}
    )
    clock = FakeClock()
    bps = FakeBps(clock, template_bandwidth_mbps=400.0)

    result = run_graph(
        config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.PASS]),
        clock,
    )

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert "exceeds template original value" in result["error"]
    assert bps.reserve_count == bps.run_count == 0


def test_pass_after_reduced_bandwidth_is_degraded_pass(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)

    result = run_graph(
        app_config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.RETRY, VerdictValue.RETRY, VerdictValue.PASS]),
        clock,
    )

    assert result["outcome"] == EvaluationOutcome.DEGRADED_PASS.value
    assert bps.bandwidth_percentages == [100.0, 80.0, 60.0]
    assert bps.run_count == 3


def test_missing_monitoring_is_inconclusive_without_another_bps_run(
    app_config: AppConfig,
) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    judge = FakeJudge([VerdictValue.PASS])

    result = run_graph(app_config, bps, FakeDut(missing_monitoring=True), judge, clock)

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert bps.run_count == 1
    assert judge.calls == 0


def test_missing_recovery_points_do_not_block_adjudication(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    judge = FakeJudge([VerdictValue.PASS])

    result = run_graph(app_config, bps, FakeDut(empty_recovery=True), judge, clock)

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert bps.run_count == 1
    assert judge.calls == 1


def test_llm_failure_does_not_launch_another_bps_run(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    judge = FakeJudge([], failure=True)

    result = run_graph(app_config, bps, FakeDut(), judge, clock)

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert bps.run_count == 1
    assert judge.calls == 1


def test_preflight_failure_does_not_create_an_attempt(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock, template_error=True)

    result = run_graph(app_config, bps, FakeDut(), FakeJudge([VerdictValue.PASS]), clock)

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert result["attempts"] == []
    assert bps.run_count == 0


def test_sqlite_checkpoint_can_resume_after_bps_run_creation(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    judge = FakeJudge([VerdictValue.PASS])
    with SqliteSaver.from_conn_string(str(app_config.storage.checkpoint_db)) as saver:
        graph = build_graph(
            EvaluationServices(
                config=app_config,
                bps=bps,
                dut=FakeDut(),
                judge=judge,
                artifacts=ArtifactStore(app_config.storage.artifact_dir),
                clock=clock,
            ),
            checkpointer=saver,
        )
        invocation_config = {"configurable": {"thread_id": "resumable-evaluation"}}
        stream = graph.stream(
            initial_state("resumable-evaluation", app_config), config=invocation_config
        )
        for event in stream:
            if "start_attempt" in event:
                break
        stream.close()

        snapshot = graph.get_state(invocation_config)
        assert snapshot.next == ("monitor_attempt",)
        result = graph.invoke(None, config=invocation_config)

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert bps.run_count == 1


def test_launch_journal_prevents_duplicate_after_node_crash(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    dut = FakeDut()
    judge = FakeJudge([VerdictValue.PASS])
    artifacts = ArtifactStore(app_config.storage.artifact_dir)
    original_write_attempt_json = artifacts.write_attempt_json
    crashed = False

    def crash_before_node_checkpoint(
        evaluation_id: str,
        attempt_number: int,
        name: str,
        value: Any,
    ) -> Path:
        nonlocal crashed
        if name == "attempt.json" and not crashed:
            crashed = True
            raise SystemExit("simulated kill after durable run ID")
        return original_write_attempt_json(evaluation_id, attempt_number, name, value)

    artifacts.write_attempt_json = crash_before_node_checkpoint  # type: ignore[method-assign]
    with SqliteSaver.from_conn_string(str(app_config.storage.checkpoint_db)) as saver:
        graph = build_graph(
            EvaluationServices(
                config=app_config,
                bps=bps,
                dut=dut,
                judge=judge,
                artifacts=artifacts,
                clock=clock,
            ),
            checkpointer=saver,
        )
        invocation = {"configurable": {"thread_id": "launch-crash"}}
        with pytest.raises(SystemExit, match="simulated kill"):
            graph.invoke(initial_state("launch-crash", app_config), config=invocation)

        artifacts.write_attempt_json = original_write_attempt_json  # type: ignore[method-assign]
        result = graph.invoke(None, config=invocation)

    assert result["outcome"] == EvaluationOutcome.PASSED.value
    assert bps.run_count == 1
    launch = ArtifactStore.read_json(
        app_config.storage.artifact_dir / "launch-crash" / "attempt-01" / "bps-launch.json"
    )
    assert launch["status"] == "released"
