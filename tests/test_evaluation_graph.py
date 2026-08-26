from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from bps_agent.artifacts import ArtifactStore
from bps_agent.graph import (
    EvaluationServices,
    build_graph,
    initial_state,
)
from bps_agent.models import (
    AppConfig,
    EvaluationOutcome,
    EvidenceBundle,
    ObservationPhase,
    ResourceObservation,
    RunCompletion,
    SupplementalSnapshot,
    VerdictDocument,
    VerdictValue,
)


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
        self.stop_count = 0
        self.export_section_ids: list[tuple[str, ...]] = []
        self.report_run_ids: list[str] = []
        self.export_run_ids: list[str] = []
        self.bandwidth_percentages: list[float] = []

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

    def set_total_bandwidth(self, percentage: float) -> None:
        self.bandwidth_percentages.append(percentage)

    def start_run(self) -> str:
        self.run_count += 1
        return f"run-{self.run_count}"

    def wait_for_completion(self, run_id: str, on_poll: Callable[[], None]) -> RunCompletion:
        self.clock.sleep(10)
        on_poll()
        return RunCompletion(terminal=True, details={"run_id": run_id, "result": "complete"})

    def wait_for_report(self, run_id: str) -> Any:
        self.report_run_ids.append(run_id)
        return self.report_toc

    def export_report(
        self,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> Path:
        self.export_run_ids.append(run_id)
        self.export_section_ids.append(section_ids)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "bps-performance-timeseries.csv":
            content = performance_report_csv()
        else:
            content = f"Test Model,performance-demo\nRun ID,{run_id}\nResult,passed\n"
        destination.write_text(content, encoding="utf-8")
        return destination

    def release_ports(self) -> None:
        self.release_count += 1

    def stop_run(self, run_id: str) -> None:
        self.stop_count += 1


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


def run_graph(
    config: AppConfig,
    bps: FakeBps,
    dut: FakeDut,
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
    assert bps.report_run_ids == ["run-1"]
    assert bps.export_run_ids == ["run-1", "run-1"]
    assert bps.export_section_ids == [
        (
            "10.4",
            "10.5",
            "10.6",
            "12.8",
            "20.3",
            "20.9",
            "30.2",
        ),
        ("30.4.5", "30.4.7", "30.4.8"),
    ]
    assert Path(result["final_artifact"]).exists()


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
    evidence = ArtifactStore.read_json(evidence_path)
    serialized_evidence = evidence_path.read_text(encoding="utf-8")
    assert "bps_report_toc" not in evidence
    assert "performance_timeseries_path" not in evidence
    assert "bps-performance-timeseries.csv" not in serialized_evidence
    assert "Timestamp" not in evidence["bps_report"]
    assert evidence["bps_performance_analysis"]["assessment"] == "normal"
    assert evidence["bps_template_total_bandwidth_mbps"] == 400.0
    assert evidence["bps_total_bandwidth_mbps"] == 400.0
    assert evidence["bps_total_bandwidth_percent"] == 100.0
    assert "aligned_samples" not in evidence["bps_performance_analysis"]
    performance_path = Path(result["attempts"][0]["performance_timeseries_path"])
    assert performance_path.name == "bps-performance-timeseries.csv"
    assert performance_path.exists()
    assert "Timestamp" in performance_path.read_text(encoding="utf-8")
    observations = evidence["dut_observations"]
    assert observations["resources"]["cpu"]["metadata"] == {"code": 0}
    assert len(observations["resources"]["cpu"]["points"]["baseline"]) == 1
    assert len(observations["resources"]["cpu"]["points"]["during"]) == 1
    assert ArtifactStore.read_json(evidence_path.parent / "dut-observations.json") == observations
    legacy_evidence = EvidenceBundle.model_validate(
        {
            **evidence,
            "bps_report_toc": [{"legacy": True}],
            "dut_observations": result["attempts"][0]["dut_observations"],
        }
    )
    upgraded = legacy_evidence.model_dump(mode="json")
    assert "bps_report_toc" not in upgraded
    assert isinstance(upgraded["dut_observations"], dict)
    assert Path(result["attempts"][0]["report_toc_path"]).exists()
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


def test_port_release_failure_does_not_erase_confirmed_run_terminal_state(
    app_config: AppConfig,
) -> None:
    class ReleaseFailingBps(FakeBps):
        def release_ports(self) -> None:
            super().release_ports()
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
    assert attempt["ports_reserved"] is True
    assert bps.stop_count == 0
    assert result["error"].startswith("BPS port release failed after confirmed terminal state:")


def test_five_retry_verdicts_reduce_bandwidth_then_end_not_passed(
    app_config: AppConfig,
) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    judge = FakeJudge([VerdictValue.RETRY] * 5)

    result = run_graph(app_config, bps, FakeDut(), judge, clock)

    assert result["outcome"] == EvaluationOutcome.NOT_PASSED.value
    assert bps.run_count == 5
    assert bps.reserve_count == bps.release_count == 5
    assert bps.bandwidth_percentages == [100.0, 80.0, 60.0, 40.0, 20.0]
    assert judge.calls == 5
    assert len(result["attempts"]) == 5
    assert [attempt["bps_total_bandwidth_mbps"] for attempt in result["attempts"]] == [
        400.0,
        320.0,
        240.0,
        160.0,
        80.0,
    ]


def test_retry_bandwidth_levels_are_relative_to_configured_initial_target(
    app_config: AppConfig,
) -> None:
    config = app_config.model_copy(
        update={"bps": app_config.bps.model_copy(update={"total_bandwidth_mbps": 300.0})}
    )
    clock = FakeClock()
    bps = FakeBps(clock)

    result = run_graph(
        config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.RETRY] * 5),
        clock,
    )

    assert result["outcome"] == EvaluationOutcome.NOT_PASSED.value
    assert bps.bandwidth_percentages == [75.0, 60.0, 45.0, 30.0, 15.0]
    assert [attempt["bps_total_bandwidth_mbps"] for attempt in result["attempts"]] == [
        300.0,
        240.0,
        180.0,
        120.0,
        60.0,
    ]


def test_bandwidth_percentage_uses_template_json_original_value(
    app_config: AppConfig,
) -> None:
    clock = FakeClock()
    bps = FakeBps(clock, template_bandwidth_mbps=800.0)

    result = run_graph(
        app_config,
        bps,
        FakeDut(),
        FakeJudge([VerdictValue.RETRY] * 5),
        clock,
    )

    assert result["outcome"] == EvaluationOutcome.NOT_PASSED.value
    assert bps.bandwidth_percentages == [50.0, 40.0, 30.0, 20.0, 10.0]
    assert all(
        attempt["bps_template_total_bandwidth_mbps"] == 800.0 for attempt in result["attempts"]
    )


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
