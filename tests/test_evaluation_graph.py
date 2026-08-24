from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from bps_agent.artifacts import ArtifactStore
from bps_agent.graph import EvaluationServices, build_graph, initial_state
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


class FakeBps:
    def __init__(self, clock: FakeClock, *, template_error: bool = False) -> None:
        self.clock = clock
        self.template_error = template_error
        self.run_count = 0
        self.reserve_count = 0
        self.release_count = 0
        self.stop_count = 0

    def find_template(self, name: str) -> dict[str, Any]:
        if self.template_error:
            raise RuntimeError("missing template")
        return {"name": name, "version": 1}

    def reserve_ports(self) -> None:
        self.reserve_count += 1

    def start_run(self) -> str:
        self.run_count += 1
        return f"run-{self.run_count}"

    def wait_for_completion(self, run_id: str, on_poll: Callable[[], None]) -> RunCompletion:
        self.clock.sleep(10)
        on_poll()
        return RunCompletion(terminal=True, details={"run_id": run_id, "result": "complete"})

    def wait_for_report(self, run_id: str) -> Any:
        return {"run_id": run_id, "sections": ["3.2"]}

    def export_report(self, run_id: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            f"Test Model,performance-demo\nRun ID,{run_id}\nResult,passed\n",
            encoding="utf-8",
        )
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
    assert bps.reserve_count == bps.release_count == 1
    assert dut.monitoring_calls == 1
    assert dut.supplemental_calls == 2
    assert dut.keepalive_calls == 1
    assert judge.calls == 1
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
    assert "bps_report_toc" not in evidence
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


def test_three_retry_verdicts_end_not_passed(app_config: AppConfig) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    judge = FakeJudge([VerdictValue.RETRY] * 3)

    result = run_graph(app_config, bps, FakeDut(), judge, clock)

    assert result["outcome"] == EvaluationOutcome.NOT_PASSED.value
    assert bps.run_count == 3
    assert bps.reserve_count == bps.release_count == 3
    assert judge.calls == 3
    assert len(result["attempts"]) == 3


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
