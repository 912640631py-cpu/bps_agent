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
    def __init__(self, *, incomplete_recovery: bool = False) -> None:
        self.incomplete_recovery = incomplete_recovery
        self.resource_calls: list[ObservationPhase] = []
        self.supplemental_calls = 0

    def collect_resources(self, phase: str) -> ResourceObservation:
        observation_phase = ObservationPhase(phase)
        self.resource_calls.append(observation_phase)
        resources: dict[str, Any] = {
            "cpu": {"code": 0, "data": [{"percent": 20}]},
            "memory": {"code": 0, "data": [{"percent": 30}]},
            "new_sessions": {"code": 0, "data": [{"count": 10}]},
            "concurrent_sessions": {"code": 0, "data": [{"count": 100}]},
            "traffic": {"T1/1": {"code": 0}, "T1/2": {"code": 0}},
        }
        if self.incomplete_recovery and observation_phase == ObservationPhase.RECOVERY:
            resources.pop("cpu")
        return ResourceObservation(
            phase=observation_phase,
            started_at="2026-08-24T00:00:00+00:00",
            finished_at="2026-08-24T00:00:01+00:00",
            resources=resources,
        )

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
    assert judge.calls == 1
    assert Path(result["final_artifact"]).exists()


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


def test_incomplete_evidence_uses_three_attempts_without_calling_llm(
    app_config: AppConfig,
) -> None:
    clock = FakeClock()
    bps = FakeBps(clock)
    judge = FakeJudge([VerdictValue.PASS])

    result = run_graph(app_config, bps, FakeDut(incomplete_recovery=True), judge, clock)

    assert result["outcome"] == EvaluationOutcome.INCONCLUSIVE.value
    assert bps.run_count == 3
    assert judge.calls == 0


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
