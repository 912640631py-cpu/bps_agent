from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bps_agent.artifacts import ArtifactStore
from bps_agent.launch import LaunchReconciliationError, RunLaunchCoordinator


class LaunchingBps:
    def __init__(self) -> None:
        self.start_calls = 0
        self.running: tuple[str, ...] = ()
        self.raise_after_start = False

    def start_run(self) -> str:
        self.start_calls += 1
        run_id = f"run-{self.start_calls}"
        if self.raise_after_start:
            raise RuntimeError("connection lost after request")
        self.running = (run_id,)
        return run_id

    def find_running_runs(self, *, template: str, group: int) -> tuple[str, ...]:
        assert template == "template"
        assert group == 10
        return self.running

    def close(self) -> None:
        return None


def coordinator(tmp_path: Path, bps: LaunchingBps) -> RunLaunchCoordinator:
    return RunLaunchCoordinator(bps, ArtifactStore(tmp_path))  # type: ignore[arg-type]


def test_known_run_id_is_reused_without_another_start(tmp_path: Path) -> None:
    bps = LaunchingBps()
    first = coordinator(tmp_path, bps).start("evaluation", 1, template="template", group=10)
    recovered = coordinator(tmp_path, bps).recover("evaluation", 1, template="template", group=10)

    assert recovered == first
    assert bps.start_calls == 1


def test_response_to_journal_crash_is_reconciled_by_running_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bps = LaunchingBps()
    store = ArtifactStore(tmp_path)
    launcher = RunLaunchCoordinator(bps, store)  # type: ignore[arg-type]
    original = store.write_attempt_json

    def crash_on_started(
        evaluation_id: str,
        attempt_number: int,
        name: str,
        value: Any,
    ) -> Path:
        if name == "bps-launch.json" and getattr(value, "status", None) == "started":
            raise SystemExit("simulated kill before run ID journal commit")
        return original(evaluation_id, attempt_number, name, value)

    monkeypatch.setattr(store, "write_attempt_json", crash_on_started)
    with pytest.raises(SystemExit, match="simulated kill"):
        launcher.start("evaluation", 1, template="template", group=10)

    monkeypatch.setattr(store, "write_attempt_json", original)
    recovered = launcher.recover("evaluation", 1, template="template", group=10)

    assert recovered is not None and recovered.run_id == "run-1"
    assert bps.start_calls == 1


def test_ambiguous_request_is_never_relaunched(tmp_path: Path) -> None:
    bps = LaunchingBps()
    bps.raise_after_start = True
    launcher = coordinator(tmp_path, bps)

    with pytest.raises(LaunchReconciliationError, match="ambiguous"):
        launcher.start("evaluation", 1, template="template", group=10)
    with pytest.raises(LaunchReconciliationError, match="refusing to start duplicate"):
        launcher.recover("evaluation", 1, template="template", group=10)

    assert bps.start_calls == 1
