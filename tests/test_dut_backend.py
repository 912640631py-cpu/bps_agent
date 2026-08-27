from __future__ import annotations

import csv
import io
import json
import threading
from pathlib import Path
from typing import Any

import paramiko

from bps_agent.adapters.dut_backend import (
    DutBackendCollector,
    DutSshBackendClient,
    _extract_json,
    render_metrics_csv,
)
from bps_agent.models import DutBackendConfig, DutCollectionMethod, DutConfig


def sample_snapshot() -> dict[str, Any]:
    return {
        "collected_at": "2026-08-27T13:53:43+08:00",
        "cpu": {
            "latest": {
                "timestamp": "1787809980",
                "time": "08-27 13:53:00",
                "mgt": 11,
                "data": 1,
            },
            "threshold": {"mgt": 90, "data": 90},
        },
        "memory": {
            "latest": {"time": "08-27 13:53:43", "percent": 71},
            "threshold": 66,
        },
        "new_sessions": {"time": "08-27 13:53:16", "count": 3},
        "concurrent_sessions": {"time": "08-27 13:53:16", "count": 25},
        "traffic": {
            "T1/1": {"time": "08-27 13:52:59", "obps": 10, "ibps": 20},
            "T1/2": {"time": "08-27 13:52:59", "obps": 30, "ibps": 40},
        },
    }


def backend_config(*, interval_seconds: float = 0.01) -> DutConfig:
    return DutConfig(
        collection_method=DutCollectionMethod.BACKEND_SSH,
        interfaces=("T1/1", "T1/2"),
        backend=DutBackendConfig(
            host="10.66.246.156",
            port=50023,
            interval_seconds=interval_seconds,
            read_attempts=1,
        ),
    )


def test_extract_json_accepts_an_informational_prefix() -> None:
    assert _extract_json("PHP notice\n" + json.dumps(sample_snapshot())) == sample_snapshot()


def test_csv_is_one_compact_row_per_sample_with_dynamic_interfaces() -> None:
    text = render_metrics_csv(
        [
            {
                "started_at": "2026-08-27T13:53:33+08:00",
                "finished_at": "2026-08-27T13:53:34+08:00",
                "resources": sample_snapshot(),
            }
        ],
        ("T1/1", "T1/2"),
    )

    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 1
    assert rows[0]["cpu_mgt_percent"] == "11"
    assert rows[0]["memory_threshold_percent"] == "66"
    assert rows[0]["traffic[T1/1].ibps"] == "20"
    assert rows[0]["traffic[T1/2].obps"] == "30"


class FakeSnapshotClient:
    def __init__(self, results: list[dict[str, Any] | Exception]) -> None:
        self.results = results
        self.connected = 0
        self.closed = 0
        self.cleared = 0
        self.calls = 0
        self.ready = threading.Event()

    def connect(self) -> None:
        self.connected += 1

    def close(self) -> None:
        self.closed += 1

    def clear_credentials(self) -> None:
        self.cleared += 1

    def read_snapshot(self, interfaces: tuple[str, ...]) -> dict[str, Any]:
        assert interfaces == ("T1/1", "T1/2")
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if self.calls >= len(self.results):
            self.ready.set()
        if isinstance(result, Exception):
            raise result
        return result


def test_collector_checkpoints_samples_and_sends_only_csv_to_evidence(tmp_path: Path) -> None:
    client = FakeSnapshotClient([sample_snapshot(), sample_snapshot()])
    collector = DutBackendCollector(
        backend_config(),
        username="dutcollector",
        password="never-write-this",
        client=client,  # type: ignore[arg-type]
    )

    collector.prepare_attempt(tmp_path)
    collector.traffic_started("2026-08-27T13:53:30+08:00")
    assert client.ready.wait(1)
    collector.traffic_finished("2026-08-27T13:54:00+08:00")
    capture = collector.finalize_attempt()

    raw_path = Path(capture.raw_artifact_path or "")
    csv_path = Path(capture.csv_artifact_path or "")
    raw_text = raw_path.read_text(encoding="utf-8")
    assert raw_path.name == "dut-metrics.json"
    assert csv_path.name == "dut-metrics.csv"
    assert "dutcollector" not in raw_text
    assert "never-write-this" not in raw_text
    assert capture.evidence.successful_sample_count >= 2  # type: ignore[union-attr]
    assert "cpu_mgt_percent" in capture.evidence.metrics_csv  # type: ignore[union-attr]
    assert "samples" not in capture.evidence.model_dump()  # type: ignore[union-attr]
    collector.close()
    assert client.cleared == 1


def test_failed_sample_is_recorded_and_collection_continues(tmp_path: Path) -> None:
    client = FakeSnapshotClient([RuntimeError("temporary failure"), sample_snapshot()])
    collector = DutBackendCollector(
        backend_config(),
        username="dutcollector",
        password="password",
        client=client,  # type: ignore[arg-type]
    )

    collector.prepare_attempt(tmp_path)
    collector.traffic_started("2026-08-27T13:53:30+08:00")
    assert client.ready.wait(1)
    collector.traffic_finished("2026-08-27T13:54:00+08:00")
    capture = collector.finalize_attempt()

    assert capture.evidence.successful_sample_count >= 1  # type: ignore[union-attr]
    assert capture.evidence.failed_sample_count == 1  # type: ignore[union-attr]
    assert "temporary failure" in capture.warnings[0]


def test_completed_attempt_can_restore_from_atomic_metrics_artifact(tmp_path: Path) -> None:
    first_client = FakeSnapshotClient([sample_snapshot()])
    first = DutBackendCollector(
        backend_config(),
        username="dutcollector",
        password="password",
        client=first_client,  # type: ignore[arg-type]
    )
    started_at = "2026-08-27T13:53:30+08:00"
    finished_at = "2026-08-27T13:54:00+08:00"
    first.prepare_attempt(tmp_path)
    first.traffic_started(started_at)
    assert first_client.ready.wait(1)
    first.traffic_finished(finished_at)
    first.finalize_attempt()

    restored_client = FakeSnapshotClient([sample_snapshot()])
    restored = DutBackendCollector(
        backend_config(),
        username="dutcollector",
        password="password",
        client=restored_client,  # type: ignore[arg-type]
    )
    restored.restore_attempt(tmp_path, started_at, finished_at)
    capture = restored.finalize_attempt()

    assert capture.evidence.successful_sample_count >= 1  # type: ignore[union-attr]
    assert restored_client.connected == 0


def test_ssh_client_does_not_load_or_persist_host_keys(monkeypatch: object) -> None:
    calls: dict[str, Any] = {}

    class Transport:
        def set_keepalive(self, seconds: int) -> None:
            calls["keepalive"] = seconds

    class SshClient:
        def set_missing_host_key_policy(self, policy: object) -> None:
            calls["policy"] = policy

        def connect(self, **kwargs: Any) -> None:
            calls["connect"] = kwargs

        def get_transport(self) -> Transport:
            return Transport()

        def close(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(paramiko, "SSHClient", SshClient)  # type: ignore[attr-defined]
    client = DutSshBackendClient(
        backend_config(),
        username="dutcollector",
        password="password",
    )

    client.connect()

    assert isinstance(calls["policy"], paramiko.AutoAddPolicy)
    assert calls["connect"]["look_for_keys"] is False
    assert calls["connect"]["allow_agent"] is False
