from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bps_agent.adapters.bps import BpsClient
from bps_agent.artifacts import ArtifactStore
from bps_agent.models import BpsConfig
from bps_agent.pdf_worker import PdfExportJob, run_pdf_job


def bps_config() -> BpsConfig:
    return BpsConfig(
        endpoint="https://bps.example.test",
        template="template",
        slot=4,
        ports=(4, 5),
        group=10,
    )


def test_scheduler_persists_secret_free_job_and_launches_isolated_process(
    tmp_path: Path, monkeypatch: Any
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-leak")
    monkeypatch.setenv("DUT_BACKEND_PASSWORD", "must-not-leak-either")
    monkeypatch.setenv("UNRELATED_SETTING", "must-not-leak")

    def popen(arguments: list[str], **kwargs: Any) -> object:
        captured["arguments"] = arguments
        captured["environment"] = dict(kwargs["env"])
        return object()

    monkeypatch.setattr("bps_agent.adapters.bps.subprocess.Popen", popen)
    client = BpsClient(bps_config(), username="bps-user", password="bps-password")
    client._session_id = "parent-session"
    client._api_key = "parent-key"
    destination = tmp_path / "bps-report-full.pdf"

    client.schedule_full_report_pdf("run-7", destination, ("1", "2"))

    job_path = tmp_path / "bps-report-full.job.json"
    raw_job = job_path.read_text(encoding="utf-8")
    assert "bps-user" not in raw_job
    assert "bps-password" not in raw_job
    assert json.loads(raw_job)["status"] == "pending"
    assert captured["environment"]["BPS_USERNAME"] == "bps-user"
    assert captured["environment"]["BPS_PASSWORD"] == "bps-password"
    assert "DEEPSEEK_API_KEY" not in captured["environment"]
    assert "DUT_BACKEND_PASSWORD" not in captured["environment"]
    assert "UNRELATED_SETTING" not in captured["environment"]
    assert set(captured["environment"]) <= {
        "BPS_PASSWORD",
        "BPS_USERNAME",
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
    assert captured["arguments"][-2:] == ["--job", str(job_path)]


def test_pdf_worker_authenticates_exports_and_logs_out(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[Any] = []

    class WorkerBps:
        def __init__(self, config: BpsConfig, *, username: str, password: str) -> None:
            calls.append(("init", config.endpoint, username, password))

        def authenticate(self) -> None:
            calls.append("authenticate")

        def export_full_report_pdf(
            self, run_id: str, destination: Path, section_ids: tuple[str, ...]
        ) -> Path:
            calls.append(("export", run_id, destination, section_ids))
            destination.write_bytes(b"%PDF-1.7\nfixture")
            return destination

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setenv("BPS_USERNAME", "worker-user")
    monkeypatch.setenv("BPS_PASSWORD", "worker-password")
    monkeypatch.setattr("bps_agent.adapters.bps.BpsClient", WorkerBps)
    destination = tmp_path / "report.pdf"
    job_path = tmp_path / "job.json"
    ArtifactStore.write_json(
        job_path,
        PdfExportJob.pending(
            config=bps_config(),
            run_id="run-8",
            destination=destination,
            section_ids=("1",),
        ),
    )

    assert run_pdf_job(job_path) == 0

    result = ArtifactStore.read_json(job_path)
    assert result["status"] == "succeeded"
    assert destination.read_bytes().startswith(b"%PDF-")
    assert calls[0] == ("init", "https://bps.example.test", "worker-user", "worker-password")
    assert calls[1] == "authenticate"
    assert calls[-1] == "close"
