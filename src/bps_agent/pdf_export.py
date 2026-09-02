"""Supplementary PDF job, worker execution, and subprocess scheduling."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from bps_agent.artifacts import ArtifactStore
from bps_agent.models.common import utc_now
from bps_agent.models.config import BpsConfig

LOGGER = logging.getLogger(__name__)

_WORKER_SYSTEM_ENVIRONMENT = frozenset(
    {
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
)


class PdfExportJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "running", "succeeded", "failed"]
    config: BpsConfig
    run_id: str
    destination: Path
    section_ids: tuple[str, ...]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    @classmethod
    def pending(
        cls,
        *,
        config: BpsConfig,
        run_id: str,
        destination: Path,
        section_ids: tuple[str, ...],
    ) -> PdfExportJob:
        return cls(
            status="pending",
            config=config,
            run_id=run_id,
            destination=destination.resolve(),
            section_ids=section_ids,
            created_at=utc_now(),
        )

    def failed(self, error: str) -> PdfExportJob:
        return self.model_copy(
            update={"status": "failed", "finished_at": utc_now(), "error": error}
        )


def run_pdf_job(job_path: Path) -> int:
    try:
        job = PdfExportJob.model_validate(ArtifactStore.read_json(job_path))
    except (ValidationError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("invalid PDF export artifact") from exc
    running = job.model_copy(update={"status": "running", "started_at": utc_now()})
    ArtifactStore.write_json(job_path, running)
    client = None
    try:
        username = os.environ.get("BPS_USERNAME", "")
        password = os.environ.get("BPS_PASSWORD", "")
        if not username or not password:
            raise RuntimeError("PDF worker omitted BPS credentials")
        from bps_agent.adapters.bps import BpsClient

        client = BpsClient(job.config, username=username, password=password)
        client.authenticate()
        client.export_full_report_pdf(job.run_id, job.destination, job.section_ids)
    except Exception as exc:
        ArtifactStore.write_json(job_path, running.failed(str(exc)))
        return 1
    finally:
        os.environ["BPS_USERNAME"] = ""
        os.environ["BPS_PASSWORD"] = ""
        if client is not None:
            with suppress(Exception):
                client.close()
    ArtifactStore.write_json(
        job_path,
        running.model_copy(
            update={"status": "succeeded", "finished_at": utc_now(), "error": None}
        ),
    )
    return 0


def schedule_full_report_pdf(
    *,
    config: BpsConfig,
    username: str,
    password: str,
    authenticated: bool,
    run_id: str,
    destination: Path,
    section_ids: tuple[str, ...],
) -> None:
    if not authenticated or not password:
        LOGGER.warning("Optional full PDF report export was not scheduled: BPS is not logged in")
        return
    job_path = destination.with_name("bps-report-full.job.json")
    job = PdfExportJob.pending(
        config=config,
        run_id=run_id,
        destination=destination,
        section_ids=section_ids,
    )
    ArtifactStore.write_json(job_path, job)
    worker_environment = {
        name.upper(): value
        for name, value in os.environ.items()
        if name.upper() in _WORKER_SYSTEM_ENVIRONMENT
    }
    worker_environment.update({"BPS_USERNAME": username, "BPS_PASSWORD": password})
    creation_flags = 0
    start_new_session = os.name != "nt"
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    try:
        subprocess.Popen(
            [sys.executable, "-m", "bps_agent.pdf_export", "--job", str(job_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            env=worker_environment,
            creationflags=creation_flags,
            start_new_session=start_new_session,
        )
    except OSError as exc:
        ArtifactStore.write_json(job_path, job.failed(f"could not launch PDF worker: {exc}"))
        raise
    finally:
        worker_environment["BPS_USERNAME"] = ""
        worker_environment["BPS_PASSWORD"] = ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BPS supplementary PDF export worker")
    parser.add_argument("--job", type=Path, required=True)
    arguments = parser.parse_args(argv)
    return run_pdf_job(arguments.job)


if __name__ == "__main__":
    raise SystemExit(main())
