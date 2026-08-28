"""Independent supplementary BPS PDF export worker."""

from __future__ import annotations

import argparse
import os
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from bps_agent.artifacts import ArtifactStore
from bps_agent.models import BpsConfig, utc_now


class PdfExportJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
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
    job = PdfExportJob.model_validate(ArtifactStore.read_json(job_path))
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
    succeeded = running.model_copy(
        update={"status": "succeeded", "finished_at": utc_now(), "error": None}
    )
    ArtifactStore.write_json(job_path, succeeded)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BPS supplementary PDF export worker")
    parser.add_argument("--job", type=Path, required=True)
    arguments = parser.parse_args(argv)
    return run_pdf_job(arguments.job)


if __name__ == "__main__":
    raise SystemExit(main())
