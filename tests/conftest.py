from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bps_agent.models.config import AppConfig


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    document: dict[str, Any] = {
        "bps": {
            "endpoint": "https://bps.example.test",
            "template": "performance-demo",
            "slot": 4,
            "ports": [4, 5],
            "group": 10,
            "poll_interval_seconds": 0.01,
            "run_timeout_seconds": 10,
            "registration_grace_seconds": 0,
            "report_poll_interval_seconds": 0.01,
            "report_attempts": 2,
        },
        "dut": {
            "collection_method": "frontend_api",
            "interfaces": ["T1/1", "T1/2"],
            "frontend": {
                "endpoint": "https://dut.example.test",
                "cooldown_seconds": 10,
                "keepalive_interval_seconds": 5,
                "read_retry_backoff_seconds": 0,
            },
        },
        "storage": {
            "artifact_dir": tmp_path / "artifacts",
            "checkpoint_db": tmp_path / "state.sqlite3",
        },
    }
    return AppConfig.model_validate(document)
