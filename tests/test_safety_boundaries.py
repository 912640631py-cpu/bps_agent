from __future__ import annotations

from pathlib import Path

import pytest

from bps_agent.artifacts import ArtifactStore
from bps_agent.models.config import BpsConfig


def test_artifact_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="unsafe path"):
        store.ensure_evaluation("../outside")


def test_configuration_rejects_credentials_in_endpoint() -> None:
    with pytest.raises(ValueError, match="credential-free"):
        BpsConfig(
            endpoint="https://user:password@bps.example.test",
            template="template",
            slot=4,
            ports=(4, 5),
            group=10,
        )
