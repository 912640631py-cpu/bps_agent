from __future__ import annotations

from pathlib import Path

import pytest

from bps_agent.artifacts import ArtifactStore
from bps_agent.locking import PortGroupLock, PortGroupLockedError
from bps_agent.models import BpsConfig


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


def test_same_evaluation_cannot_acquire_live_lock_twice(tmp_path: Path) -> None:
    first = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(4, 5),
        group=10,
        evaluation_id="evaluation-1",
    )
    second = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(4, 5),
        group=10,
        evaluation_id="evaluation-1",
    )
    first.acquire()
    try:
        with pytest.raises(PortGroupLockedError, match="active in process"):
            second.acquire()
    finally:
        first.release()


def test_preserved_lock_survives_context_exit(tmp_path: Path) -> None:
    lock = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(4, 5),
        group=10,
        evaluation_id="evaluation-1",
    )

    with lock:
        lock.preserve()

    assert lock.path.exists()
    lock.release()
