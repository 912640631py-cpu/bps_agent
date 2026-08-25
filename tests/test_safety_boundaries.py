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


def test_port_lock_is_order_independent(tmp_path: Path) -> None:
    first = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(4, 5),
        group=10,
        evaluation_id="evaluation-1",
    )
    reversed_order = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(5, 4),
        group=10,
        evaluation_id="evaluation-2",
    )

    first.acquire()
    try:
        with pytest.raises(PortGroupLockedError):
            reversed_order.acquire()
    finally:
        first.release()


def test_overlapping_port_sets_cannot_be_locked_together(tmp_path: Path) -> None:
    first = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(4, 5),
        group=10,
        evaluation_id="evaluation-1",
    )
    overlapping = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(5, 6),
        group=10,
        evaluation_id="evaluation-2",
    )

    first.acquire()
    try:
        with pytest.raises(PortGroupLockedError):
            overlapping.acquire()
    finally:
        first.release()


def test_failed_multi_port_acquire_rolls_back_partial_locks(tmp_path: Path) -> None:
    held = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(5,),
        group=10,
        evaluation_id="evaluation-1",
    )
    candidate = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(4, 5),
        group=10,
        evaluation_id="evaluation-2",
    )
    port_four = PortGroupLock(
        tmp_path,
        endpoint="https://bps.example.test",
        slot=4,
        ports=(4,),
        group=10,
        evaluation_id="evaluation-3",
    )

    held.acquire()
    try:
        with pytest.raises(PortGroupLockedError):
            candidate.acquire()
        port_four.acquire()
        port_four.release()
    finally:
        held.release()
