from __future__ import annotations

import pytest

from bps_agent.models import (
    AttemptRecord,
    DutObservations,
    ObservationPhase,
    PortReservation,
    PortReservationState,
    PortReservationStatus,
    ResourceObservation,
)


def observation(phase: ObservationPhase, started_at: str, finished_at: str) -> ResourceObservation:
    return ResourceObservation(phase=phase, started_at=started_at, finished_at=finished_at)


def test_observation_collection_uses_full_time_range() -> None:
    observations = DutObservations.from_resource_observations(
        (
            observation(
                ObservationPhase.DURING,
                "2026-08-28T08:00:00+00:00",
                "2026-08-28T08:10:00+00:00",
            ),
            observation(
                ObservationPhase.BASELINE,
                "2026-08-28T15:00:00+08:00",
                "2026-08-28T15:30:00+08:00",
            ),
            observation(
                ObservationPhase.RECOVERY,
                "2026-08-28T08:10:00+00:00",
                "2026-08-28T08:20:00+00:00",
            ),
        )
    )

    assert observations.collection_started_at == "2026-08-28T15:00:00+08:00"
    assert observations.collection_finished_at == "2026-08-28T08:20:00+00:00"


def test_duplicate_observation_phase_is_rejected() -> None:
    duplicate = observation(
        ObservationPhase.BASELINE,
        "2026-08-28T00:00:00+00:00",
        "2026-08-28T00:01:00+00:00",
    )
    with pytest.raises(ValueError, match=r"duplicate.*baseline"):
        DutObservations.from_resource_observations((duplicate, duplicate))


def test_legacy_attempt_reservation_boolean_migrates_to_the_single_state() -> None:
    attempt = AttemptRecord.model_validate(
        {
            "number": 1,
            "started_at": "2026-08-31T00:00:00+00:00",
            "ports_reserved": True,
        }
    )

    assert attempt.port_reservation_state == PortReservationState.ALL_AGENT
    assert attempt.ports_reserved is True
    assert "ports_reserved" not in attempt.model_dump()


def test_reservation_status_exposes_owner_semantics_without_graph_traversal() -> None:
    status = PortReservationStatus.classify(
        (
            PortReservation(slot=4, port=4, owner="agent", owned_by_agent=True),
            PortReservation(slot=4, port=5, owner="other", owned_by_agent=False),
        )
    )

    assert status.agent_owned_ports == (4,)
    assert status.foreign_owners == ("other",)
    assert status.has_foreign_reservation
    assert not status.is_fully_agent_owned
