from __future__ import annotations

import pytest

from bps_agent.models.bps import (
    PortReservation,
    PortReservationState,
    PortReservationStatus,
)
from bps_agent.models.common import ObservationPhase
from bps_agent.models.dut import DutObservations, ResourceObservation


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


@pytest.mark.parametrize(
    ("owners", "expected_state", "agent_ports", "foreign_owners"),
    [
        ((None, None), PortReservationState.NONE, (), ()),
        (("agent", "agent"), PortReservationState.ALL_AGENT, (4, 5), ()),
        (("agent", None), PortReservationState.PARTIAL_AGENT, (4,), ()),
        (("agent", "other"), PortReservationState.FOREIGN, (4,), ("other",)),
    ],
)
def test_reservation_status_classifies_owner_semantics(
    owners: tuple[str | None, str | None],
    expected_state: PortReservationState,
    agent_ports: tuple[int, ...],
    foreign_owners: tuple[str, ...],
) -> None:
    status = PortReservationStatus.classify(
        tuple(
            PortReservation(
                slot=4,
                port=port,
                owner=owner,
                owned_by_agent=owner == "agent",
            )
            for port, owner in zip((4, 5), owners, strict=True)
        )
    )

    assert status.state == expected_state
    assert status.agent_owned_ports == agent_ports
    assert status.foreign_owners == foreign_owners
    assert status.has_foreign_reservation == (expected_state == PortReservationState.FOREIGN)
    assert status.is_fully_agent_owned == (expected_state == PortReservationState.ALL_AGENT)
