from __future__ import annotations

import pytest

from bps_agent.models import DutObservations, ObservationPhase, ResourceObservation


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
