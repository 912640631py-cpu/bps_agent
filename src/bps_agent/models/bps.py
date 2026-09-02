"""BPS Run and physical-port domain models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import ConfigDict, Field

from bps_agent.models.common import StrictModel


class RunCompletion(StrictModel):
    terminal: bool
    details: dict[str, Any] = Field(default_factory=dict)


class PortReservation(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: int = Field(ge=0)
    port: int = Field(ge=0)
    owner: str | None = None
    owned_by_agent: bool = False


class PortReservationState(StrEnum):
    NONE = "none"
    ALL_AGENT = "all_agent"
    PARTIAL_AGENT = "partial_agent"
    FOREIGN = "foreign"


class PortReservationStatus(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: PortReservationState
    reservations: tuple[PortReservation, ...]

    @classmethod
    def classify(cls, reservations: tuple[PortReservation, ...]) -> PortReservationStatus:
        if any(
            reservation.owner is not None and not reservation.owned_by_agent
            for reservation in reservations
        ):
            state = PortReservationState.FOREIGN
        else:
            agent_count = sum(reservation.owned_by_agent for reservation in reservations)
            if agent_count == 0:
                state = PortReservationState.NONE
            elif agent_count == len(reservations):
                state = PortReservationState.ALL_AGENT
            else:
                state = PortReservationState.PARTIAL_AGENT
        return cls(state=state, reservations=reservations)

    @property
    def agent_owned_ports(self) -> tuple[int, ...]:
        return tuple(
            reservation.port for reservation in self.reservations if reservation.owned_by_agent
        )

    @property
    def foreign_owners(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    reservation.owner
                    for reservation in self.reservations
                    if reservation.owner is not None and not reservation.owned_by_agent
                }
            )
        )

    @property
    def is_fully_agent_owned(self) -> bool:
        return self.state == PortReservationState.ALL_AGENT

    @property
    def has_foreign_reservation(self) -> bool:
        return self.state == PortReservationState.FOREIGN
