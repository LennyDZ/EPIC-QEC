from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhysicalQubit:
    """Class representing a physical qubit in the quantum memory."""

    integer_index: int = Field(
        description="Index refs used in Stim",
    )

    position: tuple[float, ...] = Field(
        description="Position of the physical qubit in a multi-dimensional space.",
        default=(),
    )

    id: UUID = Field(
        description="Unique identifier of the physical qubit",
        default_factory=uuid4,
    )
