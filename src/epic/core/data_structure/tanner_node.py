from typing import Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .pauli import PauliChar


class TannerNode(BaseModel):
    """Node in a Tanner graph.

    Attributes:
        id (UUID): Unique identifier of the node.
        tag (str): human readable tag (mostly used for debugging).
        coordinates (Tuple[int, ...]): coordinates of the node (only used for visualisation), default = None.
    """

    model_config = ConfigDict(frozen=True)
    id: UUID = Field(
        description="Unique identifier of the node",
        default_factory=uuid4,
    )
    tag: str = Field(
        description="Human readable tag (mostly used for debugging)",
        default="",
    )
    coordinates: Tuple[int, ...] | None = Field(
        description="Coordinates of the node in some space (optional, for visualization or geometric codes)",
        default=None,
    )


class VariableNode(TannerNode):
    """Variable node in a Tanner graph."""

    pass


class CheckNode(TannerNode):
    """Check node in a Tanner graph."""

    check_type: PauliChar | None = Field(
        description="Type of the check node (X, Z, Y) or None when unspecified.",
        default=None,
    )

    @property
    def pauli_type(self) -> PauliChar | None:
        """Return the Pauli basis enforced by this check node, if any."""
        return self.check_type
