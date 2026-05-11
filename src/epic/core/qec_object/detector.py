from enum import Enum
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from ..data_structure.tanner_node import TannerNode
from .measurement import Measurement


@dataclass(frozen=True, slots=True)
class Detector:
    """Represents a detector in the context of quantum error correction.

    A detector is associated with one or multiple measurements, each corresponding to a specific node at a given timestep.

    Attributes
    ----------
    id : UUID
        Unique identifier for the detector.
    measurements : List[Measurement]
        List of measurements that the detector is associated with. Each measurement corresponds to a specific node at a given timestep.
    coordinates : Tuple[int, ...]
        Optional coordinates for the detector, which can be used for visualization or geometric codes.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: UUID = Field(
        description="Unique identifier for the detector",
        default_factory=uuid4,
    )
    measurements: list[Measurement] = Field(
        description="List of measurements associated with the detector. Each measurement corresponds to a specific node at a given timestep.",
        default_factory=list,
    )
    coordinates: tuple[int, ...] = Field(
        description="Optional coordinates for the detector, which can be used for visualization or geometric codes.",
        default=(),
    )
    tag: str = Field(
        description="Optional tag for the detector, which can be used for identification or debugging purposes.",
        default="",
    )


class NodeKnowledge(Enum):
    """Compilation-time knowledge tracked for a Tanner node at a detector port."""

    STABLE = "stable"
    RX = "reset-x"
    RZ = "reset-z"
    MX = "measured-x"
    MZ = "measured-z"
    UNKNOWN = "unknown"

    def basis(self) -> str | None:
        """Return the measurement or reset basis implied by this knowledge state."""
        match self:
            case NodeKnowledge.RX | NodeKnowledge.MX:
                return "X"
            case NodeKnowledge.RZ | NodeKnowledge.MZ:
                return "Z"
            case _:
                return None


@dataclass(frozen=True, slots=True)
class QubitPortState:
    """State carried by a node on the detector-graph port boundary."""

    knowledge: NodeKnowledge
    connected_nodes: set[TannerNode] = Field(
        default_factory=set,
        description="Set of nodes that are connected to this node in the detector graph. This is used to determine how detectors are formed based on the knowledge of neighboring nodes.",
    )


DetectorGraphPort = dict[TannerNode, QubitPortState]
