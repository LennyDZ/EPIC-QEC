from typing import Any, List, Set, Tuple, Union

from pydantic import Field, field_validator

from ...data_structure import TannerNode
from .qec_primitive import QECPrimitive


class ApplyGate(QECPrimitive):
    """Primitive instruction describing one or more gate applications on target nodes."""

    target_nodes: Union[Set[TannerNode], Set[Tuple[TannerNode, ...]]] = Field(
        default_factory=set,
        description=(
            "The nodes on which the gate is applied. "
            "Can be either a flat set of nodes (single-qubit gates per node) "
            "or a set of node-tuples (multi-qubit gates per tuple)."
        ),
    )
    gates: List[str] = Field(
        default_factory=list, description="The gates to be applied."
    )

    break_detector_graph: bool = Field(
        default=False,
        description=(
            "Whether to break the detector graph (i.e. node knowledge go to unknown for nodes on which a gate is applied). Default is False."
        ),
    )

    @field_validator("target_nodes")
    def validate_target_nodes_shape(cls, target_nodes):
        """Validate that target nodes are either all flat nodes or all node tuples."""
        if not target_nodes:
            return target_nodes

        first = next(iter(target_nodes))
        is_nested = isinstance(first, tuple)

        if is_nested:
            if not all(isinstance(group, tuple) for group in target_nodes):
                raise ValueError(
                    "target_nodes must be either a set of TannerNode or a set of tuple[TannerNode, ...]."
                )
            if any(len(group) == 0 for group in target_nodes):
                raise ValueError("target_nodes cannot contain empty node groups.")
        else:
            if any(isinstance(item, tuple) for item in target_nodes):
                raise ValueError(
                    "target_nodes must be either a set of TannerNode or a set of tuple[TannerNode, ...]."
                )

        return target_nodes

    @field_validator("gates")
    def validate_gates(cls, gates):
        """Validate that all requested gate mnemonics are supported."""
        valid_gates = {
            "X",
            "Y",
            "Z",
            "H",
            "S",
            "S_DAG",
            "CNOT",
            "CX",
            "CZ",
            "RX",
            "RY",
            "RZ",
            "MX",
            "MY",
            "MZ",
        }
        for g in gates:
            if g not in valid_gates:
                raise ValueError(f"Invalid gate: {g}. Valid gates are: {valid_gates}")
        return gates

    def to_payload(self) -> dict[str, Any]:
        """Serialize the primitive while preserving the raw target-node payload."""
        data = self.model_dump(exclude={"target", "target_nodes"})
        data["target"] = self.target
        data["target_nodes"] = self.target_nodes
        return data
