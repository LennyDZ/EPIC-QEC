from typing import Literal, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ..data_structure import PauliChar, PauliString, VariableNode
from .measurement import Measurement


class LogicalOperatorUpdate(BaseModel):
    """Class representing an update to a logical operator, which can be produced by the application of a gadget or a primitive.
    An update can change the logical's operator target, operator and correction. But it always preserves the logical type and id.
    E. g. an operator representing a logical X will always represent the same logical X for the same logical qubits, even if its mapping to the physical qubits change.

    Attributes:
        new_physical_support: The new logical operator after the update. If None, the logical operator is unchanged.
        new_correction: A measurement that needs to be applied as a frame correction to the logical operator. If None, no frame correction is needed.
        new_correction_mode: Whether the new correction is appended to the existing correction history (XORed with it) or if it replaces it.
    """

    new_physical_support: Set[Tuple[PauliChar, VariableNode]] = Field(
        default_factory=set,
        description="The new logical operator after the update. If None or empty, the logical operator is unchanged.",
    )
    new_correction: Set[Measurement] = Field(
        default_factory=set,
        description="A measurement that needs to be applied as a frame correction to the logical operator.",
    )
    new_correction_mode: Literal["append", "overwrite"] = Field(
        default="append",
        description="Whether the new correction is appended to the existing correction history (XORed with it) or if it overwrites it. Valid values are 'append' and 'overwrite'.",
    )


class LogicalOperator(BaseModel):
    """Class representing a logical operator in a quantum error correction code.

    Attributes:
        logical_type: The type of logical operator (X, Y, or Z).
        operator: The Pauli string representing the physical implementation of the logical operator (each element of the string applies to a corresponding variable node in the target_nodes tuple).
        target_nodes: The tuple of variable nodes the operator acts on.
        frame_correction_history: Set of measurements that have been applied to the logical operator as frame corrections (one need to apply the XOR of these to any observable involving the logical operator).
    """

    id: UUID = Field(default_factory=uuid4, init=False)
    logical_type: PauliChar
    operator: PauliString = Field(
        default=PauliString(string=()),
        description="The Pauli string representing the physical implementation of the logical operator (each element of the string applies to a corresponding variable node in the target_nodes tuple).",
    )
    target_nodes: Tuple[
        VariableNode, ...
    ]  # Tuple of variable nodes the operator acts on.

    frame_correction_history: Set[Measurement] = Field(
        default_factory=set,
        description="Set of measurements that have been applied to the logical operator as frame corrections (one need to apply the XOR of these to any observable involving the logical operator).",
    )

    def update(self, update: LogicalOperatorUpdate) -> None:
        """Apply a logical-operator update in place.

        The update can replace the physical support and either append to or overwrite
        the accumulated frame-correction history.
        """
        if update.new_physical_support:
            self.operator = PauliString(
                string=tuple(p for p, _ in update.new_physical_support)
            )
            self.target_nodes = tuple(n for _, n in update.new_physical_support)
        if update.new_correction_mode == "append":
            self.frame_correction_history ^= update.new_correction
        elif update.new_correction_mode == "overwrite":
            self.frame_correction_history = set(update.new_correction)
