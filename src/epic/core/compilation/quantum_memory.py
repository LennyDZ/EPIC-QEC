from collections.abc import Mapping, Sequence
from types import MappingProxyType

from pydantic import BaseModel, Field, PrivateAttr

from ..data_structure.tanner_node import TannerNode


class QuantumMemory(BaseModel):
    """Track slot allocation for physical qubits during compilation."""

    size: int = Field(
        description="Total number of qubits in the quantum memory.",
        default=0,
    )
    _slots: list[int] = PrivateAttr(default_factory=list)
    _free_slots: list[int] = PrivateAttr(default_factory=list)
    _allocation: dict[TannerNode, int] = PrivateAttr(default_factory=dict)

    @property
    def slots(self) -> tuple[int, ...]:
        """Return all slot indices that have been created so far."""
        return tuple(self._slots)

    @property
    def free_slots(self) -> tuple[int, ...]:
        """Return the currently unallocated slot indices."""
        return tuple(self._free_slots)

    @property
    def allocation(self) -> Mapping[TannerNode, int]:
        """Return a read-only mapping from qubits to their allocated slot."""
        # Expose a read-only view so external callers cannot mutate memory state.
        return MappingProxyType(self._allocation)

    def is_allocated(self, qubit_id: TannerNode) -> bool:
        """Return whether a qubit currently has a slot assignment."""
        return qubit_id in self._allocation

    def get_slot(self, qubit_id: TannerNode) -> int:
        """Return the slot assigned to a previously allocated qubit."""
        return self._allocation[qubit_id]

    def allocate_qubits(self, qubits: TannerNode | Sequence[TannerNode]) -> list[int]:
        """Allocate one or more qubits and return their assigned slots."""

        if isinstance(qubits, TannerNode):
            qubits = [qubits]

        if len(self._free_slots) < len(qubits):
            new_slots = len(qubits) - len(self._free_slots)
            self._slots.extend(range(self.size, self.size + new_slots))
            self._free_slots.extend(range(self.size, self.size + new_slots))
            self.size += new_slots

        for qid in qubits:
            slot = self._free_slots.pop(0)
            self._allocation[qid] = slot
        return [self._allocation[qid] for qid in qubits]

    def free_qubits(self, qubits: TannerNode | Sequence[TannerNode]) -> None:
        """Release one or more qubits and return their slots to the free pool."""
        if isinstance(qubits, TannerNode):
            qubits = [qubits]
        for qid in qubits:
            slot = self._allocation.pop(qid)
            self._free_slots.append(slot)
        return None


# class TopologicalMemory(QuantumMemory):
#     """Class representing a topological quantum memory, which is a specific type of quantum memory used in quantum error correction codes.
#     In topological memories, physical qubits are associated with a position in a n-dimensional lattice.
#     """

#     dimension: int = Field(
#         description="Dimension of the lattice in which the physical qubits are embedded.",
#         default=2,
#     )
#     bounds: Tuple[int, ...] = Field(
#         description="Bounds of the lattice in each dimension, defining the size of the lattice.",
#         default=(),
#     )
#     slot_positions: dict[int, Tuple[int, ...]] = Field(
#         description="Mapping from slot index to the coordinates of the corresponding physical qubit in the lattice.",
#         default_factory=dict,
#         init=False,
#     )
