from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Set
from uuid import UUID

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from ..data_structure import VariableNode, PhysicalQubit


class QuantumMemory(BaseModel):
    """Track slot allocation for physical qubits during compilation."""

    size_limit: int = Field(
        description="Total number of qubits in the quantum memory. -1 indicates unbounded memory that can grow as needed.",
        default=-1,
    )
    size: int = Field(
        description="Current number of qubits in the quantum memory. -1 indicates unbounded memory that can grow as needed.",
        default=0,
    )

    @model_validator(mode="after")
    def validate_size(cls, model):
        if model.size_limit == -1:
            return model
        else:
            model.size = model.size_limit
            model._existing_qubits.update(
                PhysicalQubit(integer_index=i) for i in range(model.size_limit)
            )
            model._free_qubits.update(
                PhysicalQubit(integer_index=i) for i in range(model.size_limit)
            )
            return model

    _existing_qubits: set[PhysicalQubit] = PrivateAttr(default_factory=set)
    _free_qubits: set[PhysicalQubit] = PrivateAttr(default_factory=set)

    _data_qubits_allocation: dict[VariableNode, PhysicalQubit] = PrivateAttr(
        default_factory=dict
    )

    _data_qubits_locked: dict[PhysicalQubit, UUID] = PrivateAttr(default_factory=dict)
    _ancilla_locked: dict[PhysicalQubit, UUID] = PrivateAttr(default_factory=dict)

    @property
    def slots(self) -> tuple[PhysicalQubit, ...]:
        """Return all slot indices that have been created so far."""
        return tuple(self._existing_qubits)

    @property
    def free_slots(self) -> tuple[PhysicalQubit, ...]:
        """Return the currently unallocated physical qubits."""
        return tuple(self._free_qubits)

    def data_qubits_allocation_snapshot(
        self, subset_keys: Set[VariableNode]
    ) -> MappingProxyType[VariableNode, PhysicalQubit]:
        """Return a snapshot of the current data qubits allocation for the requested subset of variable nodes."""

        subset = {
            k: self._data_qubits_allocation[k]
            for k in subset_keys
            if k in self._data_qubits_allocation
        }

        return MappingProxyType(subset)

    def is_allocated(self, variable_node: VariableNode) -> bool:
        """Return whether a variable node currently has a slot assignment."""
        return variable_node in self._data_qubits_allocation

    def get_physical_qubit(self, variable_node: VariableNode) -> PhysicalQubit:
        """Return the physical qubit assigned to a data node."""
        return self._data_qubits_allocation[variable_node]

    def swap_data_qubits(
        self, first: VariableNode, second: VariableNode
    ) -> None:
        """Swap the physical assignments of two allocated data nodes."""
        if first not in self._data_qubits_allocation:
            raise RuntimeError(
                f"Tried to swap data node {first.tag} that is not currently allocated."
            )
        if second not in self._data_qubits_allocation:
            raise RuntimeError(
                f"Tried to swap data node {second.tag} that is not currently allocated."
            )

        self._data_qubits_allocation[first], self._data_qubits_allocation[second] = (
            self._data_qubits_allocation[second],
            self._data_qubits_allocation[first],
        )

    def allocate_qubits(
        self, qubits: VariableNode | Sequence[VariableNode]
    ) -> list[PhysicalQubit]:
        """Allocate one or more qubits to data nodes and return their assigned slots."""

        if isinstance(qubits, VariableNode):
            qubits = [qubits]

        if len(self._free_qubits) < len(qubits):
            if self.size_limit != -1:
                raise RuntimeError(
                    f"Not enough free qubits to allocate {len(qubits)} qubits. Only {len(self._free_qubits)} free qubits available."
                )
            new_slots = len(qubits) - len(self._free_qubits)
            self._existing_qubits.update(
                PhysicalQubit(integer_index=i)
                for i in range(self.size, self.size + new_slots)
            )
            self._free_qubits.update(
                PhysicalQubit(integer_index=i)
                for i in range(self.size, self.size + new_slots)
            )
            self.size += new_slots

        for qid in qubits:
            slot = self._free_qubits.pop()
            self._data_qubits_allocation[qid] = slot
        return [self._data_qubits_allocation[qid] for qid in qubits]

    def free_qubits(self, qubits: VariableNode | Sequence[VariableNode]) -> None:
        """Release one or more qubits and return their slots to the free pool."""
        if isinstance(qubits, VariableNode):
            qubits = [qubits]
        for qid in qubits:
            if qid not in self._data_qubits_allocation:
                raise RuntimeError(
                    f"Tried to free a data node {qid.tag} that is not currently allocated."
                )
            pqb = self._data_qubits_allocation.pop(qid)
            self._free_qubits.add(pqb)
        return None

    def lock_data_qubits(
        self, qubits: Sequence[PhysicalQubit], requestor_id: UUID
    ) -> None:
        """Lock the given qubits for data use by a specific compiler pass."""
        for q in qubits:
            if q in self._data_qubits_locked:
                raise RuntimeError(
                    f"Physical qubit {q} is already locked for data use."
                )
            self._data_qubits_locked[q] = requestor_id

    def unlock_data_qubits(
        self, qubits: Sequence[PhysicalQubit], owner_id: UUID
    ) -> None:
        """Unlock the given qubits for data use by a specific compiler pass."""
        for q in qubits:
            if self._data_qubits_locked.get(q) != owner_id:
                raise RuntimeError(
                    f"Physical qubit {q} is not locked by requestor {owner_id}."
                )
            del self._data_qubits_locked[q]

    def lock_ancilla_qubits(self, n: int, requestor_id: UUID) -> Set[PhysicalQubit]:
        """Lock n ancilla qubits for ancilla use by a specific compiler pass."""
        if len(self._free_qubits) < n:
            if self.size_limit == -1:
                new_slots = n - len(self._free_qubits)
                self._existing_qubits.update(
                    PhysicalQubit(integer_index=i)
                    for i in range(self.size, self.size + new_slots)
                )
                self._free_qubits.update(
                    PhysicalQubit(integer_index=i)
                    for i in range(self.size, self.size + new_slots)
                )
                self.size += new_slots
            else:
                raise RuntimeError(
                    f"Not enough free qubits to lock {n} ancilla qubits. Only {len(self._free_qubits)} free qubits available."
                )

        locked_qubits = set()
        for _ in range(n):
            q = self._free_qubits.pop()
            self._ancilla_locked[q] = requestor_id
            locked_qubits.add(q)

        return locked_qubits

    def unlock_ancilla_qubits(
        self, qubits: Sequence[PhysicalQubit], owner_id: UUID
    ) -> None:
        """Unlock the given ancilla qubits for ancilla use by a specific compiler pass."""
        for q in qubits:
            if self._ancilla_locked.get(q) != owner_id:
                raise RuntimeError(
                    f"Physical qubit {q} is not locked by requestor {owner_id}."
                )
            del self._ancilla_locked[q]
            self._free_qubits.add(q)
        return None
