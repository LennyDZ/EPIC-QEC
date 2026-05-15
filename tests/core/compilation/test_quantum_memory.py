"""Tests for quantum_memory module."""

from types import MappingProxyType
from typing import Any, cast

import pytest

from epic.core.compilation.quantum_memory import QuantumMemory
from epic.core.data_structure.physical_qubit import PhysicalQubit
from epic.core.data_structure.tanner_node import VariableNode


@pytest.fixture
def memory_nodes() -> list[VariableNode]:
    return [VariableNode(tag=f"v{i}") for i in range(4)]


class TestQuantumMemory:
    def test_allocate_qubits_grows_capacity_and_assigns_slots(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()

        slots = memory.allocate_qubits(memory_nodes[:2])
        allocation = memory.data_qubits_allocation_snapshot(set(memory_nodes[:2]))

        assert len(slots) == 2
        assert all(isinstance(slot, PhysicalQubit) for slot in slots)
        assert memory.size == 1
        assert {slot.integer_index for slot in slots} == {-1, 0}
        assert {slot.integer_index for slot in memory.slots} == {-1, 0}
        assert memory.free_slots == ()
        assert allocation[memory_nodes[0]] == slots[0]
        assert allocation[memory_nodes[1]] == slots[1]
        assert memory.is_allocated(memory_nodes[0])
        assert memory.is_allocated(memory_nodes[1])

    def test_allocate_qubits_reuses_freed_slots_before_growing(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()
        initial_slots = memory.allocate_qubits(memory_nodes[:2])
        memory.free_qubits([memory_nodes[0]])

        slots = memory.allocate_qubits([memory_nodes[2]])

        assert slots == [initial_slots[0]]
        assert memory.size == 1
        assert (
            memory.data_qubits_allocation_snapshot(
                cast(set[VariableNode], {cast(Any, memory_nodes[2])})
            )[memory_nodes[2]]
            == initial_slots[0]
        )
        assert not memory.is_allocated(memory_nodes[0])
        assert memory.free_slots == ()

    def test_allocate_qubits_raises_when_request_exceeds_available_capacity(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()
        memory.allocate_qubits(memory_nodes[:2])
        memory.free_qubits([memory_nodes[1]])

        with pytest.raises(RuntimeError, match="Not enough free qubits"):
            memory.allocate_qubits(memory_nodes[2:4])

        assert memory.size == 1
        assert not memory.is_allocated(memory_nodes[2])
        assert not memory.is_allocated(memory_nodes[3])

    def test_free_qubits_releases_slots_and_updates_allocation(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()
        allocated_qubits = memory.allocate_qubits(memory_nodes[:3])

        result = memory.free_qubits(memory_nodes[:2])

        assert result is None
        assert not memory.is_allocated(memory_nodes[0])
        assert not memory.is_allocated(memory_nodes[1])
        assert memory.is_allocated(memory_nodes[2])
        assert set(memory.free_slots) == set(allocated_qubits[:2])

    def test_get_physical_qubit_raises_for_unallocated_qubit(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()

        with pytest.raises(KeyError):
            memory.get_physical_qubit(memory_nodes[0])

    def test_free_qubits_raises_for_unallocated_qubit(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()

        with pytest.raises(RuntimeError, match="is not currently allocated"):
            memory.free_qubits([memory_nodes[0]])

    def test_data_qubits_allocation_snapshot_exposes_read_only_view(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()
        allocated_qubit = memory.allocate_qubits([memory_nodes[0]])[0]

        allocation = memory.data_qubits_allocation_snapshot(
            cast(set[VariableNode], {cast(Any, memory_nodes[0])})
        )
        unsafe_allocation = cast(Any, allocation)

        assert isinstance(allocation, MappingProxyType)
        assert allocation[memory_nodes[0]] == allocated_qubit
        with pytest.raises(TypeError):
            unsafe_allocation[memory_nodes[1]] = allocated_qubit
