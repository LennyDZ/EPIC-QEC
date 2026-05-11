"""Tests for quantum_memory module."""

from types import MappingProxyType
from typing import Any, cast

import pytest

from epic.core.compilation.quantum_memory import QuantumMemory
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

        assert slots == [0, 1]
        assert memory.size == 2
        assert memory.slots == (0, 1)
        assert memory.free_slots == ()
        assert memory.get_slot(memory_nodes[0]) == 0
        assert memory.get_slot(memory_nodes[1]) == 1
        assert memory.is_allocated(memory_nodes[0])
        assert memory.is_allocated(memory_nodes[1])

    def test_allocate_qubits_reuses_freed_slots_before_growing(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()
        memory.allocate_qubits(memory_nodes[:2])
        memory.free_qubits([memory_nodes[0]])

        slots = memory.allocate_qubits([memory_nodes[2]])

        assert slots == [0]
        assert memory.size == 2
        assert memory.get_slot(memory_nodes[2]) == 0
        assert not memory.is_allocated(memory_nodes[0])
        assert memory.free_slots == ()

    def test_allocate_qubits_partially_reuses_free_slots_and_expands(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()
        memory.allocate_qubits(memory_nodes[:2])
        memory.free_qubits([memory_nodes[1]])

        slots = memory.allocate_qubits(memory_nodes[2:4])

        assert slots == [1, 2]
        assert memory.size == 3
        assert memory.slots == (0, 1, 2)
        assert memory.get_slot(memory_nodes[2]) == 1
        assert memory.get_slot(memory_nodes[3]) == 2

    def test_free_qubits_releases_slots_and_updates_allocation(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()
        memory.allocate_qubits(memory_nodes[:3])

        result = memory.free_qubits(memory_nodes[:2])

        assert result is None
        assert not memory.is_allocated(memory_nodes[0])
        assert not memory.is_allocated(memory_nodes[1])
        assert memory.is_allocated(memory_nodes[2])
        assert memory.free_slots == (0, 1)

    def test_get_slot_raises_for_unallocated_qubit(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()

        with pytest.raises(KeyError):
            memory.get_slot(memory_nodes[0])

    def test_free_qubits_raises_for_unallocated_qubit(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()

        with pytest.raises(KeyError):
            memory.free_qubits([memory_nodes[0]])

    def test_allocation_property_exposes_read_only_view(
        self, memory_nodes: list[VariableNode]
    ) -> None:
        memory = QuantumMemory()
        memory.allocate_qubits([memory_nodes[0]])

        allocation = memory.allocation
        unsafe_allocation = cast(Any, allocation)

        assert isinstance(allocation, MappingProxyType)
        assert allocation[memory_nodes[0]] == 0
        with pytest.raises(TypeError):
            unsafe_allocation[memory_nodes[1]] = 1
