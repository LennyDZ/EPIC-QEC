"""Tests for quantum_program module."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from epic.core.data_structure.quantum_program import (
    ProgramQubit,
    QProgOperation,
    QuantumProgram,
)


class TestQProgOperation:
    def test_accepts_non_negative_integer_lengths(self):
        operation = QProgOperation(name="H", length=1, targets=[])
        tick = QProgOperation(name="tick", length=0, targets=[])

        assert operation.length == 1
        assert tick.length == 0

    def test_rejects_zero_length_non_tick_operations(self):
        with pytest.raises(ValidationError, match="Only tick"):
            QProgOperation(name="H", length=0, targets=[])

    @pytest.mark.parametrize("length", [-1, 1.5])
    def test_rejects_invalid_lengths(self, length):
        with pytest.raises(ValidationError, match="length"):
            QProgOperation(name="invalid", length=length, targets=[])


class TestQuantumProgram:
    def test_add_qubit_returns_and_tracks_qubit(self):
        program = QuantumProgram(name="test")

        qubit = program.add_qubit()

        assert qubit.name == "q0"
        assert qubit in program.qubits
        assert program.width == 1

    def test_add_qubits_returns_qubits_in_allocation_order(self):
        program = QuantumProgram(name="test")

        qubits = program.add_qubits(3)

        assert [qubit.name for qubit in qubits] == ["q0", "q1", "q2"]
        assert set(qubits) == program.qubits
        assert program.width == 3

    @pytest.mark.parametrize("count", [-1, 1.5, True])
    def test_add_qubits_rejects_invalid_counts(self, count):
        program = QuantumProgram(name="test")

        with pytest.raises(ValueError, match="non-negative integer"):
            program.add_qubits(count)

    def test_add_operation_schedules_after_busy_qubit(self):
        program = QuantumProgram(name="test")
        q0 = program.add_qubit()
        q1 = program.add_qubit()

        program.add_operation(QProgOperation(name="H", length=1, targets=[q0]))
        program.add_operation(
            QProgOperation(name="CX", length=2, targets=[q0, q1])
        )

        first_operation, first_start = program.operations[0]
        second_operation, second_start = program.operations[1]
        assert (first_operation.name, first_start) == ("H", 0)
        assert (second_operation.name, second_start) == ("CX", 1)
        assert second_operation.targets == [q0, q1]
        assert program.depth == 3

    def test_rejects_qubit_not_in_program(self):
        program = QuantumProgram(name="test")
        external_qubit = ProgramQubit(name="external")

        with pytest.raises(ValueError, match="not part of the program"):
            program.add_operation(
                QProgOperation(name="H", length=1, targets=[external_qubit])
            )

    def test_add_tick_creates_zero_length_operation(self):
        program = QuantumProgram(name="test")
        qubit = program.add_qubit()

        program.add_tick([qubit])

        operation, start = program.operations[0]
        assert operation.name == "tick"
        assert operation.length == 0
        assert operation.targets == [qubit]
        assert start == 0
        assert program.depth == 0

    def test_add_operation_flattens_sub_program(self):
        program = QuantumProgram(name="test")
        a = program.add_qubit()
        b = program.add_qubit()
        c = program.add_qubit()

        sub_program = QuantumProgram(name="bell_pair")
        sub_program.add_qubit(a)
        sub_program.add_qubit(b)
        sub_program.add_operation(QProgOperation(name="H", length=1, targets=[a]))
        sub_program.add_operation(QProgOperation(name="CX", length=2, targets=[a, b]))

        program.add_operation(sub_program)
        program.add_operation(QProgOperation(name="MZ", length=1, targets=[a, b, c]))

        names_and_starts = [(operation.name, start) for operation, start in program.operations]
        assert names_and_starts == [("H", 0), ("CX", 1), ("MZ", 3)]
        assert program.depth == 4

    def test_add_operation_rejects_sub_program_with_qubit_not_in_program(self):
        program = QuantumProgram(name="test")
        program.add_qubit()

        sub_program = QuantumProgram(name="sub")
        sub_program.add_qubit()

        with pytest.raises(ValueError, match="not part of the program"):
            program.add_operation(sub_program)

    def test_add_operation_preserves_ticks_inside_sub_program(self):
        program = QuantumProgram(name="test")
        data_qubit = program.add_qubit()
        ancilla_qubit = program.add_qubit()

        def stabilizer_round():
            block = QuantumProgram(name="stabilizer_round")
            block.add_qubit(data_qubit)
            block.add_qubit(ancilla_qubit)
            block.add_operation(QProgOperation(name="RZ", length=1, targets=[ancilla_qubit]))
            block.add_tick([data_qubit, ancilla_qubit])
            block.add_operation(QProgOperation(name="CX", length=1, targets=[data_qubit, ancilla_qubit]))
            block.add_tick([data_qubit, ancilla_qubit])
            block.add_operation(QProgOperation(name="MZ", length=1, targets=[ancilla_qubit]))
            return block

        program.add_operation(stabilizer_round())
        program.add_operation(stabilizer_round())

        names_and_starts = [(operation.name, start) for operation, start in program.operations]
        assert names_and_starts == [
            ("RZ", 0),
            ("tick", 1),
            ("CX", 1),
            ("tick", 2),
            ("MZ", 2),
            ("RZ", 3),
            ("tick", 4),
            ("CX", 4),
            ("tick", 5),
            ("MZ", 5),
        ]
        assert program.depth == 6
