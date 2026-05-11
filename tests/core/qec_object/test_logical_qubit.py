"""Tests for logical_qubit module."""

from etic.core.qec_object.logical_qubit import LogicalQubit


class TestLogicalQubit:
    """Test suite for LogicalQubit class."""

    def test_initialization(self, logical_qubit: LogicalQubit):
        qubit = logical_qubit

        assert qubit.name == "L0"
        assert qubit.logical_x.logical_type.value == "X"
        assert qubit.logical_z.logical_type.value == "Z"
        assert qubit.id is not None
