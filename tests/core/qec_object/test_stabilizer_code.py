"""Tests for stabilizer_code module."""

import pytest

from core.qec_object.logical_qubit import LogicalQubit
from core.qec_object.stabilizer_code import StabilizerCode


class TestStabilizerCode:
    def test_initialization_minimal_valid(
        self,
        empty_check_tanner_graph,
        logical_qubit: LogicalQubit,
    ) -> None:
        code = StabilizerCode(
            name="toy",
            n=1,
            k=1,
            d=1,
            tanner_graph=empty_check_tanner_graph,
            logical_qubits=[logical_qubit],
        )

        assert code.name == "toy"
        assert code.n == 1
        assert code.k == 1
        assert code.d == 1

    def test_negative_parameters_raise(
        self, empty_check_tanner_graph, logical_qubit: LogicalQubit
    ) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            StabilizerCode(
                name="toy",
                n=-1,
                k=1,
                d=1,
                tanner_graph=empty_check_tanner_graph,
                logical_qubits=[logical_qubit],
            )

    def test_logical_qubit_count_validation(
        self,
        empty_check_tanner_graph,
        logical_qubit: LogicalQubit,
    ) -> None:
        with pytest.raises(ValueError, match="must equal k"):
            StabilizerCode(
                name="toy",
                n=1,
                k=0,
                d=1,
                tanner_graph=empty_check_tanner_graph,
                logical_qubits=[logical_qubit],
            )

    def test_distance_validation(
        self,
        empty_check_tanner_graph,
        logical_qubit: LogicalQubit,
    ) -> None:
        with pytest.raises(ValueError, match="weight less than d"):
            StabilizerCode(
                name="toy",
                n=1,
                k=1,
                d=2,
                tanner_graph=empty_check_tanner_graph,
                logical_qubits=[logical_qubit],
            )

    def test_disable_algebraic_validation_allows_mismatch(
        self,
        empty_check_tanner_graph,
        logical_qubit: LogicalQubit,
    ) -> None:
        code = StabilizerCode(
            name="toy",
            n=1,
            k=0,
            d=2,
            tanner_graph=empty_check_tanner_graph,
            logical_qubits=[logical_qubit],
            validate_algebraic_properties=False,
        )
        assert code.k == 0

    def test_from_pcm_builds_expected_sizes(self, one_qubit_css_pcm) -> None:
        code = StabilizerCode.from_pcm(
            code_name="pcm",
            simplectic_pcm=one_qubit_css_pcm,
            logical_qubits=[([1], [1])],
            var_coordinate={0: (0, 0)},
            check_coordinate={0: (0, 1)},
        )

        assert code.n == 1
        assert code.k == 1
        assert len(code.logical_qubits) == 1
        assert len(code.tanner_graph.variable_nodes) == 1
