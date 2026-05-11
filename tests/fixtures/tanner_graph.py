"""Shared Tanner graph fixtures for tests."""

import pytest

from core.data_structure.pauli import PauliChar
from core.data_structure.tanner_graph import TannerEdge, TannerGraph
from core.data_structure.tanner_node import CheckNode, VariableNode


@pytest.fixture
def variable_node() -> VariableNode:
    return VariableNode(tag="v0")


@pytest.fixture
def check_node_x() -> CheckNode:
    return CheckNode(tag="cx", check_type=PauliChar.X)


@pytest.fixture
def simple_graph(variable_node: VariableNode, check_node_x: CheckNode) -> TannerGraph:
    """Small valid graph with two variables, two checks, and two edges."""
    variable_node_2 = VariableNode(tag="v1")
    check_node_z = CheckNode(tag="cz", check_type=PauliChar.Z)
    edge_x = TannerEdge(
        variable_node=variable_node,
        check_node=check_node_x,
        pauli_checked=PauliChar.X,
    )
    edge_z = TannerEdge(
        variable_node=variable_node_2,
        check_node=check_node_z,
        pauli_checked=PauliChar.Z,
    )
    return TannerGraph.model_validate(
        {
            "variable_nodes": [variable_node, variable_node_2],
            "check_nodes": [check_node_x, check_node_z],
            "edges": [edge_x, edge_z],
        }
    )
