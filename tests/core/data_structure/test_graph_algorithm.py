"""Tests for graph_algorithm module."""

import pytest

from epic.core.data_structure.graph_algorithm import GraphAlgorithm
from epic.core.data_structure.pauli import PauliChar
from epic.core.data_structure.tanner_graph import TannerEdge, TannerGraph
from epic.core.data_structure.tanner_node import CheckNode, VariableNode


@pytest.fixture
def path_graph() -> TannerGraph:
    """Connected graph with a unique path across mixed check types."""
    v0 = VariableNode(tag="v0")
    v1 = VariableNode(tag="v1")
    v2 = VariableNode(tag="v2")
    cx = CheckNode(tag="cx", check_type=PauliChar.X)
    cz = CheckNode(tag="cz", check_type=PauliChar.Z)

    return TannerGraph.model_validate(
        {
            "variable_nodes": {v0, v1, v2},  # type: ignore
            "check_nodes": {cx, cz},  # type: ignore
            "edges": {
                TannerEdge(
                    variable_node=v0,
                    check_node=cx,
                    pauli_checked=PauliChar.X,
                ),
                TannerEdge(
                    variable_node=v1,
                    check_node=cx,
                    pauli_checked=PauliChar.X,
                ),
                TannerEdge(
                    variable_node=v1,
                    check_node=cz,
                    pauli_checked=PauliChar.Z,
                ),
                TannerEdge(
                    variable_node=v2,
                    check_node=cz,
                    pauli_checked=PauliChar.Z,
                ),
            },
        }
    )


class TestGraphAlgorithmShortestPath:
    def test_shortest_path_returns_path_in_bfs_order(self, path_graph: TannerGraph):
        nodes_by_tag = {
            node.tag: node
            for node in path_graph.variable_nodes | path_graph.check_nodes
        }

        path = GraphAlgorithm.shortest_path(
            path_graph, nodes_by_tag["v0"], nodes_by_tag["v2"]
        )

        assert [node.tag for node in path] == ["v0", "cx", "v1", "cz", "v2"]

    def test_shortest_path_returns_singleton_when_start_equals_end(
        self, path_graph: TannerGraph
    ):
        start = next(node for node in path_graph.variable_nodes if node.tag == "v1")

        path = GraphAlgorithm.shortest_path(path_graph, start, start)

        assert path == [start]

    def test_shortest_path_returns_empty_when_no_path(self, simple_graph: TannerGraph):
        variables = {node.tag: node for node in simple_graph.variable_nodes}

        path = GraphAlgorithm.shortest_path(
            simple_graph, variables["v0"], variables["v1"]
        )

        assert path == []

    def test_shortest_path_rejects_external_start(self, path_graph: TannerGraph):
        end = next(node for node in path_graph.variable_nodes if node.tag == "v0")

        with pytest.raises(ValueError, match="Start node is not in the Tanner graph"):
            GraphAlgorithm.shortest_path(path_graph, VariableNode(tag="external"), end)

    def test_shortest_path_rejects_external_end(self, path_graph: TannerGraph):
        start = next(node for node in path_graph.variable_nodes if node.tag == "v0")

        with pytest.raises(ValueError, match="End node is not in the Tanner graph"):
            GraphAlgorithm.shortest_path(
                path_graph, start, VariableNode(tag="external")
            )

    def test_shortest_path_filters_check_nodes_by_type(self, path_graph: TannerGraph):
        variables = {node.tag: node for node in path_graph.variable_nodes}

        path = GraphAlgorithm.shortest_path(
            path_graph,
            variables["v0"],
            variables["v2"],
            check_type=PauliChar.X,
        )

        assert path == []

    def test_shortest_path_allows_matching_check_type(self, path_graph: TannerGraph):
        variables = {node.tag: node for node in path_graph.variable_nodes}

        path = GraphAlgorithm.shortest_path(
            path_graph,
            variables["v0"],
            variables["v1"],
            check_type=PauliChar.X,
        )

        assert [node.tag for node in path] == ["v0", "cx", "v1"]
