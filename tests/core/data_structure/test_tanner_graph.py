"""Tests for tanner_graph module."""

from pydantic import ValidationError
from scipy.sparse import csr_matrix
import pytest

from core.data_structure.pauli import PauliChar, PauliString
from core.data_structure.tanner_graph import TannerEdge, TannerGraph
from core.data_structure.tanner_node import CheckNode, VariableNode
from core.qec_object.logical_operator import LogicalOperator


class TestTannerGraph:
    """Test suite for TannerGraph class."""

    def test_empty_initialization(self):
        """Test TannerGraph initialization."""
        graph = TannerGraph()
        assert graph.variable_nodes == set()
        assert graph.check_nodes == set()
        assert graph.edges == set()
        assert graph.number_of_nodes == 0

    def test_contains_nodes_and_edges(self, simple_graph):
        """Test __contains__ for variable/check nodes and edges."""
        node_v = next(iter(simple_graph.variable_nodes))
        node_c = next(iter(simple_graph.check_nodes))
        edge = next(iter(simple_graph.edges))

        assert node_v in simple_graph
        assert node_c in simple_graph
        assert edge in simple_graph

    def test_number_of_nodes(self, simple_graph):
        """Test total node count property."""
        assert simple_graph.number_of_nodes == 4

    def test_validate_edges_rejects_unknown_variable(
        self,
    ):
        """Edge endpoint nodes must be present in graph node sets."""
        v_in = VariableNode(tag="v_in")
        v_out = VariableNode(tag="v_out")
        cx = CheckNode(tag="cx")
        edge = TannerEdge(variable_node=v_out, check_node=cx, pauli_checked=PauliChar.X)

        with pytest.raises(ValidationError, match="not in variable nodes"):
            TannerGraph.model_validate(
                {
                    "variable_nodes": {v_in},  # type: ignore
                    "check_nodes": {cx},  # type: ignore
                    "edges": {edge},
                }
            )

    def test_validate_check_type_mismatch_raises(
        self,
    ):
        """Declared check type must match connected edge Pauli type."""
        v0 = VariableNode(tag="v0")
        cx = CheckNode(tag="cx", check_type=PauliChar.X)
        bad_edge = TannerEdge(
            variable_node=v0, check_node=cx, pauli_checked=PauliChar.Z
        )

        with pytest.raises(ValidationError, match="declared type"):
            TannerGraph.model_validate(
                {
                    "variable_nodes": {v0},  # type: ignore
                    "check_nodes": {cx},  # type: ignore
                    "edges": {bad_edge},
                }
            )

    def test_check_node_accepts_pauli_char_type(self):
        """Check nodes should accept PauliChar directly."""
        cx = CheckNode(tag="cx", check_type=PauliChar.X)

        assert cx.check_type == PauliChar.X
        assert cx.pauli_type == PauliChar.X

    def test_check_node_defaults_to_unspecified_type(self):
        """Check nodes with no check_type should be treated as unspecified."""
        cx = CheckNode(tag="cx")

        assert cx.check_type is None
        assert cx.pauli_type is None

    def test_tanner_edge_accepts_check_type_pauli(self):
        """Edges should accept the shared PauliChar enum directly."""
        v0 = VariableNode(tag="v0")
        cx = CheckNode(tag="cx", check_type=PauliChar.X)
        edge = TannerEdge(
            variable_node=v0,
            check_node=cx,
            pauli_checked=PauliChar.X,
        )

        assert edge.pauli_checked == PauliChar.X

    def test_graph_validation_accepts_shared_enum_inputs(self):
        """Graph validation should work when node and edge use the shared Pauli enum."""
        v0 = VariableNode(tag="v0")
        cx = CheckNode(tag="cx", check_type=PauliChar.X)
        edge = TannerEdge(
            variable_node=v0,
            check_node=cx,
            pauli_checked=PauliChar.X,
        )

        graph = TannerGraph.model_validate(
            {
                "variable_nodes": {v0},  # type: ignore
                "check_nodes": {cx},  # type: ignore
                "edges": {edge},  # type: ignore
            }
        )

        only_edge = next(iter(graph.edges))
        only_check = next(iter(graph.check_nodes))
        assert only_edge.pauli_checked == PauliChar.X
        assert only_check.check_type == PauliChar.X


class TestTannerGraphProperties:
    """Test suite for TannerGraph properties."""

    def test_indexes_have_expected_entries(self, simple_graph):
        """Index maps should include all nodes that have incident edges."""
        for node in simple_graph.variable_nodes:
            assert node in simple_graph.index_by_variable
            assert len(simple_graph.index_by_variable[node]) == 1

        for node in simple_graph.check_nodes:
            assert node in simple_graph.index_by_check
            assert len(simple_graph.index_by_check[node]) == 1

    def test_get_neighbourhood_and_degree(self, simple_graph):
        """Neighbourhood and degree should match edge connectivity."""
        var_node = next(iter(simple_graph.variable_nodes))
        check_node = next(iter(simple_graph.check_nodes))

        var_neighbors = simple_graph.get_neighbourhood(var_node)
        check_neighbors = simple_graph.get_neighbourhood(check_node)

        assert len(var_neighbors) == 1
        assert len(check_neighbors) == 1
        assert simple_graph.degree(var_node) == 1
        assert simple_graph.degree(check_node) == 1

    def test_degree_raises_for_external_node(self, simple_graph):
        """Querying degree for node outside graph should raise."""
        external = VariableNode(tag="external")
        with pytest.raises(ValueError, match="not in the Tanner graph"):
            simple_graph.degree(external)


class TestTannerGraphOperators:
    """Test suite for TannerGraph operators."""

    def test_union_and_disjoint(self, simple_graph):
        """Test graph union and disjointness checks."""
        v2 = VariableNode(tag="v2")
        c2 = CheckNode(tag="c2")
        e2 = TannerEdge(variable_node=v2, check_node=c2, pauli_checked=PauliChar.X)
        other = TannerGraph.model_validate(
            {
                "variable_nodes": {v2},  # type: ignore
                "check_nodes": {c2},  # type: ignore
                "edges": {e2},  # type: ignore
            }
        )

        joined = simple_graph | other

        assert simple_graph.is_disjoint(other)
        assert (
            joined.number_of_nodes
            == simple_graph.number_of_nodes + other.number_of_nodes
        )

    def test_parity_check_matrix_shape_and_nnz(self, simple_graph):
        """Parity-check matrix should have expected dimensions for CSS split columns."""
        matrix, var_nodes, check_nodes = simple_graph.parity_check_matrix

        assert matrix.shape == (len(check_nodes), 2 * len(var_nodes))
        assert matrix.nnz == 2

    def test_from_pcm_and_logical_operator_support(self):
        """Build from PCM and verify support extraction for a logical operator."""
        hx = csr_matrix(([1], ([0], [0])), shape=(1, 2))
        hz = csr_matrix(([1], ([0], [1])), shape=(1, 2))

        graph = TannerGraph.from_pcm(hx, hz, code_name="toy")
        var_nodes = sorted(graph.variable_nodes, key=lambda n: n.tag)

        logical_z = LogicalOperator(
            logical_type=PauliChar.Z,
            operator=PauliString(string=(PauliChar.Z, PauliChar.Z)),
            target_nodes=(var_nodes[0], var_nodes[1]),
        )

        support = graph.get_support(set(logical_z.target_nodes), logical_z.logical_type)

        assert len(support.variable_nodes) == 2
        assert len(support.check_nodes) == 1
        assert len(support.edges) == 1
        assert next(iter(support.edges)).pauli_checked == PauliChar.X
