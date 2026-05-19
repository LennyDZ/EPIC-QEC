from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Set, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.sparse import csr_matrix

from .tanner_node import TannerNode, VariableNode, CheckNode
from .pauli import PauliChar


@dataclass(frozen=True, slots=True)
class TannerEdge:
    """
    Edge in a Tanner graph, connecting a variable node to a check node.

    Attributes:
        variable_node (VariableNode): The variable node connected by the edge.
        check_node (CheckNode): The check node connected by the edge.
        pauli_checked (PauliChar): The type of Pauli operator checked by this edge (X, Z, or Y).
    """

    variable_node: VariableNode
    check_node: CheckNode
    pauli_checked: PauliChar


class TannerGraph(BaseModel):
    """Tanner graph representation of a quantum error-correcting code.

    Attributes:
        variable_nodes (Set[VariableNode]): Set of variable nodes in the Tanner graph.
        check_nodes (Set[CheckNode]): Set of check nodes in the Tanner graph.
        edges (Set[TannerEdge]): Set of edges connecting variable nodes to check nodes.

    Properties:
        number_of_nodes: Returns the total number of nodes in the Tanner graph.
        index_by_check: Returns a mapping from check nodes to their corresponding edges.
        index_by_variable: Returns a mapping from variable nodes to their corresponding edges.
    """

    variable_nodes: Set[VariableNode] = Field(
        default_factory=set, description="Set of variable nodes in the Tanner graph."
    )
    check_nodes: Set[CheckNode] = Field(
        default_factory=set, description="Set of check nodes in the Tanner graph."
    )
    edges: Set[TannerEdge] = Field(
        default_factory=set,
        description="Set of edges connecting variable nodes to check nodes.",
    )

    @field_validator("edges")
    def validate_edges(cls, edges, info):
        """Validate that edges connect existing variable and check nodes in the Tanner graph."""
        values = info.data or {}
        var_node_ids = values.get("variable_nodes", [])
        check_node_ids = values.get("check_nodes", [])

        for edge in edges:
            if edge.variable_node not in var_node_ids:
                raise ValueError(
                    f"Edge variable node {edge.variable_node.id} not in variable nodes."
                )
            if edge.check_node not in check_node_ids:
                raise ValueError(
                    f"Edge check node {edge.check_node.id} not in check nodes."
                )
        return edges

    @model_validator(mode="after")
    def validate_check_types(self) -> "TannerGraph":
        """Validate that any declared check type matches the Pauli type of attached edges."""
        for edge in self.edges:
            declared_pauli = edge.check_node.pauli_type
            if declared_pauli is not None and edge.pauli_checked != declared_pauli:
                raise ValueError(
                    f"Edge with Pauli type {edge.pauli_checked} connected to check node {edge.check_node.id} with declared type {edge.check_node.check_type}."
                )
        return self

    @property
    def number_of_nodes(self) -> int:
        """Returns the total number of nodes in the Tanner graph."""
        return len(self.variable_nodes) + len(self.check_nodes)

    @cached_property
    def index_by_check(self) -> Dict[CheckNode, Set[TannerEdge]]:
        """Returns a mapping from check nodes to the edges connected to them."""
        index = {}
        for edge in self.edges:
            if edge.check_node not in index:
                index[edge.check_node] = set()
            index[edge.check_node].add(edge)
        return index

    @cached_property
    def index_by_variable(self) -> Dict[VariableNode, Set[TannerEdge]]:
        """Returns a mapping from variable nodes to the edges connected to them."""
        index = {}
        for edge in self.edges:
            if edge.variable_node not in index:
                index[edge.variable_node] = set()
            index[edge.variable_node].add(edge)
        return index

    @cached_property
    def max_degree(self):
        """Return the maximum degree over all variable and check nodes."""
        max = 0
        for m in self.variable_nodes | self.check_nodes:
            if self.degree(m) > max:
                max = self.degree(m)
        return max

    def get_neighbourhood(self, node: VariableNode | CheckNode) -> Set[TannerNode]:
        """Returns the set of neighboring nodes connected to the given node."""
        neighbors = set()
        if node in self.variable_nodes:
            for edge in self.index_by_variable.get(node, set()):  # type: ignore
                neighbors.add(edge.check_node)
        elif node in self.check_nodes:
            for edge in self.index_by_check.get(node, set()):  # type: ignore
                neighbors.add(edge.variable_node)
        else:
            raise ValueError("Node is not in the Tanner graph.")
        return neighbors

    def degree(self, node: VariableNode | CheckNode) -> int:
        """Returns the degree of the given node in the Tanner graph."""
        if node in self.variable_nodes:
            return len(self.index_by_variable.get(node, set()))
        elif node in self.check_nodes:
            return len(self.index_by_check.get(node, set()))
        else:
            raise ValueError("Node is not in the Tanner graph.")

    def get_support(self, logical: Set[VariableNode], type: PauliChar) -> "TannerGraph":
        """Returns the Tanner graph corresponding to the support of the given logical operator.
        O(E) on first call, O(1) on subsequent calls."""
        var_nodes = set()
        check_nodes = set()
        edges = set()
        for dq in logical:
            if dq not in self.variable_nodes:
                raise ValueError(
                    f"Logical qubit {dq} not in Tanner graph variable nodes."
                )
            var_nodes.add(dq)
            for edge in self.index_by_variable.get(dq, set()):
                if edge.pauli_checked == type.dual():
                    check_nodes.add(edge.check_node)
                    edges.add(edge)

        return TannerGraph(
            variable_nodes=var_nodes,
            check_nodes=check_nodes,
            edges=edges,
        )

    @cached_property
    def parity_check_matrix(
        self,
    ) -> Tuple[csr_matrix, List[VariableNode], List[CheckNode]]:
        """Returns the parity-check matrix representation of the Tanner graph.
        O(#Check) (+ O(E) if check index not cached).

         Returns:
            Tuple[csr_matrix, List[VariableNode], List[CheckNode]]: The parity-check matrix in CSR format,
            list of variable nodes, and list of check nodes. The list are in the same order as the matrix rows and columns.
        """
        var_node_list = list(self.variable_nodes)
        check_node_list = list(self.check_nodes)
        # Map variable node to index
        var_node_index = {node.id: idx for idx, node in enumerate(var_node_list)}
        n_var = len(var_node_list)

        check_adj = self.index_by_check

        data = []
        row_indices = []
        col_indices = []
        for row_idx, check in enumerate(check_node_list):
            if check not in check_adj:
                continue
            for edge in check_adj[check]:
                row_indices.append(row_idx)
                data.append(1)
                if (
                    edge.pauli_checked == PauliChar.X
                    or edge.pauli_checked == PauliChar.Y
                ):
                    col_indices.append(var_node_index[edge.variable_node.id])
                if (
                    edge.pauli_checked == PauliChar.Z
                    or edge.pauli_checked == PauliChar.Y
                ):
                    col_indices.append(var_node_index[edge.variable_node.id] + n_var)

        return (
            csr_matrix(
                (data, (row_indices, col_indices)),
                shape=(len(check_node_list), 2 * n_var),
            ),
            var_node_list,
            check_node_list,
        )

    @staticmethod
    def from_pcm(Hx: csr_matrix, Hz: csr_matrix, code_name: str = "") -> "TannerGraph":
        """Constructs a Tanner graph from a given CSS parity-check matrix pairs.
        This method preserve the order of the columns while tagging the variable nodes.
        Variable nodes are tagged as "v_{column_index}_{code_name}" and check nodes are tagged as "c_{row_index}_{code_name}" for X checks and "c_{row_index + num_X_checks}_{code_name}" for Z checks.

        Args:
            Hx (csr_matrix): The parity-check matrix for X stabilizers in CSR format.
            Hz (csr_matrix): The parity-check matrix for Z stabilizers in CSR format.
        Returns:
            TannerGraph: The constructed Tanner graph.
        """
        x_check_count, num_vars = 0, 0
        z_check_count, num_vars_z = 0, 0
        if Hx != None:
            x_check_count, num_vars = Hx.shape  # type: ignore
        if Hz != None:
            z_check_count, num_vars_z = Hz.shape  # type: ignore
            num_vars_z = Hz.shape[1]  # type: ignore
            if num_vars != num_vars_z and Hx and Hz:
                raise ValueError("Hx and Hz must have the same number of columns.")
            num_vars = num_vars_z
        variable_nodes = [
            VariableNode(tag=f"v_{i}_{code_name}") for i in range(num_vars)
        ]

        check_nodes = [
            CheckNode(tag=f"c_{j}_{code_name}", check_type=PauliChar.X)
            for j in range(x_check_count)
        ]

        check_nodes.extend(
            [
                CheckNode(
                    tag=f"c_{j +x_check_count}_{code_name}",
                    check_type=PauliChar.Z,
                )
                for j in range(z_check_count)
            ]
        )
        edges = set()

        for i in range(x_check_count):
            start = Hx.indptr[i]
            end = Hx.indptr[i + 1]
            for idx in range(start, end):
                j = Hx.indices[idx]
                edges.add(
                    TannerEdge(
                        variable_node=variable_nodes[j],
                        check_node=check_nodes[i],
                        pauli_checked=PauliChar.X,
                    )
                )

        for i in range(z_check_count):
            start = Hz.indptr[i]
            end = Hz.indptr[i + 1]
            for idx in range(start, end):
                j = Hz.indices[idx]
                edges.add(
                    TannerEdge(
                        variable_node=variable_nodes[j],
                        check_node=check_nodes[x_check_count + i],
                        pauli_checked=PauliChar.Z,
                    )
                )

        return TannerGraph(
            variable_nodes=set(variable_nodes),
            check_nodes=set(check_nodes),
            edges=edges,
        )

    def __or__(self, other: "TannerGraph") -> "TannerGraph":
        """Returns the union of two Tanner graphs. The resulting Tanner graph contains all variable nodes, check nodes, and edges from both graphs. If there are overlapping nodes or edges (i.e. nodes or edges with the same id), they are merged into a single node or edge in the resulting graph."""

        joined_variable_nodes = self.variable_nodes.union(other.variable_nodes)
        joined_check_nodes = self.check_nodes.union(other.check_nodes)
        joined_edges = self.edges.union(other.edges)
        return TannerGraph(
            variable_nodes=joined_variable_nodes,
            check_nodes=joined_check_nodes,
            edges=joined_edges,
        )

    def __eq__(self, value):
        """
        Returns True if the Tanner graph is equal to another Tanner graph. Two Tanner graphs are equal if they have the same variable nodes, check nodes, and edges.
        Equal Tanner graphs not only have the same structure but also the same node and edge ids.
        """
        if not isinstance(value, TannerGraph):
            return NotImplemented
        if self.variable_nodes != value.variable_nodes:
            return False
        if self.check_nodes != value.check_nodes:
            return False
        if self.edges != value.edges:
            return False
        return True

    def __contains__(self, item: TannerNode | TannerEdge) -> bool:
        """Check if a node or edge is in the Tanner graph."""
        if isinstance(item, VariableNode):
            return item in self.variable_nodes
        elif isinstance(item, CheckNode):
            return item in self.check_nodes
        elif isinstance(item, TannerEdge):
            return item in self.edges
        else:
            raise ValueError("Item must be a VariableNode, CheckNode, or TannerEdge.")

    def is_disjoint(self, other: "TannerGraph") -> bool:
        """Returns True if the Tanner graph is disjoint with another Tanner graph. Meaning they do not share any node (based on node id)."""
        if not isinstance(other, TannerGraph):
            raise ValueError("Can only check disjointness with another TannerGraph.")
        return self.variable_nodes.isdisjoint(
            other.variable_nodes
        ) and self.check_nodes.isdisjoint(other.check_nodes)

    def connect_to(
        self, other: "TannerGraph", connecting_edges: Set[TannerEdge]
    ) -> "TannerGraph":
        """Join two Tanner graphs and add the explicit edges connecting them."""
        return TannerGraph(
            variable_nodes=self.variable_nodes | other.variable_nodes,
            check_nodes=self.check_nodes | other.check_nodes,
            edges=self.edges | other.edges | connecting_edges,
        )
