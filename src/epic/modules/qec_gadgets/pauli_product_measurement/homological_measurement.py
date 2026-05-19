from typing import Dict, List, Set, Tuple
from uuid import UUID
from functools import reduce
from operator import or_

import xgi
import numpy as np
import ldpc.mod2 as m2a

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.compilation.quantum_memory import QuantumMemory
from epic.core.data_structure import (
    PauliChar,
    VariableNode,
    TannerGraph,
    CheckNode,
    TannerEdge,
)
from epic.core.qec_object import (
    LogicalOperator,
    LogicalOperatorUpdate,
    LogicalQubit,
    Measurement,
    Observable,
    StabilizerCode,
)
from epic.core.qec_primitives.interfaces.apply_gate import ApplyGate
from epic.core.qec_primitives.interfaces.extract_syndrome import ExtractSyndrome
from epic.core.qec_primitives.interfaces.qec_primitive import QECPrimitive
from epic.core.qec_primitives.interfaces.readout import Readout
from epic.modules.qec_gadgets.pauli_product_measurement.ppm import PPM
from epic.modules.stabilizers_codes.css_code import CSSCode


class HomologicalMeasurement(PPM):
    """
    Pauli product measurement on any qldpc code.

    Ref: DOI: 10.1103/PhysRevX.15.021088
    """

    @staticmethod
    def _greedy_to_cheeger_of_one(matrix: np.ndarray) -> np.ndarray:
        """
        Greedy Algorithm to add edges to achieve Cheeger constant of 1.
        Ref; Algorithm 1 in the paper

        Args:
            matrix: The input matrix an hypergraph adjacency matrix.

        Returns:
            The modified matrix with additional edges added to achieve a Cheeger constant of 1.
        """
        output_graph = matrix.copy()  # just to be safe
        xgi_graph = xgi.convert.from_incidence_matrix(output_graph.T)  # type: ignore

        def boundary_size(main_graph: xgi.Hypergraph, subset_nodes: tuple) -> float:
            count = 0
            for e in main_graph.edges.members():
                if len(e & set(subset_nodes)) == 1:
                    count += 1
            return count

        def cheeger_constant(graph: xgi.Hypergraph) -> Tuple[float, tuple]:
            subset = xgi.utils.powerset(graph.nodes, max_size=graph.num_nodes // 2)
            sparsest_cut = min(subset, key=lambda s: boundary_size(graph, s) / len(s))
            return boundary_size(graph, sparsest_cut) / len(sparsest_cut), sparsest_cut

        def min_deg_vertices(graph: xgi.Hypergraph, vertices: tuple) -> list:
            min_deg = min(graph.degree(v) for v in vertices)
            min_deg_vertices = [v for v in vertices if graph.degree(v) == min_deg]

            return min_deg_vertices

        cheeger_cst, sparsest_cut = cheeger_constant(xgi_graph)

        while cheeger_cst < 1:
            # Add an edge to it to increase cheeger:
            h_star = float("-inf")
            new_edge = None
            for v1 in min_deg_vertices(xgi_graph, sparsest_cut):
                for v2 in min_deg_vertices(
                    xgi_graph, tuple(set(xgi_graph.nodes) - set(sparsest_cut))
                ):
                    xgi_graph.add_edges_from({"tmp_edge": [v1, v2]})
                    new_cst, _ = cheeger_constant(xgi_graph)
                    if new_cst > h_star:
                        h_star = new_cst
                        new_edge = (v1, v2)
                    xgi_graph.remove_edges_from(["tmp_edge"])
            if new_edge is not None:
                xgi_graph.add_edge(new_edge, idx=xgi_graph.num_edges)
                cheeger_cst, sparsest_cut = cheeger_constant(xgi_graph)
            else:
                raise RuntimeError(
                    "Failed to find an edge to increase Cheeger constant."
                )

        im = xgi.to_incidence_matrix(xgi_graph)
        return im.toarray().T  # type: ignore

    @staticmethod
    def _random_search_for_low_weight_delta_zero(
        delta1: np.ndarray,
        H_d: np.ndarray,
        f0: np.ndarray,
        num_random_samples: int = 1000,
    ) -> np.ndarray:
        """
        Random search algorithm to find a low-weight delta zero matrix.
        Ref; Algorithm 2 in the paper

        Args:
            delta1: The delta1 matrix.
            H_d: The H_d matrix.
            f0: The f0 matrix.
            num_random_samples: The number of random samples to try.

        Returns:
            The modified delta1 matrix with low-weight delta zero.
        """
        # v is any matrix whose rows form a basis of v^T*f_0 | v in ker(H_d^T)
        v_in_ker = m2a.kernel(H_d.T)
        vT_f0 = v_in_ker @ f0
        mV = m2a.row_basis(vT_f0)
        # reduced row echelon form v
        mV = m2a.reduced_row_echelon(mV)

        # define W over GF(2) so that ker(W) = im(delta1)
        mW = None

        # add rows of V to W to zero out the pivot columns of V in W
        # put W in row echelon form with zero-rows removed
        mW = m2a.reduced_row_echelon(mW)

        def rnd_invertible_matrix(rows, cols, max_attempts=1000):
            for _ in range(max_attempts):
                m = np.random.randint(0, 2, size=(rows, cols))
                if m2a.rank(m) == m.shape[0]:  # Check if the matrix is invertible
                    return m
            raise RuntimeError(
                "Failed to generate an invertible matrix after max_attempts"
            )

        delta0 = mW

        for i in range(num_random_samples):
            mA = rnd_invertible_matrix(rows, cols)
            mB = np.random.randint(0, 2, size=(rows, cols))

            prod_1 = mA @ mW
            prod_2 = mB @ mV
            d1_max_row_w = max(np.sum(delta1, axis=1))
            if max(np.sum(prod_1 + prod_2, axis=1)) < d1_max_row_w:
                delta0 = prod_1 + prod_2
            if max(np.sum(prod_1, axis=1)) < d1_max_row_w:
                delta0 = prod_1

        return delta0

    @staticmethod
    def _build_cone_code(
        codes: Set[StabilizerCode],
        logical_operators: List[LogicalOperator],
        ptype: PauliChar,
    ) -> Tuple[TannerGraph, Set[TannerEdge]]:
        """
        Build the cone code Tanner graph and the connecting edges.

        The cone code is a Tanner graph that includes the ancilla variable nodes and check nodes, as well as the edges connecting them to the original codes' Tanner graphs.

        # The comment section numbers refer to the steps in the paper's construction (algorithm 3)

        Args:
            codes: List of CSS codes involved in the measurement.
            logical_qubits: List of logical qubits corresponding to the codes.
            ptype: The type of Pauli operator being measured (X or Z).
        Returns:
            A tuple containing the ancilla Tanner graph and the connecting edges.
        """

        # Transform Tanner graph in PCM
        base_system: TannerGraph = reduce(or_, [code.tanner_graph for code in codes])
        indexed_var_nodes = {
            node: idx for idx, node in enumerate(base_system.variable_nodes)  # type: ignore
        }
        indexed_check_nodes = {
            node: idx for idx, node in enumerate(base_system.check_nodes)  # type: ignore
        }

        h_d = np.zeros(
            (len(indexed_check_nodes), len(indexed_var_nodes)), dtype=np.int8
        )
        for e in base_system.edges:
            if e.check_node.check_type == ptype.dual():
                h_d[
                    indexed_check_nodes[e.check_node],
                    indexed_var_nodes[e.variable_node],
                ] = 1

        nz = 0
        n = len(base_system.variable_nodes)  # type: ignore

        # Build f1, f0* and delta1* by removing columns and rows according to eq.49
        idx_in_supp = set()
        for lop in logical_operators:
            for tn in lop.target_nodes:
                if tn in indexed_var_nodes:
                    idx_in_supp.add(indexed_var_nodes[tn])

        eq49_matrix = np.block([[h_d, np.eye(nz)], [np.eye(n), np.zeros((n, nz))]])

        # Remove columns among the first n that are not in the support
        cols_to_remove = [i for i in range(n) if i not in idx_in_supp]
        eq49_matrix = np.delete(eq49_matrix, cols_to_remove, axis=1)

        # Remove rows among the first m that are 0 in the first n columns
        rows_to_remove = []
        for i in range(len(indexed_check_nodes)):
            if not any(eq49_matrix[i, j] for j in range(n - len(cols_to_remove))):
                rows_to_remove.append(i)
        eq49_matrix = np.delete(eq49_matrix, rows_to_remove, axis=0)

        # 1: Extract the f1, f0* and delta1* matrices
        support_width = len(idx_in_supp)
        surviving_check_rows = len(indexed_check_nodes) - len(rows_to_remove)
        f1 = eq49_matrix[-n:, :support_width]
        f0_star_T = eq49_matrix[:surviving_check_rows, support_width:]
        f0_star = f0_star_T.T
        delta1_star = eq49_matrix[:surviving_check_rows, :support_width]

        # 2: algorithm 1
        delta1 = HomologicalMeasurement._greedy_to_cheeger_of_one(delta1_star)

        # 3: add zeros columns to f0* to match the dimensions of the cone code
        h, w = delta1.shape
        num_new_cols = h - delta1_star.shape[1]
        f0 = np.hstack([f0_star, np.zeros((f0_star.shape[0], num_new_cols))])

        # 4
        delta0 = HomologicalMeasurement._random_search_for_low_weight_delta_zero(
            delta1, h_d, f0
        )

        # 8 - 18: Cellulation if sparsity not good enough
        # if bad_sparsity:
        #     ...

        # Format back to Tanner graph objects:
        # f0 columns correspond to new nodes
        new_var_nodes = [VariableNode(tag=f"av_{i}") for i in range(f0.shape[1])]
        # new checks of the ptype to measure are rows of delta1.T (or f1 transpose)
        new_c_nodes = [
            CheckNode(tag=f"asc_{i}", check_type=ptype) for i in range(delta1.shape[0])
        ]
        # new checks of the dual type are rows of delta0
        new_dual_c_nodes = [
            CheckNode(tag=f"ascd_{i}", check_type=ptype.dual())
            for i in range(delta0.shape[0])
        ]

        # edges in the ancilla system are given by delta1.T and delta0
        ancilla_edges = set()
        for row, col in np.argwhere(delta1.T == 1):
            ancilla_edges.add(
                TannerEdge(
                    variable_node=new_var_nodes[col],
                    check_node=new_dual_c_nodes[row],
                    pauli_checked=ptype,
                )
            )
        for row, col in np.argwhere(delta0 == 1):
            ancilla_edges.add(
                TannerEdge(
                    variable_node=new_var_nodes[col],
                    check_node=new_dual_c_nodes[row],
                    pauli_checked=ptype.dual(),
                )
            )
        ancilla_system = TannerGraph(
            variable_nodes=set(new_var_nodes),
            check_nodes=set(new_c_nodes) | set(new_dual_c_nodes),
            edges=ancilla_edges,
        )

        # edges connecting existing tanner to ancilla system are given by f0 and f1.T
        connecting_edges = set()
        for row, col in np.argwhere(f1.T == 1):
            connecting_edges.add(
                TannerEdge(
                    variable_node=indexed_var_nodes.inverse[col],  # type: ignore
                    check_node=new_c_nodes[row],  # type: ignore
                    pauli_checked=ptype,
                )
            )
        for row, col in np.argwhere(f0 == 1):
            connecting_edges.add(
                TannerEdge(
                    variable_node=new_var_nodes[col],  # type: ignore
                    check_node=indexed_check_nodes.inverse[row],  # type: ignore
                    pauli_checked=ptype.dual(),
                )
            )

        return ancilla_system, connecting_edges

    def compile(
        self,
        resolved_targets: List[Tuple[LogicalQubit, StabilizerCode]],
        record,
        quantum_memory: QuantumMemory,
        timestep,
        objective_distance: int,
    ):

        codes = [code for _, code in resolved_targets]
        logical_qubits = [lq for lq, _ in resolved_targets]

        codes_set = set(codes)

        if not all(isinstance(c, CSSCode) for c in codes):
            raise ValueError("All target codes must be CSS codes.")

        if self.product_to_measure.weight != len(resolved_targets):
            raise ValueError(
                "The weight of the product to measure must match the number of targets."
            )

        # For now only allow product of Xs or product of Zs. No mixed products.
        ptype = self.product_to_measure.string[0]
        if not all(p == ptype for p in self.product_to_measure.string):
            raise ValueError(
                "All factors in the product to measure must be the same Pauli type."
            )

        lop_involved = [
            lq.logical_x if ptype == PauliChar.X else lq.logical_z
            for lq in logical_qubits
        ]

        ancilla_system, connecting_edges = self._build_cone_code(
            codes_set, lop_involved, ptype
        )

        cone_system = ancilla_system.connect_to(
            reduce(or_, [code.tanner_graph for code in codes_set]), connecting_edges
        )

        qubits_for_ancilla_system = quantum_memory.lock_ancilla_qubits(
            len(ancilla_system.variable_nodes), self.id
        )
        ancilla_sys_data_qubits = {
            v: q
            for v, q in zip(ancilla_system.variable_nodes, qubits_for_ancilla_system)
        }

        qubits_for_merged_system_checks = quantum_memory.lock_ancilla_qubits(
            len(cone_system.check_nodes), self.id
        )
        checks_qubits = {
            c: q
            for c, q in zip(cone_system.check_nodes, qubits_for_merged_system_checks)
        }

        init_ancilla = ApplyGate(
            target=ancilla_system,
            target_nodes=ancilla_system.variable_nodes | ancilla_system.check_nodes,  # type: ignore
            physical_data_qubits=ancilla_sys_data_qubits,
            physical_ancilla_qubits=checks_qubits,  # type: ignore
            gates=(
                ["RX"] if ptype == PauliChar.Z else ["RZ"]
            ),  # Init in dual of the merge type.
        )

        cone_syndrome = ExtractSyndrome(
            target=cone_system,
            rounds=objective_distance,
            physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                cone_system.variable_nodes  # type: ignore
            )
            | ancilla_sys_data_qubits,
            physical_ancilla_qubits=checks_qubits,  # type: ignore
            tag=f"hm_syndrome_{self.tag}",
        )

        ancilla_readout = Readout(
            target=ancilla_system,
            physical_data_qubits=ancilla_sys_data_qubits,
            physical_ancilla_qubits=checks_qubits,  # type: ignore
            readout_basis=ptype.dual(),
            tag=f"hm_ancilla_{self.tag}",
        )

        split_syndrome = [
            ExtractSyndrome(
                target=code.tanner_graph,
                rounds=objective_distance,
                tag=f"hm_split_syndrome_{code.name}",
                physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                    code.tanner_graph.variable_nodes  # type: ignore
                ),
                physical_ancilla_qubits=checks_qubits,  # type: ignore
            )
            for code in codes
        ]

        primitives = [init_ancilla, cone_syndrome, ancilla_readout] + split_syndrome

        observable = [
            Observable(
                logical_operators_involved=lop_involved,
                measurements={
                    Measurement(
                        n.id,
                        parent_gadget_id=self.id,
                        parent_primitive_id=cone_syndrome.id,
                    )
                    for n in ancilla_system.check_nodes  # type: ignore
                },
                tag=f"hm_PP_{self.tag}",
            )
        ]

        return {}, observable, primitives
