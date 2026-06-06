from typing import Any, Dict, List, Set, Tuple, cast
from uuid import UUID
from functools import reduce
from operator import or_

import sympy as sp
import xgi
import numpy as np
import ldpc.mod2 as m2a

from epic.core.visualization.tanner_graph_vis import TannerGraphVisualizer as TGV
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
from epic.core.data_structure.graph_algorithm import GraphAlgorithm as ga


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
        xgi_graph = xgi.convert.from_incidence_matrix(output_graph.T, nodelabels=range(output_graph.shape[1]), edgelabels=range(output_graph.shape[0]))  # type: ignore

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

        im, nodes, edges = xgi.to_incidence_matrix(xgi_graph, index=True)

        node_to_row = {label: idx for idx, label in cast(Dict[int, int], nodes).items()}
        edge_to_col = {label: idx for idx, label in cast(Dict[int, int], edges).items()}

        original_node_labels = list(range(output_graph.shape[1]))
        original_edge_labels = list(range(output_graph.shape[0]))

        node_label_set = set(node_to_row)
        edge_label_set = set(edge_to_col)

        ordered_node_labels = original_node_labels + sorted(
            node_label_set - set(original_node_labels)
        )
        ordered_edge_labels = original_edge_labels + sorted(
            edge_label_set - set(original_edge_labels)
        )

        reordered = cast(Any, im).toarray()[
            [node_to_row[label] for label in ordered_node_labels],
            :,
        ][:, [edge_to_col[label] for label in ordered_edge_labels]]

        return reordered.T  # type: ignore

    @staticmethod
    def _random_search_for_low_weight_delta_zero(
        delta1: np.ndarray,
        H_d: np.ndarray,
        f0: np.ndarray,
        num_random_samples: int = 100,
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
        v_in_ker = m2a.kernel(H_d.T).toarray()  # type: ignore
        vT_f0 = (v_in_ker @ f0) % 2

        col_basis = m2a.row_basis(vT_f0.T).astype(np.int8)  # type: ignore
        mV = col_basis.T
        # reduced row echelon form v
        mV, _, _, tc = m2a.reduced_row_echelon(mV)
        mV = (mV @ tc.T) % 2
        mV, rank_v, _, pivot_cols = m2a.row_echelon(mV)
        pivot_cols = np.asarray(pivot_cols[:rank_v], dtype=int)

        # define W over GF(2) so that ker(W) = im(delta1)
        left_null_basis = m2a.nullspace(delta1.copy().T)
        mW = left_null_basis  # type: ignore

        if not np.allclose(mW @ delta1 % 2, 0):
            raise ValueError("mW @ delta1 % 2 != 0")
        # add rows of V to W to zero out the pivot columns of V in W
        for row_index in range(mW.shape[0]):
            for pivot_row_index, pivot_col in enumerate(pivot_cols):
                if mW[row_index, pivot_col]:  # type: ignore
                    mW[row_index] = (mW[row_index] + mV[pivot_row_index]) % 2  # type: ignore

        # put W in row echelon form with zero-rows removed
        mW, _, _, tc = m2a.reduced_row_echelon(mW)
        mW = (mW @ tc.T) % 2
        mW = mW[~np.all(mW == 0, axis=1)]

        if not np.allclose(mW @ delta1 % 2, 0):
            raise ValueError(f"Failed. {mW @ delta1 % 2}")

        if mW.shape[0] == 0 and mV.shape[0] == 0:
            return np.empty((0, 0), dtype=np.int8)

        def rnd_invertible_matrix(rows, cols, max_attempts=100):
            for _ in range(max_attempts):
                m = np.random.randint(0, 2, size=(rows, cols))
                if m2a.rank(m) == m.shape[0]:  # Check if the matrix is invertible
                    return m
            raise RuntimeError(
                "Failed to generate an invertible matrix after max_attempts"
            )

        delta0 = mW
        d0_row, d0_col = delta0.shape

        for i in range(num_random_samples):
            mA = rnd_invertible_matrix(mW.shape[0], mW.shape[0])

            mB = np.random.randint(0, 2, size=(d0_col, mV.shape[0]))

            prod_1 = (mA @ mW) % 2
            prod_2 = (mB @ mV) % 2
            d0_max_row_w = max(np.sum(delta0, axis=1))
            if prod_2.any():
                if max(np.sum((prod_1 + prod_2) % 2, axis=1)) < d0_max_row_w:
                    delta0 = (prod_1 + prod_2) % 2
            if max(np.sum(prod_1, axis=1)) < d0_max_row_w:
                delta0 = prod_1

        if np.any((delta1.T @ delta0.T) % 2):
            raise ValueError("Invalid delta0: not in kernel of delta1^T.")

        return delta0.astype(np.int8)  # type: ignore

    # @staticmethod
    # def _build_from_cellulation(
    #     delta1_star: np.ndarray, f0_star: np.ndarray, H_d: np.ndarray, max_cycle_weight: int = 10
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Build the cone code from a cellulation
    #     """
    #     for row in delta1:
    #         new_rows = ...
    #         new_zero_row = np.zeros_like(row)
    #         d1 = ...
    #         f0 = ...

    #     d1 = HomologicalMeasurement._greedy_to_cheeger_of_one(d1)
    #     d0 = HomologicalMeasurement._random_search_for_low_weight_delta_zero(d1, H_d, f0)

    #     for row in d0:
    #         if np.sum(row) > limit:
    #             #Add new edges (rows of ∂1, along with
    #             # corresponding zero-columns of f0) within the
    #             # cycle defined by c to break it into smaller
    #             # cycles. This results in replacing the high
    #             # weight row c with multiple lower weight rows
    #             # corresponding to the new cycles.

    #     return d0, d1, f0

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

        d_checks = [node for node in base_system.check_nodes if node.check_type == ptype.dual()]  # type: ignore
        checks = [node for node in base_system.check_nodes if node.check_type == ptype]  # type: ignore

        indexed_dual_check_nodes = {
            node: idx for idx, node in enumerate(d_checks)  # type: ignore
        }

        indexed_other_check_nodes = {
            node: idx for idx, node in enumerate(checks)  # type: ignore
        }

        h_d = np.zeros(
            (len(indexed_dual_check_nodes), len(indexed_var_nodes)), dtype=np.int8
        )
        h_o = np.zeros(
            (len(indexed_other_check_nodes), len(indexed_var_nodes)), dtype=np.int8
        )

        for e in base_system.edges:
            if e.check_node in indexed_dual_check_nodes:
                h_d[
                    indexed_dual_check_nodes[e.check_node],
                    indexed_var_nodes[e.variable_node],
                ] = 1
            elif e.check_node in indexed_other_check_nodes:
                h_o[
                    indexed_other_check_nodes[e.check_node],
                    indexed_var_nodes[e.variable_node],
                ] = 1

        nz = h_d.shape[0]
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
        for i in range(len(indexed_dual_check_nodes)):
            if not any(eq49_matrix[i, j] for j in range(n - len(cols_to_remove))):
                rows_to_remove.append(i)
        eq49_matrix = np.delete(eq49_matrix, rows_to_remove, axis=0)

        # 1: Extract the f1, f0* and delta1* matrices
        support_width = len(idx_in_supp)
        surviving_check_rows = len(indexed_dual_check_nodes) - len(rows_to_remove)
        f1 = eq49_matrix[-n:, :support_width]
        f0_star_T = eq49_matrix[:surviving_check_rows, support_width:]
        f0_star = f0_star_T.T
        delta1_star = eq49_matrix[:surviving_check_rows, :support_width]

        # 2: algorithm 1
        delta1 = HomologicalMeasurement._greedy_to_cheeger_of_one(delta1_star)

        # 3: add zeros columns to f0* to match the dimensions of the cone code
        # delta1 has shape (new_num_checks, support_width); the Cheeger algorithm
        # may have added extra checks (rows) beyond delta1_star.shape[0], each of
        # which corresponds to a new ancilla variable node (column of f0).
        h, w = delta1.shape
        num_new_cols = h - delta1_star.shape[0]

        f0 = np.hstack([f0_star, np.zeros((f0_star.shape[0], num_new_cols))])

        # 4
        delta0 = HomologicalMeasurement._random_search_for_low_weight_delta_zero(
            delta1, h_d, f0
        )

        # 8 - 18: Cellulation if sparsity not good enough
        # if bad_sparsity:
        #     delta1, delta0, f0 = HomologicalMeasurement._build_from_cellulation(
        #         delta1_star, f0_star
        #     )

        # Format back to Tanner graph objects.
        # If all existing nodes carry 4D coordinates (x, y, sys_x, sys_y), assign the
        # ancilla nodes to a new system row so the TannerGraphVisualizer 4D layout keeps
        # each sub-system in its own subplot.
        _all_existing = list(base_system.variable_nodes) + list(base_system.check_nodes)
        _coord_dims = {
            len(n.coordinates) for n in _all_existing if n.coordinates is not None
        }
        if len(_coord_dims) == 1 and next(iter(_coord_dims)) == 4:
            _existing_systems = {
                (n.coordinates[2], n.coordinates[3])
                for n in _all_existing
                if n.coordinates is not None
            }
            _anc_sys = (0, max(s[1] for s in _existing_systems) + 1)
            _anc_coords_var: list[tuple[int, ...] | None] = [
                (i, 0) + _anc_sys for i in range(f0.shape[1])
            ]
            _anc_coords_c: list[tuple[int, ...] | None] = [
                (i, -1) + _anc_sys for i in range(delta1.shape[1])
            ]
            _anc_coords_dc: list[tuple[int, ...] | None] = [
                (i, 1) + _anc_sys for i in range(delta0.shape[0])
            ]
        else:
            _anc_coords_var = [None] * f0.shape[1]
            _anc_coords_c = [None] * delta1.shape[1]
            _anc_coords_dc = [None] * delta0.shape[0]

        # f0 columns correspond to new nodes
        new_var_nodes = [
            VariableNode(tag=f"av_{i}", coordinates=_anc_coords_var[i])
            for i in range(f0.shape[1])
        ]
        # new checks of the ptype to measure are rows of delta1.T (or f1 transpose)
        new_c_nodes = [
            CheckNode(
                tag=f"asc{ptype.value}_{i}",
                check_type=ptype,
                coordinates=_anc_coords_c[i],
            )
            for i in range(delta1.shape[1])
        ]
        # new checks of the dual type are rows of delta0
        new_dual_c_nodes = [
            CheckNode(
                tag=f"ascd_{ptype.dual().value}_{i}",
                check_type=ptype.dual(),
                coordinates=_anc_coords_dc[i],
            )
            for i in range(delta0.shape[0])
        ]

        # edges in the ancilla system are given by delta1.T and delta0
        ancilla_edges = set()
        for row, col in np.argwhere(delta1.T == 1):
            ancilla_edges.add(
                TannerEdge(
                    variable_node=new_var_nodes[col],
                    check_node=new_c_nodes[row],
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
                    variable_node=next(k for k, v in indexed_var_nodes.items() if v == col),  # type: ignore
                    check_node=new_c_nodes[row],  # type: ignore
                    pauli_checked=ptype,
                )
            )
        for row, col in np.argwhere(f0 == 1):
            connecting_edges.add(
                TannerEdge(
                    variable_node=new_var_nodes[col],  # type: ignore
                    check_node=next(k for k, v in indexed_dual_check_nodes.items() if v == row),  # type: ignore
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

        anticommuting_lop = [
            lq.logical_z if ptype == PauliChar.X else lq.logical_x
            for lq in logical_qubits
        ]

        ancilla_system, connecting_edges = self._build_cone_code(
            codes, lop_involved, ptype  # type: ignore
        )

        cone_system = ancilla_system.connect_to(
            reduce(or_, [code.tanner_graph for code in codes]), connecting_edges
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
            target_nodes=ancilla_system.variable_nodes,  # type: ignore
            physical_data_qubits=ancilla_sys_data_qubits,
            physical_ancilla_qubits=checks_qubits,  # type: ignore
            gates=(
                ["RX"] if ptype == PauliChar.Z else ["RZ"]
            ),  # Init in dual of the merge type.
        )

        ptype_anc_syd = ExtractSyndrome(
            target=TannerGraph(
                variable_nodes=cone_system.variable_nodes,
                check_nodes=set(
                    [c for c in ancilla_system.check_nodes if c.check_type == ptype]
                    + [
                        c
                        for c in cone_system.check_nodes
                        if c.check_type == ptype.dual()
                    ]
                ),
                edges={
                    e
                    for e in cone_system.edges
                    if e.check_node in ancilla_system.check_nodes
                    and e.check_node.check_type == ptype
                }
                | {
                    e
                    for e in cone_system.edges
                    if e.check_node.check_type == ptype.dual()
                },
            ),  # type: ignore
            rounds=1,
            physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                cone_system.variable_nodes  # type: ignore
            )
            | ancilla_sys_data_qubits,
            physical_ancilla_qubits=checks_qubits,  # type: ignore
            tag=f"hm_1st_syndrome_{self.tag}",
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
            physical_ancilla_qubits={k: v for k, v in checks_qubits.items() if isinstance(k, CheckNode) and k in ancilla_system.check_nodes},  # type: ignore
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

        primitives = [
            init_ancilla,
            ptype_anc_syd,
            cone_syndrome,
            ancilla_readout,
        ] + split_syndrome

        observable = [
            Observable(
                logical_operators_involved=lop_involved,
                measurements={
                    Measurement(
                        n.id,
                        parent_gadget_id=self.id,
                        parent_primitive_id=ptype_anc_syd.id,
                    )
                    for n in ancilla_system.check_nodes
                    if n.check_type == ptype  # type: ignore
                },
                tag=f"hm_PP_{self.tag}",
            )
        ]

        correction = self._find_correction(
            cone_system, ancilla_system, anticommuting_lop
        )

        lop_update = LogicalOperatorUpdate(
            new_correction={
                Measurement(
                    n.id,
                    parent_gadget_id=self.id,
                    parent_primitive_id=ancilla_readout.id,
                )
                for n in correction
            },
        )

        return {anticommuting_lop[0].id: lop_update}, observable, primitives
