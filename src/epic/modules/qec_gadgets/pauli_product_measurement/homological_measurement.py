from typing import Dict, List, Set, Tuple
from uuid import UUID
from functools import reduce
from operator import or_

import numpy as np

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure.pauli import PauliChar
from epic.core.data_structure.tanner_graph import TannerEdge, TannerGraph
from epic.core.data_structure.tanner_node import CheckNode, VariableNode
from epic.core.qec_object.logical_operator import LogicalOperator, LogicalOperatorUpdate
from epic.core.qec_object.logical_qubit import LogicalQubit
from epic.core.qec_object.measurement import Measurement
from epic.core.qec_object.observable import Observable
from epic.core.qec_object.stabilizer_code import StabilizerCode
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
        modified_matrix = matrix.copy()

        # while cheeger(modified_matrix) < 1:
        #     # Find sparsest cut
        #     s = argmin(ds/d) for s in v if |s| <= |v|/2

        #     h_star = None

        return modified_matrix

    @staticmethod
    def _find_low_weight_delta_zero(
        delta1: np.ndarray, H_d: np.ndarray, f0: np.ndarray
    ) -> np.ndarray:
        return delta1

    @staticmethod
    def _build_cone_code(
        codes: Set[StabilizerCode],
        logical_operators: List[LogicalOperator],
        ptype: PauliChar,
    ) -> Tuple[TannerGraph, Set[TannerEdge]]:
        """
        Build the cone code Tanner graph and the connecting edges.

        The cone code is a Tanner graph that includes the ancilla variable nodes and check nodes, as well as the edges connecting them to the original codes' Tanner graphs.

        Args:
            codes: List of CSS codes involved in the measurement.
            logical_qubits: List of logical qubits corresponding to the codes.
            ptype: The type of Pauli operator being measured (X or Z).
        Returns:
            A tuple containing the ancilla Tanner graph and the connecting edges.
        """
        base_system: TannerGraph = reduce(or_, [code.tanner_graph for code in codes])

        indexed_var_nodes = {
            node: idx for idx, node in enumerate(base_system.variable_nodes)  # type: ignore
        }
        indexed_check_nodes = {
            node: idx for idx, node in enumerate(base_system.check_nodes)  # type: ignore
        }

        # Working with PCM here
        # take the pcm of the dual type of the measured operator (i.e H_z if we measure a X op and vice versa)
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

        # Extract the f1, f0* and delta1* matrices
        support_width = len(idx_in_supp)
        surviving_check_rows = len(indexed_check_nodes) - len(rows_to_remove)
        f1 = eq49_matrix[-n:, :support_width]
        f0_star_T = eq49_matrix[:surviving_check_rows, support_width:]
        f0_star = f0_star_T.T
        delta1_star = eq49_matrix[:surviving_check_rows, :support_width]

        delta1 = HomologicalMeasurement._greedy_to_cheeger_of_one(delta1_star)

        # add zeros columns to f0* to match the dimensions of the cone code
        h, w = delta1.shape
        num_new_cols = h - delta1_star.shape[1]
        f0 = np.hstack([f0_star, np.zeros((f0_star.shape[0], num_new_cols))])

        delta0 = HomologicalMeasurement._find_low_weight_delta_zero(delta1, h_d, f0)

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
        record: MeasurementRecordView,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:

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

        # extra_ancilla_needed = cone_code.number_of_nodes - sum(
        #     c.tanner_graph.number_of_nodes for c in set(codes)
        # )

        init_ancilla = ApplyGate(
            target=ancilla_system,
            target_nodes=ancilla_system.variable_nodes | ancilla_system.check_nodes,  # type: ignore
            gates=(
                ["RX"] if ptype == PauliChar.Z else ["RZ"]
            ),  # Init in dual of the merge type.
        )

        cone_system = ancilla_system.connect_to(
            reduce(or_, [code.tanner_graph for code in codes_set]), connecting_edges
        )
        cone_syndrome = ExtractSyndrome(
            target=cone_system,
            rounds=objective_distance,
            tag=f"hm_syndrome_{self.tag}",
        )

        ancilla_readout = Readout(
            target=ancilla_system,
            readout_basis=ptype.dual(),
            tag=f"hm_ancilla_{self.tag}",
        )

        split_syndrome = [
            ExtractSyndrome(
                target=code.tanner_graph,
                rounds=objective_distance,
                tag=f"hm_split_syndrome_{code.name}",
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
