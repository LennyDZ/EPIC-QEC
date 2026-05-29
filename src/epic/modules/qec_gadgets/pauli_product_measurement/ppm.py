from typing import List, Set
import ldpc.mod2 as m2a
import numpy as np

from epic.core.data_structure.pauli import PauliString
from epic.core.data_structure.tanner_graph import TannerGraph
from epic.core.data_structure.tanner_node import TannerNode
from epic.core.language.qec_gadget import LogicGadget
from epic.core.qec_object.logical_operator import LogicalOperator


class PPM(LogicGadget):
    product_to_measure: PauliString

    @staticmethod
    def _solve_linear_system_gf2(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        augmented = np.column_stack((matrix % 2, rhs.reshape(-1, 1) % 2)).astype(
            np.int8
        )
        rref, rank, _, _ = m2a.reduced_row_echelon(augmented)
        _, _, _, pivot_cols = m2a.row_echelon(augmented)
        num_vars = matrix.shape[1]
        pivot_cols = np.asarray(pivot_cols[:rank], dtype=int)

        if np.any(pivot_cols == num_vars):
            raise ValueError("No GF(2) solution exists for the ancilla correction.")

        solution = np.zeros(num_vars, dtype=np.int8)
        for row_index, pivot_col in enumerate(pivot_cols):
            solution[pivot_col] = int(rref[row_index, num_vars])

        return solution

    @staticmethod
    def _find_correction(
        merged_code: TannerGraph,
        ancilla_system: TannerGraph,
        anticommuting_lop: List[LogicalOperator],
    ) -> Set[TannerNode]:
        """
        Find the correction to apply based on the logical operators that anticommute with the measurement operator.
        These will be expended in the merged code, in order to fix them correctly after readout of the ancilla,
        one need to know how.

        This find a set of nodes in the ancilla graph such that the union of the support of the
        anticommuting logical operators and this set is a logical operator in the merged code.
        """

        if len(anticommuting_lop) == 0:
            return set()
        if any(
            lop.logical_type != anticommuting_lop[0].logical_type
            for lop in anticommuting_lop
        ):
            raise ValueError(
                "Currently cannot handle multiple anticommuting logical operators of different types, as the correction is ambiguous."
            )

        measurement_type = anticommuting_lop[0].logical_type.dual()

        merged_code_pcm, col_nodes, _ = merged_code.parity_check_matrix

        # split merged_code_pcm in two horizonzally, to get hx and hz
        mid = merged_code_pcm.shape[1] // 2  # type: ignore
        hx = merged_code_pcm[:, :mid]
        hz = merged_code_pcm[:, mid:]
        h_meas = hx if measurement_type == "X" else hz

        col_in_ancilla = []
        col_not_in_ancilla = []
        for i, node in enumerate(col_nodes):
            if node in ancilla_system.variable_nodes:
                col_in_ancilla.append(i)
            else:
                col_not_in_ancilla.append(i)

        ancilla_col_nodes = [col_nodes[i] for i in col_in_ancilla]
        data_col_nodes = [col_nodes[i] for i in col_not_in_ancilla]

        h_ancilla = h_meas[:, col_in_ancilla]
        h_data = h_meas[:, col_not_in_ancilla]

        l = np.zeros(h_data.shape[1], dtype=int)
        for lop in anticommuting_lop:
            for i, data_node in enumerate(data_col_nodes):
                if data_node in lop.target_nodes:
                    l[i] = l[i] ^ 1

        violated_stabs = (h_data @ l) % 2

        anticommuting_checks = {
            i for i, is_violated in enumerate(violated_stabs) if is_violated == 1
        }

        if len(anticommuting_checks) == 0:
            return set()

        # Peeling: checks with a single ancilla neighbor fully determine whether
        # this ancilla qubit is in the correction. Iteratively remove those checks
        # and qubits until no degree-1 check remains.

        # Each column corresponds to an ancilla qubit
        num_anc = h_ancilla.shape[1]
        is_sparse_matrix = hasattr(h_ancilla, "getrow") and hasattr(h_ancilla, "getcol")

        def _row_neighbors(row_index: int) -> Set[int]:
            if is_sparse_matrix:
                return set(h_ancilla.getrow(row_index).indices.tolist())  # type: ignore[attr-defined]
            row = np.asarray(h_ancilla[row_index]).ravel()
            return set(np.flatnonzero(row == 1).tolist())

        def _col_neighbors(col_index: int) -> Set[int]:
            if is_sparse_matrix:
                return set(h_ancilla.getcol(col_index).indices.tolist())  # type: ignore[attr-defined]
            col = np.asarray(h_ancilla[:, col_index]).ravel()
            return set(np.flatnonzero(col == 1).tolist())

        check_to_vars = {i: _row_neighbors(i) for i in range(h_ancilla.shape[0])}

        var_to_checks = {j: _col_neighbors(j) for j in range(h_ancilla.shape[1])}

        corrections = np.zeros(num_anc, dtype=int)
        # ancilla_col_nodes - list node order
        changed = True
        while changed:
            changed = False

            # find checks of degree 1
            for c, vars_ in list(check_to_vars.items()):
                if len(vars_) == 1:
                    v = next(iter(vars_))
                    # assign ancilla variable
                    if c in anticommuting_checks:
                        corrections[v] = 1
                        for c2 in var_to_checks[v]:
                            if c2 in anticommuting_checks:
                                anticommuting_checks.remove(c2)
                            else:
                                anticommuting_checks.add(c2)
                    else:
                        corrections[v] = 0

                    for c2 in var_to_checks[v]:
                        # remove edge connected to this var note
                        if c2 != c:
                            check_to_vars[c2].discard(v)

                    # remove check
                    check_to_vars.pop(c)
                    changed = True
                    break

        # Core system (remaining unresolved part)
        remaining_checks = list(check_to_vars.keys())

        if len(remaining_checks) > 0:
            remaining_vars = sorted(
                {var for vars_ in check_to_vars.values() for var in vars_}
            )
            h_core = h_ancilla[np.ix_(remaining_checks, remaining_vars)]
            if hasattr(h_core, "toarray"):
                h_core = h_core.toarray()
            s = np.zeros(len(remaining_checks), dtype=int)
            for i, check_index in enumerate(remaining_checks):
                if check_index in anticommuting_checks:
                    s[i] = 1

            core_solution = PPM._solve_linear_system_gf2(h_core, s)
            for local_index, value in enumerate(core_solution):
                if value == 1:
                    corrections[remaining_vars[local_index]] ^= 1

        correction_nodes = set()

        for i, val in enumerate(corrections):
            if val == 1:
                correction_nodes.add(ancilla_col_nodes[i])

                print(f"In correction: {ancilla_col_nodes[i].tag}")

        return correction_nodes
