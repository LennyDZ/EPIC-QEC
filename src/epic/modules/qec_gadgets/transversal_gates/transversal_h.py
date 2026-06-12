from typing import List, Dict, Tuple
from uuid import UUID

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.compilation.quantum_memory import QuantumMemory
from epic.core.language.qec_gadget import CodeGadget
from epic.core.qec_object import (
    LogicalOperatorUpdate,
    StabilizerCode,
    Observable,
)
from epic.core.qec_primitives.interfaces.apply_gate import ApplyGate
from epic.core.qec_primitives.interfaces.extract_syndrome import ExtractSyndrome
from epic.core.qec_primitives.interfaces.qec_primitive import QECPrimitive


class TransversalH(CodeGadget):
    """
    Transversal implementation of the logical H gate for codes with Hx = Hz.
    """

    def compile(
        self,
        resolved_targets: List[StabilizerCode],
        record: MeasurementRecordView,
        quantum_memory: QuantumMemory,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:

        primitives: List[QECPrimitive] = []

        for code in resolved_targets:
            a = ApplyGate(
                target=code.tanner_graph,
                target_nodes=code.tanner_graph.variable_nodes,  # type: ignore
                physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                    code.tanner_graph.variable_nodes
                ),
                physical_ancilla_qubits={},  # no ancillas needed for transversal H
                gates=["H"],
                tag=f"transversal_h_{code.name}",
            )

            group_by_neighborhood = {}
            for check in code.tanner_graph.check_nodes:
                neighborhood = frozenset(code.tanner_graph.get_neighbourhood(check))
                if neighborhood not in group_by_neighborhood:
                    group_by_neighborhood[neighborhood] = []
                group_by_neighborhood[neighborhood].append(check)

            dgmap = {}
            for key, checks in group_by_neighborhood.items():
                if len(checks) != 2:
                    raise ValueError(
                        f"Each check should have exactly one other check with the same neighborhood for the transversal H construction to work. Found {len(checks)} checks with neighborhood {key}."
                    )
                dgmap[checks[0]] = (checks[1],)
                dgmap[checks[1]] = (checks[0],)

            anc_for_syndrome = quantum_memory.lock_ancilla_qubits(
                len(code.tanner_graph.check_nodes), self.id
            )
            anc_for_syndrome_map = {
                n: q for n, q in zip(code.tanner_graph.check_nodes, anc_for_syndrome)
            }

            s = ExtractSyndrome(
                target=code.tanner_graph,
                physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                    code.tanner_graph.variable_nodes
                ),
                physical_ancilla_qubits=anc_for_syndrome_map,  # type: ignore
                tag=f"extract_syndrome_{code.name}",
                rounds=objective_distance,
                detector_graph_map=dgmap,
            )

            quantum_memory.unlock_ancilla_qubits(list(anc_for_syndrome), self.id)
            primitives.extend([a, s])
        return {}, [], primitives
