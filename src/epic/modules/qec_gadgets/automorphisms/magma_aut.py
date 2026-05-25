from typing import Dict, List, Tuple
from uuid import UUID
from uuid import UUID

from pydantic import Field

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.compilation.quantum_memory import QuantumMemory
from epic.core.data_structure.tanner_node import CheckNode, TannerNode
from epic.core.language.qec_gadget import CodeGadget
from epic.core.qec_object.logical_operator import LogicalOperatorUpdate
from epic.core.qec_object.observable import Observable
from epic.core.qec_object.stabilizer_code import StabilizerCode
from epic.core.qec_primitives.interfaces.apply_gate import ApplyGate
from epic.core.qec_primitives.interfaces.extract_syndrome import ExtractSyndrome
from epic.core.qec_primitives.interfaces.qec_primitive import QECPrimitive


class MagmaAut(CodeGadget):
    """A gadget that applies an automorphism of the code, as specified by the autqec automorphism package."""

    single_qubit_gates: List[Tuple[str, TannerNode]] = Field(
        description="A list of (gate, qubit_index) pairs specifying the circuit implementing the automorphism."
    )
    swaps: List[Tuple[TannerNode, TannerNode]] = Field(
        description="A list of (qubit_index_1, qubit_index_2) pairs specifying the swaps to be applied as part of the automorphism."
    )
    detector_check_map: Dict[CheckNode, Tuple[CheckNode]] = Field(
        description="A mapping from check node indices to check node indices specifying how the automorphism permutes the checks of the code. The keys and values should correspond to the indices of the check nodes in the Tanner graph of the code."
    )

    def compile(
        self,
        resolved_targets: List[StabilizerCode],
        record: MeasurementRecordView,
        quantum_memory: QuantumMemory,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:

        if len(resolved_targets) != 1:
            raise ValueError("MagmaAut gadget should only target one code.")

        code = resolved_targets[0]
        primitives: List[QECPrimitive] = []

        layers_members: List[List[TannerNode]] = [[] for _ in range(6)]
        gate_layers = ["H", "CNOT", "S", "CZ", "SQRT-X", "CXX"]

        for g, m in zip(gate_layers, layers_members):
            if any(node not in code.tanner_graph.variable_nodes for node in m):
                raise ValueError(
                    f"Qubit index {m} specified in single_qubit_gates is not in the target code."
                )
            a = ApplyGate(
                target=code.tanner_graph,
                target_nodes=set(m),
                physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(code.tanner_graph.variable_nodes),  # type: ignore
                physical_ancilla_qubits={},  # no ancillas needed for single qubit
                gates=[g],
                tag=f"{g}_layer_of_automorphism",
            )
            primitives.append(a)

        for i, j in self.swaps:
            a = ApplyGate(
                target=code.tanner_graph,
                target_nodes={i, j},  # type: ignore
                physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(code.tanner_graph.variable_nodes),  # type: ignore
                physical_ancilla_qubits={},  # no ancillas needed for swaps
                gates=["SWAP"],
                tag=f"swap_{i.tag}_{j.tag}_of_automorphism",
            )
            primitives.append(a)

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
            detector_graph_map=self.detector_check_map,
        )

        primitives.append(s)
        quantum_memory.unlock_ancilla_qubits(list(anc_for_syndrome), self.id)

        return {}, [], primitives
