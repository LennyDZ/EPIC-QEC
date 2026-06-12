from typing import Any, Dict, List, Tuple
from uuid import UUID

from pydantic import Field

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.compilation.quantum_memory import QuantumMemory
from epic.core.data_structure.tanner_node import CheckNode, TannerNode, VariableNode
from epic.core.language.qec_gadget import CodeGadget
from epic.core.qec_object.logical_operator import LogicalOperatorUpdate
from epic.core.qec_object.observable import Observable
from epic.core.qec_object.stabilizer_code import StabilizerCode
from epic.core.qec_primitives.interfaces.apply_gate import ApplyGate
from epic.core.qec_primitives.interfaces.extract_syndrome import ExtractSyndrome
from epic.core.qec_primitives.interfaces.qec_primitive import QECPrimitive


class AutQecTransversal(CodeGadget):
    """A gadget that applies an automorphism of the code, as specified by the autqec automorphism package."""

    single_qubit_gates: List[Tuple[str, TannerNode]] = Field(
        description="A list of (gate, qubit_index) pairs specifying the circuit implementing the automorphism."
    )
    swaps: List[Tuple[TannerNode, TannerNode]] = Field(
        description="A list of (qubit_index_1, qubit_index_2) pairs specifying the swaps to be applied as part of the automorphism."
    )
    detector_check_map: Dict[CheckNode, Tuple[CheckNode, ...]] = Field(
        description="A mapping from check node indices to check node indices specifying how the automorphism permutes the checks of the code. The keys and values should correspond to the indices of the check nodes in the Tanner graph of the code."
    )
    
    swap_as_gates: bool = Field(
        default=False,
        description="Whether to implement the swaps as SWAP gates in the ApplyGate primitive, or to just relabel the qubits in the quantum memory."
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
            raise ValueError("AutQecTransversal gadget should only target one code.")

        code = resolved_targets[0]
        primitives: List[QECPrimitive] = []

        gate_layers = ["H", "CNOT", "S", "CZ", "SQRT_X", "CXX"]
        layers_members: Dict[str, List[TannerNode]] = {g: [] for g in gate_layers}

        for gate, node in self.single_qubit_gates:
            if gate not in layers_members:
                raise ValueError(
                    f"Unsupported gate '{gate}' in single_qubit_gates. Supported: {gate_layers}."
                )
            if node not in code.tanner_graph.variable_nodes:
                raise ValueError(
                    f"Qubit node {node} specified in single_qubit_gates is not in the target code."
                )
            layers_members[gate].append(node)

        for gate in gate_layers:
            members = layers_members[gate]
            if not members:
                continue
            a = ApplyGate(
                target=code.tanner_graph,
                target_nodes=set(members),
                physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(code.tanner_graph.variable_nodes),  # type: ignore
                physical_ancilla_qubits={},  # no ancillas needed for single qubit
                gates=[gate],
                tag=f"{gate}_layer_of_automorphism",
            )
            primitives.append(a)

        for i, j in self.swaps:
            if not isinstance(i, VariableNode) or not isinstance(j, VariableNode):
                raise ValueError(
                    "swaps entries must be pairs of variable nodes from the target code."
                )
            if self.swap_as_gates:
                a = ApplyGate(
                    target=code.tanner_graph,
                    target_nodes={(i, j)},  # type: ignore[arg-type]
                    physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(code.tanner_graph.variable_nodes),  # type: ignore
                    physical_ancilla_qubits={},  # no ancillas needed for swaps
                    gates=["SWAP"],
                    tag=f"swap_{i.tag}_{j.tag}_of_automorphism",
                )
                primitives.append(a)
            else:
                quantum_memory.swap_data_qubits(i, j)

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
