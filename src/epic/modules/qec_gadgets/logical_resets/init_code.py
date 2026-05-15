from collections import defaultdict
from typing import Dict, List, Set, Tuple, cast
from uuid import UUID

from pydantic import Field

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.compilation.quantum_memory import QuantumMemory
from epic.core.data_structure import PauliEigenState, TannerNode
from epic.core.data_structure.physical_qubit import PhysicalQubit
from epic.core.data_structure.tanner_graph import TannerGraph
from epic.core.language import CodeGadget
from epic.core.qec_object import LogicalOperatorUpdate, Observable
from epic.core.qec_object.stabilizer_code import StabilizerCode
from epic.core.qec_primitives.interfaces import ApplyGate, QECPrimitive
from epic.core.qec_primitives.interfaces.extract_syndrome import ExtractSyndrome


class InitCode(CodeGadget):
    """
    A gadget representing the initialization of a logical qubit in a stabilizer code.
    """

    initial_state: PauliEigenState = Field(
        description="The initial state of the code. "
    )

    def compile(
        self,
        resolved_targets: List[StabilizerCode],
        record: MeasurementRecordView,
        quantum_memory: QuantumMemory,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:
        gates = []
        match self.initial_state:
            case PauliEigenState.X_plus:
                gates = ["RX"]
            case PauliEigenState.X_minus:
                gates = ["RX", "Z"]
            case PauliEigenState.Z_plus:
                gates = ["RZ"]
            case PauliEigenState.Z_minus:
                gates = ["RZ", "X"]
            case _:
                raise ValueError(
                    "Unsupported initial eigenstate, only X and Z basis are allowed"
                )
        primitives: List[QECPrimitive] = []
        # Lock 1 ancilla per checks:
        ancilla_locked: Dict[UUID, Dict[TannerNode, PhysicalQubit]] = defaultdict(dict)
        for code in resolved_targets:
            anc = quantum_memory.lock_ancilla_qubits(
                n=len(code.tanner_graph.check_nodes), requestor_id=self.id
            )
            ancilla_locked[code.id] = {
                n: q for n, q in zip(code.tanner_graph.check_nodes, anc)
            }

        for code in resolved_targets:
            primitives.append(
                ApplyGate(
                    target=code.tanner_graph,
                    physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                        code.tanner_graph.variable_nodes
                    ),
                    physical_ancilla_qubits={},
                    target_nodes=code.tanner_graph.variable_nodes,  # type: ignore
                    gates=gates,
                )
            )
            primitives.append(
                ExtractSyndrome(
                    target=code.tanner_graph,
                    physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                        code.tanner_graph.variable_nodes
                    ),
                    physical_ancilla_qubits=ancilla_locked[code.id],
                    distance=objective_distance,
                    rounds=objective_distance,
                )
            )

        for code, anc_checks_map in ancilla_locked.items():
            quantum_memory.unlock_ancilla_qubits(
                list(anc_checks_map.values()), owner_id=self.id
            )

        return {}, [], primitives
