from typing import Dict, List, Tuple
from uuid import UUID

from epic.core.compilation.measurement_record import (
    MeasurementRecordView,
)
from epic.core.data_structure import (
    PauliChar,
    PauliEigenState,
    TannerNode,
    QuantumProgram,
    QProgOperation,
)
from epic.core.qec_object import (
    Detector,
    Measurement,
    DetectorGraphPort,
    QubitPortState,
    NodeKnowledge,
)
from epic.core.qec_primitives.interfaces import ExtractSyndrome, PrimitiveImplementation
from epic.core.visualization.quantum_program_vis import draw_quantum_program


class RSCSyndromeExtraction(PrimitiveImplementation[ExtractSyndrome]):
    """RSC implementation of syndrome extraction that directly measures the stabilizers without any optimization."""

    def compile(
        self,
        instruction: ExtractSyndrome,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[QuantumProgram, List[Measurement], List[Detector], DetectorGraphPort]:

        check_nodes = instruction.target.check_nodes
        program = QuantumProgram(name=f"RSC_syndrome_extraction_{instruction.tag}")

        measurements: Dict[TannerNode, List[Measurement]] = {}
        measurements_ordered: List[Measurement] = []
        detectors: List[Detector] = []

        if len(check_nodes) == 0:
            return program, [], [], DetectorGraphPort()

        if len(check_nodes) > len(instruction.physical_ancilla_qubits):
            raise ValueError(f"""
                Not enough physical ancilla qubits provided for syndrome extraction.
                Required: {len(check_nodes)}, Provided: {len(instruction.physical_ancilla_qubits)}
                This schedule expect 1 ancilla per check node.
                """)

        checks_qubits = {
            check: instruction.physical_ancilla_qubits[check] for check in check_nodes
        }
        data_qubits = instruction.physical_data_qubits

        node_to_qubit = {**checks_qubits, **data_qubits}

        for q in node_to_qubit.values():
            program.add_qubit(q)
        tick_on_all_qb = QProgOperation(
            name="tick", length=0, targets=node_to_qubit.values()
        )

        # RESET ANCILLA

        reset_ancilla_instructions: List[QProgOperation] = []
        match instruction.ancilla_reset_state:
            case PauliEigenState.Z_plus:
                for check in check_nodes:
                    reset_ancilla_instructions.append(
                        QProgOperation(
                            name="RZ",
                            length=1,
                            targets=[node_to_qubit[check]],
                        )
                    )
            case PauliEigenState.X_plus:
                for check in check_nodes:
                    reset_ancilla_instructions.append(
                        QProgOperation(
                            name="RX",
                            length=1,
                            targets=[node_to_qubit[check]],
                        )
                    )
            case _:
                raise ValueError(
                    f"Unsupported ancilla reset state: {instruction.ancilla_reset_state}"
                )

        program.add_operations(reset_ancilla_instructions)

        # SYNDROME EXTRACTION CIRCUIT
        single_round_instructions: List[QProgOperation] = []
        node_measured = []
        x_checks = []
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        for check in check_nodes:
            neighourhood = instruction.target.get_neighbourhood(check)
            if check.check_type == PauliChar.X:
                x_checks.append(check)
            ne, se, nw, sw = None, None, None, None
            for n in neighourhood:
                if not isinstance(n.coordinates, tuple) or not isinstance(
                    check.coordinates, tuple
                ):
                    raise ValueError(
                        "Node coordinates must be tuples for the current partitioning logic."
                    )

                x_idx = 0 if n.coordinates[2] == check.coordinates[2] else 2
                y_idx = 1 if n.coordinates[3] == check.coordinates[3] else 3
                dx = n.coordinates[x_idx] > check.coordinates[x_idx]
                dy = n.coordinates[y_idx] > check.coordinates[y_idx]

                match (dx, dy):
                    case (False, True):
                        nw = n
                    case (False, False):
                        sw = n
                    case (True, False):
                        se = n
                    case (True, True):
                        ne = n
                    case _:
                        raise ValueError(
                            f"Unexpected relative coordinates between check node {check.id} and its neighbor {n.id}: {(dx, dy)}. This likely means that the partitioning logic does not match the expected layout."
                        )
            match check.check_type:
                case PauliChar.Z:
                    t1.append((se, check)) if se is not None else None
                    t2.append((ne, check)) if ne is not None else None
                    t3.append((sw, check)) if sw is not None else None
                    t4.append((nw, check)) if nw is not None else None
                case PauliChar.X:
                    t1.append((check, se)) if se is not None else None
                    t2.append((check, sw)) if sw is not None else None
                    t3.append((check, ne)) if ne is not None else None
                    t4.append((check, nw)) if nw is not None else None
                case _:
                    raise ValueError(
                        f"Unsupported check type: {check.check_type} in rotated surface code"
                    )
            node_measured.append(check)

        single_round_instructions.append(tick_on_all_qb)
        for xc in x_checks:
            single_round_instructions.append(
                QProgOperation(
                    name="H",
                    length=1,
                    targets={node_to_qubit[xc]},
                )
            )
        single_round_instructions.append(tick_on_all_qb)
        for t in [t1, t2, t3, t4]:
            for con, tar in t:
                single_round_instructions.append(
                    QProgOperation(
                        name="CX",
                        length=1,
                        targets=[node_to_qubit[con], node_to_qubit[tar]],
                    )
                )
            single_round_instructions.append(tick_on_all_qb)
        for xc in x_checks:
            single_round_instructions.append(
                QProgOperation(
                    name="H",
                    length=1,
                    targets=[node_to_qubit[xc]],
                )
            )
        single_round_instructions.append(tick_on_all_qb)

        for r in range(instruction.rounds):
            instr = single_round_instructions.copy()

            for m in node_measured:
                measurement = Measurement(
                    node_id=m.id,
                    parent_gadget_id=parent_gadget_id,
                    parent_primitive_id=instruction.id,
                    tag=f"{instruction.tag}_synd_{m.tag}_r{r}",
                )
                instr.append(
                    QProgOperation(
                        name="MRZ",
                        length=1,
                        targets=[node_to_qubit[m]],
                        measurement_id=measurement.id,
                    )
                )

                measurements.setdefault(m, []).append(measurement)
                measurements_ordered.append(measurement)
            program.add_operations(instr)

        for check in check_nodes:
            # Initial round detector
            detector_zero = instruction._detector_round_zero(
                record,
                check,
                instruction.target.get_neighbourhood(check),  # type: ignore
                det_graph_port,
                measurements[check][0],
                tag=f"{instruction.tag}_det_{check.tag}_r0",
            )
            if detector_zero is not None:
                detectors.append(detector_zero)
            # Detectors between rounds
            for r in range(1, instruction.rounds):
                previous_measurement = measurements[check][r - 1]
                current_measurement = measurements[check][r]
                detector = Detector(
                    measurements=[previous_measurement, current_measurement],
                    tag=f"{instruction.tag}_det_{check.tag}_r{r-1}_{r}",
                )
                detectors.append(detector)

        # Set next graph port state to STABLE for all nodes involved in the syndrome extraction
        new_graph_port = DetectorGraphPort()
        for node in instruction.target.check_nodes | instruction.target.variable_nodes:
            new_graph_port[node] = QubitPortState(
                knowledge=NodeKnowledge.STABLE,
                connected_nodes=instruction.target.get_neighbourhood(node),
            )

        return program, measurements_ordered, detectors, new_graph_port
