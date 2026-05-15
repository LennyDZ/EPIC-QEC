from typing import Dict, List, Tuple
from uuid import UUID


from epic.core.compilation.measurement_record import (
    MeasurementRecordView,
)
from epic.core.data_structure import PauliChar, PauliEigenState, TannerNode
from epic.core.qec_object import (
    Detector,
    Measurement,
    DetectorGraphPort,
    QubitPortState,
    NodeKnowledge,
)
from epic.core.qec_primitives.interfaces import ExtractSyndrome, PrimitiveImplementation


class RSCSyndromeExtraction(PrimitiveImplementation[ExtractSyndrome]):
    """RSC implementation of syndrome extraction that directly measures the stabilizers without any optimization."""

    def compile(
        self,
        instruction: ExtractSyndrome,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[List[str], List[Measurement], List[Detector], DetectorGraphPort]:

        check_nodes = instruction.target.check_nodes
        reset_ancilla_instructions: List[str] = []
        stim_instructions: List[str] = []
        stim_instructions.append(f"# RSC syndrome extraction {instruction.tag}")
        measurements: Dict[TannerNode, List[Measurement]] = {}
        measurements_ordered: List[Measurement] = []
        detectors: List[Detector] = []

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

        # RESET ANCILLA

        match instruction.ancilla_reset_state:
            case PauliEigenState.Z_plus:
                reset_ancilla_instructions.append(
                    f"RZ {" ".join([str(node_to_qubit[check].integer_index) for check in check_nodes])}"
                )
            case PauliEigenState.X_plus:
                reset_ancilla_instructions.append(
                    f"RX {" ".join([str(node_to_qubit[check].integer_index) for check in check_nodes])}"
                )
            case _:
                raise ValueError(
                    f"Unsupported ancilla reset state: {instruction.ancilla_reset_state}"
                )

        stim_instructions.extend(reset_ancilla_instructions)

        # SYNDROME EXTRACTION CIRCUIT
        single_round_instructions: List[str] = []
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

        single_round_instructions.append("TICK")
        single_round_instructions.append(
            f"H {" ".join(str(node_to_qubit[xc].integer_index) for xc in x_checks)}"
        )
        single_round_instructions.append("TICK")
        for t in [t1, t2, t3, t4]:
            single_round_instructions.append(
                f"CX {" ".join(f"{str(node_to_qubit[con].integer_index)} {str(node_to_qubit[tar].integer_index)}" for con, tar in t)}"
            )
            single_round_instructions.append("TICK")
        single_round_instructions.append(
            f"H {" ".join(str(node_to_qubit[xc].integer_index) for xc in x_checks)}"
        )
        single_round_instructions.append("TICK")
        single_round_instructions.append(
            f"MRZ {" ".join(str(node_to_qubit[c].integer_index) for c in node_measured)}"
        )

        stim_instructions.append(f"REPEAT {instruction.rounds} {{")
        stim_instructions.extend([f"   {instr}" for instr in single_round_instructions])
        stim_instructions.append("}")
        for r in range(instruction.rounds):
            for m in node_measured:
                measurement = Measurement(
                    node_id=m.id,
                    parent_gadget_id=parent_gadget_id,
                    parent_primitive_id=instruction.id,
                    tag=f"{instruction.tag}_synd_{m.tag}_r{r}",
                )
                measurements.setdefault(m, []).append(measurement)
                measurements_ordered.append(measurement)

        for check in check_nodes:
            # Initial round detector
            detector_zero = instruction._detector_round_zero(
                record,
                check,
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

        return stim_instructions, measurements_ordered, detectors, new_graph_port
