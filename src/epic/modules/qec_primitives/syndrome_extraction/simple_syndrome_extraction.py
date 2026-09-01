from types import MappingProxyType
from typing import Dict, List, Tuple
from uuid import UUID
import warnings

from epic.core.compilation.measurement_record import (
    MeasurementRecordView,
)

from epic.core.data_structure import PauliChar, PauliEigenState, TannerNode, CheckNode, QuantumProgram, QProgOperation
from epic.core.qec_object import (
    Detector,
    Measurement,
    DetectorGraphPort,
    QubitPortState,
    NodeKnowledge,
)
from epic.core.qec_primitives.interfaces import ExtractSyndrome, PrimitiveImplementation


class SimpleSyndromeExtraction(PrimitiveImplementation[ExtractSyndrome]):
    """Simple implementation of syndrome extraction that directly measures the stabilizers without any optimization."""

    def compile(
        self,
        instruction: ExtractSyndrome,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[QuantumProgram, List[Measurement], List[Detector], DetectorGraphPort]:

        check_nodes = instruction.target.check_nodes
        
        program = QuantumProgram(name=f"Simple_syndrome_extraction_{instruction.tag}")
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

        program.add_qubits(list(checks_qubits.values()) + list(data_qubits.values()))

        # RESET ANCILLA
        reset_ancilla_instructions: List[QProgOperation] = []
        if instruction.ancilla_reset_state == PauliEigenState.Z_plus:
            for check in check_nodes:
                reset_ancilla_instructions.append(
                    QProgOperation(
                        name="RZ",
                        length=1,
                        targets=[checks_qubits[check]],
                    )
                )
        elif instruction.ancilla_reset_state == PauliEigenState.X_plus:
            for check in check_nodes:
                reset_ancilla_instructions.append(
                    QProgOperation(
                        name="RX",
                        length=1,
                        targets=[checks_qubits[check]],
                    )
                )
        else:
            raise ValueError(
                f"Unsupported ancilla reset state: {instruction.ancilla_reset_state}"
            )

        program.add_operations(reset_ancilla_instructions)
        single_round_instructions: List[QProgOperation] = []
        node_measured = []
        # SYNDROME EXTRACTION CIRCUIT
        for check in check_nodes:
            if check.check_type:
                check_circuit = self._extract_check_circuit(
                    checks_qubits[check].integer_index,
                    [
                        data_qubits[n].integer_index  # type: ignore
                        for n in instruction.target.get_neighbourhood(check)
                    ],
                    check.check_type,
                )
                single_round_instructions.extend(check_circuit)
                node_measured.append(check)
            

        for r in range(instruction.rounds):
            instr = single_round_instructions.copy()
            
            for m in node_measured:
                measurement = Measurement(
                    node_id=m.id,
                    parent_gadget_id=parent_gadget_id,
                    parent_primitive_id=instruction.id,
                    tag=f"synd_{m.tag}_r{r}",
                )
                instr.append(
                    QProgOperation(
                        name="MRZ",
                        length=1,
                        targets=[checks_qubits[m]],
                        measurement_id=measurement.id,
                    )
                )
                measurements.setdefault(m, []).append(measurement)
                measurements_ordered.append(measurement)
            program.add_operations(instr)
            

        for check in check_nodes:
            # Initial round detector
            neighbourhood = instruction.target.get_neighbourhood(check)
            detector_zero = self._detector_round_zero(
                record,
                check,
                det_graph_port,
                measurements[check][0],
            )
            if detector_zero is not None:
                detectors.append(detector_zero)
            # Detectors between rounds
            for r in range(1, instruction.rounds):
                previous_measurement = measurements[check][r - 1]
                current_measurement = measurements[check][r]
                detector = Detector(
                    measurements=[previous_measurement, current_measurement],
                    tag=f"detector_{check.tag}_r{r-1}_{r}",
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

    @staticmethod
    def _extract_check_circuit(
        check: int, neighbours: List[int], check_type: PauliChar
    ) -> List[QProgOperation]:
        instructions: List[QProgOperation] = []
        match (check_type):
            case PauliChar.Z:
                for dq in neighbours:
                    instructions.append(
                        QProgOperation(
                            name="CX",
                            length=1,
                            targets=[dq, check],
                        )
                    )
            case PauliChar.X:
                instructions.append(
                    QProgOperation(
                        name="H",
                        length=1,
                        targets=[check],
                    )
                )
                for dq in neighbours:
                    instructions.append(
                        QProgOperation(
                            name="CX",
                            length=1,
                            targets=[check, dq],
                        )
                    )
                instructions.append(
                    QProgOperation(
                        name="H",
                        length=1,
                        targets=[check],
                    )
                )
            case PauliChar.Y:
                instructions.append(
                    QProgOperation(
                        name="H",
                        length=1,
                        targets=[check],
                    )
                )
                for dq in neighbours:
                    instructions.append(
                        QProgOperation(
                            name="CY",
                            length=1,
                            targets=[check, dq],
                        )
                    )
                instructions.append(
                    QProgOperation(
                        name="H",
                        length=1,
                        targets=[check],
                    )
                )
            case _:
                raise ValueError(f"Non-CSS stabiliser type not supported")

        return instructions

    @staticmethod
    def _detector_round_zero(
        record: MeasurementRecordView,
        check: CheckNode,
        dgp: DetectorGraphPort,
        round_zero_measurement: Measurement,
    ) -> Detector | None:
        measurement_in_detectors = [round_zero_measurement]
        check_knowledge = dgp[check].knowledge
        match check_knowledge:
            case NodeKnowledge.STABLE:
                # By default, if a check was stable, we expect it to have the same parity as the previous round
                latest = record.latest_by_node_id(check.id)
                if latest is None:
                    raise ValueError(
                        f"No measurement found in record for stable check node {check.id}"
                    )
                measurement_in_detectors.append(latest)
            case NodeKnowledge.UNKNOWN:
                return None  # If the check is in unknown state, we cannot be sure about the outcome, so no detector is formed
            case NodeKnowledge.MZ | NodeKnowledge.MX:
                if check_knowledge.basis() != check.check_type:
                    return None  # If the check was measured in a different basis, we cannot be sure about the outcome, so no detector is formed
                else:  # if the check was measured in the same basis, the last measurement is included in the detector. (it may flip the expected parity)
                    latest = record.latest_by_node_id(check.id)
                if latest is None:
                    raise ValueError(
                        f"No measurement found in record for stable check node {check.id}"
                    )
                measurement_in_detectors.append(latest)

            case NodeKnowledge.RX | NodeKnowledge.RZ:
                # if it was reseted in the oposite bais, we cannot be sure about the outcome, so no detector is formed
                if check_knowledge.basis() != check.check_type:
                    return None
            case _:
                raise ValueError(f"Invalid known check state: {known_check_state}")

        # Handle neighbors know state.
        extra_measurements = []
        for v in dgp[check].connected_nodes:
            match dgp[v].knowledge:
                case NodeKnowledge.RZ | NodeKnowledge.RX:
                    # If some neighbor was reset, in a different basis than the check, we cannot be sure about the outcome, so no detector is formed.
                    if dgp[v].knowledge.basis() != check_knowledge.basis():
                        return None
                case NodeKnowledge.MZ | NodeKnowledge.MX:
                    # If some neighbor was measured, it is fine as long as it is in the same basis,
                    # but we need to include the latest measurement of that neighbor in the detector
                    if dgp[v].knowledge.basis() != check.check_type:
                        return None
                    lm = record.latest_by_node_id(v.id)
                    if lm is not None:
                        extra_measurements.append(lm)
                    else:
                        warnings.warn(
                            f"Neighbor {v.id} of stable check {check.id} was measured but no measurement found in record. This neighbor will be ignored in the detector formation, which may lead to missed detection events."
                        )
                case NodeKnowledge.STABLE:
                    pass  # If some neighbor was stable, it does not affect the detector formation
                case NodeKnowledge.UNKNOWN:
                    # If some of its neighbors are in unknown state, we cannot be sure about the outcome, so no detector is formed
                    return None
                case _:
                    pass

        return Detector(
            measurements=measurement_in_detectors + extra_measurements,
            tag=f"check_'{check.tag}'_r0",
        )
