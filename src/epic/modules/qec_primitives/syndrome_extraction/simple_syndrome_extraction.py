from types import MappingProxyType
from typing import Dict, List, Tuple
from uuid import UUID
import warnings

from epic.core.qec_object.detector import NodeKnowledge
from epic.core.compilation.measurement_record import (
    MeasurementRecord,
    MeasurementRecordView,
)
from epic.core.compilation.quantum_memory import QuantumMemory
from epic.core.data_structure import PauliChar, PauliEigenState, TannerNode
from epic.core.data_structure.tanner_node import CheckNode
from epic.core.qec_object import Detector, Measurement
from epic.core.qec_object.detector import DetectorGraphPort, QubitPortState
from epic.core.qec_primitives.interfaces import ExtractSyndrome, PrimitiveImplementation


class SimpleSyndromeExtraction(PrimitiveImplementation[ExtractSyndrome]):
    """Simple implementation of syndrome extraction that directly measures the stabilizers without any optimization."""

    def compile(
        self,
        instruction: ExtractSyndrome,
        memory: QuantumMemory,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[List[str], List[Measurement], List[Detector], DetectorGraphPort]:

        check_nodes = instruction.target.check_nodes
        reset_ancilla_instructions: List[str] = []
        stim_instructions: List[str] = []
        measurements: Dict[TannerNode, List[Measurement]] = {}
        measurements_ordered: List[Measurement] = []
        detectors: List[Detector] = []
        to_alloc: List[TannerNode] = []

        for n in instruction.target.variable_nodes | instruction.target.check_nodes:
            if not memory.is_allocated(n):
                to_alloc.append(n)
        memory.allocate_qubits(to_alloc)

        # RESET ANCILLA

        if instruction.ancilla_reset_state == PauliEigenState.Z_plus:
            reset_ancilla_instructions.append(
                f"RZ {" ".join([str(memory.get_slot(check)) for check in check_nodes])}"
            )
        elif instruction.ancilla_reset_state == PauliEigenState.X_plus:
            reset_ancilla_instructions.append(
                f"RX {" ".join([str(memory.get_slot(check)) for check in check_nodes])}"
            )
        else:
            raise ValueError(
                f"Unsupported ancilla reset state: {instruction.ancilla_reset_state}"
            )

        stim_instructions.extend(reset_ancilla_instructions)
        single_round_instructions: List[str] = []
        node_measured = []
        # SYNDROME EXTRACTION CIRCUIT
        for check in check_nodes:
            single_round_instructions.append(
                f"# Stab: {check.tag}, type: {check.check_type}"
            )  # for clarity in the generated stim code
            if check.check_type:
                check_circuit = self._extract_check_circuit(
                    memory.get_slot(check),
                    [
                        memory.get_slot(n)
                        for n in instruction.target.get_neighbourhood(check)
                    ],
                    check.check_type,
                )
                single_round_instructions.extend(check_circuit)
                node_measured.append(check)
        # R round of circuit extraction into MRZ
        stim_instructions.append(f"REPEAT {instruction.rounds} {{")
        stim_instructions.extend(
            [f"    {instr}" for instr in single_round_instructions]
        )
        stim_instructions.append(
            f"    MRZ {" ".join(str(memory.get_slot(n)) for n in node_measured)}"
        )
        stim_instructions.append("}")

        for r in range(instruction.rounds):
            for m in node_measured:
                measurement = Measurement(
                    node_id=m.id,
                    parent_gadget_id=parent_gadget_id,
                    parent_primitive_id=instruction.id,
                    tag=f"synd_{m.tag}_r{r}",
                )
                measurements.setdefault(m, []).append(measurement)
                measurements_ordered.append(measurement)

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

        return stim_instructions, measurements_ordered, detectors, new_graph_port

    @staticmethod
    def _extract_check_circuit(
        check: int, neighbours: List[int], check_type: PauliChar
    ) -> List[str]:
        stim_instructions: List[str] = []
        match (check_type):
            case PauliChar.Z:
                for dq in neighbours:
                    stim_instructions.append(f"CX {dq} {check}")
            case PauliChar.X:
                stim_instructions.append(f"H {check}")
                for dq in neighbours:
                    stim_instructions.append(f"CX {check} {dq}")
                stim_instructions.append(f"H {check}")
            case PauliChar.Y:
                stim_instructions.append(f"H {check}")
                for dq in neighbours:
                    stim_instructions.append(f"CY {check}, {dq}")
                stim_instructions.append(f"H {check}")
            case _:
                raise ValueError(f"Non-CSS stabiliser type not supported")

        return stim_instructions

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
