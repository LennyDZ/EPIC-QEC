from types import MappingProxyType
from typing import Dict, List, Tuple
from unittest import case
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


class RSCSyndromeExtraction(PrimitiveImplementation[ExtractSyndrome]):
    """RSC implementation of syndrome extraction that directly measures the stabilizers without any optimization."""

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
        stim_instructions.append(f"# RSC syndrome extraction {instruction.tag}")
        measurements: Dict[TannerNode, List[Measurement]] = {}
        measurements_ordered: List[Measurement] = []
        detectors: List[Detector] = []

        to_alloc: List[TannerNode] = []
        for n in instruction.target.variable_nodes | instruction.target.check_nodes:
            if not memory.is_allocated(n):
                to_alloc.append(n)
        memory.allocate_qubits(to_alloc)

        # RESET ANCILLA

        match instruction.ancilla_reset_state:
            case PauliEigenState.Z_plus:
                reset_ancilla_instructions.append(
                    f"RZ {" ".join([str(memory.get_slot(check)) for check in check_nodes])}"
                )
            case PauliEigenState.X_plus:
                reset_ancilla_instructions.append(
                    f"RX {" ".join([str(memory.get_slot(check)) for check in check_nodes])}"
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
            f"H {" ".join(str(memory.get_slot(xc)) for xc in x_checks)}"
        )
        single_round_instructions.append("TICK")
        for t in [t1, t2, t3, t4]:
            single_round_instructions.append(
                f"CX {" ".join(f"{str(memory.get_slot(con))} {str(memory.get_slot(tar))}" for con, tar in t)}"
            )
            single_round_instructions.append("TICK")
        single_round_instructions.append(
            f"H {" ".join(str(memory.get_slot(xc)) for xc in x_checks)}"
        )
        single_round_instructions.append("TICK")
        single_round_instructions.append(
            f"MRZ {" ".join(str(memory.get_slot(c)) for c in node_measured)}"
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
            detector_zero = self._detector_round_zero(
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

    @staticmethod
    def _detector_round_zero(
        record: MeasurementRecordView,
        check: CheckNode,
        dgp: DetectorGraphPort,
        round_zero_measurement: Measurement,
        tag: str,
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
            tag=tag,
        )
