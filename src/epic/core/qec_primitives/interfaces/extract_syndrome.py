from typing import Set

from pydantic import Field, field_validator
import warnings

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure.pauli import PauliChar
from epic.core.data_structure.tanner_node import CheckNode, VariableNode
from epic.core.qec_object.detector import Detector, DetectorGraphPort, NodeKnowledge
from epic.core.qec_object.measurement import Measurement

from ...data_structure import PauliEigenState
from .qec_primitive import QECPrimitive


class ExtractSyndrome(QECPrimitive):
    """Primitive instruction for extracting the syndrome of a quantum error correcting code."""

    rounds: int
    ancilla_reset_state: PauliEigenState = Field(
        default=PauliEigenState.Z_plus,
        description="Initial state for the ancilla qubits used in stabiliser checks.",
    )

    @field_validator("rounds")
    def validate_rounds(cls, rounds):
        """Ensure syndrome extraction runs for at least one round."""
        if rounds < 1:
            raise ValueError("Rounds must be a positive integer.")
        return rounds

    @staticmethod
    def _detector_round_zero(
        record: MeasurementRecordView,
        check: CheckNode,
        neighbors: Set[VariableNode],
        dgp: DetectorGraphPort,
        round_zero_measurement: Measurement,
        tag: str,
    ) -> Detector | None:
        measurement_in_detectors = [round_zero_measurement]
        if check not in dgp:
            check_knowledge = NodeKnowledge.RZ
        else:
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
                pass
            case _:
                raise ValueError(f"Invalid known check state: {check_knowledge}")

        # Handle neighbors know state.
        extra_measurements = []
        connected_nodes = dgp[check].connected_nodes if check in dgp else set()
        for v in neighbors | connected_nodes:
            if v not in dgp:
                neighbor_knowledge = NodeKnowledge.UNKNOWN
            else:
                neighbor_knowledge = dgp[v].knowledge
            match neighbor_knowledge:
                case NodeKnowledge.RZ | NodeKnowledge.RX:
                    # If some neighbor was reset, in a different basis than the check, we cannot be sure about the outcome, so no detector is formed.
                    if neighbor_knowledge.basis() != check.check_type:
                        return None

                case NodeKnowledge.MZ | NodeKnowledge.MX:
                    # If some neighbor was measured, it is fine as long as it is in the same basis,
                    # but we need to include the latest measurement of that neighbor in the detector

                    if neighbor_knowledge.basis() == check.check_type:
                        extra_measurements.append(record.latest_by_node_id(v.id))
                    else:
                        return None
                case NodeKnowledge.STABLE:
                    if check_knowledge != NodeKnowledge.STABLE:
                        if check not in dgp[v].connected_nodes:
                            pass
                        else:
                            # If the check is reset stable but some of its neighbors are stable, we cannot be sure about the outcome, so no detector is formed
                            return None
                    elif v not in connected_nodes:
                        return None  # If the check is stable but some of its neighbors are stable but not connected to the check, we cannot be sure about the outcome, so no detector is formed
                case NodeKnowledge.UNKNOWN:
                    # If some of its neighbors are in unknown state, we cannot be sure about the outcome, so no detector is formed
                    return None
                case _:
                    pass

        return Detector(
            measurements=measurement_in_detectors + extra_measurements,
            tag=tag,
        )
