from pydantic import Field, field_validator
import warnings

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure.tanner_node import CheckNode
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
