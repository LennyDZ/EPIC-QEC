from typing import Dict, Self, Tuple

from pydantic import Field, PrivateAttr, field_validator, model_validator
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
    detector_graph_map: Dict[CheckNode, Tuple[CheckNode, ...]] = Field(
        default_factory=dict,
        description="Define relationship between checks of the previous round, to build detectors. Default is identity mapping, i.e. each check is only connected to itself in the previous round.",
    )

    _has_detector_graph_map: bool = PrivateAttr(default=False, init=False)

    @field_validator("rounds")
    def validate_rounds(cls, rounds):
        """Ensure syndrome extraction runs for at least one round."""
        if rounds < 1:
            raise ValueError("Rounds must be a positive integer.")
        return rounds

    @model_validator(mode="after")
    def validate_detector_graph_map(self) -> Self:
        if not self.detector_graph_map:
            # If no mapping is provided, we assume an identity mapping (each check is only connected to itself in the previous round)
            self.detector_graph_map = {
                check: (check,) for check in self.target.check_nodes
            }
        else:
            for k, v in self.detector_graph_map.items():
                if k not in self.target.check_nodes:
                    raise ValueError(
                        f"Key {k} in detector_graph_map is not a check node in the target Tanner graph."
                    )
                if any(
                    mapped_check not in self.target.check_nodes for mapped_check in v
                ):
                    raise ValueError(
                        f"One or more values in tuple for key {k} are not check nodes in the target Tanner graph."
                    )
            self._has_detector_graph_map = True
        return self

    @model_validator(mode="after")
    def validate_target_if_has_detector_graph_map(self) -> Self:
        if self._has_detector_graph_map:
            if not all(
                check in self.detector_graph_map for check in self.target.check_nodes
            ):
                raise ValueError(
                    "When a custom detector graph map is provided, all check nodes in the target Tanner graph must be included as keys in it."
                )
        return self

    def _detector_round_zero(
        self,
        record: MeasurementRecordView,
        check: CheckNode,
        neighbors: Set[VariableNode],
        dgp: DetectorGraphPort,
        round_zero_measurement: Measurement,
        tag: str,
    ) -> Detector | None:
        measurement_in_detectors = [round_zero_measurement]

        if self._has_detector_graph_map:
            # If a custom mapping is provided, we use it to determine the connections in the detector graph
            if check not in self.detector_graph_map:
                raise ValueError(
                    f"Check {check} is not in the provided detector_graph_map."
                )
            if dgp[check].knowledge != NodeKnowledge.STABLE:
                raise ValueError(f"""
                    Check {check} is expected to be stable when using a custom detector_graph_map, but it is in state {dgp[check].knowledge}.
                    This is required because mapping checks to other than themselves is only possible if they were already measured in the previous round.
                    This is required because mapping checks to other than themselves is not possible if the detector should rely on neighbors measurements.
                    """)
            neighbors = self.target.get_neighbourhood(check)  # type: ignore
            if not dgp[check].connected_nodes.issubset(neighbors):
                raise ValueError(f"""
                    Check {check} is expected to only have relations with node included in this syndrome measurement.
                    This is required because otherwise, effects of nodes measured in external scope would be wrongly combine with the mapping's target.
                    """)

            measurement_in_detectors.extend(
                [record.latest_by_node_id(n.id) for n in self.detector_graph_map[check]]
            )
            return Detector(
                measurements=measurement_in_detectors,
                tag=f"det_from_custom_map_[{self.detector_graph_map[check][0].tag}]-[{check.tag}]",
            )

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
                return None  # If the check was in unknown state, we cannot be sure about the outcome, so no detector is formed
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
