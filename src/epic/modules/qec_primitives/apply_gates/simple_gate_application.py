from typing import List, Set, Tuple, Union, cast
from uuid import UUID

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure import TannerNode, VariableNode
from epic.core.qec_object import Detector, Measurement, DetectorGraphPort, NodeKnowledge

from epic.core.qec_object.detector import QubitPortState
from epic.core.qec_primitives.interfaces import ApplyGate, PrimitiveImplementation


class SimpleGateApplication(PrimitiveImplementation[ApplyGate]):
    """Simple gate compilation that emits direct circuit instructions."""

    @staticmethod
    def _sanitize_target_nodes(
        target_nodes: Union[Set[TannerNode], Set[Tuple[TannerNode, ...]]],
    ) -> Set[Tuple[VariableNode, ...]]:
        """Normalize targets to a set of tuples and validate homogeneous tuple sizes."""
        if not target_nodes:
            raise ValueError("ApplyGate instruction must specify target_nodes.")

        first = next(iter(target_nodes))

        if isinstance(first, tuple):
            sanitized_targets = set(
                cast(Tuple[VariableNode, ...], group) for group in target_nodes
            )

            tuple_lengths = {len(group) for group in sanitized_targets}
            if 0 in tuple_lengths:
                raise ValueError("target_nodes cannot contain empty tuples.")
            if len(tuple_lengths) != 1:
                raise ValueError(
                    "target_nodes must contain tuples with the same length."
                )

            return sanitized_targets

        if any(isinstance(item, tuple) for item in target_nodes):
            raise ValueError(
                "target_nodes must be either a set of VariableNode or a set of tuple[VariableNode, ...]."
            )

        return {(cast(VariableNode, node),) for node in target_nodes}

    def compile(
        self,
        instruction: ApplyGate,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[List[str], List[Measurement], List[Detector], DetectorGraphPort]:

        if any(g in {"M", "MZ", "MX"} for g in instruction.gates):
            raise ValueError(
                "ApplyGate cannot compile measurement gates like M, MX, or MZ."
            )

        stim_instructions: List[str] = []
        new_dg_port = DetectorGraphPort()

        sanitized_targets = self._sanitize_target_nodes(instruction.target_nodes)

        gate_to_knowledge = {
            "RZ": NodeKnowledge.RZ,
            "RX": NodeKnowledge.RX,
        }

        mem = {
            **instruction.physical_data_qubits,
            **instruction.physical_ancilla_qubits,
        }

        for t in sanitized_targets:
            for node in t:
                if instruction.gates[-1] in gate_to_knowledge:
                    new_dg_port[node] = QubitPortState(
                        knowledge=gate_to_knowledge[instruction.gates[-1]]
                    )
                elif instruction.break_detector_graph:
                    new_dg_port[node] = QubitPortState(knowledge=NodeKnowledge.UNKNOWN)

        for gate in instruction.gates:
            slots = " ".join(
                str(mem[node].integer_index) for t in sanitized_targets for node in t
            )
            stim_instructions.append(f"{gate} {slots}")

        return stim_instructions, [], [], new_dg_port
