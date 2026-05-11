from typing import List, Set, Tuple, Union, cast
from uuid import UUID

from etic.core.compilation import QuantumMemory
from etic.core.compilation.measurement_record import MeasurementRecordView
from etic.core.data_structure import TannerNode
from etic.core.qec_object import Detector, Measurement
from etic.core.qec_object.detector import DetectorGraphPort, NodeKnowledge, QubitPortState
from etic.core.qec_primitives.interfaces import ApplyGate, PrimitiveImplementation


class SimpleGateApplication(PrimitiveImplementation[ApplyGate]):
    """Simple gate compilation that emits direct circuit instructions."""

    @staticmethod
    def _sanitize_target_nodes(
        target_nodes: Union[Set[TannerNode], Set[Tuple[TannerNode, ...]]],
    ) -> Set[Tuple[TannerNode, ...]]:
        """Normalize targets to a set of tuples and validate homogeneous tuple sizes."""
        if not target_nodes:
            raise ValueError("ApplyGate instruction must specify target_nodes.")

        first = next(iter(target_nodes))

        if isinstance(first, tuple):
            sanitized_targets = set(
                cast(Tuple[TannerNode, ...], group) for group in target_nodes
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
                "target_nodes must be either a set of TannerNode or a set of tuple[TannerNode, ...]."
            )

        return {(cast(TannerNode, node),) for node in target_nodes}

    def compile(
        self,
        instruction: ApplyGate,
        memory: QuantumMemory,
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

        if instruction.gates[-1] in gate_to_knowledge:
            node_knowledge = gate_to_knowledge[instruction.gates[-1]]
        else:
            node_knowledge = NodeKnowledge.UNKNOWN

        for targets in sanitized_targets:
            to_alloc = []
            for node in targets:
                if not memory.is_allocated(node):
                    to_alloc.append(node)
                new_dg_port[node] = QubitPortState(knowledge=node_knowledge)

            memory.allocate_qubits(to_alloc)

        for gate in instruction.gates:
            slots = " ".join(
                str(memory.get_slot(node)) for t in sanitized_targets for node in t
            )
            stim_instructions.append(f"{gate} {slots}")

        return stim_instructions, [], [], new_dg_port
