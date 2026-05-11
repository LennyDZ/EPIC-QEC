from types import MappingProxyType
from typing import Dict, List, Tuple, cast
from uuid import UUID

from etic.core.compilation import QuantumMemory
from etic.core.qec_object.detector import NodeKnowledge
from etic.core.compilation.measurement_record import MeasurementRecordView
from etic.core.data_structure import CheckNode, TannerNode
from etic.core.data_structure.pauli import PauliChar
from etic.core.qec_object import Detector, Measurement
from etic.core.qec_object.detector import DetectorGraphPort, QubitPortState
from etic.core.qec_primitives.interfaces import PrimitiveImplementation, Readout


class NaiveReadout(PrimitiveImplementation[Readout]):
    """Represents a simple readout of a code where all variable nodes are measured in the given basis."""

    def compile(
        self,
        instruction: Readout,
        memory: QuantumMemory,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[List[str], List[Measurement], List[Detector], DetectorGraphPort]:
        instructions: List[str] = []
        new_measurements: Dict[UUID, Measurement] = {}

        new_port_state = (
            NodeKnowledge.MX
            if instruction.readout_basis == PauliChar.X
            else NodeKnowledge.MZ
        )
        nm = []
        for node in instruction.target.variable_nodes:
            if not memory.is_allocated(node):
                raise ValueError(
                    f"Cannot measure node {node.id} in gadget={parent_gadget_id} as it is not allocated in memory."
                )
            nm.append(str(memory.get_slot(node)))
            new_m = Measurement(
                node_id=node.id,
                parent_gadget_id=parent_gadget_id,
                parent_primitive_id=instruction.id,
                tag=f"readout_{node.tag}",
            )
            new_measurements[node.id] = new_m
        instructions.append(f"M{instruction.readout_basis.value} {' '.join(nm)}")

        detectors: List[Detector] = []
        for v in list(instruction.target.variable_nodes):
            var_port = det_graph_port.get(
                v, QubitPortState(knowledge=NodeKnowledge.RZ)
            )  # Default to RZ if not present
            match var_port.knowledge:
                case NodeKnowledge.RZ | NodeKnowledge.RX:
                    if instruction.readout_basis == var_port.knowledge.basis():
                        detectors.append(
                            Detector(measurements=[new_measurements[v.id]])
                        )
                case NodeKnowledge.MZ | NodeKnowledge.MX:
                    if instruction.readout_basis == var_port.knowledge.basis():
                        lm = record.latest_by_node_id(v.id)
                        if lm is None:
                            raise ValueError(
                                f"No previous measurement found for node {v.id} required to form detector."
                            )
                        detectors.append(
                            Detector(measurements=[new_measurements[v.id], lm])
                        )
                case NodeKnowledge.STABLE:
                    continue
                case NodeKnowledge.UNKNOWN:
                    continue  # No detectors formed with variable nodes in UNKNOWN state
                case _:
                    raise ValueError(
                        f"Unexpected port state {var_port} for variable node {v.tag}"
                    )

        for c in list(instruction.target.check_nodes):
            if (
                c.check_type == instruction.readout_basis
                and det_graph_port[c].knowledge == NodeKnowledge.STABLE
            ):
                neighbors_var = instruction.target.get_neighbourhood(c)
                if all(
                    det_graph_port[nei].knowledge == NodeKnowledge.STABLE
                    for nei in neighbors_var
                ):
                    # All neighbors are STABLE, so we can form a detector with just the new measurement
                    last_check_measurement = record.latest_by_node_id(c.id)
                    neighbors_meas = [new_measurements[v.id] for v in neighbors_var]

                    detectors.append(
                        Detector(
                            measurements=neighbors_meas + [last_check_measurement],
                            tag=f"readout_{c.tag}_and_neighbors",
                        )
                    )

        new_dg_port = DetectorGraphPort()
        for node in instruction.target.variable_nodes:
            new_dg_port[node] = QubitPortState(knowledge=new_port_state)
        memory.free_qubits(list(instruction.target.variable_nodes))

        return instructions, list(new_measurements.values()), detectors, new_dg_port
