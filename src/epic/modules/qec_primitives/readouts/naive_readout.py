from typing import Dict, List, Tuple
from uuid import UUID

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure import PauliChar
from epic.core.qec_object import (
    Detector,
    Measurement,
    DetectorGraphPort,
    QubitPortState,
    NodeKnowledge,
)
from epic.core.qec_primitives.interfaces import PrimitiveImplementation, Readout


class NaiveReadout(PrimitiveImplementation[Readout]):
    """Represents a simple readout of a code where all variable nodes are measured in the given basis."""

    def compile(
        self,
        instruction: Readout,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[List[str], List[Measurement], List[Detector], DetectorGraphPort]:
        instructions: List[str] = []
        new_measurements: Dict[UUID, Measurement] = {}
        new_measurement_ordered = []

        new_port_state = (
            NodeKnowledge.MX
            if instruction.readout_basis == PauliChar.X
            else NodeKnowledge.MZ
        )

        nm = []
        for node in instruction.target.variable_nodes:
            nm.append(str(instruction.physical_data_qubits[node].integer_index))
            new_m = Measurement(
                node_id=node.id,
                parent_gadget_id=parent_gadget_id,
                parent_primitive_id=instruction.id,
                tag=f"readout_{node.tag}",
            )
            new_measurements[node.id] = new_m
            new_measurement_ordered.append(new_m)

        instructions.append(f"M{instruction.readout_basis.value} {' '.join(nm)}")

        detectors: List[Detector] = []
        checks_stable_with_measured_vars = set()

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
                    for c in det_graph_port[v].connected_nodes:
                        if c.check_type == instruction.readout_basis:  # type: ignore
                            if det_graph_port[c].knowledge == NodeKnowledge.STABLE:
                                checks_stable_with_measured_vars.add(c)

                case NodeKnowledge.UNKNOWN:
                    continue  # No detectors formed with variable nodes in UNKNOWN state
                case _:
                    raise ValueError(
                        f"Unexpected port state {var_port} for variable node {v.tag}"
                    )

        for c in checks_stable_with_measured_vars:
            relatives = det_graph_port[c].connected_nodes

            if all(v in instruction.target.variable_nodes for v in relatives):
                # If all neighbors of the stable check are measured variable nodes, we can form a detector with the latest measurement of the check and the new measurements of its neighbors.
                lm = record.latest_by_node_id(c.id)
                if lm is None:
                    raise ValueError(
                        f"No previous measurement found for stable check node {c.id} required to form detector."
                    )
                for v in relatives:
                    if v.id not in new_measurements:
                        raise ValueError(
                            f"Expected measurement for variable node {v.tag} not found in new measurements."
                        )
                neighbor_measurements = [new_measurements[v.id] for v in relatives]

                detectors.append(
                    Detector(
                        measurements=[lm] + neighbor_measurements,
                        tag=f"readout_spans_{c.tag}",
                    )
                )

        new_dg_port = DetectorGraphPort()
        for node in instruction.target.variable_nodes:
            new_dg_port[node] = QubitPortState(knowledge=new_port_state)

        return instructions, new_measurement_ordered, detectors, new_dg_port
