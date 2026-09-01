from typing import Dict, List, Tuple
from uuid import UUID

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure import (
    PauliChar,
    PauliEigenState,
    TannerNode,
    TannerEdge,
    QuantumProgram,
    QProgOperation,
)
from epic.core.qec_object import (
    Detector,
    DetectorGraphPort,
    NodeKnowledge,
    QubitPortState,
    Measurement,
)
from epic.core.qec_primitives.interfaces.extract_syndrome import ExtractSyndrome
from epic.core.qec_primitives.interfaces.qec_primitive import PrimitiveImplementation


class ZXColoringExtraction(PrimitiveImplementation[ExtractSyndrome]):
    """
    Implement the syndrome extraction using ZX-coloring for CSS codes.
    1. Take X and Z stabilizers as input
    2. Color the Edges for each type (X and Z) separately to get CNOT layers
    """

    @staticmethod
    def _color_edges(edges: set[TannerEdge]) -> Tuple[Dict[TannerEdge, int], int]:
        # Greedily assign each edge to the first layer where neither endpoint is used.
        coloring: Dict[TannerEdge, int] = {}
        used_colors_by_node: Dict[TannerNode, set[int]] = {}
        max_color = -1

        for edge in edges:
            qubit = edge.variable_node
            check = edge.check_node
            unavailable = used_colors_by_node.get(
                qubit, set()
            ) | used_colors_by_node.get(check, set())

            color = 0
            while color in unavailable:
                color += 1

            coloring[edge] = color
            used_colors_by_node.setdefault(qubit, set()).add(color)
            used_colors_by_node.setdefault(check, set()).add(color)
            max_color = max(max_color, color)

        return coloring, max_color

    def compile(
        self,
        instruction: ExtractSyndrome,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[QuantumProgram, List[Measurement], List[Detector], DetectorGraphPort]:

        check_nodes = instruction.target.check_nodes
        program = QuantumProgram(
            name=f"ZX_coloring_syndrome_extraction_{instruction.tag}"
        )
        detectors = []
        measurements = []

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

        node_to_qubit = {**checks_qubits, **data_qubits}

        for qubit in node_to_qubit.values():
            program.add_qubit(qubit)

        # Separate X and Z edges
        tanner = instruction.target
        x_edges = set()
        z_edges = set()
        for edge in tanner.edges:
            if edge.check_node.check_type is None:
                raise ValueError(
                    f"Check node {edge.check_node} does not have a check type. The Tanner Graph may not be CSS."
                )
            if (
                edge.check_node.check_type == PauliChar.X
                or edge.pauli_checked == PauliChar.X
            ):
                x_edges.add(edge)
            elif (
                edge.check_node.check_type == PauliChar.Z
                or edge.pauli_checked == PauliChar.Z
            ):
                z_edges.add(edge)
            else:
                raise ValueError(
                    f"Edge {edge} is not properly labeled as X or Z. Check the Tanner graph."
                )

        # Reset ancilla qubits
        check_nodes = instruction.target.check_nodes
        ordered_checks = list(check_nodes)
        x_checks = [
            check for check in ordered_checks if check.check_type == PauliChar.X
        ]

        # Reset ALL ancilla:
        match instruction.ancilla_reset_state:
            case PauliEigenState.Z_plus:
                program.add_operations(
                    [
                        QProgOperation(
                            name="RZ",
                            length=1,
                            targets=[node_to_qubit[check]],
                        )
                        for check in ordered_checks
                    ]
                )
            case PauliEigenState.X_plus:
                program.add_operations(
                    [
                        QProgOperation(
                            name="RX",
                            length=1,
                            targets=[node_to_qubit[check]],
                        )
                        for check in ordered_checks
                    ]
                )
            case _:
                raise ValueError(
                    f"Unsupported ancilla reset state: {instruction.ancilla_reset_state}"
                )
        x_coloring, x_color_count = self._color_edges(x_edges)
        z_coloring, z_color_count = self._color_edges(z_edges)
        single_round_instructions: List[QProgOperation] = []

        # Apply CNOTs for X and Z edges separately
        # X checks:
        if x_checks:

            single_round_instructions.extend(
                [
                    QProgOperation(
                        name="H",
                        length=1,
                        targets=[node_to_qubit[xc]],
                    )
                    for xc in x_checks
                ]
            )
            single_round_instructions.append(
                QProgOperation(
                    name="tick",
                    length=0,
                    targets=node_to_qubit.values(),
                )
            )
        x_cnot_steps = [[] for _ in range(x_color_count + 1)]
        for edge in x_edges:
            var_slot = node_to_qubit[edge.variable_node]
            check_slot = node_to_qubit[edge.check_node]
            color = x_coloring[edge]
            x_cnot_steps[color].append((check_slot, var_slot))
        for step in x_cnot_steps:
            if not step:
                continue
            for check, var in step:
                single_round_instructions.append(
                    QProgOperation(
                        name="CX",
                        length=1,
                        targets=[check, var],
                    )
                )

            single_round_instructions.append(
                QProgOperation(
                    name="tick",
                    length=0,
                    targets=node_to_qubit.values(),
                )
            )
        if x_checks:
            single_round_instructions.extend(
                [
                    QProgOperation(
                        name="H",
                        length=1,
                        targets=[node_to_qubit[xc]],
                    )
                    for xc in x_checks
                ]
            )
            single_round_instructions.append(
                QProgOperation(
                    name="tick",
                    length=0,
                    targets=node_to_qubit.values(),
                )
            )

        # Z checks:
        z_cnot_steps = [[] for _ in range(z_color_count + 1)]
        for edge in z_edges:
            var_qubit = node_to_qubit[edge.variable_node]
            check_qubit = node_to_qubit[edge.check_node]
            color = z_coloring[edge]
            z_cnot_steps[color].append((check_qubit, var_qubit))
        for step in z_cnot_steps:
            if not step:
                continue
            for check, var in step:
                single_round_instructions.append(
                    QProgOperation(
                        name="CX",
                        length=1,
                        targets=[var, check],
                    )
                )
            single_round_instructions.append(
                QProgOperation(
                    name="tick",
                    length=0,
                    targets=node_to_qubit.values(),
                )
            )

        measurements_by_node = {}
        for r in range(instruction.rounds):
            instr= single_round_instructions.copy()
            
            for m in ordered_checks:
                instr.append(
                    QProgOperation(
                        name="MRZ",
                        length=1,
                        targets=[node_to_qubit[m]],
                        measurement_id=measurement.id,
                    )
                )
                measurement = Measurement(
                    node_id=m.id,
                    parent_gadget_id=parent_gadget_id,
                    parent_primitive_id=instruction.id,
                    tag=f"{instruction.tag}_synd_{m.tag}_r{r}",
                )
                measurements_by_node.setdefault(m.id, []).append(measurement)
                measurements.append(measurement)
            program.add_operations(instr)
        
        for check in ordered_checks:
            detector_zero = instruction._detector_round_zero(
                record,
                check,
                instruction.target.get_neighbourhood(check),  # type: ignore
                det_graph_port,
                measurements_by_node[check.id][0],
                tag=f"{instruction.tag}_det_{check.tag}_r0",
            )
            if detector_zero is not None:
                detectors.append(detector_zero)
            for r in range(1, instruction.rounds):
                previous_measurement = measurements_by_node[check.id][r - 1]
                current_measurement = measurements_by_node[check.id][r]
                detector = Detector(
                    measurements=[previous_measurement, current_measurement],
                    tag=f"{instruction.tag}_det_{check.tag}_r{r-1}_{r}",
                )
                detectors.append(detector)

        new_graph_port = DetectorGraphPort()
        for node in instruction.target.check_nodes:
            new_graph_port[node] = QubitPortState(
                knowledge=NodeKnowledge.STABLE,
                connected_nodes=instruction.target.get_neighbourhood(node),
            )
            for nei in instruction.target.get_neighbourhood(node):
                if nei in new_graph_port:
                    new_graph_port[nei].connected_nodes.add(node)
                else:
                    new_graph_port[nei] = QubitPortState(
                        knowledge=NodeKnowledge.STABLE,
                        connected_nodes={node},  # type: ignore
                    )

        return program, measurements, detectors, new_graph_port
