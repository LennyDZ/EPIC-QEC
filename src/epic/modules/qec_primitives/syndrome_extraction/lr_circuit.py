from typing import Dict, List, Tuple
from uuid import UUID

import numpy as np
from lr_circuits import LRCode

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure import QProgOperation, QuantumProgram, TannerNode
from epic.core.qec_object import (
    Detector,
    DetectorGraphPort,
    Measurement,
    NodeKnowledge,
    QubitPortState,
)
from epic.core.qec_primitives.interfaces import ExtractSyndrome, PrimitiveImplementation

# Opcodes emitted by lr_circuits.construct_cnot_circuit, mapped to their stim-style equivalent.
_OPCODE_TO_GATE = {"R": "RZ", "M": "MZ", "H": "H", "CNOT": "CX"}


class LRCircuit(PrimitiveImplementation[ExtractSyndrome]):
    """Syndrome extraction using the LR-circuits package.

    This builds a Stim-style circuit with the package's documented LR-circuit workflow,
    converts the generated operations into a QuantumProgram, creates measurements from
    the generated ancilla measurements, and forms detectors following the same pattern
    as the RSC syndrome-extraction primitive.

    References:
        [1] Armands Strikis and Dan E. Browne and Michael E. Beverland (2026).
            High-performance syndrome extraction circuits for quantum codes
            https://arxiv.org/abs/2603.05481
    """

    split: int | None = None

    def compile(
        self,
        instruction: ExtractSyndrome,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[QuantumProgram, List[Measurement], List[Detector], DetectorGraphPort]:

        target = instruction.target
        pcm, var_nodes, check_nodes = target.parity_check_matrix

        num_vars = pcm.shape[1] // 2
        x_part = pcm[:, :num_vars]
        z_part = pcm[:, num_vars:]
        # CSS codes have no check acting with both X and Z on the same row (i.e. no Y checks).
        if x_part.multiply(z_part).count_nonzero() > 0:
            raise ValueError(
                "LRCircuit requires a CSS Tanner graph: some check nodes mix X and Z Pauli terms."
            )

        Hx = np.asarray(x_part.toarray(), dtype=int)
        Hz = np.asarray(z_part.toarray(), dtype=int)
        # Rows are shared across both halves; drop the all-zero rows introduced by the other Pauli type's checks.
        x_mask = Hx.any(axis=1)
        z_mask = Hz.any(axis=1)
        Hx = Hx[x_mask]
        Hz = Hz[z_mask]
        x_check_nodes = [node for node, keep in zip(check_nodes, x_mask) if keep]
        z_check_nodes = [node for node, keep in zip(check_nodes, z_mask) if keep]
        # Qubit index in the raw circuit is the position in variable_nodes then Z-check_nodes then X-check_nodes.
        index_to_node: List[TannerNode] = [*var_nodes, *z_check_nodes, *x_check_nodes]
        node_to_qubit = {**instruction.physical_data_qubits, **instruction.physical_ancilla_qubits}

        if self.split is None:
            self.split = max(1, len(var_nodes) // 2)

        code = LRCode(
            Hx,
            Hz,
            self.split,
            f"epic_{instruction.tag or instruction.id}",
        )
        code.LR_colouring(ancilla_optimised_colouring=False)
        code.find_LR_residuals()

        attempts = 100
        BPOSD_params = {
            "error_rate": 0.01,
            "bp_method": "ms",
            "max_iter": 200,
            "osd_method": "osd_cs",
            "osd_order": 3,
            "ms_scaling_factor": 0  
        }
        num_workers = 10 
        batch_size = 10
        code.evaluate_residual_dists(attempts, BPOSD_params, max_workers=num_workers, batch_size=batch_size)

        max_col = 1000
        num_orders = 1
        code.find_CNOT_orders(max_col, num_orders)

        # The following filtering step is optional 
        attempts = 1000
        BPOSD_params = {
            "error_rate": 0.01,
            "bp_method": "ms",
            "max_iter": 500,
            "osd_method": "osd_cs",
            "osd_order": 7,
            "ms_scaling_factor": 0  
        }
        code.filter_orders_with_extended_distance(attempts, BPOSD_params, max_workers=10)

        circuit = code.construct_circuit_simple(
            rounds=instruction.rounds,
        )

        program = QuantumProgram(name=f"LR_syndrome_extraction_{instruction.tag}")
        for qubit in node_to_qubit.values():
            program.add_qubit(qubit)
        tick_on_all_qb = QProgOperation(
            name="tick", length=0, targets=list(node_to_qubit.values())
        )

        measurements_by_node: Dict[TannerNode, List[Measurement]] = {}
        measurements_ordered: List[Measurement] = []
        detectors: List[Detector] = []

        for instr in circuit:
            # instr is a string, first word is the operation, rest are the qubit indices
            parts = instr.split()
            if not parts:
                continue
            op, indices = parts[0], parts[1:]

            if op == "TICK":
                program.add_operation(tick_on_all_qb)
                continue

            nodes = [index_to_node[int(idx)] for idx in indices]
            qubits = [node_to_qubit[node] for node in nodes]
            gate = _OPCODE_TO_GATE[op]

            if op == "M":
                # Split simultaneous measurements so each ancilla gets its own Measurement record.
                for node, qubit in zip(nodes, qubits):
                    measurement = Measurement(
                        node_id=node.id,
                        parent_gadget_id=parent_gadget_id,
                        parent_primitive_id=instruction.id,
                        tag=f"{instruction.tag}_synd_{node.tag}_r{len(measurements_by_node.get(node, []))}",
                    )
                    program.add_operation(
                        QProgOperation(
                            name=gate,
                            length=1,
                            targets=[qubit],
                            measurement_id=measurement.id,
                        )
                    )
                    measurements_by_node.setdefault(node, []).append(measurement)
                    measurements_ordered.append(measurement)
            else:
                program.add_operation(
                    QProgOperation(name=gate, length=1, targets=qubits)
                )

        for check in check_nodes:
            # Initial round detector
            detector_zero = instruction._detector_round_zero(
                record,
                check,
                instruction.target.get_neighbourhood(check),  # type: ignore
                det_graph_port,
                measurements_by_node[check][0],
                tag=f"{instruction.tag}_det_{check.tag}_r0",
            )
            if detector_zero is not None:
                detectors.append(detector_zero)
            # Detectors between rounds
            for r in range(1, instruction.rounds):
                previous_measurement = measurements_by_node[check][r - 1]
                current_measurement = measurements_by_node[check][r]
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

        return program, measurements_ordered, detectors, new_graph_port


