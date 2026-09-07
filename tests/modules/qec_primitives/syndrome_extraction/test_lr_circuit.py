import numpy as np
from scipy.sparse import csr_matrix

from epic.core.compilation.measurement_record import MeasurementRecord
from epic.core.data_structure import CheckNode, PauliChar, TannerGraph
from epic.core.data_structure.physical_qubit import PhysicalQubit
from epic.core.qec_object import DetectorGraphPort
from epic.core.qec_primitives.interfaces import ExtractSyndrome
from epic.modules.qec_primitives.syndrome_extraction.lr_circuit import LRCircuit
from epic.modules.stabilizers_codes.rotated_surface_code import RotatedSurfaceCode


def test_lr_circuit_builds_quantum_program_from_lr_code() -> None:
    hx = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=int)
    hz = np.array([[1, 1, 1, 1]], dtype=int)
    target = TannerGraph.from_pcm(csr_matrix(hx), csr_matrix(hz), code_name="lr_test")

    data_qubits = {
        node: PhysicalQubit(integer_index=i)
        for i, node in enumerate(sorted(target.variable_nodes, key=lambda n: n.tag))
    }
    ancilla_qubits = {
        node: PhysicalQubit(integer_index=len(data_qubits) + i)
        for i, node in enumerate(sorted(target.check_nodes, key=lambda n: n.tag))
    }

    instruction = ExtractSyndrome(
        target=target,
        rounds=1,
        physical_data_qubits=data_qubits,
        physical_ancilla_qubits=ancilla_qubits,
        tag="lr_test",
    )

    implementation = LRCircuit()
    program, measurements, detectors, new_graph_port = implementation.compile(
        instruction=instruction,
        record=MeasurementRecord().view(),
        det_graph_port=DetectorGraphPort(),
        parent_gadget_id=instruction.id,
    )

    assert program.operations
    assert any(op.name == "MZ" for op, _ in program.operations)
    assert len(measurements) == len(target.check_nodes) * instruction.rounds
    assert len(detectors) >= 0
    assert all(node in new_graph_port for node in target.check_nodes | target.variable_nodes)



def _make_syndrome_extraction_instruction(target: TannerGraph, rounds: int, tag: str) -> ExtractSyndrome:
    data_qubits = {
        node: PhysicalQubit(integer_index=i)
        for i, node in enumerate(sorted(target.variable_nodes, key=lambda n: n.tag))
    }
    ancilla_qubits = {
        node: PhysicalQubit(integer_index=len(data_qubits) + i)
        for i, node in enumerate(sorted(target.check_nodes, key=lambda n: n.tag))
    }
    return ExtractSyndrome(
        target=target,
        rounds=rounds,
        physical_data_qubits=data_qubits,
        physical_ancilla_qubits=ancilla_qubits,
        tag=tag,
    )


def test_lr_circuit_is_coherent_for_rotated_surface_code_distance_3() -> None:
    code = RotatedSurfaceCode.from_distance(distance=3, code_name="rsc3")
    target = code.tanner_graph
    rounds = 2
    instruction = _make_syndrome_extraction_instruction(target, rounds=rounds, tag="rsc3")

    implementation = LRCircuit()
    program, _, _, _ = implementation.compile(
        instruction=instruction,
        record=MeasurementRecord().view(),
        det_graph_port=DetectorGraphPort(),
        parent_gadget_id=instruction.id,
    )

    node_to_qubit = {**instruction.physical_data_qubits, **instruction.physical_ancilla_qubits}
    qubit_to_node = {qubit.integer_index: node for node, qubit in node_to_qubit.items()}

    ordered_ops = [op for op, _ in sorted(program.operations, key=lambda x: x[1])]

    # Every CX must connect a check node to a variable node that is one of its Tanner-graph neighbours.
    cx_ops = [op for op in ordered_ops if op.name == "CX"]
    assert cx_ops
    for op in cx_ops:
        qubits = [q.integer_index for q in op.targets]
        assert len(qubits) == 2
        control_node = qubit_to_node[qubits[0]]
        target_node = qubit_to_node[qubits[1]]
        check_node, var_node = (
            (control_node, target_node)
            if isinstance(control_node, CheckNode)
            else (target_node, control_node)
        )
        assert isinstance(check_node, CheckNode)
        assert not isinstance(var_node, CheckNode)
        assert var_node in target.get_neighbourhood(check_node)

    # X-check ancillas must be Hadamard-conjugated around the CNOT layers (pipelined once between rounds).
    x_check_qubits = {
        node_to_qubit[node].integer_index
        for node in target.check_nodes
        if node.check_type == PauliChar.X
    }
    h_ops = [op for op in ordered_ops if op.name == "H"]
    assert h_ops
    for op in h_ops:
        assert {q.integer_index for q in op.targets} == x_check_qubits

    # Any CX touching an X-check qubit must be sandwiched between an earlier and a later H on that qubit.
    x_check_cx_indices = [
        i
        for i, op in enumerate(ordered_ops)
        if op.name == "CX"
        and x_check_qubits & {q.integer_index for q in op.targets}
    ]
    assert x_check_cx_indices
    h_indices = [i for i, op in enumerate(ordered_ops) if op.name == "H"]
    assert min(h_indices) < min(x_check_cx_indices)
    assert max(h_indices) > max(x_check_cx_indices)
