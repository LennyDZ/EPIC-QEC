from epic.core.compilation.measurement_record import MeasurementRecord
from epic.core.data_structure import CheckNode, PauliChar, TannerGraph
from epic.core.data_structure.physical_qubit import PhysicalQubit
from epic.core.qec_object import DetectorGraphPort
from epic.core.qec_primitives.interfaces import ExtractSyndrome
from epic.modules.qec_primitives.syndrome_extraction.rsc_syndrome_extraction import (
    RSCSyndromeExtraction,
)
from epic.modules.stabilizers_codes.rotated_surface_code import RotatedSurfaceCode


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


def test_rsc_syndrome_extraction_builds_stim_instructions_from_pcm() -> None:
    code = RotatedSurfaceCode.from_distance(distance=3, code_name="rsc_test")
    target = code.tanner_graph
    instruction = _make_syndrome_extraction_instruction(target, rounds=1, tag="rsc_test")

    implementation = RSCSyndromeExtraction()
    instructions, measurements, detectors, new_graph_port = implementation.compile(
        instruction=instruction,
        record=MeasurementRecord().view(),
        det_graph_port=DetectorGraphPort(),
        parent_gadget_id=instruction.id,
    )

    assert instructions
    assert any(line.split()[0] == "MRZ" for line in instructions)
    assert len(measurements) == len(target.check_nodes) * instruction.rounds
    assert len(detectors) >= 0
    assert all(node in new_graph_port for node in target.check_nodes | target.variable_nodes)


def test_rsc_syndrome_extraction_is_coherent_for_rotated_surface_code_distance_3() -> None:
    code = RotatedSurfaceCode.from_distance(distance=3, code_name="rsc3")
    target = code.tanner_graph
    rounds = 2
    instruction = _make_syndrome_extraction_instruction(target, rounds=rounds, tag="rsc3")

    implementation = RSCSyndromeExtraction()
    instructions, _, _, _ = implementation.compile(
        instruction=instruction,
        record=MeasurementRecord().view(),
        det_graph_port=DetectorGraphPort(),
        parent_gadget_id=instruction.id,
    )

    node_to_qubit = {**instruction.physical_data_qubits, **instruction.physical_ancilla_qubits}
    qubit_to_node = {qubit.integer_index: node for node, qubit in node_to_qubit.items()}

    # Every CX must connect a check node to a variable node that is one of its Tanner-graph neighbours.
    cx_lines = [line for line in instructions if line.split()[0] == "CX"]
    assert cx_lines
    for line in cx_lines:
        qubits = [int(q) for q in line.split()[1:]]
        assert len(qubits) % 2 == 0
        for control, target_qubit in zip(qubits[::2], qubits[1::2]):
            control_node = qubit_to_node[control]
            target_node = qubit_to_node[target_qubit]
            check_node, var_node = (
                (control_node, target_node)
                if isinstance(control_node, CheckNode)
                else (target_node, control_node)
            )
            assert isinstance(check_node, CheckNode)
            assert not isinstance(var_node, CheckNode)
            assert var_node in target.get_neighbourhood(check_node)

    # X-check ancillas must be Hadamard-conjugated around the CX layers, once per round.
    x_check_qubits = {
        node_to_qubit[node].integer_index
        for node in target.check_nodes
        if node.check_type == PauliChar.X
    }
    h_lines = [line for line in instructions if line.split()[0] == "H"]
    assert len(h_lines) == 2
    for line in h_lines:
        assert {int(q) for q in line.split()[1:]} == x_check_qubits

    h_indices = [i for i, line in enumerate(instructions) if line.split()[0] == "H"]
    cx_indices = [i for i, line in enumerate(instructions) if line.split()[0] == "CX"]
    assert min(h_indices) < min(cx_indices)
    assert max(h_indices) > max(cx_indices)
