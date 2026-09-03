## QEC Primitives

This guide presents details on how to create an implementation for a primitive in EPIC-QEC. It is important to understand the separation between a primitive instruction and its implementations. A `QECPrimitive` instruction is an abstract interface that describes the meaning of the instruction. It can then be implemented by several `PrimitiveImplementation`s that describe different ways of performing it.

All `QECPrimitive` instructions share a common structure with:
- a target, which is a `TannerGraph`
- a human readable tag
- a distance (indicative only)
- an ID

There are four instructions defined by EPIC-QEC:
- `ApplyGate(target_nodes, gates)`: simply applies the given gates to the given nodes of the Tanner graph.
- `ExtractSyndrome(rounds, ancilla_reset_state)`: performs `rounds` rounds of measurement of the checks, or syndromes, in the Tanner graph. One can also specify the state in which the ancilla used to measure the checks are reset between rounds.
- `Readout(readout_basis)`: Measure all variable nodes in the given Tanner graph in the requested Pauli basis.
- `CustomPrimitive(implementation_class)`: allows one to directly call a specific custom implementation class, bypassing the compilation setting.

It is then possible to build specific implementations. For example, the syndrome-extraction circuit can vary significantly between codes, and one may also have several designs for reading out qubits in a fault-tolerant way.

Implementing primitives is probably one of the trickiest parts, mostly because we must build the detector graph in a valid way. The basic structure of an implementation is as follows:

```python
class ImplementationName(PrimitiveImplementation[ApplyGate]):
    """Simple gate compilation that emits direct circuit instructions."""

    def compile(
        self,
        instruction: ApplyGate,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[QuantumProgram, List[Measurement], List[Detector], DetectorGraphPort]:
        ...
```

The class must inherit from `PrimitiveImplementation[<...>]`, where the placeholder is replaced by the specific
`QECPrimitive` instruction that is implemented. The `compile` method then builds the expected information.
It receives the following as input:
- An instruction, which is the implemented `QECPrimitive`.
- A view of the measurement record, which can be used when building detectors between two primitives.
- A detector graph port, which provides information on the states of the qubits in the previous primitive in order to build detectors correctly.
- The ID of the gadget using the primitive.

From these data, the `compile` method must build the following:
- A `QuantumProgram` containing `QProgOperation`s for the Stim instructions or gates. Add physical qubits with `program.add_qubit(...)` and operations with `program.add_operation(...)`.
- A list of `Measurement`s. Each measurement operation should carry the corresponding measurement's ID in its `measurement_id` field.
- A list of `Detector`s. They define parity constraints as sets of `Measurement`s.
- A new, updated `DetectorGraphPort`. This is a dictionary that contains knowledge about the state of each node, or physical qubit, when entering the primitive. It is used to build detectors that depend on measurements from both the previous and the current primitive.

The example below gives an overview of how to build a primitive for syndrome measurement in a rotated
surface code. The full code can be found in `modules/syndrome_extraction/rsc_syndrome_extraction.py`.

```python
# This implements the ExtractSyndrome primitive
class RSCSyndromeExtraction(PrimitiveImplementation[ExtractSyndrome]):
    """RSC implementation of syndrome extraction that directly measures the stabilizers without any optimization."""

    def compile(
        self,
        instruction: ExtractSyndrome,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[QuantumProgram, List[Measurement], List[Detector], DetectorGraphPort]:

        check_nodes = instruction.target.check_nodes
        program = QuantumProgram(name=f"RSC_syndrome_extraction_{instruction.tag}")

        measurements: Dict[TannerNode, List[Measurement]] = {}
        measurements_ordered: List[Measurement] = []
        detectors: List[Detector] = []

        if len(check_nodes) == 0:
            return [], [], [], DetectorGraphPort()

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

        for q in node_to_qubit.values():
            program.add_qubit(q)
        tick_on_all_qb = QProgOperation(
            name="tick", length=0, targets=node_to_qubit.values()
        )

        # RESET ANCILLA

        reset_ancilla_instructions: List[QProgOperation] = []
        match instruction.ancilla_reset_state:
            case PauliEigenState.Z_plus:
                for check in check_nodes:
                    reset_ancilla_instructions.append(
                        QProgOperation(
                            name="RZ",
                            length=1,
                            targets=[node_to_qubit[check]],
                        )
                    )
            case PauliEigenState.X_plus:
                for check in check_nodes:
                    reset_ancilla_instructions.append(
                        QProgOperation(
                            name="RX",
                            length=1,
                            targets=[node_to_qubit[check]],
                        )
                    )
            case _:
                raise ValueError(
                    f"Unsupported ancilla reset state: {instruction.ancilla_reset_state}"
                )

        program.add_operations(reset_ancilla_instructions)

        # SYNDROME EXTRACTION CIRCUIT
        single_round_instructions: List[QProgOperation] = []
        node_measured = []
        x_checks = []
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        for check in check_nodes:
            neighourhood = instruction.target.get_neighbourhood(check)
            if check.check_type == PauliChar.X:
                x_checks.append(check)
            ne, se, nw, sw = None, None, None, None
            for n in neighourhood:
                if not isinstance(n.coordinates, tuple) or not isinstance(
                    check.coordinates, tuple
                ):
                    raise ValueError(
                        "Node coordinates must be tuples for the current partitioning logic."
                    )

                x_idx = 0 if n.coordinates[2] == check.coordinates[2] else 2
                y_idx = 1 if n.coordinates[3] == check.coordinates[3] else 3
                dx = n.coordinates[x_idx] > check.coordinates[x_idx]
                dy = n.coordinates[y_idx] > check.coordinates[y_idx]

                match (dx, dy):
                    case (False, True):
                        nw = n
                    case (False, False):
                        sw = n
                    case (True, False):
                        se = n
                    case (True, True):
                        ne = n
                    case _:
                        raise ValueError(
                            f"Unexpected relative coordinates between check node {check.id} and its neighbor {n.id}: {(dx, dy)}. This likely means that the partitioning logic does not match the expected layout."
                        )
            match check.check_type:
                case PauliChar.Z:
                    t1.append((se, check)) if se is not None else None
                    t2.append((ne, check)) if ne is not None else None
                    t3.append((sw, check)) if sw is not None else None
                    t4.append((nw, check)) if nw is not None else None
                case PauliChar.X:
                    t1.append((check, se)) if se is not None else None
                    t2.append((check, sw)) if sw is not None else None
                    t3.append((check, ne)) if ne is not None else None
                    t4.append((check, nw)) if nw is not None else None
                case _:
                    raise ValueError(
                        f"Unsupported check type: {check.check_type} in rotated surface code"
                    )
            node_measured.append(check)

        single_round_instructions.append(tick_on_all_qb)
        for xc in x_checks:
            single_round_instructions.append(
                QProgOperation(
                    name="H",
                    length=1,
                    targets={node_to_qubit[xc]},
                )
            )
        single_round_instructions.append(tick_on_all_qb)
        for t in [t1, t2, t3, t4]:
            for con, tar in t:
                single_round_instructions.append(
                    QProgOperation(
                        name="CX",
                        length=1,
                        targets=[node_to_qubit[con], node_to_qubit[tar]],
                    )
                )
            single_round_instructions.append(tick_on_all_qb)
        for xc in x_checks:
            single_round_instructions.append(
                QProgOperation(
                    name="H",
                    length=1,
                    targets=[node_to_qubit[xc]],
                )
            )
        single_round_instructions.append(tick_on_all_qb)

        for r in range(instruction.rounds):
            instr = single_round_instructions.copy()

            for m in node_measured:
                measurement = Measurement(
                    node_id=m.id,
                    parent_gadget_id=parent_gadget_id,
                    parent_primitive_id=instruction.id,
                    tag=f"{instruction.tag}_synd_{m.tag}_r{r}",
                )
                instr.append(
                    QProgOperation(
                        name="MRZ",
                        length=1,
                        targets=[node_to_qubit[m]],
                        measurement_id=measurement.id,
                    )
                )

                measurements.setdefault(m, []).append(measurement)
                measurements_ordered.append(measurement)
            program.add_operations(instr)

        for check in check_nodes:
            # Initial round detector
            detector_zero = instruction._detector_round_zero(
                record,
                check,
                instruction.target.get_neighbourhood(check),  # type: ignore
                det_graph_port,
                measurements[check][0],
                tag=f"{instruction.tag}_det_{check.tag}_r0",
            )
            if detector_zero is not None:
                detectors.append(detector_zero)
            # Detectors between rounds
            for r in range(1, instruction.rounds):
                previous_measurement = measurements[check][r - 1]
                current_measurement = measurements[check][r]
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
```

This snippet skips the hard part that implements the detector linked with the previous primitive by wrapping it into the `_detector_round_zero` method. A detailed description of how this function works is provided in the next section.

### Detectors graph

In order to build detectors whose measurements are spread between two primitives, we use what we call a `DetectorGraphPort`, a dictionary that assigns to each node a `QubitPortState` describing the knowledge we have about the state of the qubits at the end of the previous primitive. The possible states, described by the `NodeKnowledge` enum, are the following:
- Stable: the node just went through a round of syndrome measurement.
- RX (respectively RZ): the node was freshly reset and is therefore in the state $|+\rangle$ (respectively $|0\rangle$).
- MX (respectively MZ): the node was just measured. It is in an eigenstate of the basis, and the last measurement in the record associated with this node indicates into which state it was projected.
- Unknown: there is no knowledge on the node's state. It can be in any state.

The `QubitPortState` is a pair consisting of a `NodeKnowledge` and a set of `connected_node`s. The set is empty for all states except the Stable one. In that case, it contains the neighboring nodes that were included in the stabilizer measurement. This is necessary because, for example, when we extract syndromes just after splitting a patch in lattice surgery, the previous round was performed on the merged patch, so some nodes that were involved in it will not be included in the current extraction-process input.

The example below shows how we use the `DetectorGraphPort` to build detectors between the previous state and the first round of syndrome measurements for a given check:

```python
def _detector_round_zero(
        record: MeasurementRecordView,
        check: CheckNode,
        dgp: DetectorGraphPort,
        round_zero_measurement: Measurement,
    ) -> Detector | None:
        # This is the first measurement of the current primitive.
        measurement_in_detectors = [round_zero_measurement]
        check_knowledge = dgp[check].knowledge
        match check_knowledge:
            # If a check was stable, we expect it to have the same parity as in the previous round.
            case NodeKnowledge.STABLE:
                latest = record.latest_by_node_id(check.id)
                measurement_in_detectors.append(latest)
            case NodeKnowledge.UNKNOWN:
            # If the check is in an unknown state, we cannot be sure about the outcome, so no detector is formed.
                return None  
            case NodeKnowledge.MZ | NodeKnowledge.MX:
                # If the check was measured in a different basis, the parity between the two
                # rounds is random, so we cannot create a detector.
                if check_knowledge.basis() != check.check_type:
                    return None  
                
                # If the check was measured in the same basis, the last measurement is included
                # in the detector. It may flip the expected parity.
                else:  
                    latest = record.latest_by_node_id(check.id)
                if latest is None:
                    raise ValueError(
                        f"No measurement found in record for stable check node {check.id}"
                    )
                measurement_in_detectors.append(latest)

            case NodeKnowledge.RX | NodeKnowledge.RZ:
                # If it was reset in the opposite basis, we cannot infer anything about the outcome,
                # so no detector is formed.
                if check_knowledge.basis() != check.check_type:
                    return None
            case _:
                raise ValueError(f"Invalid known check state: {known_check_state}")

        # We also need to look at the neighboring-node states, as they may be included in
        # the detectors. There is connected_nodes only if the check was in "Stable" state.
        # Otherwise this does nothing.
        extra_measurements = []
        for v in dgp[check].connected_nodes:
            match dgp[v].knowledge:
                case NodeKnowledge.RZ | NodeKnowledge.RX:
                    # If a neighbor was reset in a different basis than the check, we cannot be sure about the outcome, so no detector is formed.
                    if dgp[v].knowledge.basis() != check_knowledge.basis():
                        return None
                case NodeKnowledge.MZ | NodeKnowledge.MX:
                    # If a neighbor was measured, it is fine as long as it is in the same basis,
                    # but we need to include the latest measurement of that neighbor in the detector
                    # Typically, when we read out the ancillas in lattice surgery, we need to add the outcome
                    # to the detector parity.
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
                    pass  # If the neighbor was stable, it does not affect detector formation.
                case NodeKnowledge.UNKNOWN:
                    # If a neighbor is in an unknown state, we cannot be sure about the outcome, so no detector is formed.
                    return None
                case _:
                    pass

        return Detector(
            measurements=measurement_in_detectors + extra_measurements,
            tag=f"check_'{check.tag}'_r0",
        )
```
