## QEC Gadget in EPIC-QEC

This guide presents details on how to implement a gadget in EPIC-QEC. All QEC gadgets inherit from one of the four `QECGadget` subclasses:
- `LogicGadget`
- `CodeGadget`
- `AllocCode`
- `FreeCode`

The last two are memory gadgets, and they are treated separately by the compiler to allocate or free the physical memory associated with the given codes and logical qubits, while also recording them as variables referred to by the given variable names. In the current version of the package, they are not meant to be further subclassed. The first two are operational gadgets. They both have very similar behavior: they take a list of targets, strings corresponding to variable names, as inputs and provide a `compile` method. The only difference is that `LogicGadget` targets specific logical qubits, which may be supported on an arbitrary set of codes. One can have two logical qubits on two different codes, but we assume a single logical qubit is always fully supported within a single code. On the other hand, `CodeGadget` acts on stabilizer codes without targeting specific logical qubits.

The compiler takes care of parsing the input according to the parent class of the gadget. Therefore, implementing a gadget is mostly about the `compile` function. The detailed signature of this function is as follows:

```python
def compile(
        self,
        resolved_targets: List[StabilizerCode], # List[Tuple[LogicalQubit, StabilizerCode]]
        record: MeasurementRecordView,
        quantum_memory: QuantumMemory,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:
```

The function is given the resolved targets, which are codes for `CodeGadget` and logical qubits with their host codes for `LogicGadget`. It also receives the following information from the compiler:
- A view of the measurement record, which is an object maintained by the compiler that contains the measurements performed so far. *Why though? I'm not sure anymore, and I should remove this parameter ASAP.*
- An access to the quantum memory, to reserve eventally needed ancilla and to be aware of the data qubits assignement.
- The timestep, which indicates the position at which this gadget was processed by the compiler.
- The objective distance of the whole experiment. This can typically be used to determine how many rounds of error correction should be performed.

Then it must return the following information:
- A dictionary that includes any correction to apply to the logical operators
- A list of observables
- A list of primitive instructions

If any additional parameters are needed, they can be added as class attributes, and they will be provided when using the gadget while writing the program. They will then naturally be available in the method because it is not static.

To provide a better understanding, the following section goes through the implementation of a gadget for lattice surgery on rotated surface codes. The full implementation can be found in `RSCSurgery.py`; to keep this document readable, only the relevant information is included.

The protocol, described in the figure below, can be described in four steps, which correspond to the primitives the gadget will compile into:
1. Initialize the ancilla in the right basis
2. Measure the syndrome of the merged system d-times
3. Read out the ancilla qubits
4. Measure the syndrome of each of the patches individually

We also want to create an observable associated with the outcome of the logical Pauli-product measurement. Finally, we will need to correct one of the logical operators based on the outcome of the ancilla-qubit readout. The code for our gadget class will look like this:

```python
class RSCSurgery(LogicGadget):
    product_to_measure: PauliString # Either XX or ZZ

    def compile(
        self,
        resolved_targets: List[Tuple[LogicalQubit, StabilizerCode]],
        record,
        quantum_memory,
        timestep,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:
        # Helper to check that the operation is valid (code position/orientation, etc.)
        _check_validity(resolved_targets, self.product_to_measure)

        # create useful refs
        code1, code2 = (rt[1] for rt in resolved_targets)
        lq1, lq2 = (rt[0] for rt in resolved_targets)
        m_type = "X" if self.product_to_measure == "XX" else "Z"
        if m_type == "X":
            involved_lop1 = lq1.logical_x
            involved_lop2 = lq2.logical_x
            anticommuting_lop1 = lq1.logical_z
            anticommuting_lop2 = lq2.logical_z
        else:
            # Same thing, just swapping the logical Pauli operators.
            ...

        # Find the axis (vertical or horizontal) and the boundary nodes involved in the merge on both sides.
        axis, boundaries = _find_axis_and_boundary(resolved_targets, self.product_to_measure)

        # Build a Tanner graph with the ancilla and find the edges that connect it to the boundaries.
        ancilla_system, connecting_edges = _build_ancilla_system(boundaries)
        
        # Create a Tanner graph for the merged system.
        merged_system: TannerGraph = ancilla_system.connect_to(
            code1.tanner_graph | code2.tanner_graph, connecting_edges
        )

        # Get ancilla for the ancilla data qubits in the merge
        # And for the checks, so that we can use them to measure the stabilizser
        # Make sure to unlock them at the end
        ancilla = quantum_memory.lock_ancilla_qubits(n=len(ancilla_system.variable_nodes + merged_code.check_nodes), requestor_id=self.id)

        # map ancilla to node so we always reuse the same
        # This is not strictly necessary, and we could for example reserve less ancilla and use them to measure several different stabilizers
        ancilla_qubits_to_node = {...}

        # Primitive 1: initialize the ancilla qubits in the right basis.
        init_ancilla = ApplyGate(
            target=ancilla_system,
            target_nodes=ancilla_system.variable_nodes | ancilla_system.check_nodes,  # type: ignore
            physical_data_qubits={k: v for k, v in ancilla_qubits_to_node.items() if isinstance(k, VariableNode)},  # type: ignore
            physical_ancilla_qubits=ancilla_qubits_to_node,
            gates=(
                ["RX"] if m_type == PauliChar.Z else ["RZ"]
            ),  # Initialize in the dual basis of the merge type.
        )

        # Primitive 2: do d rounds of syndrome measurement on the merged system.
        merged_syndrome = ExtractSyndrome(
            target=merged_system,
            physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                merged_system.variable_nodes
            )
            | {k: v for k, v in ancilla_qubits_to_node.items() if isinstance(k, VariableNode)},
            physical_ancilla_qubits=ancilla_qubits_to_node,
            rounds=objective_distance,
            tag=f"rsc_surgery_merged_syndrome_{self.tag}",
        )

        # Primitive 3: read out the ancilla system in the right basis.
        ancilla_readout = Readout(
            target=ancilla_system,
            physical_data_qubits={
                k: v
                for k, v in ancilla_qubits_to_node.items()
                if isinstance(k, VariableNode)
            },
            physical_ancilla_qubits=ancilla_qubits_to_node,
            readout_basis=m_type.dual(),
            tag=f"rsc_surgery_measurement_{self.tag}",
        )

        # Primitive 4: do d rounds of syndrome measurement on each original code after the split.
        split_syndrome = [
            ExtractSyndrome(
                target=initial_code,
                physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                    initial_code.variable_nodes
                ),
                physical_ancilla_qubits=ancilla_qubits_to_node,
                rounds=objective_distance,
                tag=f"rsc_surgery_split_syndrome_{self.tag}",
            )
            for initial_code in [code1.tanner_graph, code2.tanner_graph]
        ]

        primitives = [init_ancilla, merged_syndrome, ancilla_readout] + split_syndrome

        # Create an observable for the Pauli-product outcome.
        # The included measurements are the last stabilizer measurements for each check
        # of the corresponding type that lies between the two logical operators (red dots in the figure).
        check_in_between = _find_check_in_between(lq1, lq2, m_type)

        observable = [
            Observable(
                logical_operators_involved=[involved_lop1, involved_lop2],
                measurements={
                    Measurement(
                        n.id,
                        parent_gadget_id=self.id,
                        parent_primitive_id=merged_syndrome.id,
                    )
                    for n in check_in_between
                },
                tag=f"rsc_surgery_{self.tag}",
            )
        ]

        # We create a correction based on the ancilla readout for one of the anticommuting logical operators.
        # We find the involved nodes, that is, a chain linking the two anticommuting operators in the merged system.
        node_in_correction = _find_correction(merged_system, anticommuting_lop1, anticommuting_lop2)
        correction = {
            anticommuting_lop1.id: LogicalOperatorUpdate(
                new_correction={
                    Measurement(
                        n.id,
                        parent_gadget_id=self.id,
                        parent_primitive_id=ancilla_readout.id,
                    )
                    for n in node_in_correction
                }
            ),
        }

        quantum_memory.unlock_ancilla_qubits(
            qubits=list(ancilla), owner_id=self.id
        )

        return correction, observable, primitives
```

Although the full implementation may require some tedious computation to navigate the node positions, the overall logic remains quite simple. This allows us to obtain a circuit with detectors, while the corrections and observables are handled as part of the computation as well.

We now state some important points to remember when implementing a gadget:
- The state, meaning the Tanner-graph structure, of the stabilizer-code object must not be changed by the gadget. To build the merged code, for example, we build a Tanner graph that uses references to the code's graph, but we do not modify it.
- The logical operators must not be changed directly. We only provide `LogicalOperatorUpdate`, which will be processed by the compiler after the gadget is processed. These corrections will be effective only in the next gadget. A logical-operator update can also remap the logical operator to some other support, as long as it stays within the same code.
- All the ancilla reserved (locked) should be unlock. Otherwise they will not be reusable by the next gadgets.
- The measurements in the observables and corrections must include the primitive ID and the gadget ID. Then the compiler will associate them with the last measurement instruction of the given node ID that was added in the corresponding primitive.
- The observable's tag is used as the variable name for it.
- If the observable corresponds to a logical measurement, it must specify which logical operators are associated with it. This is required so that the compiler can complete it with the corrections previously added to these logical operators.
- To preserve compilation performance, we want to avoid complex computation during gadget compilation. Ideally, we want the `compile()` function to have constant or linear complexity, but anything polynomial is still acceptable.

More information can be found in the API documentation. If you'd like to add your own gadget to the project, have a look at the contributor guide.