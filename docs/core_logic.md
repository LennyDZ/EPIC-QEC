# Core compilation logic

EPIC-QEC compilation logic is implemented by the `QECCompiler` object. A compiler is built according to a
specific configuration and can then process any program, which is a list of `QECGadget`s. It holds
a `CompilationContext`, which contains the various registers required for compilation, as well as a `PrimitiveCompiler` object, which will automatically allow the compiler to use the specific primitives implementations given in the configuration. The context starts empty and is updated at each step. The logic is very simple and can be summarized
by the following pseudocode:

```python
ctx = CompilationContext()
primitive_compiler = PrimitiveCompiler(config)

def compile(program, ctx):
    # Loop through gadgets and primitives to compile them
    for gadget in program:
        if gadget is alloc/free code:
            ctx.register/free_code(gadget.target)
        else:
            observables, lop_updates, primitives = gadget.compile(...)

            for p in primitives:
                stim, measurement, detectors, dg_port = primitive_compiler.compile(p, ...)

                ctx.add(stim, measurements, detectors)
            ctx.add_updates(lop_updates)
            ctx.resolve_and_add(observables)
    return ctx.to_compile_experiment()
```

While looping through the gadgets and primitives, the context is updated to track all the necessary information. The value that is returned is simply an immutable object containing the final Stim instructions, detectors, and observables.

The context stores the following data:
- `uuid_memory`: a register that maps UUIDs to objects. It can contain any object, such as codes, logical operators, logical qubits, or observables.
- `naming_registry`: a map from assigned variable names to the corresponding object's ID.
- `observables`: a set of the UUIDs of all existing observables, stored in `uuid_memory`.
- `detectors`: a list of the detectors that have been added. They are fully stored here and not included in `uuid_memory`, since we do not need to view or modify them after they are added.
- `operator_to_qubits`: a map from a logical operator's ID to the ID of the logical qubit it belongs to.
- `qubit_to_code`: similar to the previous one, but for logical qubits belonging to a code.
- `detector_port`: a state object used to create detectors between primitives. See the [primitives guide](qec_primitives.md) for more details.
- `quantum_memory`: a `QuantumMemory` object that stores the assignment of physical qubits to the data nodes and a pool of ancilla available for the gadget to reserve.
- `measurement_record`: a `MeasurementRecord` object that stores all measurements, alongside indexing that allows them to be explored efficiently.
- `compilation_time`: a kind of internal clock tracking which cycle we are in. This is not really used, though.

Although the logic is quite straightforward, the following points are worth noting:
 - First, when codes are allocated, they are added to the context's memory alongside their logical qubits and the corresponding logical operators. The context stores these structures until they are freed. While they are stored, we expect that nothing will modify them, except for logical-operator corrections, which are handled only by the compiler itself. When allocated, physical qubits are assigned to each data qubits of the code (not to checks, as they aren't necessary map 1-1 with physical qubits).

- Second, logical-operator corrections are added to the logical operators at the end of each gadget, in `ctx.add_updates(lop_updates)` in the pseudocode. They are stored within the `LogicalOperator` object itself as a list of measurements, which will flip the observable outcome if their outcome parity is odd.

- Finally, observables are "resolved" when they are added to the context. This means that we append to their set of measurements the corrections that exist in each of the logical operators involved in the observables, and we correctly refer the measurements to those created by the primitives.

Also, the context is supposed to be modified only by the compiler, so we tried, as much as possible, to provide only views of the different objects to gadgets and primitives. Quantum memory is given to gadgets, that are allowed to lock/free ancillas. This is not ideal, and we'll eventually find a way to restrict the interface.
 



