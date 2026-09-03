# EPIC-QEC

## Introduction

EPIC-QEC is a Python package that provides tools to compile quantum programs with integrated quantum error-correction solutions. It is designed to allow as much customization as possible and to support as many QEC ideas as possible. Its main purpose is to allow researchers to implement new ideas easily and integrate them with existing ones to create runnable simulations. This provides a well-defined framework in which competing solutions can be compared under circuit-level noise models.



> [!IMPORTANT]
> This package is still under development, and this documentation describes only the features that are already available.
>
> Ultimately, we aim to support the entire compilation stack for QEC programs, that is, to transform any quantum algorithm written in a standard gate-like language, such as X, CNOT, and so on, or formats such as QASM and Stim, into a new equivalent program with error-correction capabilities, including detectors, observables, and possibly a decoder.
>
> Descriptions of desirable features to be implemented in future versions are available here. Any contributions, ideas, and comments are welcome.




### Main features

To understand how the compilation works, we define below the three main objects that are used:

*Stabilizer codes* are at the core of modern quantum error correction, and the package follows the usual definitions. At a low level, they are stored and processed as `TannerGraph`, a graph-like object whose nodes are either `VariableNode`s, representing data qubits, or `CheckNode`s, representing checks associated with stabilizers. Edges encode how data qubits are involved in stabilizer checks, so they always connect a data-qubit node to a check node. At a higher level, the `StabilizerCode` object describes a code by combining its Tanner graph, a name, its parameters `(n, d, k)`, and a reference to the logical qubits it supports, along with how the logical operators are mapped to the `VariableNode`s.

[*QEC gadgets*](qec_gadget.md) are used to describe high-level logical instructions. There are two types of gadgets: memory gadgets and operational gadgets. Memory gadgets are used to allocate or free quantum resources, meaning physical qubits, for the logical qubits and their codes, while operational gadgets represent logical operations such as Clifford gates or logical Pauli-product measurements for PBC programs. They are described by the `QECGadget` class and its children. Each operational gadget provides a `compile` method that, given some logical qubits or codes as input, returns logical-operator updates, observables, primitive instructions describing the action of the gadget on the hardware, and the physical qubits used by the gadget. The compiler applies the updates and observables and releases the used physical qubits after the scheduled operation finishes.

[*QEC primitives*](qec_primitives.md) describe low-level instructions used in QEC. Their implementations take Tanner graphs, physical-qubit mappings, and possibly specifically targeted nodes as input. They compile into a `QuantumProgram` containing `QProgOperation` objects, alongside descriptions of the measurements included in those operations and information related to detectors. EPIC-QEC defines a finite set of primitives in its core as interfaces, which are then mapped to specific implementations according to the configuration chosen for compilation. The main example is `ExtractSyndrome`, which describes `rounds` rounds of measurement of the checks in the Tanner graph. Different implementations can describe different extraction circuits or schedules.

Using these components, the main workflow provided by EPIC-QEC lets users choose the codes and primitive implementations used to compile a `QuantumProgram`. A program is built with `add_qubit(...)` and `add_operation(...)`: code allocation is represented by an `AllocCode` gadget, while operational gadgets are attached to `QProgOperation` objects. The compiler schedules those operations, resolves their targets, allocates memory, creates detectors, and records which physical measurements correspond to which logical observables. It ultimately produces a `CompiledExperiment` that can be paired with noise and feedforward corrections to create a runnable Stim benchmark experiment. The scheme below summarizes the main steps of the compilation.

<<ADD SCHEME>>

Any researcher can easily create their own gadgets, primitives, or codes according to the interfaces defined by EPIC-QEC and combine them with existing implementations of their choice to compare them with competing ideas. While codes are simply defined using the symplectic parity-check-matrix formalism, implementing gadgets or primitives requires more specific understanding. The guides below provide the relevant information alongside simple implementation examples:

- [How to create a QEC Gadget](qec_gadget.md)
- [How to create custom implementation of a QEC primitive](qec_primitives.md)

More details on how the compilation process, memory management, frame tracking, and the rest of the core logic work are described in [this guide](core_logic.md) and in the arXiv paper associated with the initial version of this project.