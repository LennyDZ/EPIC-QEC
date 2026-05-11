# ETIC-QEC

## Introduction

ETIC-QEC is a Python package that provides tools to compile quantum programs with integrated quantum error-correction solutions. It is designed to allow as much customization as possible and to support as many QEC ideas as possible. Its main purpose is to allow researchers to implement new ideas easily and integrate them with existing ones to create runnable simulations. This provides a well-defined framework in which competing solutions can be compared under circuit-level noise models.



> [!IMPORTANT]
> This package is still under development, and this documentation describes only the features that are already available.
>
> Ultimately, we aim to support the entire compilation stack for QEC programs, that is, to transform any quantum algorithm written in a standard gate-like language, such as X, CNOT, and so on, or formats such as QASM and Stim, into a new equivalent program with error-correction capabilities, including detectors, observables, and possibly a decoder.
>
> Descriptions of desirable features to be implemented in future versions are available here. Any contributions, ideas, and comments are welcome.




### Main features

To understand how the compilation works, we define below the three main objects that are used:

*Stabilizer codes* are at the core of modern quantum error correction, and the package follows the usual definitions. At a low level, they are stored and processed as `TannerGraph`, a graph-like object whose nodes are either `VariableNode`s, representing data qubits, or `CheckNode`s, representing checks associated with stabilizers. Edges encode how data qubits are involved in stabilizer checks, so they always connect a data-qubit node to a check node. At a higher level, the `StabilizerCode` object describes a code by combining its Tanner graph, a name, its parameters `(n, d, k)`, and a reference to the logical qubits it supports, along with how the logical operators are mapped to the `VariableNode`s.

[*QEC gadgets*](qec_gadget.md) are used to describe high-level logical instructions. There are two types of gadgets: memory gadgets and operational gadgets. Memory gadgets are used to allocate or free quantum resources, meaning physical qubits, for the logical qubits and their codes, while operational gadgets represent logical operations such as Clifford gates or logical Pauli-product measurements for PBC programs. They are described by the `QECGadget` class and its children. Each operational gadget provides a `compile` method that, given some logical qubits or codes as input, returns a list of `QECPrimitives` instructions describing the action of the gadget on the hardware, alongside one or more observables and corrections to the logical information required to maintain the expected state.

[*QEC primitives*](qec_primitives.md) describe low-level instructions used in QEC. Their implementations take Tanner graphs, and possibly specifically targeted nodes, as input and compile them into a list of Stim instructions, alongside a description of the measurements included in those instructions and information related to detectors. ETIC-QEC defines a finite set of primitives in its core as interfaces, which are then mapped to specific implementations according to the configuration chosen for the compilation. The main example is `ExtractSyndrome`, which describes `d` rounds of measurement of the checks in the Tanner graph. Different implementations can describe different extraction circuits(schedules).

Using these components, the main workflow provided by ETIC-QEC lets users choose the codes and primitives used to compile a program described as a list of gadgets. The compiler then takes care of allocating memory, creating detectors, and recording which physical measurements correspond to which logical observables, ultimately producing a `CompiledExperiment` object that can be paired with noise and feedforward correction to create a runnable Stim benchmark experiment. The scheme below summarizes the main steps of the compilation.

<<ADD SCHEME>>

Any researcher can easily create their own gadgets, primitives, or codes according to the interfaces defined by ETIC-QEC and combine them with existing implementations of their choice to compare them with competing ideas. While codes are simply defined using the symplectic parity-check-matrix formalism, implementing gadgets or primitives requires more specific understanding. The guides below provide the relevant information alongside simple implementation examples:

- [How to create a QEC Gadget](qec_gadget.md)
- [How to create custom implementation of a QEC primitive](qec_primitives.md)

More details on how the compilation process, memory management, frame tracking, and the rest of the core logic work are described in [this guide](core_logic.md) and in the arXiv paper associated with the initial version of this project.