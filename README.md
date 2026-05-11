# EPIC-QEC

EPIC-QEC (Experimental Package for Implementing and Compiling Quantum Error Correction) is a Python package that aims to facilitate benchmarking of QEC solutions with ciruict level noise. It provide a core structure that allows the users to build fully customized experiments that includes all the components of quantum error correction into an exiting quantum algorithm. The construction allows for wide range of customization, including which stabilizers codes to use, how to measure the syndromes of the codes, how to perform each logical operations, etc...

The notebook [getting_started.ipynb](docs/getting_started.ipynb) provide a brief overview of the package's capabilities.

For more complete documentation, especially if you want to implement your own modules please have a look at [the documentation overview](docs/introduction.md).

For advanced imlementation detail go see [the core compilation logic guide](docs/core_logic.md).

Contributions, suggestions and feedbacks are welcome, please have a look at <contrib_guide> for more informations.


## Setup

This project uses `uv` and requires Python 3.12 or newer.

```bash
uv sync --all-groups
source .venv/bin/activate
```

You can then run the test suite with:

```bash
pytest
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for the full text.



As this package is in an early development stage, it comes with several limitations:
- Only Clifford are supported for now. The package use Stim as IR, since we cannot perform large scale simulations on universal quantum space, we choose to rely on it for its simplicity and efficiency.
- As for now, the parallelization of gadgets and primitives is not supported. Therefore, it is not possible to compare the depth of the circuits
- Since the package was primarily meant for qLDPC experiment, it is not relying on precise topology of the physical qubits, it assumes full connectivity and rely only on tanner graph to describe the structures. In order to implement solutions that need to specify positions for the qubits (like surface code surgery), one need to do some patchy work with the node's coordinates.
