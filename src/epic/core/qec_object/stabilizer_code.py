from typing import Dict, List, Tuple
from uuid import UUID, uuid4

from scipy.sparse import csr_matrix

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..data_structure import (
    CheckNode,
    PauliChar,
    PauliString,
    TannerEdge,
    TannerGraph,
    VariableNode,
)
from .logical_operator import LogicalOperator
from .logical_qubit import LogicalQubit


class StabilizerCode(BaseModel):
    """Class representing a stabilizer code defined by its Tanner graph and logical operators."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the stabilizer code.",
    )

    name: str = Field(
        default_factory=lambda: f"SC_{uuid4().hex[:8]}",
        description="Human-readable name for the stabilizer code.",
    )

    n: int = Field(
        description="Number of physical qubits (variable nodes) in the code.",
        default=0,
    )
    k: int = Field(
        description="Number of logical qubits in the code.",
        default=0,
    )
    d: int = Field(
        description="Distance of the code.",
        default=0,
    )

    tanner_graph: TannerGraph
    logical_qubits: List[LogicalQubit]

    validate_algebraic_properties: bool = Field(
        default=True,
        description="Whether to validate algebraic properties of the code upon initialization.",
    )

    @field_validator("n", "k", "d")
    def validate_non_negative_parameters(cls, value: int) -> int:
        """Ensure the basic code parameters are non-negative."""
        if value < 0:
            raise ValueError("Parameters n, k, and d must be non-negative.")
        return value

    @model_validator(mode="after")
    def validate_logical_operators_count(self) -> "StabilizerCode":
        """Ensure the declared number of logical qubits matches ``k``."""
        if not self.validate_algebraic_properties:
            return self
        if len(self.logical_qubits) != self.k:
            raise ValueError("Number of logical operators must equal k.")
        return self

    @model_validator(mode="after")
    def validate_logical_operators_distance(self) -> "StabilizerCode":
        """Ensure each logical X and Z operator has weight at least ``d``."""
        if not self.validate_algebraic_properties:
            return self
        for lq in self.logical_qubits:
            if lq.logical_x.operator.weight < self.d:
                raise ValueError(
                    f"Logical qubit {lq} has logical X operator with weight less than d."
                )
            if lq.logical_z.operator.weight < self.d:
                raise ValueError(
                    f"Logical qubit {lq} has logical Z operator with weight less than d."
                )
        return self

    @model_validator(mode="after")
    def validate_logical_operators_commutation(self) -> "StabilizerCode":
        """Ensure the logical operators commute with the stabilizer checks."""
        if not self.validate_algebraic_properties:
            return self
        pcm, var, check = self.tanner_graph.parity_check_matrix
        var_to_col_index = {v.id: i for i, v in enumerate(var)}
        logical_operators = []
        for lq in self.logical_qubits:
            for lop in [lq.logical_x, lq.logical_z]:
                op_simplectic_vector = [0] * (2 * self.n)
                for pauli, var_node in zip(lop.operator.string, lop.target_nodes):
                    match pauli:
                        case "X":
                            op_simplectic_vector[var_to_col_index[var_node.id]] = 1
                        case "Z":
                            op_simplectic_vector[
                                var_to_col_index[var_node.id] + self.n
                            ] = 1
                        case "Y":
                            op_simplectic_vector[var_to_col_index[var_node.id]] = 1
                            op_simplectic_vector[
                                var_to_col_index[var_node.id] + self.n
                            ] = 1
                        case _:
                            continue
                logical_operators.append(op_simplectic_vector)

        H_X = pcm[:, : self.n]
        H_Z = pcm[:, self.n :]
        for lop in logical_operators:
            # Check commutation relations
            lop_X = lop[: self.n]
            lop_Z = lop[self.n :]
            s = (H_X.dot(lop_Z) + H_Z.dot(lop_X)) % 2
            if not all(c == 0 for c in s):
                raise ValueError(
                    f"Logical operators {lop} do not commute with stabilizers."
                )

        # TODO check logical operators aren't product of stabilizers
        # TODO check logical operators X and Z of a given qubit anticommute

        return self

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StabilizerCode):
            return NotImplemented
        return self.id == other.id

    @classmethod
    def from_pcm(
        cls,
        code_name: str,
        simplectic_pcm: csr_matrix,
        logical_qubits: List[Tuple[List[int], List[int]]],
        var_coordinate: Dict[int, Tuple[int, ...]] = {},
        check_coordinate: Dict[int, Tuple[int, ...]] = {},
    ) -> "StabilizerCode":
        """Construct a code from a given parity check matrix and a set of logical qubits.

        Parameters
        ----------
        code_name : str
            Name of the stabilizer code.
        simplectic_pcm : csr_matrix
            Parity check matrix in simplectic representation.
        logical_qubits : List[Tuple[List[int], List[int]]]
            List of tuples representing the logical X and Z operators for each logical qubit.
            Logical operators are represented as binary vectors mapping to the columns
            of the parity check matrix. If 1 then the qubit is part of the logical operator, if 0 it is not.

        Returns
        -------
        StabilizerCode
            Constructed stabilizer code instance.

        Raises
        ------
        ValueError
            If the parity check matrix has an odd number of columns.
        ValueError
            If the logical operators do not match the number of physical qubits.
        """

        n_row, n_col = simplectic_pcm.shape  # type: ignore
        if n_col % 2 != 0:
            raise ValueError(
                "Parity check matrix in simplectic representation must have an even number of columns."
            )

        n = n_col // 2
        k = len(logical_qubits)
        d = min(
            min(sum(1 for p in lop[0] if p != 0) for lop in logical_qubits),
            min(sum(1 for p in lop[1] if p != 0) for lop in logical_qubits),
        )
        if var_coordinate is not None and len(var_coordinate) != n:
            raise ValueError(
                "Variable coordinate dictionary length must match the number of physical qubits."
            )
        if var_coordinate is not None:
            var_nodes = [
                VariableNode(tag=f"v_{i}_{code_name}", coordinates=var_coordinate[i])
                for i in range(n)
            ]
        else:
            var_nodes = [VariableNode(tag=f"v_{i}_{code_name}") for i in range(n)]

        if check_coordinate is not None and len(check_coordinate) != n_row:
            raise ValueError(
                "Check coordinate dictionary length must match the number of checks."
            )

        logical_qubits_list = []
        for i, lq in enumerate(logical_qubits):
            lx_target = []
            lz_target = []
            for j in range(n):
                if lq[0][j] == 1:
                    lx_target.append(var_nodes[j])
                if lq[1][j] == 1:
                    lz_target.append(var_nodes[j])
            l = LogicalQubit(
                logical_x=LogicalOperator(
                    operator=PauliString(string=tuple([PauliChar.X] * len(lx_target))),
                    target_nodes=tuple(lx_target),
                    logical_type=PauliChar.X,
                ),
                logical_z=LogicalOperator(
                    operator=PauliString(string=tuple([PauliChar.Z] * len(lz_target))),
                    target_nodes=tuple(lz_target),
                    logical_type=PauliChar.Z,
                ),
                name=f"{code_name}_lq_{i}",
            )
            logical_qubits_list.append(l)

        check_nodes = (
            [
                CheckNode(tag=f"c_{i}_{code_name}", coordinates=check_coordinate[i])
                for i in range(n_row)
            ]
            if check_coordinate is not None
            else [CheckNode(tag=f"c_{i}_{code_name}") for i in range(n_row)]
        )
        edges = []
        for i in range(n_row):
            for j in range(n):
                if simplectic_pcm[i, j] == 1:
                    edges.append(
                        TannerEdge(
                            check_node=check_nodes[i],
                            variable_node=var_nodes[j],
                            pauli_checked=PauliChar.X,
                        )
                    )
                if simplectic_pcm[i, j + n] == 1:
                    edges.append(
                        TannerEdge(
                            check_node=check_nodes[i],
                            variable_node=var_nodes[j],
                            pauli_checked=PauliChar.Z,
                        )
                    )

        tanner_graph = TannerGraph(
            variable_nodes=set(var_nodes),
            check_nodes=set(check_nodes),
            edges=set(edges),
        )

        return cls(
            n=n,
            k=k,
            d=d,
            name=code_name,
            tanner_graph=tanner_graph,
            logical_qubits=logical_qubits_list,
            validate_algebraic_properties=False,
        )
