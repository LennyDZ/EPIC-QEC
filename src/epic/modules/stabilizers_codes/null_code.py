from typing import List
from uuid import uuid4

from pydantic import Field, model_validator

from epic.core.data_structure import PauliChar, PauliString, TannerGraph, VariableNode
from epic.core.qec_object import LogicalOperator, LogicalQubit, StabilizerCode


class NullCode(StabilizerCode):

    name: str = Field(
        default_factory=lambda: f"NullCode_{uuid4().hex[:8]}",
        description="Human-readable name for the stabilizer code.",
    )

    n: int = Field(
        description="Number of physical qubits in the code.",
        default=0,
    )
    k: int = Field(
        description="Number of logical qubits in the code.", default=0, init=False
    )
    d: int = Field(
        description="Distance of the code.",
        default=0,
        init=False,
    )

    validate_algebraic_properties: bool = Field(
        default=False,
        description="NullCode is generated directly and does not require algebraic consistency checks.",
    )

    tanner_graph: TannerGraph = Field(default_factory=TannerGraph, init=False)
    logical_qubits: List[LogicalQubit] = Field(default_factory=list, init=False)

    @model_validator(mode="after")
    def initialize_null_code(self) -> "NullCode":
        """Post-init canonicalization: k=n, n variable nodes, and one logical qubit per node."""
        variable_nodes = [VariableNode(tag=f"v_{i}_{self.name}") for i in range(self.n)]

        self.k = self.n
        self.tanner_graph = TannerGraph(
            variable_nodes=set(variable_nodes),
            check_nodes=set(),
            edges=set(),
        )

        self.logical_qubits = [
            LogicalQubit(
                name=f"{self.name}_lq_{i}",
                logical_x=LogicalOperator(
                    logical_type=PauliChar.X,
                    operator=PauliString(string=(PauliChar.X,)),
                    target_nodes=(node,),
                ),
                logical_z=LogicalOperator(
                    logical_type=PauliChar.Z,
                    operator=PauliString(string=(PauliChar.Z,)),
                    target_nodes=(node,),
                ),
            )
            for i, node in enumerate(variable_nodes)
        ]

        return self
