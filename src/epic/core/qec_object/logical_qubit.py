from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from .logical_operator import LogicalOperator


class LogicalQubit(BaseModel):
    """Model representing a logical qubit in a quantum error-correcting code.

    Attributes:
        name (str): Name of the logical qubit.
        id (UUID): Unique identifier of the logical qubit.
        logical_x (LogicalOperator): Logical X operator associated with the logical qubit.
        logical_z (LogicalOperator): Logical Z operator associated with the logical qubit.
    """

    id: UUID = Field(
        description="Unique identifier of the logical qubit",
        default_factory=uuid4,
    )
    name: str = Field(description="Name of the logical qubit")

    logical_x: LogicalOperator = Field(
        description="Logical X operator associated with the logical qubit",
    )

    logical_z: LogicalOperator = Field(
        description="Logical Z operator associated with the logical qubit",
    )
