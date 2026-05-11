from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Measurement:
    """Lightweight measurement event produced during compilation.

    Each measurement records the measured node together with the parent gadget and
    primitive identifiers that emitted it.
    """

    node_id: UUID
    parent_gadget_id: UUID
    parent_primitive_id: UUID
    tag: str = Field(
        default="", description="Optional tag describing the type of measurement event."
    )

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the measurement."
    )
