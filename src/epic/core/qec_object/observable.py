from typing import List, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .logical_operator import LogicalOperator
from .measurement import Measurement


class Observable(BaseModel):
    """Represents an observable in the context of quantum error correction.

    An observable is associated with a parity of recorded measurements and, when
    applicable, with the logical operators whose frame corrections should be folded in.

    Attributes
    ----------
    id : UUID
        Unique identifier for the observable.
    measurements : Set[Measurement]
        Set of measurements that the observable is associated with. Each measurement corresponds to a specific node at a given timestep.
    tag : str
        Optional human-readable tag for the observable, which can be used for debugging or visualization purposes.
    """

    id: UUID = Field(
        description="Unique identifier for the observable",
        default_factory=uuid4,
    )

    logical_operators_involved: List[LogicalOperator] = Field(
        default_factory=list,
        description="Set of logical operators that the observable is associated with if any, e.g. if the observable is a measurement of X1*X2, it should be associated with the logical operators representing X1 and X2. This is used to keep track of the frame corrections that need to be applied to the observable when the logical operators are updated by the application of gadgets or primitives.",
    )

    measurements: Set[Measurement] = Field(
        description="Set of measurements associated with the observable. Each measurement corresponds to a specific node at a given timestep.",
        default_factory=set,
    )

    tag: str = Field(
        description="Optional human-readable tag for the observable, which can be used for debugging or visualization purposes.",
        default="",
    )

    def apply_corrective_frame(self):
        """XOR logical-operator frame corrections into the observable measurement set."""
        for op in self.logical_operators_involved:
            self.measurements ^= op.frame_correction_history
        return self
