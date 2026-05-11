from typing import Dict, List, Tuple
from uuid import UUID

from pydantic import Field

from etic.core.data_structure.pauli import PauliChar
from .qec_primitive import QECPrimitive


class Readout(QECPrimitive):
    """Represents a simple readout of a code where all variable nodes are measured in the given basis."""

    readout_basis: PauliChar = Field(
        description="The basis in which the readout is performed. Valid values are 'X', 'Y', and 'Z'.",
        default=PauliChar.Z,
    )
