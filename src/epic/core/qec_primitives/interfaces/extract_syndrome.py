from pydantic import Field, field_validator

from ...data_structure import PauliEigenState
from .qec_primitive import QECPrimitive


class ExtractSyndrome(QECPrimitive):
    """Primitive instruction for extracting the syndrome of a quantum error correcting code."""

    rounds: int
    ancilla_reset_state: PauliEigenState = Field(
        default=PauliEigenState.Z_plus,
        description="Initial state for the ancilla qubits used in stabiliser checks.",
    )

    @field_validator("rounds")
    def validate_rounds(cls, rounds):
        """Ensure syndrome extraction runs for at least one round."""
        if rounds < 1:
            raise ValueError("Rounds must be a positive integer.")
        return rounds
