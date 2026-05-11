from typing import List, Tuple
from pydantic import BaseModel, ConfigDict
from enum import Enum


class PauliChar(str, Enum):
    """Enum representing the single-qubit Pauli operators."""

    X = "X"
    Y = "Y"
    Z = "Z"
    I = "I"

    def dual(self) -> "PauliChar":
        """Return the dual Pauli basis."""
        match self:
            case PauliChar.X:
                return PauliChar.Z
            case PauliChar.Z:
                return PauliChar.X
            case PauliChar.Y:
                return PauliChar.Y
            case PauliChar.I:
                return PauliChar.I


class PauliEigenState(Enum):
    """Enum representing the eigenstates of the single-qubit Pauli operators."""

    Z_plus = "0"
    Z_minus = "1"
    X_plus = "+"
    X_minus = "-"
    Y_plus = "+i"
    Y_minus = "-i"

    def pauli_from_zero(self) -> List[str]:
        """Return the sequence of gates to prepare this eigenstate from the |0> state."""
        match self:
            case PauliEigenState.Z_plus:
                return []
            case PauliEigenState.Z_minus:
                return [PauliChar.X]
            case PauliEigenState.X_plus:
                return ["H"]
            case PauliEigenState.X_minus:
                return ["H", PauliChar.Z]
            case PauliEigenState.Y_plus:
                return ["H", "S"]
            case PauliEigenState.Y_minus:
                return [PauliChar.X, "H", "S"]


class PauliString(BaseModel):
    """Representation of a Pauli string acting on multiple qubits.

    Attributes:
        string (Tuple[PauliChar, ...]): List of Pauli characters representing the Pauli string.
    """

    model_config = ConfigDict(frozen=True)
    string: Tuple[PauliChar, ...]

    @property
    def weight(self) -> int:
        """Calculate the weight of the Pauli string (number of non-identity operators)."""
        return sum(1 for p in self.string if p != PauliChar.I)

    def commutes_with(self, other: "PauliString") -> bool:
        """Check if this Pauli string commutes with another pauli string."""
        if len(self.string) != len(other.string):
            raise ValueError(
                "Pauli strings must be of the same length to check commutation."
            )

        anti_commute_count = 0
        for p1, p2 in zip(self.string, other.string):
            if p1 == PauliChar.I or p2 == PauliChar.I:
                continue
            if p1 != p2:
                anti_commute_count += 1

        return anti_commute_count % 2 == 0
