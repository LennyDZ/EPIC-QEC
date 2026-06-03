from enum import Enum
from typing import List

from pydantic import BaseModel, field_validator, model_validator

RESET_GATES = {"R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"}
MEASUREMENT_GATES = {
    "M",
    "MX",
    "MY",
    "MZ",
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "MXX",
    "MYY",
    "MZZ",
    "MPP",
}
CLIFFORD_GATES = {
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "SQRT_X",
    "SQRT_Y",
    "CX",
    "CY",
    "CZ",
    "CNOT",
    "SWAP",
}

SCALAR_PROBABILITY_INSTRUCTIONS = {
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "E",
    "ELSE_CORRELATED_ERROR",
    "HERALDED_ERASE",
    "I_ERROR",
    "II_ERROR",
}

THREE_TERM_CHANNEL_INSTRUCTIONS = {"PAULI_CHANNEL_1"}
FIFTEEN_TERM_CHANNEL_INSTRUCTIONS = {"PAULI_CHANNEL_2"}
FOUR_TERM_CHANNEL_INSTRUCTIONS = {"HERALDED_PAULI_CHANNEL_1"}


class NoiseInstruction(Enum):
    # Explicit Stim noise instructions.
    DEPOLARIZE1 = "DEPOLARIZE1"
    DEPOLARIZE2 = "DEPOLARIZE2"
    X_ERROR = "X_ERROR"
    Y_ERROR = "Y_ERROR"
    Z_ERROR = "Z_ERROR"
    PAULI_CHANNEL_1 = "PAULI_CHANNEL_1"
    PAULI_CHANNEL_2 = "PAULI_CHANNEL_2"
    # HERALDED_PAULI_CHANNEL_1 = "HERALDED_PAULI_CHANNEL_1"
    # HERALDED_ERASE = "HERALDED_ERASE"
    # E = "E"
    # ELSE_CORRELATED_ERROR = "ELSE_CORRELATED_ERROR"
    # I_ERROR = "I_ERROR"
    # II_ERROR = "II_ERROR"


class NoiseApplicationMode(Enum):
    BEFORE_CLIFFORD = "BEFORE_CLIFFORD"
    AFTER_CLIFFORD = "AFTER_CLIFFORD"
    AFTER_RESET = "AFTER_RESET"
    BEFORE_MEASUREMENT = "BEFORE_MEASUREMENT"
    BEFORE_ROUND = "BEFORE_ROUND"


class NoiseSpecification(BaseModel):
    instruction: NoiseInstruction
    probability: float | List[float]
    application_mode: NoiseApplicationMode
    target_gates: List[str]

    @model_validator(mode="after")
    def validate_application_mode(self):
        if not self.target_gates:
            raise ValueError("target_gates must not be empty.")

        targets = [g.upper() for g in self.target_gates]

        if self.application_mode == NoiseApplicationMode.AFTER_RESET:
            invalid = [g for g in targets if g not in RESET_GATES]
            if invalid:
                raise ValueError(
                    f"AFTER_RESET only supports reset-like gates. Invalid targets: {invalid}"
                )

        elif self.application_mode == NoiseApplicationMode.BEFORE_MEASUREMENT:
            invalid = [g for g in targets if g not in MEASUREMENT_GATES]
            if invalid:
                raise ValueError(
                    "BEFORE_MEASUREMENT only supports measurement-like gates. "
                    f"Invalid targets: {invalid}"
                )

        elif self.application_mode in {
            NoiseApplicationMode.BEFORE_CLIFFORD,
            NoiseApplicationMode.AFTER_CLIFFORD,
        }:
            invalid = [g for g in targets if g not in CLIFFORD_GATES]
            if invalid:
                raise ValueError(
                    "BEFORE/AFTER_CLIFFORD only supports Clifford gates. "
                    f"Invalid targets: {invalid}"
                )

        self.target_gates = targets
        return self

    @field_validator("probability")
    def validate_probability(cls, v):
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Probability list must not be empty.")
            if not all(0 <= p <= 1 for p in v):
                raise ValueError("All probabilities must be between 0 and 1.")
        elif not (0 <= v <= 1):
            raise ValueError("Probability must be between 0 and 1.")
        return v

    @model_validator(mode="after")
    def validate_instruction_probability_shape(self):
        p = self.probability
        instruction_name = self.instruction.value

        if instruction_name in SCALAR_PROBABILITY_INSTRUCTIONS:
            if isinstance(p, list):
                raise ValueError(
                    f"{self.instruction.value} expects a scalar probability, not a list."
                )

        elif instruction_name in THREE_TERM_CHANNEL_INSTRUCTIONS:
            if not isinstance(p, list) or len(p) != 3:
                raise ValueError("PAULI_CHANNEL_1 expects a list of 3 probabilities.")
            if sum(p) > 1:
                raise ValueError("PAULI_CHANNEL_1 probabilities must sum to <= 1.")

        elif instruction_name in FIFTEEN_TERM_CHANNEL_INSTRUCTIONS:
            if not isinstance(p, list) or len(p) != 15:
                raise ValueError("PAULI_CHANNEL_2 expects a list of 15 probabilities.")
            if sum(p) > 1:
                raise ValueError("PAULI_CHANNEL_2 probabilities must sum to <= 1.")

        elif instruction_name in FOUR_TERM_CHANNEL_INSTRUCTIONS:
            if not isinstance(p, list) or len(p) != 4:
                raise ValueError(
                    "HERALDED_PAULI_CHANNEL_1 expects a list of 4 probabilities."
                )
            if sum(p) > 1:
                raise ValueError(
                    "HERALDED_PAULI_CHANNEL_1 probabilities must sum to <= 1."
                )

        return self
