from pydantic import BaseModel, Field

from epic.core.experiment.noise_specification import (
    NoiseApplicationMode,
    NoiseInstruction,
    NoiseSpecification,
)

ONE_QUBIT_CLIFFORD_GATES = ["I", "X", "Y", "Z", "H", "S"]
TWO_QUBIT_CLIFFORD_GATES = ["CX", "CY", "CZ", "SWAP"]
RESET_GATES = ["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]
MEASUREMENT_GATES = [
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
]


class NoiseModel(BaseModel):
    noises: list[NoiseSpecification] = Field(
        default_factory=list,
        description="List of noise specifications to be applied to the compiled program.",
    )

    def apply_model(self, compiled_program: str) -> str:
        """Apply the noise model to the compiled program by inserting the specified noise instructions at the appropriate locations in the circuit.

        Args:
            compiled_program: The compiled program as a string.

        Returns:
            The modified compiled program with noise instructions inserted.
        """
        if not compiled_program:
            return compiled_program

        noisy_lines: list[str] = []

        for line in compiled_program.splitlines():
            gate_name, targets = self._parse_stim_line(line)

            if gate_name is None:
                noisy_lines.append(line)
                continue

            # Apply "before" rules first.
            for spec in self.noises:
                if (
                    spec.application_mode == NoiseApplicationMode.BEFORE_CLIFFORD
                    and self._matches(spec, gate_name)
                ):
                    noisy_lines.append(self._format_noise_instruction(spec, targets))
                elif (
                    spec.application_mode == NoiseApplicationMode.BEFORE_MEASUREMENT
                    and self._matches(spec, gate_name)
                ):
                    noisy_lines.append(self._format_noise_instruction(spec, targets))

            noisy_lines.append(line)

            # Apply "after" rules after the gate.
            for spec in self.noises:
                if (
                    spec.application_mode == NoiseApplicationMode.AFTER_CLIFFORD
                    and self._matches(spec, gate_name)
                ):
                    noisy_lines.append(self._format_noise_instruction(spec, targets))
                elif (
                    spec.application_mode == NoiseApplicationMode.AFTER_RESET
                    and self._matches(spec, gate_name)
                ):
                    noisy_lines.append(self._format_noise_instruction(spec, targets))

        return "\n".join(noisy_lines)

    def _matches(self, spec: NoiseSpecification, gate_name: str) -> bool:
        normalized_targets = {self._normalize_gate_name(g) for g in spec.target_gates}
        return gate_name in normalized_targets

    @staticmethod
    def _normalize_gate_name(gate_name: str) -> str:
        gate = gate_name.upper()
        if gate == "CNOT":
            return "CX"
        return gate

    def _format_noise_instruction(self, spec: NoiseSpecification, targets: str) -> str:
        probability = self._format_probability(spec.probability)
        line = f"{spec.instruction.value}({probability})"
        if targets:
            return f"{line} {targets}"
        return line

    @staticmethod
    def _format_probability(probability: float | list[float]) -> str:
        if isinstance(probability, list):
            return ", ".join(f"{p:.12g}" for p in probability)
        return f"{probability:.12g}"

    def _parse_stim_line(self, line: str) -> tuple[str | None, str]:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            return None, ""

        first_token = stripped.split(maxsplit=1)[0]
        gate_name = self._normalize_gate_name(first_token.split("(", 1)[0])

        # Skip annotation/structural lines.
        if gate_name in {
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "SHIFT_COORDS",
            "TICK",
            "QUBIT_COORDS",
        }:
            return None, ""

        targets = stripped[len(first_token) :].strip()
        return gate_name, targets


class StimLikeNoiseModel(NoiseModel):
    """Convenience noise model that accepts Stim-generated-like probability knobs.

    This class does not attempt to replicate every detail of Stim scheduling,
    but provides a close gate-based approximation using the same probability
    parameters commonly used with `stim.Circuit.generated(...)`.
    """

    before_round_data_depolarization: float | None = None
    before_measure_flip_probability: float | None = None
    after_reset_flip_probability: float | None = None
    after_clifford_depolarization: float | None = None

    @classmethod
    def from_stim_like_probabilities(
        cls,
        *,
        before_round_data_depolarization: float | None = None,
        before_measure_flip_probability: float | None = None,
        after_reset_flip_probability: float | None = None,
        after_clifford_depolarization: float | None = None,
    ) -> "StimLikeNoiseModel":
        noises: list[NoiseSpecification] = []

        # Temporarily disabled: before-round depolarization needs explicit
        # data-qubit targeting semantics before it can be safely applied.
        # if before_round_data_depolarization is not None:
        #     noises.append(
        #         NoiseSpecification(
        #             instruction=NoiseInstruction.DEPOLARIZE1,
        #             probability=before_round_data_depolarization,
        #             application_mode=NoiseApplicationMode.BEFORE_ROUND,
        #             target_gates=ONE_QUBIT_CLIFFORD_GATES,
        #         )
        #     )

        if before_measure_flip_probability is not None:
            # Use an anticommuting flip for X-basis measurements.
            noises.append(
                NoiseSpecification(
                    instruction=NoiseInstruction.Z_ERROR,
                    probability=before_measure_flip_probability,
                    application_mode=NoiseApplicationMode.BEFORE_MEASUREMENT,
                    target_gates=["MX", "MRX"],
                )
            )
            noises.append(
                NoiseSpecification(
                    instruction=NoiseInstruction.X_ERROR,
                    probability=before_measure_flip_probability,
                    application_mode=NoiseApplicationMode.BEFORE_MEASUREMENT,
                    target_gates=[
                        "M",
                        "MY",
                        "MZ",
                        "MR",
                        "MRY",
                        "MRZ",
                    ],
                )
            )

        if after_reset_flip_probability is not None:
            # Use an anticommuting flip for X-basis resets.
            noises.append(
                NoiseSpecification(
                    instruction=NoiseInstruction.Z_ERROR,
                    probability=after_reset_flip_probability,
                    application_mode=NoiseApplicationMode.AFTER_RESET,
                    target_gates=["RX", "MRX"],
                )
            )
            noises.append(
                NoiseSpecification(
                    instruction=NoiseInstruction.X_ERROR,
                    probability=after_reset_flip_probability,
                    application_mode=NoiseApplicationMode.AFTER_RESET,
                    target_gates=["R", "RY", "RZ", "MR", "MRY", "MRZ"],
                )
            )

        if after_clifford_depolarization is not None:
            noises.append(
                NoiseSpecification(
                    instruction=NoiseInstruction.DEPOLARIZE1,
                    probability=after_clifford_depolarization,
                    application_mode=NoiseApplicationMode.AFTER_CLIFFORD,
                    target_gates=ONE_QUBIT_CLIFFORD_GATES,
                )
            )
            noises.append(
                NoiseSpecification(
                    instruction=NoiseInstruction.DEPOLARIZE2,
                    probability=after_clifford_depolarization,
                    application_mode=NoiseApplicationMode.AFTER_CLIFFORD,
                    target_gates=TWO_QUBIT_CLIFFORD_GATES,
                )
            )

        return cls(
            noises=noises,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
            after_clifford_depolarization=after_clifford_depolarization,
        )

    def apply_model(self, compiled_program: str) -> str:
        return super().apply_model(compiled_program)
