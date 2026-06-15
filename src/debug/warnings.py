class CompilerWarning(Warning):
    """Base warning for mycompiler."""

    pass


class WeirdAllocationWarning(CompilerWarning):
    """Warning for weird qubit allocation patterns."""

    pass


class CodeBelowDistanceWarning(CompilerWarning):
    """Warning raised when allocating a code below the objective distance."""

    def __init__(self, code_distance: int, objective_distance: int, gadget_tag: str):
        message = f"Code distance {code_distance} is below objective distance {objective_distance} (gadget: {gadget_tag})"
        super().__init__(message)


class EmptyInputWarning(CompilerWarning):
    """Warning raised when empty input."""

    def __init__(self):
        message = "Received an empty list of targets."
        super().__init__(message)


class MissingNodeStatusWarning(CompilerWarning):
    """Warning raised when missing node statuses are defaulted to UNKNOWN."""

    pass


class MemOverwriteWarning(CompilerWarning):
    """Warning raised when overwriting an existing key in RegisterMemory."""

    pass
