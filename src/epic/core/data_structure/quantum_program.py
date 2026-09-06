from __future__ import annotations

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass
from typing import Dict, Set, List, Tuple, TYPE_CHECKING
from uuid import UUID, uuid4

from .physical_qubit import PhysicalQubit

# `language` imports back from this module, so it's only imported for type
# checking here; runtime uses import it lazily to avoid a circular import.
if TYPE_CHECKING:
    from ..language import QECGadget, AllocCode


class ProgramQubit(BaseModel):
    """
    A class representing a qubit in a quantum program.

    Attributes:
        id (UUID): The unique identifier of the qubit.
        name (str): The name of the qubit.
        used_until (int): The position until which the qubit is used.
    """

    id: UUID = Field(default_factory=uuid4, init=False)
    name: str

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProgramQubit):
            return NotImplemented
        return self.id == other.id

@dataclass(frozen=True)
class QProgOperation:
    """
    A class representing a quantum operation in a quantum program.

    Attributes:
        name (str): The name of the quantum operation.
        length (int): The duration of the operation in time steps.
        targets (List[ProgramQubit | PhysicalQubit]): A list of qubits involved in the operation.
        implementation (QECGadget): An optional QECGadget that implement the operation.
    """

    name: str
    length: int
    targets: List[ProgramQubit | PhysicalQubit | str] = Field(default_factory=list)

    implementation: QECGadget | None = None
    measurement_id: UUID | None = None

    @field_validator("length")
    @classmethod
    def validate_length(cls, length: int, info: ValidationInfo) -> int:
        if length < 0:
            raise ValueError("Operation length must be non-negative.")
        return length



class QuantumProgram(BaseModel):
    """
    A class representing a quantum program.

    Attributes:
        name (str): The name of the quantum program.
        qubits (list): A list of qubits used in the program.
        gates (list): A list of quantum gates applied in the program.
        operations (list): A list of quantum operations in the program along with their start times.
        width (int): The number of qubits in the program.
        depth (int): The total duration of the program in time steps.
    """

    name: str = Field(default="QuantumProgram")
    id: UUID = Field(default_factory=uuid4, init=False)
    
    qubits: Set[ProgramQubit | PhysicalQubit] = Field(default_factory=set)
    qubits_latest_usage: Dict[ProgramQubit | PhysicalQubit, int] = Field(default_factory=dict, init=False)

    width: int = Field(default=0, init=False)
    depth: int = Field(default=0, init=False)

    operations: List[Tuple[QProgOperation, int]] = Field(default_factory=list)


    def add_qubit(self, qubit: ProgramQubit | PhysicalQubit | AllocCode | None = None,) -> ProgramQubit:
        """
        Add a qubit to the quantum program.

        Returns:
            ProgramQubit: The newly added qubit.
        """
        from ..language import AllocCode  # local import to avoid a circular import

        if qubit is None:
            qubit = ProgramQubit(name=f"q{self.width}")

        if isinstance(qubit, AllocCode):
            self.operations.append((QProgOperation(name="alloc_code", length=0, targets=[], implementation=qubit), self.depth))
            qubit = ProgramQubit(name=qubit.code_varname)

        self.qubits.add(qubit)
        self.qubits_latest_usage[qubit] = 0
        self.width += 1
        return qubit

    def add_qubits(self, count: int) -> list[ProgramQubit]:
        """Add multiple qubits to the quantum program.

        Args:
            count: The number of qubits to add.

        Returns:
            The newly added qubits in allocation order.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Qubit count must be a non-negative integer.")

        return [self.add_qubit() for _ in range(count)]

    def add_operation(self, operation: "QProgOperation | QuantumProgram") -> None:
        """
        Add a quantum operation to the program.

        Args:
            operation (QProgOperation | QuantumProgram): The quantum operation or sub-program to be added.
        """
        if isinstance(operation, QuantumProgram):
            for qubit in operation.qubits:
                if qubit is None or qubit not in self.qubits:
                    self.add_qubit(qubit)
            # Re-add the sub-program's operations in order so scheduling is
            # driven by the parent's own qubit occupancy, not double-booked.
            for sub_operation, _ in sorted(operation.operations, key=lambda x: x[1]):
                self.add_operation(
                    QProgOperation(
                        name=sub_operation.name,
                        length=sub_operation.length,
                        targets=sub_operation.targets,
                        implementation=sub_operation.implementation,
                        measurement_id=sub_operation.measurement_id,
                    )
                )
            return

        targets = operation.targets
        op_length = operation.length

        earliest_open_position = 0
        for qubit in targets:
            if qubit is None or qubit not in self.qubits:
                self.add_qubit(qubit)
            earliest_open_position = max(earliest_open_position, self.qubits_latest_usage.get(qubit, 0))

        for qubit in targets:
            self.qubits_latest_usage[qubit] = earliest_open_position + op_length

        self.depth = max(self.depth, earliest_open_position + op_length)

        self.operations.append((operation, earliest_open_position))

    def add_operations(self, operations: List[QProgOperation]) -> None:
        """
        Add multiple quantum operations to the program.

        Args:
            operations (List[QProgOperation]): A list of quantum operations to be added.
        """
        for operation in operations:
            self.add_operation(operation)

    def add_tick(self, target: List[ProgramQubit]) -> None:
        """
        Add a tick operation to the program.

        Args:
            target (List[ProgramQubit]): A list of qubits to which the tick operation is applied.
        """
        tick_operation = QProgOperation(name="tick", length=0, targets=target)
        self.add_operation(tick_operation)

    def operation_in_time(self) -> Tuple[Dict[int, List[QProgOperation]], Dict[int, List[QProgOperation]]]:
        """
        Get the operations in the program organized by time.

        Returns:
            Tuple[Dict[int, List[QProgOperation]], Dict[int, List[QProgOperation]]]: A tuple containing two dictionaries:
                - The first dictionary maps time steps to a list of operations starting at that time.
                - The second dictionary maps time steps to a list of operations ending at that time, including ticks.
        """
        operations_by_start_time: Dict[int, List[QProgOperation]] = {}
        operations_by_end_time: Dict[int, List[QProgOperation]] = {}

        for operation, start_time in self.operations:
            end_time = start_time + operation.length
            operations_by_start_time.setdefault(start_time, []).append(operation)
            operations_by_end_time.setdefault(end_time, []).append(operation)

        return operations_by_start_time, operations_by_end_time



                