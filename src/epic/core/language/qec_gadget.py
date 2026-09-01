from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from epic.core.data_structure.physical_qubit import PhysicalQubit

from ..compilation.measurement_record import MeasurementRecordView
from ..compilation.quantum_memory import QuantumMemory
from ..qec_object.logical_operator import LogicalOperatorUpdate
from ..qec_object.logical_qubit import LogicalQubit
from ..qec_object.observable import Observable
from ..qec_object.stabilizer_code import StabilizerCode

if TYPE_CHECKING:
    from ..qec_primitives.interfaces.qec_primitive import QECPrimitive


class QECGadget(ABC, BaseModel):
    """Base type for high-level gadgets."""

    model_config = ConfigDict(frozen=True)
    tag: str = Field(default="", frozen=True)

    id: UUID = Field(default_factory=uuid4, init=False)

    @property
    def estimated_gate_cost(self) -> int:
        """Return the cost of implementing this gadget in terms of gates."""
        return 0

    @property
    def estimated_ancilla_cost(self) -> int:
        """Return the cost of implementing this gadget in terms of ancilla qubits."""
        return 0

    # def __mul__(self, other: "QECGadget"):
    #     """Combine two gadgets sequentially."""
    #     pass

    # def __matmul__(self, other):
    #     """Combine two gadgets in parallel, i.e., the gates of the two gadgets can be interleaved."""
    #     pass


class LogicGadget(QECGadget):
    """A gadget that operates on logical operators."""

    targets: List[str] = Field(default_factory=list)

    @abstractmethod
    def compile(
        self,
        resolved_targets: List[Tuple[LogicalQubit, StabilizerCode]],
        record: MeasurementRecordView,
        quantum_memory: QuantumMemory,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive], List[PhysicalQubit]]:
        """Compile this gadget into a sequence of primitive code instructions, along with any logical
        operator updates and observables produced by the gadget."""
        pass


class CodeGadget(QECGadget):
    """A gadget that operates on one or more stabilizer codes."""

    targets: List[str] = Field(default_factory=list)

    @abstractmethod
    def compile(
        self,
        resolved_targets: List[StabilizerCode],
        record: MeasurementRecordView,
        quantum_memory: QuantumMemory,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive], List[PhysicalQubit]]:
        """Compile this gadget into a sequence of primitive code instructions, along with any logical operator updates and observables produced by the gadget."""
        pass


class AllocCode(QECGadget):
    """A gadget that allocates qubits in the quantum memory and create new logical qubits and code in the context."""

    target_code: StabilizerCode
    code_varname: str
    logical_qubits_varnames: List[str] = Field(default_factory=list)


class FreeCode(QECGadget):
    """Release a registered code from the context and free its allocated qubits."""

    code_varname: str
