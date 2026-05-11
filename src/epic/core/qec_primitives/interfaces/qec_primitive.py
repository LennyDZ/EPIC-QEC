from abc import ABC
from typing import Any, Generic, List, Protocol, Tuple, TypeVar
from uuid import UUID, uuid4


from pydantic import BaseModel, Field, field_validator

from ...qec_object.detector import DetectorGraphPort
from ...compilation.measurement_record import MeasurementRecordView
from ...compilation.quantum_memory import QuantumMemory
from ...data_structure import TannerGraph
from ...qec_object import Detector, Measurement


class QECPrimitive(ABC, BaseModel):
    """Abstract base class for primitives compiled against a Tanner graph."""

    target: TannerGraph
    tag: str = ""
    distance: int = 0

    id: UUID = Field(default_factory=uuid4, init=False)

    @field_validator("distance")
    def validate_distance(cls, distance):
        """Ensure the requested objective distance is non-negative."""
        if distance < 0:
            raise ValueError("Distance must be a non-negative integer.")
        return distance

    def to_payload(self) -> dict[str, Any]:
        """Serialize the primitive while preserving its Tanner-graph target."""
        data = self.model_dump(exclude={"target"})
        data["target"] = self.target
        return data

    def get_implementation_class(self, registry) -> type:
        """Look up the configured implementation class for this primitive type."""
        return registry.get(type(self))


T = TypeVar("T", bound="QECPrimitive", contravariant=True)


class PrimitiveImplementation(Protocol, Generic[T]):
    """Protocol implemented by backend-specific primitive compilers."""

    def compile(
        self,
        instruction: T,
        memory: QuantumMemory,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[List[str], List[Measurement], List[Detector], DetectorGraphPort]:
        """Compile one primitive into circuit instructions, outputs, and port state."""
        ...
