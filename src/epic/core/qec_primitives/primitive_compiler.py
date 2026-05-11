from importlib import import_module
from types import MappingProxyType
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, PrivateAttr

from etic.core.data_structure.tanner_node import TannerNode
from etic.core.qec_object.detector import DetectorGraphPort, QubitPortState

from ..compilation.measurement_record import MeasurementRecordView
from ..compilation.quantum_memory import QuantumMemory
from ..qec_object import Detector, Measurement
from .interfaces import QECPrimitive


class PrimitiveRegistry(BaseModel):
    """Registry for available primitive implementations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _registry: dict[type[QECPrimitive], type] = PrivateAttr(default_factory=dict)

    def register(self, instr_cls: type[QECPrimitive], impl_cls: type):
        """Register the implementation class used for a primitive instruction type."""
        self._registry[instr_cls] = impl_cls

    def get(self, instr_cls: type[QECPrimitive]) -> type:
        """Return the registered implementation class for a primitive type."""
        try:
            return self._registry[instr_cls]
        except KeyError as exc:
            raise KeyError(
                f"No primitive implementation registered for {instr_cls.__module__}.{instr_cls.__name__}."
            ) from exc

    def load_from_config(self, config: dict):
        """Populate the registry from dotted-path entries in the compiler config."""
        mapping = config.get("primitives", {})
        if not isinstance(mapping, dict):
            raise ValueError("config['primitives'] must be a dictionary.")

        for instr_path, impl_path in mapping.items():
            instr_module, instr_name = instr_path.rsplit(".", 1)
            impl_module, impl_name = impl_path.rsplit(".", 1)

            instr_cls = getattr(import_module(instr_module), instr_name)
            impl_cls = getattr(import_module(impl_module), impl_name)

            if not isinstance(impl_cls, type):
                raise TypeError(
                    f"Configured primitive implementation {impl_path} must be a class."
                )

            self.register(instr_cls, impl_cls)


class PrimitiveCompiler(BaseModel):
    """Compile primitive instructions via config-selected implementations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: dict[str, Any]
    _registry: PrimitiveRegistry = PrivateAttr(default_factory=PrimitiveRegistry)

    def model_post_init(self, __context: Any) -> None:
        """Load primitive implementations once the model has been initialized."""
        self._registry.load_from_config(self.config)

    def compile(
        self,
        primitive_instruction: QECPrimitive,
        memory: QuantumMemory,
        record: MeasurementRecordView,
        det_graph_port: MappingProxyType[TannerNode, QubitPortState],
        parent_gadget_id: UUID,
    ) -> tuple[list[str], list[Measurement], list[Detector], DetectorGraphPort]:
        """Compile a primitive by dispatching to its configured implementation class."""

        impl_cls = primitive_instruction.get_implementation_class(self._registry)

        implementation = impl_cls()

        result = implementation.compile(
            instruction=primitive_instruction,
            memory=memory,
            record=record,
            det_graph_port=det_graph_port,
            parent_gadget_id=parent_gadget_id,
        )

        instructions, measurements, detectors, detector_stitch = result
        return list(instructions), list(measurements), list(detectors), detector_stitch
