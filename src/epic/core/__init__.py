"""Public exports for the core QEC domain model."""

from pydantic.dataclasses import rebuild_dataclass

from . import compilation, data_structure, experiment, language, qec_object, qec_primitives
from .compilation import *
from .data_structure import *
from .experiment import *
from .language import *
from .qec_object import *
from .qec_primitives import *

# `data_structure.quantum_program` only imports `language` under TYPE_CHECKING (to avoid
# a circular import), so its forward-referenced `QECGadget` field must be resolved here
# now that `language` is fully loaded.
rebuild_dataclass(data_structure.QProgOperation, _types_namespace={"QECGadget": language.QECGadget})

__all__ = [
	"compilation",
	"data_structure",
	"experiment",
	"language",
	"qec_object",
	"qec_primitives",
]
__all__ += compilation.__all__
__all__ += data_structure.__all__
__all__ += experiment.__all__
__all__ += language.__all__
__all__ += qec_object.__all__
__all__ += qec_primitives.__all__
