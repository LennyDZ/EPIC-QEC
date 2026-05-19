"""Public package facade for epic."""

from importlib.metadata import PackageNotFoundError, version

from .core.compilation import (
    CompiledExperiment,
    CompilationContext,
    MeasurementRecord,
    QECCompiler,
    QuantumMemory,
)
from .core.data_structure import (
    CheckNode,
    GraphAlgorithm,
    PauliChar,
    PauliEigenState,
    PauliString,
    TannerEdge,
    TannerGraph,
    TannerNode,
    VariableNode,
)
from .core.experiment import (
    NoiseApplicationMode,
    NoiseInstruction,
    NoiseModel,
    NoiseSpecification,
    StimLikeNoiseModel,
)
from .core.language import AllocCode, CodeGadget, FreeCode, LogicGadget, QECGadget
from .core.qec_object import (
    Detector,
    LogicalOperator,
    LogicalOperatorUpdate,
    LogicalQubit,
    Measurement,
    Observable,
    StabilizerCode,
)
from .modules import (
    CSSCode,
    InitCode,
    NaiveLogicalMeasurement,
    NaiveReadout,
    NullCode,
    RSCSurgery,
    RSCSyndromeExtraction,
    ReadoutCode,
    RotatedSurfaceCode,
    SimpleGateApplication,
    SimpleSyndromeExtraction,
    SurfaceCode,
    ToricCode,
)

try:
    __version__ = version("epic")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "AllocCode",
    "CSSCode",
    "CheckNode",
    "CodeGadget",
    "CompilationContext",
    "CompiledExperiment",
    "Detector",
    "FreeCode",
    "GraphAlgorithm",
    "InitCode",
    "LogicGadget",
    "LogicalOperator",
    "LogicalOperatorUpdate",
    "LogicalQubit",
    "Measurement",
    "MeasurementRecord",
    "NaiveLogicalMeasurement",
    "NaiveReadout",
    "NoiseApplicationMode",
    "NoiseInstruction",
    "NoiseModel",
    "NoiseSpecification",
    "NullCode",
    "Observable",
    "PauliChar",
    "PauliEigenState",
    "PauliString",
    "QECCompiler",
    "QECGadget",
    "QuantumMemory",
    "RSCSurgery",
    "RSCSyndromeExtraction",
    "ReadoutCode",
    "RotatedSurfaceCode",
    "SimpleGateApplication",
    "SimpleSyndromeExtraction",
    "StabilizerCode",
    "StimLikeNoiseModel",
    "SurfaceCode",
    "TannerEdge",
    "TannerGraph",
    "TannerNode",
    "ToricCode",
    "VariableNode",
    "__version__",
]
