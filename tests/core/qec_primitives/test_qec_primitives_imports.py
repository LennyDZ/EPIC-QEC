from epic.core.qec_primitives import PrimitiveCompiler, PrimitiveRegistry
from epic.core.qec_primitives.interfaces import (
    ApplyGate,
    CustomPrimitive,
    ExtractSyndrome,
    PrimitiveImplementation,
    QECPrimitive,
    QECProcedure,
    Readout,
)


def test_qec_primitives_exports_are_importable() -> None:
    assert PrimitiveCompiler is not None
    assert PrimitiveRegistry is not None
    assert ApplyGate is not None
    assert CustomPrimitive is not None
    assert ExtractSyndrome is not None
    assert PrimitiveImplementation is not None
    assert QECPrimitive is not None
    assert QECProcedure is not None
    assert Readout is not None
