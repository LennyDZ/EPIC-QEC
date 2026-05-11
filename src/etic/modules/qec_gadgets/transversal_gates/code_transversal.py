from uuid import UUID

from pydantic import Field, field_validator
from typing import Dict, List, Set, Tuple

from etic.core.data_structure import TannerGraph
from etic.core.compilation.measurement_record import MeasurementRecordView
from etic.core.language import CodeGadget
from etic.core.qec_object import LogicalOperatorUpdate, Observable, StabilizerCode
from etic.core.qec_primitives.interfaces import ApplyGate, QECPrimitive, QECProcedure


class TransversalCNOT(CodeGadget):
    """A gadget applying transversal CNOT between all the qubits of two given codes."""

    tag: str = Field(default="transversal_cnot", init=False)
    gates: List[str] = Field(default_factory=lambda: ["CNOT"], init=False)
