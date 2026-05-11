from types import MappingProxyType
from typing import Dict, List, Tuple
from uuid import UUID

from epic.core.qec_object.detector import NodeKnowledge
from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure import TannerNode
from epic.core.qec_object import Detector, Measurement
from epic.core.qec_object.detector import DetectorGraphPort
from epic.core.qec_primitives.interfaces import PrimitiveImplementation, QECProcedure


class EmptyProcedure(PrimitiveImplementation[QECProcedure]):
    def compile(
        self,
        instruction: QECProcedure,
        memory,
        record: MeasurementRecordView,
        det_graph_port: DetectorGraphPort,
        parent_gadget_id: UUID,
    ) -> Tuple[List[str], List[Measurement], List[Detector], DetectorGraphPort]:
        return [], [], [], {}
