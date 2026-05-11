from typing import Dict, List, Tuple
from uuid import UUID

from epic.core.compilation.measurement_record import MeasurementRecordView
from epic.core.data_structure.pauli import PauliChar
from epic.core.language.qec_gadget import CodeGadget
from epic.core.qec_object import LogicalOperatorUpdate, Observable
from epic.core.qec_object.measurement import Measurement
from epic.core.qec_object.stabilizer_code import StabilizerCode
from epic.core.qec_primitives.interfaces import QECPrimitive
from epic.core.qec_primitives.interfaces.readout import Readout


class ReadoutCode(CodeGadget):
    """Reads out a stabilizer code by measuring all data qubits in the Z basis and
    building the final detectors that close the detector graph."""

    def compile(
        self,
        resolved_targets: List[StabilizerCode],
        record: MeasurementRecordView,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:
        primitives: List[QECPrimitive] = []
        observables: List[Observable] = []
        for code in resolved_targets:
            ro = Readout(
                target=code.tanner_graph,
                readout_basis=PauliChar.Z,
                tag=f"readout_{code.name}",
            )
            primitives.append(ro)
            obs = Observable(
                tag=f"readout_{self.tag}",
                logical_operators_involved=[code.logical_qubits[0].logical_z],
                measurements={
                    Measurement(
                        n.id, parent_gadget_id=self.id, parent_primitive_id=ro.id
                    )
                    for n in code.logical_qubits[0].logical_z.target_nodes
                },
            )
            observables.append(obs)
        return {}, observables, primitives
