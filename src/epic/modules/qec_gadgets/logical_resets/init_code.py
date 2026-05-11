from typing import Dict, List, Set, Tuple, cast
from uuid import UUID

from pydantic import Field

from etic.core.compilation.measurement_record import MeasurementRecordView
from etic.core.data_structure import PauliEigenState, TannerNode
from etic.core.language import CodeGadget
from etic.core.qec_object import LogicalOperatorUpdate, Observable
from etic.core.qec_object.stabilizer_code import StabilizerCode
from etic.core.qec_primitives.interfaces import ApplyGate, QECPrimitive
from etic.core.qec_primitives.interfaces.extract_syndrome import ExtractSyndrome


class InitCode(CodeGadget):
    """
    A gadget representing the initialization of a logical qubit in a stabilizer code.
    """

    initial_state: PauliEigenState = Field(
        description="The initial state of the code. "
    )

    def compile(
        self,
        resolved_targets: List[StabilizerCode],
        record: MeasurementRecordView,
        timestep: int,
        objective_distance: int,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:
        gates = []
        match self.initial_state:
            case PauliEigenState.X_plus:
                gates = ["RX"]
            case PauliEigenState.X_minus:
                gates = ["RX", "Z"]
            case PauliEigenState.Z_plus:
                gates = ["RZ"]
            case PauliEigenState.Z_minus:
                gates = ["RZ", "X"]
            case _:
                raise ValueError(
                    "Unsupported initial eigenstate, only X and Z basis are allowed"
                )
        primitives: List[QECPrimitive] = []
        for code in resolved_targets:
            target_nodes = cast(
                Set[TannerNode],
                code.tanner_graph.variable_nodes | code.tanner_graph.check_nodes,
            )
            primitives.append(
                ApplyGate(
                    target=code.tanner_graph,
                    target_nodes=target_nodes,
                    gates=gates,
                )
            )
            primitives.append(
                ExtractSyndrome(
                    target=code.tanner_graph,
                    distance=objective_distance,
                    rounds=objective_distance,
                )
            )

        return {}, [], primitives
