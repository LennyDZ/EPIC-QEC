from typing import Dict, List, Tuple
from uuid import UUID

from pydantic import model_validator

from epic.core.compilation.quantum_memory import QuantumMemory
from epic.core.data_structure import PauliChar, TannerGraph
from epic.core.language import LogicGadget
from epic.core.qec_object import LogicalOperatorUpdate, Measurement, Observable
from epic.core.qec_object.logical_operator import LogicalOperator
from epic.core.qec_object.logical_qubit import LogicalQubit
from epic.core.qec_primitives.interfaces import QECPrimitive, Readout


class NaiveLogicalMeasurement(LogicGadget):

    basis: List[PauliChar] = [PauliChar.Z]
    free_qubits: bool = False

    @model_validator(mode="after")
    def validate_basis_len(self):
        if len(self.basis) != len(self.targets):
            raise ValueError(
                "NaiveLogicalMeasurement requires one measurement basis per target logical qubit. "
                f"Got {len(self.basis)} basis entries for {len(self.targets)} targets."
            )
        return self

    def compile(
        self,
        resolved_targets,
        record,
        quantum_memory: QuantumMemory,
        timestep,
        objective_distance,
    ) -> Tuple[Dict[UUID, LogicalOperatorUpdate], List[Observable], List[QECPrimitive]]:
        primitives = []
        lop_targets: List[LogicalOperator] = []
        for lop, basis in zip(resolved_targets, self.basis):
            match basis:
                case PauliChar.X:
                    lop_targets.append(lop[0].logical_x)
                case PauliChar.Z:
                    lop_targets.append(lop[0].logical_z)
                case _:
                    raise ValueError(
                        f"Unsupported measurement basis {basis} for logical operator measurement. Only X and Z are supported."
                    )

        for lop in lop_targets:
            target = TannerGraph(
                variable_nodes=set(lop.target_nodes),
                check_nodes=set(),
                edges=set(),
            )
            primitives.append(
                Readout(
                    target=target,
                    physical_data_qubits=quantum_memory.data_qubits_allocation_snapshot(
                        target.variable_nodes
                    ),
                    physical_ancilla_qubits={},  # no ancilla needed since no check in target nodes
                    readout_basis=lop.logical_type,
                    tag=f"measurement_{self.tag}_{lop.id}",
                )
            )
        observable = Observable(
            tag=f"measurement_{self.tag}",
            logical_operators_involved=list(lop_targets),
            measurements={
                Measurement(
                    n.id, parent_gadget_id=self.id, parent_primitive_id=primitives[i].id
                )
                for i, lop in enumerate(lop_targets)
                for n in lop.target_nodes
            },
        )

        return dict(), [observable], primitives
