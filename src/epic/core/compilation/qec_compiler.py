from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TypeVar
from uuid import UUID
import warnings

from debug.warnings import CodeBelowDistanceWarning
from epic.core.data_structure.physical_qubit import PhysicalQubit
from epic.core.data_structure.quantum_program import QuantumProgram
from epic.core.qec_object.logical_operator import LogicalOperatorUpdate
from epic.core.qec_primitives import PrimitiveCompiler

from .compiled_experiment import CompiledExperiment
from .compilation_context import CompilationContext
from ..qec_object import LogicalQubit, StabilizerCode, Observable
from ..language import QECGadget, AllocCode, CodeGadget, FreeCode, LogicGadget
from ..qec_primitives.interfaces import QECPrimitive

from epic.core.visualization.tanner_graph_vis import TannerGraphVisualizer


class QECCompiler:
    """Compile high-level QEC gadgets into a compiled experiment."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the compiler and its compilation context.

        Args:
            config: Compiler configuration, including primitive implementation mapping
                and target distance requirements.
        """
        self.config = config
        self.distance = self.config.get("objective_distance", 0)
        self.quantum_memory_limit = self.config.get("physical_qubits_limit", -1)
        self.ctx = CompilationContext(memory_size=self.quantum_memory_limit)
        self.primitive_compiler = PrimitiveCompiler(config=config)

    def _visualize_util():
        pass

    def compile(
        self,
        program: QuantumProgram,
        visual_output_path: str | Path | None = None,
        show_progress: bool = False,
    ) -> CompiledExperiment:
        """Compile a QEC program into a concrete experiment description.

        Args:
            program: Ordered gadget sequence to compile.
            visual_output_path: Optional path used to emit per-primitive visualizations.
            show_progress: If true, print progress and the current gadget/primitive.

        Returns:
            The compiled experiment containing circuit instructions, detectors, and
            observables.
        """
        CompiledEffect: type = tuple[
            dict[UUID, LogicalOperatorUpdate],
            list[Observable],
            list[PhysicalQubit],
        ]

        gadgets_effect_buffer: Dict[UUID, CompiledEffect] = {}  # type: ignore

        timings_starts, timings_ends = program.operation_in_time()
        for timestep in range(program.depth+1):
            operations_start = timings_starts.get(timestep, [])
            operations_end = timings_ends.get(timestep, [])

            def _validate_operation_content(operation) -> List[QECGadget]:
                if operation.implementation is None:
                    return []
                if isinstance(operation.implementation, QECGadget):
                    return [operation.implementation]
                elif isinstance(operation.implementation, list) and all(
                    isinstance(g, QECGadget) for g in operation.implementation
                ):
                    return operation.implementation
                else:
                    raise ValueError(
                        f"Unsupported operation implementation type: {type(operation.implementation)}"
                    )

            for operation in operations_end:
                
                gadgets = _validate_operation_content(operation)

                for gadget in gadgets:
                    if isinstance(gadget, (AllocCode, FreeCode)):
                        continue

                    gadget_compiled_effect = gadgets_effect_buffer.pop(
                        gadget.id, (None, None, None)
                    )
                    lop_updates, observables, physical_qubits_used = (
                        gadget_compiled_effect
                    )

                    if observables is not None:
                        for obs in observables:
                            resolved_obs = self.ctx.resolve_observable(obs)
                            self.ctx.add_observable(resolved_obs)

                    if lop_updates:
                        for op, update in lop_updates.items():
                            self.ctx.get_logical_op_by_id(op).update(update)

                    self.ctx.quantum_memory.unlock_qubits(
                        physical_qubits_used, owner_id=gadget.id
                    )

            for operation in operations_start:
                gadgets = _validate_operation_content(operation)

                for gadget in gadgets:
                    match gadget:
                        case AllocCode():
                            if gadget.target_code.d < self.distance:
                                warnings.warn(
                                    CodeBelowDistanceWarning(
                                        gadget.target_code.d, self.distance, gadget.tag
                                    )
                                )
                            pq_allocated = self.ctx.register_code(
                                gadget.logical_qubits_varnames,
                                gadget.target_code,
                                gadget.code_varname,
                            )

                            for q in pq_allocated:
                                self.ctx._output_program.add_qubit(q)

                            continue
                        case FreeCode():
                            self.ctx.unregister_code(gadget.code_varname)
                            continue
                        case CodeGadget():
                            resolved_targets = self.ctx.resolve_targets_varname(
                                gadget.targets, StabilizerCode
                            )
                            (
                                lop_updates,
                                observables,
                                primitive_code_instructions,
                                physical_qubits_used,
                            ) = gadget.compile(
                                resolved_targets,
                                self.ctx.measurement_record.view(),
                                self.ctx.quantum_memory,
                                self.ctx.t_gadget,
                                self.distance,
                            )
                            self.ctx.quantum_memory.lock_qubits(
                                physical_qubits_used, requestor_id=gadget.id
                            )
                        case LogicGadget():
                            resolved_targets = self.ctx.resolve_targets_varname(
                                gadget.targets, LogicalQubit
                            )
                            (
                                lop_updates,
                                observables,
                                primitive_code_instructions,
                                physical_qubits_used,
                            ) = gadget.compile(
                                resolved_targets,
                                self.ctx.measurement_record.view(),
                                self.ctx.quantum_memory,
                                self.ctx.t_gadget,
                                self.distance,
                            )
                            self.ctx.quantum_memory.lock_qubits(
                                physical_qubits_used, requestor_id=gadget.id
                            )
                        case _:
                            raise ValueError(f"Unsupported gadget type: {type(gadget)}")

                    measurement_in_gadget = []
                    gadget_primitive_length = len(primitive_code_instructions)
                    for primitive_index, p_op in enumerate(
                        primitive_code_instructions, start=1
                    ):
                        instruction, measurements, detectors, new_dg_port = (
                            self.primitive_compiler.compile(
                                p_op,
                                self.ctx.measurement_record.view(),
                                self.ctx.detector_graph_port_view(),
                                parent_gadget_id=gadget.id,
                            )
                        )

                        self.ctx.measurement_record.add_measurement(measurements)
                        measurement_in_gadget.extend(measurements)
                        for detector in detectors:
                            self.ctx.add_detector(detector)
                        for k in new_dg_port.keys():
                            self.ctx.update_dg_port(k, new_dg_port[k])
                        self.ctx.add_circuit_operation(instruction)

                    gadgets_effect_buffer[gadget.id] = (
                        lop_updates,
                        observables,
                        physical_qubits_used,
                    )

        return self.ctx.to_compiled_experiment()
