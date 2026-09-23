from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID
import warnings

from debug.warnings import CodeBelowDistanceWarning
from epic.core.data_structure.quantum_program import ProgramQubit, QProgOperation
from epic.core.visualization.quantum_program_vis import QuantumProgramVisualizer

from ..qec_object import LogicalQubit, StabilizerCode, Observable, LogicalOperatorUpdate
from ..language import QECGadget, AllocCode, CodeGadget, FreeCode, LogicGadget
from ..data_structure import PhysicalQubit, QuantumProgram
from ..qec_primitives import PrimitiveCompiler

from .compiled_experiment import CompiledExperiment
from .compilation_context import CompilationContext



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

    def _visualize_util(self):
        pass

    @staticmethod
    def _append_log(log_path: Path, markdown: str) -> None:
        """Append a Markdown section to a compilation report."""
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{markdown.rstrip()}\n\n")

    @staticmethod
    def _display_tag(tag: str) -> str:
        """Return a readable label for an optional gadget or primitive tag."""
        return tag or "untagged"

    @staticmethod
    def parse_to_program(gadgets: List[QECGadget]) -> QuantumProgram:
        program = QuantumProgram(name="Parsed_QEC_Program")
        qubits: dict[str, ProgramQubit] = {}
        logical_in_patches: dict[str, list[ProgramQubit]] = {}
        for gadget in gadgets:
            if isinstance(gadget, AllocCode):
                logical_in_patches[gadget.code_varname] = []
                for q in gadget.logical_qubits_varnames:
                    qubits[q] = ProgramQubit(name=q)
                    program.add_qubit(qubits[q])
                    logical_in_patches[gadget.code_varname].append(qubits[q])
                program.add_operation(QProgOperation(
                    name="AllocCode",
                    length=1,
                    targets=logical_in_patches[gadget.code_varname],
                    implementation=gadget
                ))
            elif isinstance(gadget, LogicGadget):
                program.add_operation(QProgOperation(
                    name=gadget.tag,
                    length=1,
                    targets=[qubits[q] for q in gadget.targets],
                    implementation=gadget
                ))
            elif isinstance(gadget, CodeGadget):
                for target in gadget.targets:
                    program.add_operation(QProgOperation(
                        name=gadget.tag,
                        length=1,
                        targets=logical_in_patches[target],
                        implementation=gadget
                    ))
            elif isinstance(gadget, FreeCode):
                program.add_operation(QProgOperation(
                    name=gadget.tag,
                    length=1,
                    targets=logical_in_patches[gadget.code_varname],
                    implementation=gadget,
                ))
        return program

    def compile(
        self,
        program: QuantumProgram | List[QECGadget],
        visual_output_path: str | Path | None = None,
        show_progress: bool = False,
        log_output_path: str | Path | None = None,
    ) -> CompiledExperiment:
        """Compile a QEC program into a concrete experiment description.

        Args:
            program: Ordered gadget sequence to compile.
            visual_output_path: Optional path used to emit per-primitive visualizations.
            show_progress: If true, print progress and the current gadget/primitive.
            log_output_path: Optional Markdown file that records compilation details
                and an image of the input program.

        Returns:
            The compiled experiment containing circuit instructions, detectors, and
            observables.
        """
        CompiledEffect: type = tuple[
            dict[UUID, LogicalOperatorUpdate],
            list[Observable],
            list[PhysicalQubit],
        ]

        log_path = Path(log_output_path) if log_output_path is not None else None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("# Compilation Report\n\n", encoding="utf-8")

        input_was_gadget_list = isinstance(program, list)
        if input_was_gadget_list and all(isinstance(g, QECGadget) for g in program):
            program = self.parse_to_program(program)
            if log_path is not None:
                self._append_log(
                    log_path,
                    "## Input\n\n"
                    "- Source: list of `QECGadget` objects\n"
                    "- Parsing: succeeded",
                )
        elif log_path is not None:
            self._append_log(
                log_path,
                "## Input\n\n"
                "- Source: `QuantumProgram`\n"
                "- Parsing: not required",
            )

        if log_path is not None:
            image_path = log_path.parent / f"{log_path.stem}_input_program.png"
            QuantumProgramVisualizer.visualize(
                program,
                output_path=image_path,
                title="Input QuantumProgram",
            )
            self._append_log(
                log_path,
                "### Input QuantumProgram\n\n"
                f"![Input QuantumProgram]({image_path.name})\n\n"
                f"- Qubits: {len(program.qubits)}\n"
                f"- Operations: {len(program.operations)}\n"
                f"- Depth: {program.depth}",
            )

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
                    if log_path is not None:
                        self._append_log(
                            log_path,
                            f"### Gadget: {self._display_tag(gadget.tag)}\n\n"
                            f"- Type: `{type(gadget).__name__}`",
                        )
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

                            if log_path is not None:
                                self._append_log(log_path, "- Primitives: none")
                            continue
                        case FreeCode():
                            self.ctx.unregister_code(gadget.code_varname)
                            if log_path is not None:
                                self._append_log(log_path, "- Primitives: none")
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
                    if log_path is not None:
                        primitive_list = "\n".join(
                            f"{primitive_index}. `{self._display_tag(p_op.tag)}` "
                            f"(`{type(p_op).__name__}`)"
                            for primitive_index, p_op in enumerate(
                                primitive_code_instructions, start=1
                            )
                        )
                        self._append_log(
                            log_path,
                            f"- Primitives:\n{primitive_list or '  - none'}",
                        )
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

        compiled_experiment = self.ctx.to_compiled_experiment()
        if log_path is not None:
            observable_tags = "\n".join(
                f"  - `{self._display_tag(observable.tag)}`"
                for observable in compiled_experiment.observables
            ) or "  - none"
            self._append_log(
                log_path,
                "## Output Summary\n\n"
                f"- Instructions: {len(compiled_experiment.program.operations)}\n"
                f"- Physical qubits: {len(compiled_experiment.program.qubits)}\n"
                f"- Measurements: {len(compiled_experiment.record.view().measurements())}\n"
                f"- Detectors: {len(compiled_experiment.detectors)}\n"
                f"- Observables: {len(compiled_experiment.observables)}\n"
                "- Observable tags:\n"
                + observable_tags,
            )

        return compiled_experiment
