from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from uuid import UUID
import warnings

from debug.warnings import CodeBelowDistanceWarning
from epic.core.qec_object.logical_operator import LogicalOperatorUpdate

from .compiled_experiment import CompiledExperiment
from .compilation_context import CompilationContext
from ..qec_object import LogicalQubit, StabilizerCode, Observable
from ..language import QECGadget, AllocCode, CodeGadget, FreeCode, LogicGadget
from ..qec_primitives.interfaces import QECPrimitive
from ..qec_primitives import PrimitiveCompiler

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
        # self.gadget_compiler = GadgetCompiler(config)

    def _visualize_util(
        self,
        path: str | Path,
        primitive: QECPrimitive,
        resolved_targets: Sequence[
            StabilizerCode | tuple[LogicalQubit, StabilizerCode]
        ],
        observables: List[Observable] | None,
        lop_updates: Dict[UUID, LogicalOperatorUpdate],
        gadget_tag: str,
    ) -> None:
        """Render a Tanner-graph view for a compiled primitive within a gadget.

        Args:
            path: Output path for the generated visualization.
            primitive: Primitive instruction being visualized.
            resolved_targets: Concrete gadget targets resolved from context.
            observables: Observables emitted by the gadget, if any.
            lop_updates: Updates to logical operators within the gadget.
            gadget_tag: Human-readable gadget label used in the visualization title.
        """
        highlights = []
        systems: Mapping[tuple[int, int], str] = {}
        primitive_target = primitive.target.model_copy(deep=True)
        resolved_targets_view = tuple(resolved_targets)
        observables_view = (
            tuple(obs.model_copy(deep=True) for obs in observables)
            if observables is not None
            else None
        )

        lop = []
        for t in resolved_targets_view:

            if isinstance(t, StabilizerCode):
                representative = next(iter(t.tanner_graph.variable_nodes), None)
                if (
                    representative is not None
                    and representative.coordinates is not None
                    and len(representative.coordinates) >= 4
                ):
                    systems[
                        (representative.coordinates[2], representative.coordinates[3])
                    ] = t.name
                for q in t.logical_qubits:
                    lop.append(q.logical_x)
                    lop.append(q.logical_z)
            elif (
                isinstance(t, Tuple)
                and len(t) == 2
                and isinstance(t[0], LogicalQubit)
                and isinstance(t[1], StabilizerCode)
            ):
                q, t = t
                lop.append(q.logical_x)
                lop.append(q.logical_z)
                representative = next(iter(t.tanner_graph.variable_nodes), None)
                if (
                    representative is not None
                    and representative.coordinates is not None
                    and len(representative.coordinates) >= 4
                ):
                    systems[
                        (representative.coordinates[2], representative.coordinates[3])
                    ] = t.name

        for op in lop:
            highlights.append(
                (
                    {t.id for t in op.target_nodes},
                    "auto",
                    f"logical op {op.logical_type}",
                )
            )
        if observables_view is not None:
            for obs in observables_view:
                highlights.append(
                    (
                        {t.node_id for t in obs.measurements},
                        "auto",
                        f"observable: {obs.tag}",
                    )
                )

        for lop_id, lop_update in lop_updates.items():
            for node in lop_update.new_correction:
                highlights.append(
                    (
                        {t.node_id for t in lop_update.new_correction},
                        "auto",
                        f"lop update",
                    )
                )

        TannerGraphVisualizer.visualize(
            primitive_target,
            highlight_nodes=[
                (set(nodes), color, label) for nodes, color, label in highlights
            ],
            system_labels=systems,
            output_path=path,
            title=f"Gadget: {gadget_tag}, Primitive: {primitive.__class__.__name__}-{primitive.tag}",
        )

    @staticmethod
    def _describe_gadget(gadget: QECGadget) -> str:
        """Return a short human-readable description for progress reporting."""
        gadget_name = gadget.__class__.__name__
        target_names = getattr(gadget, "targets", None)

        if isinstance(gadget, AllocCode):
            return (
                f"{gadget_name} '{gadget.tag}' allocating code '{gadget.code_varname}' "
                f"with target code '{gadget.target_code.name}'"
            )

        if isinstance(gadget, FreeCode):
            return f"{gadget_name} '{gadget.tag}' freeing code '{gadget.code_varname}'"

        if target_names:
            return f"{gadget_name} '{gadget.tag}' targeting {list(target_names)}"

        return f"{gadget_name} '{gadget.tag}'"

    @staticmethod
    def _describe_primitive(primitive: QECPrimitive) -> str:
        """Return a short human-readable description for progress reporting."""
        primitive_name = primitive.__class__.__name__
        primitive_tag = f" '{primitive.tag}'" if primitive.tag else ""
        return f"{primitive_name}{primitive_tag}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration for progress output."""
        return f"{seconds:.3f}s"

    def compile(
        self,
        program: List[QECGadget],
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
        total_gadgets = len(program)
        compile_start = perf_counter()

        if show_progress:
            print(f"Starting compilation of {total_gadgets} gadget(s)")

        for gadget_index, gadget in enumerate(program, start=1):
            gadget_start = perf_counter()
            if show_progress:
                print(
                    f"[{gadget_index}/{total_gadgets}] Compiling "
                    f"{self._describe_gadget(gadget)}"
                )

            match gadget:
                case AllocCode():
                    if gadget.target_code.d < self.distance:
                        warnings.warn(
                            CodeBelowDistanceWarning(
                                gadget.target_code.d, self.distance, gadget.tag
                            )
                        )
                    self.ctx.register_code(
                        gadget.logical_qubits_varnames,
                        gadget.target_code,
                        gadget.code_varname,
                    )
                    if show_progress:
                        elapsed = self._format_duration(perf_counter() - gadget_start)
                        print(
                            f"[{gadget_index}/{total_gadgets}] Registered code "
                            f"'{gadget.code_varname}' in {elapsed}"
                        )
                    continue
                case FreeCode():
                    self.ctx.unregister_code(gadget.code_varname)
                    if show_progress:
                        elapsed = self._format_duration(perf_counter() - gadget_start)
                        print(
                            f"[{gadget_index}/{total_gadgets}] Released code "
                            f"'{gadget.code_varname}' in {elapsed}"
                        )
                    continue
                case CodeGadget():
                    resolved_targets = self.ctx.resolve_targets_varname(
                        gadget.targets, StabilizerCode
                    )
                    lop_updates, observables, primitive_code_instructions = (
                        gadget.compile(
                            resolved_targets,
                            self.ctx.measurement_record.view(),
                            self.ctx.quantum_memory,
                            self.ctx.t_gadget,
                            self.distance,
                        )
                    )
                case LogicGadget():
                    resolved_targets = self.ctx.resolve_targets_varname(
                        gadget.targets, LogicalQubit
                    )
                    lop_updates, observables, primitive_code_instructions = (
                        gadget.compile(
                            resolved_targets,
                            self.ctx.measurement_record.view(),
                            self.ctx.quantum_memory,
                            self.ctx.t_gadget,
                            self.distance,
                        )
                    )
                case _:
                    raise ValueError(f"Unsupported gadget type: {type(gadget)}")

            gadget_measurements = []
            total_primitives = len(primitive_code_instructions)
            for primitive_index, p_op in enumerate(
                primitive_code_instructions, start=1
            ):
                if show_progress:
                    print(
                        f"[{gadget_index}/{total_gadgets}] Primitive "
                        f"{primitive_index}/{total_primitives}: "
                        f"{self._describe_primitive(p_op)}"
                    )
                primitive_start = perf_counter()
                c_instructions, measurements, detectors, new_dg_port = (
                    self.primitive_compiler.compile(
                        p_op,
                        self.ctx.measurement_record.view(),
                        self.ctx.detector_graph_port_view(),
                        parent_gadget_id=gadget.id,
                    )
                )
                self.ctx.measurement_record.add_measurement(measurements)
                gadget_measurements.extend(measurements)
                for detector in detectors:
                    self.ctx.add_detector(detector)
                for k in new_dg_port.keys():
                    self.ctx.update_dg_port(k, new_dg_port[k])
                for instruction in c_instructions:
                    self.ctx.add_circuit_instruction(instruction)
                self.ctx.incr_primitive_timestep()

                if show_progress:
                    primitive_elapsed = self._format_duration(
                        perf_counter() - primitive_start
                    )
                    print(
                        f"[{gadget_index}/{total_gadgets}] Primitive "
                        f"{primitive_index}/{total_primitives} complete in "
                        f"{primitive_elapsed}"
                    )

                if visual_output_path is not None:
                    self._visualize_util(
                        visual_output_path,
                        p_op,
                        resolved_targets,
                        observables,
                        lop_updates,
                        gadget.tag,
                    )

            # Should I apply the frame correction the Observable with or without the gadget's own logical correction ? ???
            if observables is not None:
                for obs in observables:
                    resolved_obs = self.ctx.resolve_observable(obs)
                    self.ctx.add_observable(resolved_obs)

            if lop_updates:
                for op, update in lop_updates.items():
                    self.ctx.get_logical_op_by_id(op).update(update)

            self.ctx.incr_gadget_timestep()

            if show_progress:
                gadget_elapsed = self._format_duration(perf_counter() - gadget_start)
                print(
                    f"[{gadget_index}/{total_gadgets}] Completed "
                    f"{self._describe_gadget(gadget)} in {gadget_elapsed}"
                )

        self.ctx.assert_no_allocated_codes()
        if show_progress:
            total_elapsed = self._format_duration(perf_counter() - compile_start)
            print(f"Compilation complete in {total_elapsed}")
        return self.ctx.to_compiled_experiment()
