from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple
import warnings

from debug.warnings import CodeBelowDistanceWarning

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
        gadget_tag: str,
    ) -> None:
        """Render a Tanner-graph view for a compiled primitive within a gadget.

        Args:
            path: Output path for the generated visualization.
            primitive: Primitive instruction being visualized.
            resolved_targets: Concrete gadget targets resolved from context.
            observables: Observables emitted by the gadget, if any.
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
        TannerGraphVisualizer.visualize(
            primitive_target,
            highlight_nodes=[
                (set(nodes), color, label) for nodes, color, label in highlights
            ],
            system_labels=systems,
            output_path=path,
            title=f"Gadget: {gadget_tag}, Primitive: {primitive.__class__.__name__}-{primitive.tag}",
        )

    def compile(
        self, program: List[QECGadget], visual_output_path: str | Path | None = None
    ) -> CompiledExperiment:
        """Compile a QEC program into a concrete experiment description.

        Args:
            program: Ordered gadget sequence to compile.
            visual_output_path: Optional path used to emit per-primitive visualizations.

        Returns:
            The compiled experiment containing circuit instructions, detectors, and
            observables.
        """
        for gadget in program:
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
                    continue
                case FreeCode():
                    self.ctx.unregister_code(gadget.code_varname)
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
            for p_op in primitive_code_instructions:
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

                if visual_output_path is not None:
                    self._visualize_util(
                        visual_output_path,
                        p_op,
                        resolved_targets,
                        observables,
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

        self.ctx.assert_no_allocated_codes()
        return self.ctx.to_compiled_experiment()
