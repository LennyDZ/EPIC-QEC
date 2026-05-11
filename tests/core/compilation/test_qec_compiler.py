"""Tests for qec_compiler module."""

from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import Field

from epic.core.compilation.qec_compiler import QECCompiler
from epic.core.data_structure.tanner_graph import TannerGraph
from epic.core.qec_object.detector import NodeKnowledge, QubitPortState
from epic.core.qec_object.logical_operator import LogicalOperatorUpdate
from epic.core.qec_object.measurement import Measurement
from epic.core.qec_object.observable import Observable
from epic.core.qec_object.stabilizer_code import StabilizerCode
from epic.core.language.qec_gadget import CodeGadget, LogicGadget
from epic.core.qec_primitives.interfaces.qec_primitive import QECPrimitive
from debug.warnings import CodeBelowDistanceWarning


class DummyPrimitive(QECPrimitive):
    pass


class DummyCodeGadget(CodeGadget):
    primitive: DummyPrimitive
    emitted_observables: list[Observable] = Field(default_factory=list)
    updates: dict[UUID, LogicalOperatorUpdate] = Field(default_factory=dict)
    call_log: dict[str, Any] = Field(default_factory=dict)

    def compile(self, resolved_targets, record, timestep, objective_distance):
        self.call_log["resolved_targets"] = resolved_targets
        self.call_log["record_size"] = len(record.measurements())
        self.call_log["timestep"] = timestep
        self.call_log["objective_distance"] = objective_distance
        primitives = cast(list[QECPrimitive], [self.primitive])
        return self.updates, self.emitted_observables, primitives


class DummyLogicGadget(LogicGadget):
    primitive: DummyPrimitive
    emitted_observables: list[Observable] = Field(default_factory=list)
    updates: dict[UUID, LogicalOperatorUpdate] = Field(default_factory=dict)
    call_log: dict[str, Any] = Field(default_factory=dict)

    def compile(self, resolved_targets, record, timestep, objective_distance):
        self.call_log["resolved_targets"] = resolved_targets
        self.call_log["record_size"] = len(record.measurements())
        self.call_log["timestep"] = timestep
        self.call_log["objective_distance"] = objective_distance
        primitives = cast(list[QECPrimitive], [self.primitive])
        return self.updates, self.emitted_observables, primitives


class TestQECCompiler:
    def test_compile_runs_code_gadget_and_collects_outputs(
        self,
        qec_compiler: QECCompiler,
        monkeypatch,
        alloc_code_gadget,
        free_code_gadget,
        stabilizer_code: StabilizerCode,
        logical_qubit,
        measurement_a: Measurement,
        measurement_b: Measurement,
        detector,
        observable_a: Observable,
        variable_node,
    ) -> None:
        primitive = DummyPrimitive(target=stabilizer_code.tanner_graph, tag="p0")
        gadget = DummyCodeGadget(
            tag="code-step",
            targets=["code0"],
            primitive=primitive,
            emitted_observables=[observable_a],
            updates={
                logical_qubit.logical_x.id: LogicalOperatorUpdate(
                    new_correction={measurement_b},
                    new_correction_mode="overwrite",
                )
            },
        )

        def fake_compile(
            self,
            primitive_instruction,
            memory,
            record,
            det_graph_port,
            parent_gadget_id,
        ):
            assert primitive_instruction == primitive
            assert parent_gadget_id == gadget.id
            return (
                ["H 0", "M 0"],
                [measurement_a],
                [detector],
                {
                    variable_node: QubitPortState(
                        knowledge=NodeKnowledge.MZ,
                        connected_nodes={variable_node},
                    )
                },
            )

        monkeypatch.setattr(
            type(qec_compiler.primitive_compiler), "compile", fake_compile
        )

        compiled = qec_compiler.compile([alloc_code_gadget, gadget, free_code_gadget])

        assert gadget.call_log["resolved_targets"] == [stabilizer_code]
        assert gadget.call_log["record_size"] == 0
        assert gadget.call_log["timestep"] == 0
        assert gadget.call_log["objective_distance"] == 1
        assert compiled.record.view().measurements() == (measurement_a,)
        assert compiled.circuit_instructions == ["H 0", "M 0"]
        assert compiled.detectors == [detector]
        assert len(compiled.observables) == 1
        assert compiled.observables[0].tag == "obs_a"
        assert compiled.observables[0].measurements == {measurement_a}
        assert logical_qubit.logical_x.frame_correction_history == {measurement_b}
        assert (
            qec_compiler.ctx.detector_graph_port_view()[variable_node].knowledge
            == NodeKnowledge.MZ
        )

    def test_compile_runs_logic_gadget_and_visualization_hook(
        self,
        qec_compiler: QECCompiler,
        monkeypatch,
        alloc_code_gadget,
        free_code_gadget,
        stabilizer_code: StabilizerCode,
        logical_qubit,
        measurement_a: Measurement,
        observable_a: Observable,
    ) -> None:
        primitive = DummyPrimitive(target=stabilizer_code.tanner_graph, tag="p1")
        gadget = DummyLogicGadget(
            tag="logic-step",
            targets=[logical_qubit.name],
            primitive=primitive,
            emitted_observables=[observable_a],
        )
        visualized: dict[str, Any] = {}

        def fake_compile(
            self,
            primitive_instruction,
            memory,
            record,
            det_graph_port,
            parent_gadget_id,
        ):
            return (["M 0"], [measurement_a], [], {})

        def fake_visualize(
            path, primitive_arg, resolved_targets, observables, gadget_tag
        ):
            visualized["path"] = path
            visualized["primitive"] = primitive_arg
            visualized["resolved_targets"] = resolved_targets
            visualized["observables"] = observables
            visualized["gadget_tag"] = gadget_tag

        monkeypatch.setattr(
            type(qec_compiler.primitive_compiler), "compile", fake_compile
        )
        monkeypatch.setattr(qec_compiler, "_visualize_util", fake_visualize)

        compiled = qec_compiler.compile(
            [alloc_code_gadget, gadget, free_code_gadget],
            visual_output_path=Path("visuals/out.svg"),
        )

        assert gadget.call_log["resolved_targets"] == [(logical_qubit, stabilizer_code)]
        assert compiled.circuit_instructions == ["M 0"]
        assert visualized["path"] == Path("visuals/out.svg")
        assert visualized["primitive"] == primitive
        assert visualized["resolved_targets"] == [(logical_qubit, stabilizer_code)]
        assert visualized["observables"] == [observable_a]
        assert visualized["gadget_tag"] == "logic-step"

    def test_compile_warns_when_allocating_code_below_objective_distance(
        self,
        compiler_config: dict,
        alloc_code_gadget,
        free_code_gadget,
    ) -> None:
        compiler = QECCompiler(config={**compiler_config, "objective_distance": 2})

        with pytest.warns(
            CodeBelowDistanceWarning,
            match="Code distance 1 is below objective distance 2",
        ):
            compiled = compiler.compile([alloc_code_gadget, free_code_gadget])

        assert compiled.record.view().measurements() == ()
        assert compiled.circuit_instructions == []
        assert compiled.detectors == []
        assert compiled.observables == []

    def test_compile_rejects_unsupported_gadget(
        self, qec_compiler: QECCompiler
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported gadget type"):
            qec_compiler.compile(cast(Any, [object()]))
