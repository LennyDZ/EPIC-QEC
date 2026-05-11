"""Tests for compilation_context module."""

from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

import pytest

from epic.core.compilation.compilation_context import CompilationContext
from epic.core.qec_object.detector import NodeKnowledge, QubitPortState
from epic.core.qec_object.measurement import Measurement
from epic.core.qec_object.observable import Observable
from epic.core.qec_object.stabilizer_code import StabilizerCode


@pytest.fixture
def stabilizer_code(empty_check_tanner_graph, logical_qubit) -> StabilizerCode:
    return StabilizerCode(
        name="toy",
        n=1,
        k=1,
        d=1,
        tanner_graph=empty_check_tanner_graph,
        logical_qubits=[logical_qubit],
    )


@pytest.fixture
def registered_context(
    stabilizer_code: StabilizerCode,
    logical_qubit,
) -> CompilationContext:
    ctx = CompilationContext()
    ctx.register_code([logical_qubit.name], stabilizer_code, "code0")
    return ctx


class TestCompilationContext:
    def test_register_code_populates_lookup_and_resolution_maps(
        self,
        registered_context: CompilationContext,
        stabilizer_code: StabilizerCode,
        logical_qubit,
    ) -> None:
        resolved_qubits = registered_context.resolve_targets_varname(
            [logical_qubit.name], type(logical_qubit)
        )
        resolved_codes = registered_context.resolve_targets_varname(
            ["code0"], type(stabilizer_code)
        )

        assert registered_context.read_variable("code0") == stabilizer_code
        assert registered_context.read_variable(logical_qubit.name) == logical_qubit
        assert registered_context.allocated_code_varnames() == ["code0"]
        assert resolved_qubits == [(logical_qubit, stabilizer_code)]
        assert resolved_codes == [stabilizer_code]
        assert registered_context.lqb_from_op(logical_qubit.logical_x) == logical_qubit
        assert registered_context.code_from_lqb(logical_qubit) == stabilizer_code
        assert (
            registered_context.get_logical_op_by_id(logical_qubit.logical_z.id)
            == logical_qubit.logical_z
        )

    def test_unregister_code_removes_registered_objects(
        self,
        registered_context: CompilationContext,
        logical_qubit,
    ) -> None:
        with pytest.raises(ValueError, match="allocated codes still registered"):
            registered_context.assert_no_allocated_codes()

        registered_context.unregister_code("code0")

        assert registered_context.allocated_code_varnames() == []
        registered_context.assert_no_allocated_codes()
        with pytest.raises(KeyError, match="Variable name code0 not found"):
            registered_context.read_variable("code0")
        with pytest.raises(KeyError, match="Variable name L0 not found"):
            registered_context.read_variable(logical_qubit.name)

    def test_resolve_targets_varname_rejects_missing_or_wrong_type(
        self,
        registered_context: CompilationContext,
        logical_qubit,
        stabilizer_code: StabilizerCode,
    ) -> None:
        with pytest.raises(KeyError, match="Target variable name missing not found"):
            registered_context.resolve_targets_varname(["missing"], type(logical_qubit))

        with pytest.raises(TypeError, match="is not a valid target type"):
            registered_context.resolve_targets_varname(["code0"], type(logical_qubit))

        with pytest.raises(TypeError, match="is not a valid target type"):
            registered_context.resolve_targets_varname(
                [logical_qubit.name],
                type(stabilizer_code),
            )

    def test_timestep_counters_advance_as_expected(self) -> None:
        ctx = CompilationContext()

        ctx.incr_primitive_timestep()
        ctx.incr_primitive_timestep()
        ctx.incr_gadget_timestep()

        assert ctx.t_gadget == 1
        assert ctx.t_primitive == 0

    def test_resolve_observable_rebinds_measurements_and_applies_frame_correction(
        self,
        registered_context: CompilationContext,
        logical_qubit,
        variable_node,
    ) -> None:
        primitive_a = uuid4()
        primitive_b = uuid4()
        gadget_a = uuid4()
        gadget_b = uuid4()

        recorded_measurement = Measurement(
            node_id=variable_node.id,
            parent_gadget_id=gadget_a,
            parent_primitive_id=primitive_a,
            tag="recorded",
        )
        recorded_correction = Measurement(
            node_id=variable_node.id,
            parent_gadget_id=gadget_b,
            parent_primitive_id=primitive_b,
            tag="correction",
        )
        registered_context.measurement_record.add_measurement(
            [recorded_measurement, recorded_correction]
        )

        unresolved_measurement = Measurement(
            node_id=variable_node.id,
            parent_gadget_id=uuid4(),
            parent_primitive_id=primitive_a,
            tag="recorded",
        )
        unresolved_correction = Measurement(
            node_id=variable_node.id,
            parent_gadget_id=uuid4(),
            parent_primitive_id=primitive_b,
            tag="correction",
        )
        logical_qubit.logical_x.frame_correction_history = {unresolved_correction}

        resolved = registered_context.resolve_observable(
            Observable(
                logical_operators_involved=[logical_qubit.logical_x],
                measurements={unresolved_measurement},
                tag="obs0",
            )
        )

        assert resolved.measurements == {recorded_measurement, recorded_correction}

    def test_add_observable_warns_on_duplicate_tag_and_latest_is_returned(
        self,
        registered_context: CompilationContext,
    ) -> None:
        first = Observable(tag="obs")
        second = Observable(tag="obs")

        registered_context.add_observable(first)
        with pytest.warns(UserWarning, match="Observable tag obs already exists"):
            registered_context.add_observable(second)

        assert registered_context.get_observable_by_id(first.id) == first
        assert registered_context.get_observable_by_tag("obs") == second

    def test_detector_graph_port_view_is_read_only_and_reflects_updates(
        self,
        registered_context: CompilationContext,
        variable_node,
    ) -> None:
        state = QubitPortState(
            knowledge=NodeKnowledge.MX,
            connected_nodes={variable_node},
        )
        registered_context.update_dg_port(variable_node, state)

        view = registered_context.detector_graph_port_view()
        unsafe_view = cast(Any, view)

        assert isinstance(view, MappingProxyType)
        assert view[variable_node] == state
        with pytest.raises(TypeError):
            unsafe_view[variable_node] = QubitPortState(knowledge=NodeKnowledge.UNKNOWN)

    def test_to_compiled_experiment_collects_recorded_outputs(
        self,
        registered_context: CompilationContext,
        measurement_a,
        detector,
    ) -> None:
        observable = Observable(tag="obs")
        registered_context.measurement_record.add_measurement(measurement_a)
        registered_context.add_detector(detector)
        registered_context.add_circuit_instruction(["H 0", "M 0"])
        registered_context.add_observable(observable)

        compiled = registered_context.to_compiled_experiment()

        assert compiled.record is registered_context.measurement_record
        assert compiled.circuit_instructions == ["H 0", "M 0"]
        assert compiled.detectors == [detector]
        assert compiled.observables == [observable]

    def test_resolve_observable_raises_when_measurement_missing(
        self,
        registered_context: CompilationContext,
        measurement_a,
    ) -> None:
        with pytest.raises(ValueError, match="not found in measurement record"):
            registered_context.resolve_observable(
                Observable(measurements={measurement_a})
            )
