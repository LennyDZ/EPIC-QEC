"""Tests for logical_operator module."""

from core.data_structure.pauli import PauliChar, PauliString
from core.data_structure.tanner_node import VariableNode
from core.qec_object.logical_operator import LogicalOperator, LogicalOperatorUpdate


class TestLogicalOperator:
    """Test suite for LogicalOperator class."""

    def test_initialization(self, variable_node: VariableNode):
        op = LogicalOperator(
            logical_type=PauliChar.X,
            operator=PauliString(string=(PauliChar.X,)),
            target_nodes=(variable_node,),
        )

        assert op.logical_type == PauliChar.X
        assert op.operator.string == (PauliChar.X,)
        assert op.target_nodes == (variable_node,)
        assert op.id is not None

    def test_update_new_physical_support(self, variable_node: VariableNode):
        op = LogicalOperator(
            logical_type=PauliChar.X,
            operator=PauliString(string=(PauliChar.X,)),
            target_nodes=(variable_node,),
        )
        new_node = VariableNode(tag="v1")
        update = LogicalOperatorUpdate(
            new_physical_support={(PauliChar.Z, new_node)},
        )

        op.update(update)

        assert op.operator.string == (PauliChar.Z,)
        assert op.target_nodes == (new_node,)

    def test_update_append_correction_xors(
        self,
        logical_x: LogicalOperator,
        measurement_a,
        measurement_b,
    ):
        logical_x.frame_correction_history = {measurement_a}
        update = LogicalOperatorUpdate(
            new_correction={measurement_a, measurement_b},
            new_correction_mode="append",
        )

        logical_x.update(update)

        assert logical_x.frame_correction_history == {measurement_b}

    def test_update_overwrite_replaces_history(
        self,
        logical_x: LogicalOperator,
        measurement_a,
        measurement_b,
    ):
        logical_x.frame_correction_history = {measurement_a}
        update = LogicalOperatorUpdate(
            new_correction={measurement_b},
            new_correction_mode="overwrite",
        )

        logical_x.update(update)

        assert logical_x.frame_correction_history == {measurement_b}


class TestLogicalOperatorProperties:
    def test_instances_have_distinct_ids(self, logical_x: LogicalOperator):
        other = LogicalOperator(
            logical_type=logical_x.logical_type,
            operator=logical_x.operator,
            target_nodes=logical_x.target_nodes,
        )

        assert logical_x.id != other.id

    def test_model_equality_via_model_validate(self, logical_z: LogicalOperator):
        cloned = LogicalOperator.model_validate(logical_z.model_dump())
        assert logical_z == cloned

    def test_update_default_mode_is_append(self):
        update = LogicalOperatorUpdate()
        assert update.new_correction_mode == "append"
