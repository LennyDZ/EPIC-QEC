"""Tests for measurement, detector and observable modules."""

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from core.qec_object.detector import NodeKnowledge
from core.qec_object.observable import Observable


class TestMeasurement:
    def test_measurement_fields(self, measurement_a, variable_node) -> None:
        assert measurement_a.node_id == variable_node.id
        assert isinstance(measurement_a.parent_gadget_id, UUID)
        assert isinstance(measurement_a.parent_primitive_id, UUID)
        assert measurement_a.tag == "m_a"
        assert measurement_a.id is not None

    def test_measurements_have_distinct_ids(self, measurement_a, measurement_b) -> None:
        assert measurement_a.id != measurement_b.id


class TestDetector:
    def test_detector_initialization(self, detector, measurement_a) -> None:
        assert detector.measurements == [measurement_a]
        assert detector.coordinates == (1, 2)
        assert detector.id is not None

    def test_detector_is_frozen(self, detector) -> None:
        with pytest.raises(FrozenInstanceError):
            detector.coordinates = (9, 9)


class TestObservable:
    def test_apply_corrective_frame_xors_measurements(
        self, observable_with_ops, measurement_a, measurement_b
    ) -> None:
        observable = observable_with_ops.apply_corrective_frame()
        assert observable.measurements == {measurement_a, measurement_b}


def test_observable_defaults() -> None:
    obs = Observable()
    assert obs.logical_operators_involved == []
    assert obs.measurements == set()
    assert obs.tag == ""
