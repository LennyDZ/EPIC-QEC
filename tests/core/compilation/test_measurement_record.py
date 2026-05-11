from uuid import uuid4

import pytest

from core.compilation.measurement_record import MeasurementRecord
from core.qec_object.measurement import Measurement


@pytest.fixture
def indexed_measurements() -> tuple[Measurement, Measurement, Measurement]:
    node_a = uuid4()
    node_b = uuid4()
    gadget_shared = uuid4()
    primitive_shared = uuid4()

    m1 = Measurement(
        node_id=node_a,
        parent_gadget_id=gadget_shared,
        parent_primitive_id=primitive_shared,
        tag="m1",
    )
    m2 = Measurement(
        node_id=node_a,
        parent_gadget_id=uuid4(),
        parent_primitive_id=primitive_shared,
        tag="m2",
    )
    m3 = Measurement(
        node_id=node_b,
        parent_gadget_id=gadget_shared,
        parent_primitive_id=uuid4(),
        tag="m3",
    )

    return m1, m2, m3


def test_view_measurements_returns_immutable_ordered_sequence() -> None:
    record = MeasurementRecord()

    m1 = Measurement(
        node_id=uuid4(), parent_gadget_id=uuid4(), parent_primitive_id=uuid4(), tag="m1"
    )
    m2 = Measurement(
        node_id=uuid4(), parent_gadget_id=uuid4(), parent_primitive_id=uuid4(), tag="m2"
    )

    record.add_measurement([m1, m2])

    measurements_view = record.view().measurements()

    assert measurements_view == (m1, m2)
    assert isinstance(measurements_view, tuple)


def test_add_measurement_accepts_single_measurement(measurement_a) -> None:
    record = MeasurementRecord()

    record.add_measurement(measurement_a)

    assert record.view().measurements() == (measurement_a,)


def test_add_measurement_ignores_empty_iterable() -> None:
    record = MeasurementRecord()

    record.add_measurement([])

    assert record.view().measurements() == ()


def test_view_indexes_measurements_by_node_id(
    indexed_measurements: tuple[Measurement, Measurement, Measurement],
) -> None:
    record = MeasurementRecord()
    m1, m2, m3 = indexed_measurements

    record.add_measurement([m1, m2, m3])

    assert record.view().by_node_id(m1.node_id) == (m1, m2)
    assert record.view().by_node_id(m3.node_id) == (m3,)


def test_view_indexes_measurements_by_parent_gadget_id(
    indexed_measurements: tuple[Measurement, Measurement, Measurement],
) -> None:
    record = MeasurementRecord()
    m1, m2, m3 = indexed_measurements

    record.add_measurement([m1, m2, m3])

    assert record.view().by_parent_gadget_id(m1.parent_gadget_id) == (m1, m3)
    assert record.view().by_parent_gadget_id(m2.parent_gadget_id) == (m2,)


def test_view_indexes_measurements_by_parent_primitive_id(
    indexed_measurements: tuple[Measurement, Measurement, Measurement],
) -> None:
    record = MeasurementRecord()
    m1, m2, m3 = indexed_measurements

    record.add_measurement([m1, m2, m3])

    assert record.view().by_parent_primitive_id(m1.parent_primitive_id) == (m1, m2)
    assert record.view().by_parent_primitive_id(m3.parent_primitive_id) == (m3,)


def test_view_indexes_measurements_by_node_id_and_primitive_id(
    indexed_measurements: tuple[Measurement, Measurement, Measurement],
) -> None:
    record = MeasurementRecord()
    m1, m2, m3 = indexed_measurements

    record.add_measurement([m1, m2, m3])

    assert record.view().by_node_id_and_primitive_id(
        m1.node_id,
        m1.parent_primitive_id,
    ) == (m1, m2)
    assert record.view().by_node_id_and_primitive_id(
        m3.node_id,
        m3.parent_primitive_id,
    ) == (m3,)


def test_latest_by_node_id_returns_most_recent_match(
    indexed_measurements: tuple[Measurement, Measurement, Measurement],
) -> None:
    record = MeasurementRecord()
    m1, m2, m3 = indexed_measurements

    record.add_measurement(m1)
    record.add_measurement([m3, m2])

    assert record.view().latest_by_node_id(m1.node_id) == m2
    assert record.view().latest_by_node_id(m3.node_id) == m3


def test_latest_by_node_id_raises_for_unknown_node() -> None:
    record = MeasurementRecord()

    with pytest.raises(ValueError, match="No measurements found for node_id"):
        record.view().latest_by_node_id(uuid4())


def test_view_returns_empty_tuples_for_unknown_indexes(
    indexed_measurements: tuple[Measurement, Measurement, Measurement],
) -> None:
    record = MeasurementRecord()
    m1, _, _ = indexed_measurements

    record.add_measurement([m1])

    assert record.view().by_node_id(uuid4()) == ()
    assert record.view().by_parent_gadget_id(uuid4()) == ()
    assert record.view().by_parent_primitive_id(uuid4()) == ()
    assert record.view().by_node_id_and_primitive_id(m1.node_id, uuid4()) == ()
