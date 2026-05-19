from uuid import UUID
from collections.abc import Iterable

from pydantic import BaseModel, PrivateAttr

from ..qec_object import Measurement


class MeasurementRecord(BaseModel):
    """Store recorded measurements and indexes for common lookup paths."""

    _measurements: list[Measurement] = PrivateAttr(default_factory=list, init=False)
    _by_node_id: dict[UUID, list[Measurement]] = PrivateAttr(
        default_factory=dict, init=False
    )
    _by_parent_gadget_id: dict[UUID, list[Measurement]] = PrivateAttr(
        default_factory=dict, init=False
    )
    _by_parent_primitive_id: dict[UUID, list[Measurement]] = PrivateAttr(
        default_factory=dict, init=False
    )
    _by_node_and_primitive_id: dict[tuple[UUID, UUID], list[Measurement]] = PrivateAttr(
        default_factory=dict, init=False
    )

    def _index_measurement(self, measurement: Measurement) -> None:
        """Index a single measurement by node id, gadget timestep, primitive timestep, and event type."""
        self._by_node_id.setdefault(measurement.node_id, []).append(measurement)
        self._by_parent_gadget_id.setdefault(measurement.parent_gadget_id, []).append(
            measurement
        )
        self._by_parent_primitive_id.setdefault(
            measurement.parent_primitive_id, []
        ).append(measurement)
        self._by_node_and_primitive_id.setdefault(
            (measurement.node_id, measurement.parent_primitive_id), []
        ).append(measurement)

    def _rebuild_indexes(self) -> None:
        """Rebuild indexes from the current measurement list."""
        self._by_node_id.clear()
        self._by_parent_gadget_id.clear()
        self._by_parent_primitive_id.clear()
        self._by_node_and_primitive_id.clear()
        for measurement in self._measurements:
            self._index_measurement(measurement)

    def add_measurement(
        self, measurements: Measurement | Iterable[Measurement]
    ) -> None:
        """Add one or many measurements to the record.

        Args:
            measurements: A single measurement or an iterable of measurements.
        """
        if isinstance(measurements, Measurement):
            self._measurements.append(measurements)
            self._index_measurement(measurements)
            return

        batch = list(measurements)
        if not batch:
            return

        self._measurements.extend(batch)
        for measurement in batch:
            self._index_measurement(measurement)

    def view(self) -> "MeasurementRecordView":
        """Return a read-only view over the current measurement record."""
        return MeasurementRecordView(self)


class MeasurementRecordView:
    """Read-only view over a MeasurementRecord (no mutation allowed)."""

    def __init__(self, record: MeasurementRecord):
        """Wrap a measurement record with tuple-based read accessors."""
        self._record = record

    def measurements(self) -> tuple[Measurement, ...]:
        """Return all measurements in insertion order as an immutable view."""
        return tuple(self._record._measurements)

    def by_node_id(self, node_id: UUID) -> tuple[Measurement, ...]:
        """Return all measurements recorded for a given node id."""
        return tuple(self._record._by_node_id.get(node_id, []))

    def latest_by_node_id(self, node_id: UUID) -> Measurement:
        """Return the latest measurement recorded for a given node id."""
        matches = self._record._by_node_id.get(node_id)
        out = matches[-1] if matches else None
        if out is None:
            raise ValueError(f"No measurements found for node_id {node_id}")
        return out

    def by_parent_gadget_id(self, parent_gadget_id: UUID) -> tuple[Measurement, ...]:
        """Return all measurements emitted by a given parent gadget."""
        return tuple(self._record._by_parent_gadget_id.get(parent_gadget_id, []))

    def by_parent_primitive_id(
        self, parent_primitive_id: UUID
    ) -> tuple[Measurement, ...]:
        """Return all measurements emitted by a given parent primitive."""
        return tuple(self._record._by_parent_primitive_id.get(parent_primitive_id, []))

    def by_node_id_and_primitive_id(
        self, node_id: UUID, parent_primitive_id: UUID
    ) -> tuple[Measurement, ...]:
        """Return measurements matching both a node id and a parent primitive id."""
        return tuple(
            self._record._by_node_and_primitive_id.get(
                (node_id, parent_primitive_id), []
            )
        )
