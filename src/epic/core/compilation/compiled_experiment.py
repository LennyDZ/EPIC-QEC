from functools import cached_property
from typing import Dict, List
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from ..qec_object import Detector, Observable, Measurement
from ..data_structure import PhysicalQubit, QuantumProgram
from ..experiment import NoiseModel

from .measurement_record import MeasurementRecord


class CompiledExperiment(BaseModel):
    """Immutable compiled representation of an experiment ready for Stim export."""

    model_config = ConfigDict(frozen=True)

    record: MeasurementRecord
    program: QuantumProgram
    detectors: list[Detector]
    observables: list[Observable]

    measurement_to_index: Dict[Measurement, int] = {}

    def _rec_negative_index(self, measurement: Measurement) -> int:
        """Return the Stim ``rec`` offset for a recorded measurement."""
        if measurement not in self.measurement_to_index:
            raise ValueError(
                f"Measurement {measurement.id} used by detector/observable was not found in the measurement record."
            )
        stream_index = self.measurement_to_index[measurement]
        total_measurements = len(self.record.view().measurements())
        # In stim, rec[-1] is the latest measurement and rec[-N] is the earliest in scope.
        rec_offset = total_measurements - stream_index
        return -rec_offset

    def _measurement_to_rec_term(self, measurement: Measurement) -> str:
        """Format a measurement reference as a Stim ``rec[...]`` term."""
        return f"rec[{self._rec_negative_index(measurement)}]"

    def _format_detector_line(self, detector: Detector) -> str:
        """Build the Stim line for a detector and its measurement references."""
        rec_terms = [self._measurement_to_rec_term(m) for m in detector.measurements]
        if detector.coordinates:
            coords = ", ".join(str(c) for c in detector.coordinates)
            return f"DETECTOR({coords}) {' '.join(rec_terms)} # {detector.tag}"
        return f"DETECTOR {' '.join(rec_terms)} # {detector.tag}"

    def _format_observable_line(
        self, observable: Observable, index: int, verbose: bool = False
    ) -> str:
        """Build the Stim line for an observable include block."""

        measurements = sorted(
            observable.measurements,
            key=lambda m: self.measurement_to_index.get(m, -1),
        )
        rec_terms = [self._measurement_to_rec_term(m) for m in measurements]
        verb = f" # {observable.tag}" if verbose else ""
        if rec_terms:
            return f"OBSERVABLE_INCLUDE({index}) {' '.join(rec_terms)}" + verb
        return f"OBSERVABLE_INCLUDE({index})" + verb

    def to_stim_program(
        self,
        observables: List[List[str]],
        noise_model: NoiseModel | None = None,
        verbose: bool = False,
    ) -> str:

        lines = []

        op_by_start_time = sorted(self.program.operations, key=lambda x: x[1])
        idx = 0
        existing_observable_by_tag = {obs.tag: obs for obs in self.observables}
        m_by_id = self.record.by_measurement_id
        previous_start_time = 0
        for op, start_time in op_by_start_time:
            if start_time != previous_start_time:
                lines.append("TICK")
                previous_start_time = start_time

            if op.name == "tick":
                continue

            if not all(isinstance(t, PhysicalQubit) for t in op.targets):
                raise ValueError(f"Operation {op.name} has non-physical qubit targets: {op.targets}. When compiling to stim, it is expected that all qubits refers to physical qubits.")
            lines.append(f"{op.name} {' '.join(str(t.integer_index) for t in op.targets)}")
            if op.name in {"M", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"}:
                if not op.measurement_id:
                    raise ValueError(f"Operation {op.name} is missing a measurement_id.")
                self.measurement_to_index[m_by_id[op.measurement_id]] = idx
                idx += 1

        for i, detector in enumerate(self.detectors):
            pre = f"""# Detector {detector.tag} includes measurements: {[m.tag for m in detector.measurements]}"""
            verb = f" - Det Idx: {i}"
            l = self._format_detector_line(detector)
            if verbose:
                lines.append(pre)
                lines.append(l + verb)
            else:
                lines.append(l)


        stim_observable = []
        for ob in observables:
            new_ob_lops = []
            new_ob_measurements = set()
            new_ob_tag = "+".join(ob)
            for op_included in ob:
                if op_included not in existing_observable_by_tag:
                    raise ValueError(
                        f"Observable with tag {op_included} not found in compiled experiment."
                    )
                existing_ob = existing_observable_by_tag[op_included]
                new_ob_lops.extend(existing_ob.logical_operators_involved)
                new_ob_measurements ^= existing_ob.measurements
            stim_observable.append(
                Observable(
                    tag=new_ob_tag,
                    logical_operators_involved=new_ob_lops,
                    measurements=new_ob_measurements,
                )
            )

        for i, stim_ob in enumerate(stim_observable):
            lines.append(self._format_observable_line(stim_ob, i, verbose=verbose))

        program = "\n".join(lines)

        if noise_model is None:
            return program
        return noise_model.apply_model(program)