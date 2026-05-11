from functools import cached_property
from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from epic.core.experiment.noise_model import NoiseModel

from ..qec_object import Measurement
from .measurement_record import MeasurementRecord
from ..qec_object import Detector, Observable


class CompiledExperiment(BaseModel):
    """Immutable compiled representation of an experiment ready for Stim export."""

    model_config = ConfigDict(frozen=True)

    record: MeasurementRecord
    circuit_instructions: list[str]
    detectors: list[Detector]
    observables: list[Observable]

    @cached_property
    def measurement_to_index(self) -> Dict[Measurement, int]:
        """Map each recorded measurement to its position in the Stim record stream."""
        return {m: idx for idx, m in enumerate(self.record.view().measurements())}

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
            return f"DETECTOR({coords}) {' '.join(rec_terms)}"
        return f"DETECTOR {' '.join(rec_terms)}"

    def _format_observable_line(self, observable: Observable, index: int) -> str:
        """Build the Stim line for an observable include block."""

        measurements = sorted(
            observable.measurements,
            key=lambda m: self.measurement_to_index.get(m, -1),
        )
        rec_terms = [self._measurement_to_rec_term(m) for m in measurements]
        if rec_terms:
            return f"OBSERVABLE_INCLUDE({index}) {' '.join(rec_terms)}"
        return f"OBSERVABLE_INCLUDE({index})"

    def to_stim_program(
        self,
        observables: List[List[str]],
        noise_model: NoiseModel,
    ) -> str:
        """Render the compiled experiment as a Stim program and apply noise.

        Args:
            observables: Groups of observable tags to combine into Stim observables.
            noise_model: Noise model applied to the rendered program.

        Returns:
            The final Stim program after detector, observable, and noise expansion.
        """
        lines = []
        existing_observable_by_tag = {obs.tag: obs for obs in self.observables}
        for instruction in self.circuit_instructions:
            lines.append(instruction)
        for detector in self.detectors:
            lines.append(self._format_detector_line(detector))
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
            lines.append(self._format_observable_line(stim_ob, i))

        program = "\n".join(lines)

        return noise_model.apply_model(program)
