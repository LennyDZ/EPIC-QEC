"""Tests for compiled_experiment module."""

from uuid import uuid4

import pytest

from epic.core.compilation.compiled_experiment import CompiledExperiment
from epic.core.compilation.measurement_record import MeasurementRecord
from epic.core.qec_object.measurement import Measurement
from epic.core.qec_object.observable import Observable


class TestCompiledExperiment:
    def test_measurement_to_index_uses_record_insertion_order(
        self,
        compiled_experiment: CompiledExperiment,
        measurement_a: Measurement,
        measurement_b: Measurement,
    ) -> None:
        assert compiled_experiment.measurement_to_index == {
            measurement_a: 0,
            measurement_b: 1,
        }

    def test_rec_negative_index_counts_back_from_latest(
        self,
        compiled_experiment: CompiledExperiment,
        measurement_a: Measurement,
        measurement_b: Measurement,
    ) -> None:
        assert compiled_experiment._rec_negative_index(measurement_a) == -2
        assert compiled_experiment._rec_negative_index(measurement_b) == -1

    def test_rec_negative_index_raises_for_unrecorded_measurement(
        self,
        compiled_experiment: CompiledExperiment,
    ) -> None:
        missing = Measurement(
            node_id=uuid4(),
            parent_gadget_id=uuid4(),
            parent_primitive_id=uuid4(),
            tag="missing",
        )

        with pytest.raises(ValueError, match="not found in the measurement record"):
            compiled_experiment._rec_negative_index(missing)

    def test_measurement_to_rec_term_formats_stim_reference(
        self,
        compiled_experiment: CompiledExperiment,
        measurement_b: Measurement,
    ) -> None:
        assert compiled_experiment._measurement_to_rec_term(measurement_b) == "rec[-1]"

    def test_format_detector_line_includes_coordinates(
        self,
        compiled_experiment: CompiledExperiment,
        detector,
    ) -> None:
        assert (
            compiled_experiment._format_detector_line(detector)
            == "DETECTOR(1, 2) rec[-2]"
        )

    def test_format_observable_line_sorts_measurements_by_record_order(
        self,
        compiled_experiment: CompiledExperiment,
        measurement_a: Measurement,
        measurement_b: Measurement,
    ) -> None:
        observable = Observable(
            tag="combined",
            measurements={measurement_b, measurement_a},
        )

        assert (
            compiled_experiment._format_observable_line(observable, 3)
            == "OBSERVABLE_INCLUDE(3) rec[-2] rec[-1]"
        )

    def test_to_stim_program_emits_circuit_detectors_and_observables(
        self,
        compiled_experiment: CompiledExperiment,
        identity_noise_model,
    ) -> None:
        stim_program = compiled_experiment.to_stim_program(
            [["obs_a"], ["obs_a", "obs_b"]],
            identity_noise_model,
        )

        assert stim_program == "\n".join(
            [
                "H 0",
                "M 0",
                "DETECTOR(1, 2) rec[-2]",
                "OBSERVABLE_INCLUDE(0) rec[-2]",
                "OBSERVABLE_INCLUDE(1) rec[-2] rec[-1]",
            ]
        )

    def test_to_stim_program_xors_measurements_across_combined_observables(
        self,
        measurement_record_with_measurements: MeasurementRecord,
        identity_noise_model,
        measurement_a: Measurement,
        observable_a: Observable,
    ) -> None:
        compiled = CompiledExperiment(
            record=measurement_record_with_measurements,
            circuit_instructions=[],
            detectors=[],
            observables=[
                observable_a,
                Observable(measurements={measurement_a}, tag="obs_a_again"),
            ],
        )

        stim_program = compiled.to_stim_program(
            [["obs_a", "obs_a_again"]],
            identity_noise_model,
        )

        assert stim_program == "OBSERVABLE_INCLUDE(0)"

    def test_to_stim_program_raises_for_unknown_observable_tag(
        self,
        compiled_experiment: CompiledExperiment,
        identity_noise_model,
    ) -> None:
        with pytest.raises(ValueError, match="Observable with tag missing not found"):
            compiled_experiment.to_stim_program([["missing"]], identity_noise_model)
