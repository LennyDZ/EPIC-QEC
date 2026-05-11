"""Shared QEC object fixtures for tests."""

from uuid import uuid4

from scipy.sparse import csr_matrix
import pytest

from core.compilation.compiled_experiment import CompiledExperiment
from core.compilation.measurement_record import MeasurementRecord
from core.compilation.qec_compiler import QECCompiler
from core.data_structure.pauli import PauliChar, PauliString
from core.data_structure.tanner_graph import TannerGraph
from core.data_structure.tanner_node import VariableNode
from core.qec_object.detector import Detector
from core.qec_object.logical_operator import LogicalOperator
from core.qec_object.logical_qubit import LogicalQubit
from core.qec_object.measurement import Measurement
from core.qec_object.observable import Observable
from core.qec_object.stabilizer_code import StabilizerCode
from core.language.qec_gadget import AllocCode, FreeCode
from experiment.noise_model import NoiseModel


@pytest.fixture
def measurement_a(variable_node: VariableNode) -> Measurement:
    return Measurement(
        node_id=variable_node.id,
        parent_gadget_id=uuid4(),
        parent_primitive_id=uuid4(),
        tag="m_a",
    )


@pytest.fixture
def measurement_b(variable_node: VariableNode) -> Measurement:
    return Measurement(
        node_id=variable_node.id,
        parent_gadget_id=uuid4(),
        parent_primitive_id=uuid4(),
        tag="m_b",
    )


@pytest.fixture
def logical_x(variable_node: VariableNode) -> LogicalOperator:
    return LogicalOperator(
        logical_type=PauliChar.X,
        operator=PauliString(string=(PauliChar.X,)),
        target_nodes=(variable_node,),
    )


@pytest.fixture
def logical_z(variable_node: VariableNode) -> LogicalOperator:
    return LogicalOperator(
        logical_type=PauliChar.Z,
        operator=PauliString(string=(PauliChar.Z,)),
        target_nodes=(variable_node,),
    )


@pytest.fixture
def logical_qubit(
    logical_x: LogicalOperator, logical_z: LogicalOperator
) -> LogicalQubit:
    return LogicalQubit(name="L0", logical_x=logical_x, logical_z=logical_z)


@pytest.fixture
def empty_check_tanner_graph(variable_node: VariableNode) -> TannerGraph:
    return TannerGraph.model_validate(
        {
            "variable_nodes": [variable_node],
            "check_nodes": [],
            "edges": [],
        }
    )


@pytest.fixture
def observable_with_ops(
    logical_x: LogicalOperator,
    logical_z: LogicalOperator,
    measurement_a: Measurement,
    measurement_b: Measurement,
) -> Observable:
    logical_x.frame_correction_history = {measurement_a}
    logical_z.frame_correction_history = {measurement_b}
    return Observable(logical_operators_involved=[logical_x, logical_z])


@pytest.fixture
def one_qubit_css_pcm() -> csr_matrix:
    return csr_matrix(([1], ([0], [0])), shape=(1, 2))


@pytest.fixture
def detector(measurement_a: Measurement) -> Detector:
    return Detector(measurements=[measurement_a], coordinates=(1, 2))


@pytest.fixture
def measurement_record() -> MeasurementRecord:
    return MeasurementRecord()


@pytest.fixture
def measurement_record_with_measurements(
    measurement_record: MeasurementRecord,
    measurement_a: Measurement,
    measurement_b: Measurement,
) -> MeasurementRecord:
    measurement_record.add_measurement([measurement_a, measurement_b])
    return measurement_record


@pytest.fixture
def observable_a(measurement_a: Measurement) -> Observable:
    return Observable(measurements={measurement_a}, tag="obs_a")


@pytest.fixture
def observable_b(measurement_b: Measurement) -> Observable:
    return Observable(measurements={measurement_b}, tag="obs_b")


@pytest.fixture
def identity_noise_model() -> NoiseModel:
    return NoiseModel()


@pytest.fixture
def compiled_experiment(
    measurement_record_with_measurements: MeasurementRecord,
    detector: Detector,
    observable_a: Observable,
    observable_b: Observable,
) -> CompiledExperiment:
    return CompiledExperiment(
        record=measurement_record_with_measurements,
        circuit_instructions=["H 0", "M 0"],
        detectors=[detector],
        observables=[observable_a, observable_b],
    )


@pytest.fixture
def stabilizer_code(
    empty_check_tanner_graph: TannerGraph,
    logical_qubit: LogicalQubit,
) -> StabilizerCode:
    return StabilizerCode(
        name="toy",
        n=1,
        k=1,
        d=1,
        tanner_graph=empty_check_tanner_graph,
        logical_qubits=[logical_qubit],
    )


@pytest.fixture
def compiler_config() -> dict:
    return {"primitives": {}, "objective_distance": 1}


@pytest.fixture
def qec_compiler(compiler_config: dict) -> QECCompiler:
    return QECCompiler(config=compiler_config)


@pytest.fixture
def alloc_code_gadget(
    stabilizer_code: StabilizerCode,
    logical_qubit: LogicalQubit,
) -> AllocCode:
    return AllocCode(
        tag="alloc",
        target_code=stabilizer_code,
        code_varname="code0",
        logical_qubits_varnames=[logical_qubit.name],
    )


@pytest.fixture
def free_code_gadget() -> FreeCode:
    return FreeCode(tag="free", code_varname="code0")
