"""Compilation-time context and registries for QEC programs."""

from types import MappingProxyType
from typing import Any, Dict, List, Set, Tuple, TypeVar, overload
from uuid import UUID
import warnings

from etic.core.qec_object.detector import (
    Detector,
    DetectorGraphPort,
    QubitPortState,
)
from etic.core.qec_object.measurement import Measurement

from .compiled_experiment import CompiledExperiment
from .measurement_record import MeasurementRecord
from .quantum_memory import QuantumMemory
from ..data_structure.tanner_node import TannerNode
from ..qec_object import LogicalOperator, LogicalQubit, Observable, StabilizerCode

TargetT = TypeVar("TargetT", LogicalQubit, StabilizerCode)


class CompilationContext:
    """
    Experiment Context for quantum error correction simulations.
    A context contains tracks all sorts of registers used during the compilation.

    The context is updated at each step of the compilation process, the logical operators may evolve
    and the context will record the measurements, detectors, and observables produced by the computation.

    Attributes:
        quantum_memory: Stores the allocated quantum state during compilation.
        measurement_record: Tracks measurements emitted by compiled primitives.
    """

    def __init__(self):
        """Initialize the compilation state and all its registers. At the start, everything is empty."""
        self._uuid_memory: Dict[UUID, Any] = {}
        self._naming_registry: Dict[str, UUID] = {}
        self._observables: Set[UUID] = set()
        self._operator_to_qubit: Dict[UUID, UUID] = {}
        self._qubit_to_code: Dict[UUID, UUID] = {}
        self._detector_port: DetectorGraphPort = DetectorGraphPort()
        self._circuit_instructions = []
        self._detectors = []

        self.quantum_memory = QuantumMemory()
        self.measurement_record = MeasurementRecord()

        self._compilation_time = (0, 0)

    def read_variable(self, varname: str) -> Any:
        """Return the object registered under a variable name."""
        if varname not in self._naming_registry:
            raise KeyError(f"Variable name {varname} not found in context.")
        var_id = self._naming_registry[varname]
        return self._uuid_memory[var_id]

    def incr_gadget_timestep(self):
        """Advance to the next gadget timestep and reset the primitive timestep."""
        self._compilation_time = (self._compilation_time[0] + 1, 0)

    def incr_primitive_timestep(self):
        """Advance the primitive timestep within the current gadget timestep."""
        self._compilation_time = (
            self._compilation_time[0],
            self._compilation_time[1] + 1,
        )

    @property
    def t_gadget(self) -> int:
        """Return the current gadget timestep."""
        return self._compilation_time[0]

    @property
    def t_primitive(self) -> int:
        """Return the current primitive timestep."""
        return self._compilation_time[1]

    def to_compiled_experiment(self) -> CompiledExperiment:
        """Build a compiled experiment from the recorded instructions and outputs."""
        return CompiledExperiment(
            record=self.measurement_record,
            circuit_instructions=self._circuit_instructions,
            detectors=self._detectors,
            observables=[self._uuid_memory[obs_id] for obs_id in self._observables],
        )

    def _resolve_measurement(self, measurement: Measurement) -> Measurement:
        """Resolve a measurement reference to its latest recorded instance."""
        matches = self.measurement_record.view().by_node_id_and_primitive_id(
            measurement.node_id,
            measurement.parent_primitive_id,
        )
        if not matches:
            raise ValueError(
                f"Measurement {measurement} not found in measurement record."
            )
        return matches[-1]

    def resolve_observable(self, observable: Observable) -> Observable:
        """Return an observable with all referenced measurements resolved."""
        resolved_measurements: Set[Measurement] = set()
        for m in observable.measurements:
            resolved_measurements.add(self._resolve_measurement(m))

        resolved_observable = Observable(
            logical_operators_involved=observable.logical_operators_involved,
            measurements=resolved_measurements,
            tag=observable.tag,
        )
        resolved_observable.apply_corrective_frame()
        resolved_observable.measurements = {
            self._resolve_measurement(measurement)
            for measurement in resolved_observable.measurements
        }
        return resolved_observable

    def add_observable(self, observable: Observable):
        """Register an observable in the context and naming registry."""
        self._observables.add(observable.id)
        self._uuid_memory[observable.id] = observable
        if observable.tag in self._naming_registry:
            warnings.warn(
                f"Observable tag {observable.tag} already exists in naming registry. Overwriting previous entry."
            )
        self._naming_registry[observable.tag] = observable.id

    def add_detector(self, detector: Detector | List[Detector]):
        """Append one detector or a list of detectors to the compiled output."""
        if isinstance(detector, list):
            self._detectors.extend(detector)
        else:
            self._detectors.append(detector)

    def add_circuit_instruction(self, instruction: str | List[str]):
        """Append one circuit instruction or a list of instructions."""
        if isinstance(instruction, list):
            self._circuit_instructions.extend(instruction)
        else:
            self._circuit_instructions.append(instruction)

    def get_observable_by_id(self, obs_id: UUID) -> Observable:
        """Return an observable by UUID."""
        if obs_id not in self._uuid_memory:
            raise KeyError(f"Observable with id {obs_id} not found in context.")
        obs = self._uuid_memory[obs_id]
        if not isinstance(obs, Observable):
            raise TypeError(f"Object with id {obs_id} is not an Observable.")
        return obs

    def get_observable_by_tag(self, tag: str) -> Observable:
        """Return an observable by its tag."""
        if tag not in self._naming_registry:
            raise KeyError(f"Observable with tag {tag} not found in context.")
        obs_id = self._naming_registry[tag]
        return self.get_observable_by_id(obs_id)

    def register_code(
        self, lqb_name: List[str], code: StabilizerCode, code_varname: str
    ):
        """Register a code, its logical qubits, and their logical operators."""
        self._uuid_memory[code.id] = code
        self._naming_registry[code_varname] = code.id
        for idx, qubit in enumerate(code.logical_qubits):
            self._uuid_memory[qubit.id] = qubit
            self._naming_registry[lqb_name[idx]] = qubit.id
            self._qubit_to_code[qubit.id] = code.id
            for op in [qubit.logical_x, qubit.logical_z]:
                self._uuid_memory[op.id] = op
                self._operator_to_qubit[op.id] = qubit.id

    def unregister_code(self, code_varname: str):
        """Remove a registered code and all objects derived from it."""
        if code_varname not in self._naming_registry:
            raise KeyError(f"Code variable name {code_varname} not found in context.")
        code_id = self._naming_registry[code_varname]
        code = self._uuid_memory[code_id]
        if not isinstance(code, StabilizerCode):
            raise TypeError(f"Object with id {code_id} is not a StabilizerCode.")
        for qubit in code.logical_qubits:
            del self._uuid_memory[qubit.id]
            qubit_varname = next(
                (
                    name
                    for name, var_id in self._naming_registry.items()
                    if var_id == qubit.id
                ),
                None,
            )
            if qubit_varname is not None:
                del self._naming_registry[qubit_varname]
            del self._qubit_to_code[qubit.id]
            for op in [qubit.logical_x, qubit.logical_z]:
                del self._uuid_memory[op.id]
                del self._operator_to_qubit[op.id]
        del self._uuid_memory[code_id]
        del self._naming_registry[code_varname]

    def allocated_code_varnames(self) -> List[str]:
        """Return the variable names of all currently registered codes."""
        return [
            varname
            for varname, var_id in self._naming_registry.items()
            if var_id in self._uuid_memory
            and isinstance(self._uuid_memory[var_id], StabilizerCode)
        ]

    def assert_no_allocated_codes(self):
        """Raise if any stabilizer codes are still registered."""
        allocated_codes = self.allocated_code_varnames()
        if allocated_codes:
            raise ValueError(
                "Compilation finished with allocated codes still registered: "
                f"{', '.join(sorted(allocated_codes))}."
            )

    def lqb_from_op(self, op: LogicalOperator) -> LogicalQubit:
        """Return the logical qubit that owns a logical operator."""
        if op.id not in self._operator_to_qubit:
            raise KeyError(f"Logical operator with id {op.id} not found in context.")
        qubit_id = self._operator_to_qubit[op.id]
        qubit = self._uuid_memory[qubit_id]
        if not isinstance(qubit, LogicalQubit):
            raise TypeError(f"Object with id {qubit_id} is not a LogicalQubit.")
        return qubit

    def code_from_lqb(self, lqb: LogicalQubit) -> StabilizerCode:
        """Return the stabilizer code that owns a logical qubit."""
        if lqb.id not in self._qubit_to_code:
            raise KeyError(f"Logical qubit with id {lqb.id} not found in context.")
        code_id = self._qubit_to_code[lqb.id]
        code = self._uuid_memory[code_id]
        if not isinstance(code, StabilizerCode):
            raise TypeError(f"Object with id {code_id} is not a StabilizerCode.")
        return code

    def get_logical_op_by_id(self, log_id: UUID) -> LogicalOperator:
        """Return a logical operator by UUID."""
        if log_id not in self._uuid_memory:
            raise KeyError(f"Logical operator with id {log_id} not found in context.")
        log_op = self._uuid_memory[log_id]
        if not isinstance(log_op, LogicalOperator):
            raise TypeError(f"Object with id {log_id} is not a LogicalOperator.")
        return log_op

    def detector_graph_port_view(self) -> MappingProxyType[TannerNode, QubitPortState]:
        """Return a read-only view of the detector graph port state."""
        return MappingProxyType(self._detector_port)

    def update_dg_port(self, node: TannerNode, knowledge: QubitPortState):
        """Store the current detector-graph port state for a Tanner node."""
        self._detector_port[node] = knowledge

    @overload
    def resolve_targets_varname(
        self, targets: List[str], expected_type: type[LogicalQubit]
    ) -> List[Tuple[LogicalQubit, StabilizerCode]]: ...
    @overload
    def resolve_targets_varname(
        self, targets: List[str], expected_type: type[StabilizerCode]
    ) -> List[StabilizerCode]: ...

    def resolve_targets_varname(
        self, targets: List[str], expected_type: type[TargetT]
    ) -> List[Tuple[LogicalQubit, StabilizerCode]] | List[StabilizerCode]:
        """Resolve named targets to typed objects expected by a primitive or gadget."""
        resolved = []
        for t in targets:
            if t not in self._naming_registry:
                raise KeyError(f"Target variable name {t} not found in context.")
            target_id = self._naming_registry[t]
            target_obj = self._uuid_memory[target_id]
            if not isinstance(target_obj, expected_type):
                raise TypeError(
                    f"Object with id {target_id} is not a valid target type "
                    f"({type(target_obj).__name__}); expected {expected_type.__name__}."
                )
            if expected_type is LogicalQubit:
                assert isinstance(target_obj, LogicalQubit)
                resolved.append((target_obj, self.code_from_lqb(target_obj)))
            else:
                resolved.append(target_obj)
        return resolved
