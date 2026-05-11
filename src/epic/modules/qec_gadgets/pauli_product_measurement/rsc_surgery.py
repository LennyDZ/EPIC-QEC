from typing import List, Set, Tuple

from pydantic import field_validator

from epic.core.data_structure.pauli import PauliChar
from epic.core.data_structure.tanner_graph import TannerEdge, TannerGraph
from epic.core.data_structure.tanner_node import CheckNode, CheckNode
from epic.core.data_structure.tanner_node import VariableNode
from epic.core.data_structure.graph_algorithm import GraphAlgorithm as ga
from epic.core.qec_object.logical_operator import LogicalOperatorUpdate
from epic.core.qec_object.logical_qubit import LogicalQubit
from epic.core.qec_object.measurement import Measurement
from epic.core.qec_object.observable import Observable
from epic.core.qec_object.stabilizer_code import StabilizerCode
from epic.core.qec_primitives.interfaces.apply_gate import ApplyGate
from epic.core.qec_primitives.interfaces.extract_syndrome import ExtractSyndrome
from epic.core.qec_primitives.interfaces.readout import Readout
from epic.modules.qec_gadgets.pauli_product_measurement.ppm import PPM
from epic.modules.stabilizers_codes.rotated_surface_code import RotatedSurfaceCode


class RSCSurgery(PPM):
    """
    A gadget representing the surgery of two logical qubits in a stabilizer code using the RSC protocol.
    """

    @field_validator("product_to_measure")
    def validate_product_to_measure(cls, v):
        if v.weight != 2:
            raise ValueError(
                f"RSCSurgery requires a product of two Pauli operators to measure. Got {v.weight} operators."
            )
        if not all(x == v.string[0] for x in v.string):
            raise ValueError(
                f"RSCSurgery only supports measuring XX or ZZ products. Got {v.string}."
            )
        return v

    def _check_axis(
        self,
        resolved_targets: List[Tuple[LogicalQubit, StabilizerCode]],
        lop_involved,
    ) -> int:

        # Check operators are either all vertical or all horizontally in the lattice.
        lop0, lop1 = lop_involved
        axis = []
        for lop in lop_involved:
            x1, y1 = lop.target_nodes[0].coordinates[:2]
            x2, y2 = lop.target_nodes[1].coordinates[:2]
            if x1 == x2:
                axis.append(1)
            elif y1 == y2:
                axis.append(0)
        if not all(a == axis[0] for a in axis):
            raise ValueError(
                f"RSCSurgery requires the two logical operators to be adjacent along the same axis. Got axes {axis} for the two logical operators."
            )
        merge_axis = 1 - axis[0]
        syst1 = lop1.target_nodes[0].coordinates[2:]
        syst0 = lop0.target_nodes[0].coordinates[2:]
        if (
            syst0[axis[0]] != syst1[axis[0]]
            or abs(syst0[merge_axis] - syst1[merge_axis]) != 1
        ):
            raise ValueError(
                f"RSCSurgery requires the two logical operators to be adjacent along the same axis. Got coordinates {syst0} and {syst1} for the two logical operators."
            )

        return axis[0]

    def _build_ancilla_system(
        self, lop_supports: List[TannerGraph], axis: int, merge_type: PauliChar
    ) -> Tuple[TannerGraph, Set[TannerEdge]]:
        connecting_edges = set()
        merge_axis = 1 - axis
        ordered_supports = sorted([sorted(list(ls.check_nodes | ls.variable_nodes), key=lambda n: n.coordinates[axis]) for ls in lop_supports], key=lambda x: x[0].coordinates[merge_axis + 2])  # type: ignore
        anc_check = set()
        anc_var = set()
        ancilla_nodes = []

        # Check the two-body stabs layout:
        def get_anc_coord(sys_coord, layer, side):
            if axis == 0:
                return (layer, side, sys_coord[0], sys_coord[1])
            else:
                return (side, layer, sys_coord[0], sys_coord[1])

        sys_coord = ordered_supports[0][0].coordinates[2:]  # type: ignore
        first_check_coord = ordered_supports[0][0].coordinates[1 - axis]  # type: ignore
        anc_var_fixed_coord = lop_supports[0].number_of_nodes + 1
        inverser = -1 if axis == 0 else 1
        side = +1
        # add outer 2 body stabs
        c = CheckNode(
            tag=f"anc_check_0",
            check_type=merge_type,
            coordinates=get_anc_coord(
                sys_coord, -1, anc_var_fixed_coord + side * inverser
            ),
        )
        side = -side
        ancilla_nodes.append(c)
        anc_check.add(c)
        for i, (code1_node, code2_node) in enumerate(
            zip(ordered_supports[0], ordered_supports[1])
        ):
            if isinstance(code1_node, CheckNode) and isinstance(code2_node, CheckNode):
                c = CheckNode(
                    tag=f"anc_check_{i+1}",
                    check_type=merge_type,
                    coordinates=get_anc_coord(
                        sys_coord, i, anc_var_fixed_coord + side * inverser
                    ),
                )
                side = -side
                anc_check.add(c)
                ancilla_nodes.append(c)
            elif isinstance(code1_node, VariableNode) and isinstance(
                code2_node, VariableNode
            ):
                v = VariableNode(
                    tag=f"anc_var_{i+1}",
                    coordinates=get_anc_coord(sys_coord, i, anc_var_fixed_coord),
                )
                anc_var.add(v)
                ancilla_nodes.append(v)

            else:
                raise ValueError(
                    f"Unexpected node types in logical operator supports. Got {type(code1_node)} and {type(code2_node)} for nodes at position {i} in the supports."
                )
        c = CheckNode(
            tag=f"anc_check_{len(ordered_supports[0])+1}",
            check_type=merge_type,
            coordinates=get_anc_coord(
                sys_coord,
                len(ordered_supports[0]),
                anc_var_fixed_coord + side * inverser,
            ),
        )
        anc_check.add(c)
        ancilla_nodes.append(c)

        ancilla_edges = set()
        for an in range(len(ancilla_nodes) - 1):
            if isinstance(ancilla_nodes[an], CheckNode) and isinstance(
                ancilla_nodes[an + 1], VariableNode
            ):
                ancilla_edges.add(
                    TannerEdge(
                        variable_node=ancilla_nodes[an + 1],
                        check_node=ancilla_nodes[an],
                        pauli_checked=merge_type,
                    )
                )
            elif isinstance(ancilla_nodes[an], VariableNode) and isinstance(
                ancilla_nodes[an + 1], CheckNode
            ):
                ancilla_edges.add(
                    TannerEdge(
                        variable_node=ancilla_nodes[an],
                        check_node=ancilla_nodes[an + 1],
                        pauli_checked=merge_type,
                    )
                )

        connecting_edges = set()

        # This is a bit criptic, it basically create edges between the codes and the ancilla system to for the expected layout.
        # Based on the orientation of the codes as well as the orientation of the merge.
        if inverser == 1:
            connecting_edges.add(TannerEdge(variable_node=ordered_supports[axis][0], check_node=ancilla_nodes[0], pauli_checked=merge_type))  # type: ignore
            connecting_edges.add(TannerEdge(variable_node=ordered_supports[1 - axis][-1], check_node=ancilla_nodes[-1], pauli_checked=merge_type))  # type: ignore
        else:
            connecting_edges.add(TannerEdge(variable_node=ordered_supports[axis][0], check_node=ancilla_nodes[0], pauli_checked=merge_type))  # type: ignore
            connecting_edges.add(TannerEdge(variable_node=ordered_supports[1 - axis][-1], check_node=ancilla_nodes[-1], pauli_checked=merge_type))  # type: ignore

        for i, anc_node in enumerate(ancilla_nodes[1:-1]):
            if isinstance(anc_node, CheckNode):
                if anc_node.coordinates[1 - axis] < anc_var_fixed_coord:  # type: ignore
                    connecting_edges.add(TannerEdge(variable_node=ordered_supports[0][i + 1], check_node=anc_node, pauli_checked=merge_type))  # type: ignore
                    connecting_edges.add(TannerEdge(variable_node=ordered_supports[0][i - 1], check_node=anc_node, pauli_checked=merge_type))  # type: ignore
                else:
                    connecting_edges.add(TannerEdge(variable_node=ordered_supports[1][i + 1], check_node=anc_node, pauli_checked=merge_type))  # type: ignore
                    connecting_edges.add(TannerEdge(variable_node=ordered_supports[1][i - 1], check_node=anc_node, pauli_checked=merge_type))  # type: ignore
            else:
                checkside = (
                    1 if (i % 4 == 2 and axis == 0) or (i % 4 == 0 and axis == 1) else 0
                )
                if i < len(ancilla_nodes) - 3:
                    connecting_edges.add(TannerEdge(variable_node=anc_node, check_node=ordered_supports[checkside][i + 1], pauli_checked=merge_type.dual()))  # type: ignore
                if i > 1:
                    connecting_edges.add(TannerEdge(variable_node=anc_node, check_node=ordered_supports[1 - checkside][i - 1], pauli_checked=merge_type.dual()))  # type: ignore

        return (
            TannerGraph(
                variable_nodes=anc_var,
                check_nodes=anc_check,
                edges=ancilla_edges,
            ),
            connecting_edges,
        )

    def compile(
        self,
        resolved_targets: List[Tuple[LogicalQubit, StabilizerCode]],
        record,
        timestep,
        objective_distance: int,
    ):
        code1, code2 = resolved_targets[0][1], resolved_targets[1][1]
        if code1.d != code2.d:
            raise ValueError(
                f"RSCSurgery requires the two logical qubits to be encoded in codes of the same distance. Got codes of distance {code1.d} and {code2.d}."
            )

        if not isinstance(code1, RotatedSurfaceCode) and not isinstance(
            code2, RotatedSurfaceCode
        ):
            raise ValueError(
                f"RSCSurgery only supports logical qubits encoded in the Rotated Surface Code. Got code of type {type(code1)} and {type(code2)}."
            )

        # Extract Logical op involved in the pauli product to measure, as well as the anticommuting ones that may require corrections
        anticommuting_lop = []
        lop_involved = []
        if self.product_to_measure.string[0] == "X":
            for lop, code in resolved_targets:
                lop_involved.append(lop.logical_x)
                anticommuting_lop.append(lop.logical_z)
        else:
            for lop, code in resolved_targets:
                lop_involved.append(lop.logical_z)
                anticommuting_lop.append(lop.logical_x)

        # Get merge type (X or Z) from the product to measure
        merge_type = self.product_to_measure.string[0]

        # Get axis along which the merge occurs, horizontal (0) or vertical (1)
        axis = self._check_axis(resolved_targets, lop_involved)

        ## Compute translated lop, i.e. the chain that is on the merge boundary.
        # The current lop.target_nodes may represent a chain far from the merge boundary.
        # For each code we select the equivalent chain (same logical, different stabilizer-coset
        # representative) whose perpendicular coordinate sits at the edge facing the other code.
        syst0 = lop_involved[0].target_nodes[0].coordinates[2:]
        syst1 = lop_involved[1].target_nodes[0].coordinates[2:]
        merge_axis = 1 - axis
        c1_is_lower = syst0[merge_axis] < syst1[merge_axis]

        checks_in_between = set()
        lop_supports = []
        for i, (lop, code) in enumerate(zip(lop_involved, [code1, code2])):
            is_lower = c1_is_lower if i == 0 else not c1_is_lower
            if is_lower:
                checks_in_between.update(
                    n for n in code.tanner_graph.check_nodes if n.coordinates[1 - axis] > lop.target_nodes[0].coordinates[1 - axis] and n.check_type == merge_type  # type: ignore
                )
            else:
                checks_in_between.update(
                    n for n in code.tanner_graph.check_nodes if n.coordinates[1 - axis] < lop.target_nodes[0].coordinates[1 - axis] and n.check_type == merge_type  # type: ignore
                )
            perp_coords = [n.coordinates[1 - axis] for n in code.tanner_graph.variable_nodes]  # type: ignore
            boundary_coord = max(perp_coords) if is_lower else min(perp_coords)
            boundary_nodes = {
                n for n in code.tanner_graph.variable_nodes if n.coordinates[1 - axis] == boundary_coord  # type: ignore
            }
            lop_supports.append(
                code.tanner_graph.get_support(boundary_nodes, lop.logical_type)
            )

        ancilla_system, connecting_edges = self._build_ancilla_system(
            lop_supports, axis, merge_type
        )

        checks_in_between.update(n for n in ancilla_system.check_nodes)  # type: ignore

        # 3
        merged_system = ancilla_system.connect_to(
            code1.tanner_graph | code2.tanner_graph, connecting_edges
        )

        # 4
        primitives = []
        lop_updates = {}

        init_ancilla = ApplyGate(
            target=ancilla_system,
            target_nodes=ancilla_system.variable_nodes | ancilla_system.check_nodes,  # type: ignore
            gates=(
                ["RX"] if merge_type == PauliChar.Z else ["RZ"]
            ),  # Init in dual of the merge type.
        )
        merged_syndrome = ExtractSyndrome(
            target=merged_system,
            rounds=objective_distance,
            tag=f"rsc_surgery_merged_syndrome_{self.tag}",
        )
        ancilla_readout = Readout(
            target=ancilla_system,
            readout_basis=merge_type.dual(),
            tag=f"rsc_surgery_measurement_{self.tag}",
        )

        split_syndrome = [
            ExtractSyndrome(
                target=initial_code,
                rounds=objective_distance,
                tag=f"rsc_surgery_split_syndrome_{self.tag}",
            )
            for initial_code in [code1.tanner_graph, code2.tanner_graph]
        ]

        primitives.extend(
            [init_ancilla, merged_syndrome, ancilla_readout] + split_syndrome
        )

        # Find shortest path between endpoints of anticommuting_lop[0] and anticommuting_lop[1] (aligned along axis)
        nodes0 = sorted(anticommuting_lop[0].target_nodes, key=lambda n: n.coordinates[1 - axis])  # type: ignore
        nodes1 = sorted(anticommuting_lop[1].target_nodes, key=lambda n: n.coordinates[1 - axis])  # type: ignore

        # Try both endpoint pairings and pick the shorter path
        path1 = ga.shortest_path(graph=merged_system, start=nodes0[-1], end=nodes1[0], check_type=merge_type)  # type: ignore
        path2 = ga.shortest_path(graph=merged_system, start=nodes0[0], end=nodes1[-1], check_type=merge_type)  # type: ignore

        correction_chain = (
            path1 if (path1 and (not path2 or len(path1) <= len(path2))) else path2
        )

        correction_chain = [c for c in correction_chain if isinstance(c, VariableNode) and c in ancilla_system.variable_nodes] if correction_chain else []  # type: ignore
        lop_updates = {
            anticommuting_lop[0].id: LogicalOperatorUpdate(
                new_correction={
                    Measurement(
                        n.id,
                        parent_gadget_id=self.id,
                        parent_primitive_id=ancilla_readout.id,
                    )
                    for n in correction_chain
                }
            ),
        }

        observable = [
            Observable(
                logical_operators_involved=lop_involved,
                measurements={
                    Measurement(
                        n.id,
                        parent_gadget_id=self.id,
                        parent_primitive_id=merged_syndrome.id,
                    )
                    for n in checks_in_between
                },
                tag=f"rsc_surgery_{self.tag}",
            )
        ]

        return lop_updates, observable, primitives
