from typing import Dict, List, Tuple

import numpy as np
from pydantic import Field

from .css_code import CSSCode


class RotatedSurfaceCode(CSSCode):
    """Pydantic rotated surface code model built from lattice distance."""

    @staticmethod
    def _build_css_pcm(
        distance: int,
    ) -> Tuple[
        np.ndarray, np.ndarray, Dict[int, Tuple[int, int]], Dict[int, Tuple[int, int]]
    ]:
        if distance < 3 or distance % 2 == 0:
            raise ValueError("RotatedSurfaceCode requires an odd distance >= 3.")

        d = distance
        n = d * d

        # Data qubits on a d x d grid. Coordinates are scaled by 2 to keep
        # check centers and qubits on integer lattice points without overlap.
        data_index: Dict[Tuple[int, int], int] = {}
        var_coordinate: Dict[int, Tuple[int, int]] = {}
        for y in range(d):
            for x in range(d):
                i = y * d + x
                data_index[(x, y)] = i
                var_coordinate[i] = (2 * x, 2 * y)

        x_supports: List[List[int]] = []
        z_supports: List[List[int]] = []
        x_check_coords: List[Tuple[int, int]] = []
        z_check_coords: List[Tuple[int, int]] = []

        # Interior 4-body plaquettes on a checkerboard.
        for y in range(d - 1):
            for x in range(d - 1):
                support = [
                    data_index[(x, y)],
                    data_index[(x + 1, y)],
                    data_index[(x, y + 1)],
                    data_index[(x + 1, y + 1)],
                ]
                check_coord = (2 * x + 1, 2 * y + 1)
                if (x + y) % 2 == 0:
                    x_supports.append(support)
                    x_check_coords.append(check_coord)
                else:
                    z_supports.append(support)
                    z_check_coords.append(check_coord)

        # Boundary 2-body checks (alternating pattern) for the rotated patch.
        # Top/bottom are X-type; left/right are Z-type.
        for x in range(d - 1):
            if x % 2 == 1:
                x_supports.append([data_index[(x, 0)], data_index[(x + 1, 0)]])
                x_check_coords.append((2 * x + 1, -1))
            if x % 2 == 0:
                x_supports.append([data_index[(x, d - 1)], data_index[(x + 1, d - 1)]])
                x_check_coords.append((2 * x + 1, 2 * d - 1))

        for y in range(d - 1):
            if y % 2 == 0:
                z_supports.append([data_index[(0, y)], data_index[(0, y + 1)]])
                z_check_coords.append((-1, 2 * y + 1))
            if y % 2 == 1:
                z_supports.append([data_index[(d - 1, y)], data_index[(d - 1, y + 1)]])
                z_check_coords.append((2 * d - 1, 2 * y + 1))

        hx = np.zeros((len(x_supports), n), dtype=np.uint8)
        hz = np.zeros((len(z_supports), n), dtype=np.uint8)

        for row, support in enumerate(x_supports):
            hx[row, support] = 1
        for row, support in enumerate(z_supports):
            hz[row, support] = 1

        check_coordinate: Dict[int, Tuple[int, int]] = {}
        for i, coord in enumerate(x_check_coords):
            check_coordinate[i] = coord
        x_count = len(x_check_coords)
        for i, coord in enumerate(z_check_coords):
            check_coordinate[x_count + i] = coord

        return hx, hz, var_coordinate, check_coordinate

    @staticmethod
    def _build_default_logicals(
        distance: int, var_coordinate: Dict[int, Tuple[int, ...]]
    ) -> List[Tuple[List[int], List[int]]]:
        _ = distance
        n = len(var_coordinate)
        logical_x = [0] * n
        logical_z = [0] * n

        # Boundary logicals for rotated patch (open boundaries).
        for i, coord in var_coordinate.items():
            x, y = coord[:2]
            if x == 0:
                logical_x[i] = 1
            if y == 0:
                logical_z[i] = 1

        return [(logical_x, logical_z)]

    @classmethod
    def from_distance(
        cls,
        distance: int,
        code_name: str = "rotated_surface",
        system_coordinate: Tuple[int, int] = (0, 0),
    ) -> "RotatedSurfaceCode":
        hx, hz, var_coordinate, check_coordinate = cls._build_css_pcm(distance)

        var_coordinate = {
            i: (
                coord[0],
                coord[1],
            )
            + system_coordinate
            for i, coord in var_coordinate.items()
        }
        check_coordinate = {
            i: (
                coord[0],
                coord[1],
            )
            + system_coordinate
            for i, coord in check_coordinate.items()
        }

        logical_qubits = cls._build_default_logicals(distance, var_coordinate)

        code = cls.from_css_pcm(
            code_name=code_name,
            hx=hx,
            hz=hz,
            logical_qubits=logical_qubits,
            var_coordinate=var_coordinate,
            check_coordinate=check_coordinate,
        )
        return code
