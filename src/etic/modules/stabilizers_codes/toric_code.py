from typing import Dict, List, Tuple

import numpy as np

from epic.modules.stabilizers_codes.css_code import CSSCode


class ToricCode(CSSCode):
    """Pydantic toric code model built from lattice distance."""

    @staticmethod
    def _build_css_pcm(
        distance: int,
    ) -> Tuple[
        np.ndarray, np.ndarray, Dict[int, Tuple[int, int]], Dict[int, Tuple[int, int]]
    ]:
        if distance < 2:
            raise ValueError("distance must be >= 2")

        d = distance
        L = 2 * d

        data_coords: List[Tuple[int, int]] = []
        data_index: Dict[Tuple[int, int], int] = {}
        x_checks: List[Tuple[int, int]] = []
        z_checks: List[Tuple[int, int]] = []

        for r in range(L):
            for c in range(L):
                if (r + c) % 2 == 1:
                    data_index[(r, c)] = len(data_coords)
                    data_coords.append((r, c))
                elif r % 2 == 0 and c % 2 == 0:
                    x_checks.append((r, c))
                else:
                    z_checks.append((r, c))

        n = len(data_coords)
        hx = np.zeros((len(x_checks), n), dtype=np.uint8)
        hz = np.zeros((len(z_checks), n), dtype=np.uint8)

        def wrap(x: int) -> int:
            return x % L

        def neighbors(r: int, c: int) -> List[Tuple[int, int]]:
            return [
                (wrap(r), wrap(c - 1)),
                (wrap(r), wrap(c + 1)),
                (wrap(r - 1), wrap(c)),
                (wrap(r + 1), wrap(c)),
            ]

        for row, (r, c) in enumerate(x_checks):
            for q in neighbors(r, c):
                hx[row, data_index[q]] = 1

        for row, (r, c) in enumerate(z_checks):
            for q in neighbors(r, c):
                hz[row, data_index[q]] = 1

        var_coordinate = {i: coord for i, coord in enumerate(data_coords)}
        all_checks = x_checks + z_checks
        check_coordinate = {i: coord for i, coord in enumerate(all_checks)}

        return hx, hz, var_coordinate, check_coordinate

    @staticmethod
    def _build_default_logicals(
        distance: int, var_coordinate: Dict[int, Tuple[int, ...]]
    ) -> List[Tuple[List[int], List[int]]]:
        L = 2 * distance
        n = len(var_coordinate)

        # Z loops on primal edges (horizontal/vertical cycles).
        z_h = [0] * n
        z_v = [0] * n

        # X loops on dual edges so they are not the same support as Z loops.
        x_h = [0] * n
        x_v = [0] * n

        for i, coord in var_coordinate.items():
            r, c = coord[:2]
            # In this checkerboard embedding:
            # - horizontal edges are (even r, odd c)
            # - vertical edges are (odd r, even c)

            # Horizontal homology class.
            if r == 0 and c % 2 == 1:
                z_h[i] = 1
            if r == 1 and c % 2 == 0:
                x_h[i] = 1

            # Vertical homology class.
            if c == 0 and r % 2 == 1:
                z_v[i] = 1
            if c == 1 and r % 2 == 0:
                x_v[i] = 1

        return [
            (x_h, z_v),
            (x_v, z_h),
        ]

    @classmethod
    def from_distance(
        cls,
        distance: int,
        code_name: str = "toric",
        system_coordinate: Tuple[int, int] = (0, 0),
    ) -> "ToricCode":
        hx, hz, var_coordinate, check_coordinate = ToricCode._build_css_pcm(distance)

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

        logical_qubits = ToricCode._build_default_logicals(distance, var_coordinate)

        code = cls.from_css_pcm(
            code_name=code_name,
            hx=hx,
            hz=hz,
            logical_qubits=logical_qubits,
            var_coordinate=var_coordinate,
            check_coordinate=check_coordinate,
        )
        return code
