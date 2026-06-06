from typing import Dict, List, Tuple

import numpy as np

from epic.modules.stabilizers_codes.css_code import CSSCode


class SurfaceCode(CSSCode):
    """Represents a planar surface code with open boundaries. The code is defined by its distance, which determines the size of the lattice and the number of qubits and checks. The code has one logical qubit, with the logical X operator connecting the top and bottom boundaries (smooth) and the logical Z operator connecting the left and right boundaries (rough)."""

    @staticmethod
    def _build_css_pcm(
        distance: int,
    ) -> Tuple[
        np.ndarray, np.ndarray, Dict[int, Tuple[int, int]], Dict[int, Tuple[int, int]]
    ]:
        if distance < 2:
            raise ValueError("distance must be >= 2")

        d = distance
        L = 2 * d - 1

        data_coords: List[Tuple[int, int]] = []
        data_index: Dict[Tuple[int, int], int] = {}
        x_checks: List[Tuple[int, int]] = []
        z_checks: List[Tuple[int, int]] = []

        # Checkerboard embedding with open boundaries.
        # Data qubits at all four corners: (r+c) % 2 == 0
        # X-checks at (r odd, c even), Z-checks at (r even, c odd)
        # Top/bottom boundaries are smooth (no X-checks), left/right are rough (X-checks present).
        for r in range(L):
            for c in range(L):
                if (r + c) % 2 == 0:
                    data_index[(r, c)] = len(data_coords)
                    data_coords.append((r, c))
                elif r % 2 == 1 and c % 2 == 0:
                    # X-checks at odd rows (rough boundaries on left/right)
                    x_checks.append((r, c))
                elif r % 2 == 0 and c % 2 == 1:
                    # Z-checks at even rows (smooth boundaries on top/bottom)
                    z_checks.append((r, c))

        n = len(data_coords)
        hx = np.zeros((len(x_checks), n), dtype=np.uint8)
        hz = np.zeros((len(z_checks), n), dtype=np.uint8)

        def neighbors_open(r: int, c: int) -> List[Tuple[int, int]]:
            candidates = [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)]
            return [q for q in candidates if q in data_index]

        for row, (r, c) in enumerate(x_checks):
            for q in neighbors_open(r, c):
                hx[row, data_index[q]] = 1

        for row, (r, c) in enumerate(z_checks):
            for q in neighbors_open(r, c):
                hz[row, data_index[q]] = 1

        var_coordinate = {i: coord for i, coord in enumerate(data_coords)}
        all_checks = x_checks + z_checks
        check_coordinate = {i: coord for i, coord in enumerate(all_checks)}

        return hx, hz, var_coordinate, check_coordinate

    @staticmethod
    def _build_default_logicals(
        distance: int, var_coordinate: Dict[int, Tuple[int, int]]
    ) -> List[Tuple[List[int], List[int]]]:
        n = len(var_coordinate)

        boundary_left = [0] * n  # Z logical: connects left (rough) to right (rough)
        boundary_top = [0] * n  # X logical: connects top (smooth) to bottom (smooth)

        for i, (r, c) in var_coordinate.items():
            # Z logical: data qubits on the left boundary (c=0, r even)
            if c == 0 and r % 2 == 0:
                boundary_left[i] = 1
            # X logical: data qubits on the top boundary (r=0, c even)
            if r == 0 and c % 2 == 0:
                boundary_top[i] = 1

        # One logical qubit for planar surface code.
        # Return as (X, Z) where X is on top/bottom (smooth) and Z is on left/right (rough)
        return [(boundary_top, boundary_left)]

    @classmethod
    def from_distance(
        cls,
        distance: int,
        code_name: str = "surface",
        system: tuple[int, int] | None = None,
    ) -> "SurfaceCode":
        """Construct a planar surface code of the given distance.

        Parameters
        ----------
        distance : int
            Code distance (>= 2).
        code_name : str, optional
            Human-readable name for the code.
        system : tuple[int, int] | None, optional
            When provided, all node coordinates are extended to 4D as
            ``(x, y, system[0], system[1])``.  This lets multiple codes live in
            distinct sub-plots of the ``TannerGraphVisualizer`` 4D layout.
        """
        hx, hz, var_coordinate, check_coordinate = cls._build_css_pcm(distance)
        logical_qubits = cls._build_default_logicals(distance, var_coordinate)
        if system is not None:
            var_coordinate = {k: v + system for k, v in var_coordinate.items()}
            check_coordinate = {k: v + system for k, v in check_coordinate.items()}
        code = cls.from_css_pcm(
            code_name=code_name,
            hx=hx,
            hz=hz,
            logical_qubits=logical_qubits,
            var_coordinate=var_coordinate,
            check_coordinate=check_coordinate,
        )
        return code
