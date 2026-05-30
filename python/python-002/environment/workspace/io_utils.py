# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, Any, Optional, Tuple


class IOUtils:

    @staticmethod
    def serialize_stellar_model(model_data: Dict[str, Any], filename: str):
        np.savez(filename, **model_data)

    @staticmethod
    def deserialize_stellar_model(filename: str) -> Dict[str, Any]:
        data = np.load(filename, allow_pickle=True)
        return {k: data[k] for k in data.files}

    @staticmethod
    def write_evolution_track(times: np.ndarray, luminosities: np.ndarray,
                              radii: np.ndarray, temperatures: np.ndarray,
                              filename: str):
        header = "# time[yr]  L[L_sun]  R[R_sun]  Teff[K]"
        data = np.column_stack([times, luminosities, radii, temperatures])
        np.savetxt(filename, data, header=header, fmt='%.6e')

    @staticmethod
    def read_evolution_track(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.loadtxt(filename)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    @staticmethod
    def grid_to_mass_coordinates(node_r: np.ndarray, element_nodes: np.ndarray,
                                 node_mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        node_r = np.asarray(node_r, dtype=np.float64)
        node_mass = np.asarray(node_mass, dtype=np.float64)
        element_nodes = np.asarray(element_nodes, dtype=int)

        max_idx = element_nodes.max()
        if max_idx >= len(node_r):
            element_nodes = element_nodes - 1
        return node_mass, element_nodes

    @staticmethod
    def magic_square(n: int) -> np.ndarray:
        if n < 3 or n % 2 == 0:

            return np.arange(1, n * n + 1).reshape(n, n)
        M = np.zeros((n, n), dtype=int)
        i, j = 0, n // 2
        for num in range(1, n * n + 1):
            M[i, j] = num
            new_i, new_j = (i - 1) % n, (j + 1) % n
            if M[new_i, new_j] != 0:
                i = (i + 1) % n
            else:
                i, j = new_i, new_j
        return M

    @staticmethod
    def test_matrix_condition(n: int = 5) -> Tuple[np.ndarray, float]:
        M = IOUtils.magic_square(n).astype(np.float64)
        cond = np.linalg.cond(M)
        return M, cond
