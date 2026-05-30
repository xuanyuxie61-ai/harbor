# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple


class ShearBuildingModel:

    def __init__(
        self,
        n_story: int = 10,
        story_heights: np.ndarray = None,
        story_masses: np.ndarray = None,
        story_stiffness: np.ndarray = None,
        damping_ratio: float = 0.05,
    ):
        if n_story < 1:
            raise ValueError("n_story must be >= 1")
        self.n_story = n_story
        self.n_dof = n_story + 1


        if story_heights is None:
            story_heights = np.full(n_story, 3.5)
        if story_masses is None:

            story_masses = np.linspace(1.2e6, 8.0e5, n_story)
        if story_stiffness is None:

            story_stiffness = np.linspace(8.0e8, 2.0e8, n_story)

        self.story_heights = np.asarray(story_heights, dtype=float)
        self.story_masses = np.asarray(story_masses, dtype=float)
        self.story_stiffness = np.asarray(story_stiffness, dtype=float)
        self.damping_ratio = float(damping_ratio)


        if self.story_heights.shape != (n_story,):
            raise ValueError("story_heights must have shape (n_story,)")
        if self.story_masses.shape != (n_story,):
            raise ValueError("story_masses must have shape (n_story,)")
        if self.story_stiffness.shape != (n_story,):
            raise ValueError("story_stiffness must have shape (n_story,)")


        self._build_mass_matrix()
        self._build_stiffness_matrix()
        self._build_damping_matrix()
        self._build_influence_vector()
        self._build_node_coordinates()




    def _build_mass_matrix(self):
        n = self.n_dof
        M = np.zeros((n, n), dtype=float)


        M[0, 0] = 2.0e6


        for i in range(self.n_story):
            idx = i + 1
            M[idx, idx] = self.story_masses[i]



        for i in range(n - 1):
            coupling = 0.01 * np.sqrt(M[i, i] * M[i + 1, i + 1])
            M[i, i + 1] = coupling
            M[i + 1, i] = coupling

        self.M = M




    def _build_stiffness_matrix(self):
        n = self.n_dof
        K = np.zeros((n, n), dtype=float)


        k_iso = 1.5e7
        k_1 = self.story_stiffness[0]

        K[0, 0] = k_iso + k_1
        K[0, 1] = -k_1
        K[1, 0] = -k_1


        for i in range(1, self.n_story):
            idx = i
            k_below = self.story_stiffness[i - 1]
            k_above = self.story_stiffness[i]
            K[idx, idx] = k_below + k_above
            K[idx, idx + 1] = -k_above
            K[idx + 1, idx] = -k_above


        K[self.n_story, self.n_story] = self.story_stiffness[-1]

        self.K = K
        self.K_elastic = K.copy()




    def _build_damping_matrix(self):



        pass

    def _compute_natural_frequencies(self) -> np.ndarray:



        try:


            M_sqrt = np.diag(np.sqrt(np.diag(self.M)))
            M_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.M)))
            K_tilde = M_inv_sqrt @ self.K @ M_inv_sqrt
            eigvals = np.linalg.eigvalsh(K_tilde)

            eigvals = np.where(eigvals < 0, 0, eigvals)
            omegas = np.sqrt(eigvals)
            omegas = np.sort(omegas)
        except np.linalg.LinAlgError:

            omegas = np.sqrt(np.diag(self.K) / np.diag(self.M))
            omegas = np.sort(omegas)
        return omegas




    def _build_influence_vector(self):
        self.Gamma = np.ones(self.n_dof, dtype=float)




    def _build_node_coordinates(self):
        n = self.n_dof
        coords = np.zeros((n, 3), dtype=float)


        coords[0, :] = [0.0, 0.0, 0.0]


        z = 0.0
        for i in range(self.n_story):
            z += self.story_heights[i]
            coords[i + 1, :] = [0.0, 0.0, z]

        self.node_coords = coords
        self.total_height = z




    def get_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.M.copy(), self.C.copy(), self.K.copy()

    def get_influence_vector(self) -> np.ndarray:
        return self.Gamma.copy()

    def get_node_coordinates(self) -> np.ndarray:
        return self.node_coords.copy()

    def get_natural_frequencies(self) -> np.ndarray:
        return self._compute_natural_frequencies()

    def update_isolation_stiffness(self, k_iso: float):
        if k_iso <= 0:
            raise ValueError("Isolation stiffness must be positive")
        k_1 = self.story_stiffness[0]
        self.K[0, 0] = k_iso + k_1
        self.K[0, 1] = -k_1
        self.K[1, 0] = -k_1

        self._build_damping_matrix()
