# -*- coding: utf-8 -*-
"""
structure_model.py
==================
Structure model for a multi-story building with base isolation.

Core science:
  - Assembles mass matrix M, stiffness matrix K, and Rayleigh damping C
    for a shear-building model with n_dof degrees of freedom.
  - Incorporates 3-D node coordinate management (from xyz_display seed).
  - FEM-based consistent mass / stiffness formulation inspired by
    the 1-D reaction-diffusion finite-element assembly (377_fem_neumann).

Physical formulas:
  - Story stiffness:   k_i = (12 * E * I) / h_i^3      [N/m]
  - Story mass:        m_i = (w_i * h_i * A_i) / g     [kg]
  - Rayleigh damping:  C = alpha * M + beta * K
    where alpha = 2 * zeta * omega_1 * omega_2 / (omega_1 + omega_2)
          beta  = 2 * zeta / (omega_1 + omega_2)
"""

import numpy as np
from typing import Tuple


class ShearBuildingModel:
    """
    n-story shear building with base isolation layer.
    
    Parameters
    ----------
    n_story : int
        Number of stories above isolation layer.
    story_heights : np.ndarray, shape (n_story,)
        Height of each story [m].
    story_masses : np.ndarray, shape (n_story,)
        Mass of each story [kg].
    story_stiffness : np.ndarray, shape (n_story,)
        Lateral stiffness of each story [N/m].
    damping_ratio : float
        Target damping ratio (default 0.05 for 5% critical damping).
    """

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
        self.n_dof = n_story + 1   # +1 for isolation layer (base)

        # Default parameters: 10-story reinforced concrete building
        if story_heights is None:
            story_heights = np.full(n_story, 3.5)   # 3.5 m per story
        if story_masses is None:
            # Typical floor mass: 800 tons to 1200 tons
            story_masses = np.linspace(1.2e6, 8.0e5, n_story)
        if story_stiffness is None:
            # Stiffness decreases from bottom to top
            story_stiffness = np.linspace(8.0e8, 2.0e8, n_story)

        self.story_heights = np.asarray(story_heights, dtype=float)
        self.story_masses = np.asarray(story_masses, dtype=float)
        self.story_stiffness = np.asarray(story_stiffness, dtype=float)
        self.damping_ratio = float(damping_ratio)

        # Validate dimensions
        if self.story_heights.shape != (n_story,):
            raise ValueError("story_heights must have shape (n_story,)")
        if self.story_masses.shape != (n_story,):
            raise ValueError("story_masses must have shape (n_story,)")
        if self.story_stiffness.shape != (n_story,):
            raise ValueError("story_stiffness must have shape (n_story,)")

        # Build structural matrices
        self._build_mass_matrix()
        self._build_stiffness_matrix()
        self._build_damping_matrix()
        self._build_influence_vector()
        self._build_node_coordinates()

    # ------------------------------------------------------------------ #
    # Mass matrix (lumped + consistent FEM formulation)
    # ------------------------------------------------------------------ #
    def _build_mass_matrix(self):
        """
        Assemble the global mass matrix M.
        
        For a shear-building model, the lumped mass matrix is diagonal:
          M_ii = mass of floor i
        
        We also add a small consistent mass coupling (off-diagonal)
        inspired by the FEM mass matrix from the reaction-diffusion seed:
          M_consistent = (h/6) * [[2, 1], [1, 2]]
        """
        n = self.n_dof
        M = np.zeros((n, n), dtype=float)

        # Isolation layer mass (typically large concrete base slab)
        M[0, 0] = 2.0e6   # 2000 tons base slab

        # Story masses
        for i in range(self.n_story):
            idx = i + 1
            M[idx, idx] = self.story_masses[i]

        # Small consistent-mass off-diagonal coupling (numerical stability)
        # M_{i,i+1} = 0.01 * sqrt(M_{ii} * M_{i+1,i+1}) to avoid zero modes
        for i in range(n - 1):
            coupling = 0.01 * np.sqrt(M[i, i] * M[i + 1, i + 1])
            M[i, i + 1] = coupling
            M[i + 1, i] = coupling

        self.M = M

    # ------------------------------------------------------------------ #
    # Stiffness matrix (tridiagonal shear-building)
    # ------------------------------------------------------------------ #
    def _build_stiffness_matrix(self):
        """
        Assemble the global stiffness matrix K for a shear-building.
        
        The story stiffness k_i relates inter-story drift to shear force:
          V_i = k_i * (u_i - u_{i-1})
        
        The global stiffness matrix is tridiagonal:
          K_{ii}     = k_i + k_{i+1}
          K_{i,i+1}  = -k_{i+1}
          K_{i+1,i}  = -k_{i+1}
        
        Boundary condition at base: the isolation layer stiffness is
        treated separately via the isolation_bearing module.
        """
        n = self.n_dof
        K = np.zeros((n, n), dtype=float)

        # Isolation layer to first story
        k_iso = 1.5e7    # Initial isolation stiffness [N/m]
        k_1 = self.story_stiffness[0]

        K[0, 0] = k_iso + k_1
        K[0, 1] = -k_1
        K[1, 0] = -k_1

        # Intermediate stories
        for i in range(1, self.n_story):
            idx = i
            k_below = self.story_stiffness[i - 1]
            k_above = self.story_stiffness[i]
            K[idx, idx] = k_below + k_above
            K[idx, idx + 1] = -k_above
            K[idx + 1, idx] = -k_above

        # Top story (only below stiffness)
        K[self.n_story, self.n_story] = self.story_stiffness[-1]

        self.K = K
        self.K_elastic = K.copy()   # Store elastic stiffness for reference

    # ------------------------------------------------------------------ #
    # Rayleigh damping
    # ------------------------------------------------------------------ #
    def _build_damping_matrix(self):
        """
        Build Rayleigh damping matrix:  C = alpha * M + beta * K
        
        The coefficients alpha, beta are chosen so that the damping ratio
        is approximately zeta at two target frequencies omega_1 and omega_2:
          | 1/omega_1   omega_1 | |alpha|   |2*zeta|
          | 1/omega_2   omega_2 | |beta | = |2*zeta|
        
        Solving:
          alpha = 2*zeta*omega_1*omega_2 / (omega_1 + omega_2)
          beta  = 2*zeta / (omega_1 + omega_2)
        """
        # TODO: Hole 4 - Implement Rayleigh damping matrix construction
        # Compute alpha, beta from natural frequencies and damping ratio,
        # then build C = alpha * M + beta * K
        pass

    def _compute_natural_frequencies(self) -> np.ndarray:
        """Compute undamped natural frequencies (rad/s) from K and M."""
        # Solve generalized eigenvalue problem  K * phi = omega^2 * M * phi
        # For symmetric positive-definite matrices, use scipy if available,
        # otherwise use numpy's eigh with mass-orthogonal transformation.
        try:
            # K and M are symmetric; M should be positive definite
            # We solve M^{-1/2} K M^{-1/2} v = lambda v
            M_sqrt = np.diag(np.sqrt(np.diag(self.M)))
            M_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.M)))
            K_tilde = M_inv_sqrt @ self.K @ M_inv_sqrt
            eigvals = np.linalg.eigvalsh(K_tilde)
            # Filter out numerical noise (negative tiny values)
            eigvals = np.where(eigvals < 0, 0, eigvals)
            omegas = np.sqrt(eigvals)
            omegas = np.sort(omegas)
        except np.linalg.LinAlgError:
            # Fallback: approximate with diagonal M
            omegas = np.sqrt(np.diag(self.K) / np.diag(self.M))
            omegas = np.sort(omegas)
        return omegas

    # ------------------------------------------------------------------ #
    # Influence vector (ground motion coupling)
    # ------------------------------------------------------------------ #
    def _build_influence_vector(self):
        """
        Influence vector Gamma maps ground acceleration to nodal forces:
          F_eq(t) = -M * Gamma * u_g''(t)
        
        For a lumped-mass shear building, all DOFs move with the ground,
        so Gamma = [1, 1, ..., 1]^T.
        """
        self.Gamma = np.ones(self.n_dof, dtype=float)

    # ------------------------------------------------------------------ #
    # 3-D node coordinates (from xyz_display seed idea)
    # ------------------------------------------------------------------ #
    def _build_node_coordinates(self):
        """
        Construct 3-D node coordinates for the building.
        
        Nodes are placed at the center of each floor.  The z-coordinate
        increases with story height.  We use a simple rectangular footprint
        for geometric properties used in quadrature and stress analysis.
        """
        n = self.n_dof
        coords = np.zeros((n, 3), dtype=float)

        # Base isolation layer at z = 0
        coords[0, :] = [0.0, 0.0, 0.0]

        # Cumulative height for stories
        z = 0.0
        for i in range(self.n_story):
            z += self.story_heights[i]
            coords[i + 1, :] = [0.0, 0.0, z]

        self.node_coords = coords
        self.total_height = z

    # ------------------------------------------------------------------ #
    # Public getters
    # ------------------------------------------------------------------ #
    def get_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (M, C, K) matrices."""
        return self.M.copy(), self.C.copy(), self.K.copy()

    def get_influence_vector(self) -> np.ndarray:
        """Return influence vector Gamma."""
        return self.Gamma.copy()

    def get_node_coordinates(self) -> np.ndarray:
        """Return (n_dof, 3) node coordinate array."""
        return self.node_coords.copy()

    def get_natural_frequencies(self) -> np.ndarray:
        """Return sorted natural frequencies [rad/s]."""
        return self._compute_natural_frequencies()

    def update_isolation_stiffness(self, k_iso: float):
        """
        Update the isolation layer stiffness (used during nonlinear analysis
        or parameter optimization).  This modifies K[0,0], K[0,1], K[1,0].
        """
        if k_iso <= 0:
            raise ValueError("Isolation stiffness must be positive")
        k_1 = self.story_stiffness[0]
        self.K[0, 0] = k_iso + k_1
        self.K[0, 1] = -k_1
        self.K[1, 0] = -k_1
        # Rebuild damping since K changed
        self._build_damping_matrix()
