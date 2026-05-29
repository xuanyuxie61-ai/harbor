# -*- coding: utf-8 -*-
"""
modal_analysis.py
=================
Modal analysis of the building structure using generalized eigenvalue
solution, incorporating ideas from the Chladni figures seed project.

Core science:
  - Generalized eigenvalue problem:  (K - omega^2 M) * phi = 0
  - Modal mass normalization:        phi_i^T * M * phi_i = 1
  - Modal participation factor:      Gamma_i = phi_i^T * M * Gamma
  - Effective modal mass:            m_eff,i = Gamma_i^2

The Chladni figures seed (172) solves a biharmonic eigenvalue problem for
plate vibration.  Here we adapt the discrete operator philosophy to the
shear-building stiffness/mass matrices, extracting modal properties that
govern the seismic response.
"""

import numpy as np
from typing import Tuple, Optional


class ModalAnalysis:
    """
    Modal analysis for a multi-DOF structural system.
    
    Attributes
    ----------
    omega : np.ndarray
        Natural circular frequencies [rad/s], sorted ascending.
    period : np.ndarray
        Natural periods [s].
    phi : np.ndarray
        Mode shape matrix, column i is mode i, mass-normalized.
    gamma : np.ndarray
        Modal participation factors.
    meff : np.ndarray
        Effective modal masses.
    meff_ratio : np.ndarray
        Cumulative effective modal mass ratio.
    """

    def __init__(self, M: np.ndarray, K: np.ndarray, n_modes: Optional[int] = None):
        """
        Parameters
        ----------
        M : np.ndarray, shape (n_dof, n_dof)
            Mass matrix (symmetric positive definite).
        K : np.ndarray, shape (n_dof, n_dof)
            Stiffness matrix (symmetric positive semi-definite).
        n_modes : int, optional
            Number of modes to retain.  If None, all modes are computed.
        """
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.n_dof = self.M.shape[0]

        if n_modes is None:
            n_modes = self.n_dof
        self.n_modes = min(n_modes, self.n_dof)

        self._solve_eigenproblem()
        self._normalize_modes()
        self._compute_participation()

    # ------------------------------------------------------------------ #
    # Generalized eigenvalue solver
    # ------------------------------------------------------------------ #
    def _solve_eigenproblem(self):
        """
        Solve  K * phi = omega^2 * M * phi  by transforming to standard form.
        
        Transformation:
          M = L * L^T   (Cholesky factorization, from 989_r8po seed)
          Let v = L^T * phi, then
            (L^{-1} * K * L^{-T}) * v = omega^2 * v
        
        The matrix  A = L^{-1} * K * L^{-T}  is symmetric,
        so we use numpy.linalg.eigh for numerical stability.
        """
        M = self.M
        K = self.K

        # Cholesky factorization of M (M is SPD for structural systems)
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError as e:
            # Fallback: regularize M slightly if not strictly SPD
            eps = 1e-10 * np.max(np.diag(M))
            M_reg = M + eps * np.eye(self.n_dof)
            L = np.linalg.cholesky(M_reg)

        # Solve L^{-1} * K * L^{-T}
        L_inv = np.linalg.inv(L)
        A = L_inv @ K @ L_inv.T

        # Symmetric eigenvalue problem
        eigvals, eigvecs = np.linalg.eigh(A)

        # Filter numerical noise and sort
        eigvals = np.where(eigvals < 0, 0, eigvals)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Natural frequencies
        self.omega = np.sqrt(eigvals[:self.n_modes])
        self.period = np.where(self.omega > 0, 2.0 * np.pi / self.omega, np.inf)

        # Mode shapes in physical coordinates: phi = L^{-T} * v
        self.phi = (L_inv.T @ eigvecs)[:, :self.n_modes]

    # ------------------------------------------------------------------ #
    # Mass normalization
    # ------------------------------------------------------------------ #
    def _normalize_modes(self):
        """
        Normalize mode shapes such that:
          phi_i^T * M * phi_i = 1
        
        This is the standard modal mass normalization used in earthquake
        engineering.  With this normalization, the generalized mass is unity
        and the generalized stiffness equals omega_i^2.
        """
        for i in range(self.n_modes):
            phi_i = self.phi[:, i]
            m_i = float(phi_i @ self.M @ phi_i)
            if m_i > 0:
                self.phi[:, i] = phi_i / np.sqrt(m_i)
            else:
                self.phi[:, i] = 0.0

    # ------------------------------------------------------------------ #
    # Modal participation factors & effective masses
    # ------------------------------------------------------------------ #
    def _compute_participation(self):
        """
        Compute modal participation factors and effective modal masses.
        
        Participation factor for mode i:
          Gamma_i = phi_i^T * M * 1_vec
        
        Effective modal mass:
          m_eff,i = Gamma_i^2
        
        Cumulative mass ratio:
          sum(m_eff,i) / M_total  should approach 1.0 as more modes are included.
        """
        one_vec = np.ones(self.n_dof, dtype=float)
        self.gamma = np.zeros(self.n_modes, dtype=float)
        self.meff = np.zeros(self.n_modes, dtype=float)

        for i in range(self.n_modes):
            phi_i = self.phi[:, i]
            gamma_i = float(phi_i @ self.M @ one_vec)
            self.gamma[i] = gamma_i
            self.meff[i] = gamma_i ** 2

        M_total = float(one_vec @ self.M @ one_vec)
        self.meff_ratio = np.cumsum(self.meff) / M_total

    # ------------------------------------------------------------------ #
    # Modal truncation error estimate
    # ------------------------------------------------------------------ #
    def truncation_error(self, target_ratio: float = 0.9) -> int:
        """
        Return the minimum number of modes required to achieve the target
        cumulative effective mass ratio (commonly 90% per building codes).
        """
        n_needed = np.searchsorted(self.meff_ratio, target_ratio, side="left") + 1
        return int(min(n_needed, self.n_modes))

    # ------------------------------------------------------------------ #
    # Spectral displacement / acceleration (response spectrum concept)
    # ------------------------------------------------------------------ #
    def spectral_quantities(
        self,
        spectrum_func,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given a response spectrum function S_a(T), compute for each mode:
          S_di = S_ai / omega_i^2       (spectral displacement)
          S_vi = S_ai / omega_i         (spectral velocity)
          S_ai = spectrum_func(T_i)     (spectral acceleration)
        
        Parameters
        ----------
        spectrum_func : callable
            Function S_a(T) returning spectral acceleration [m/s^2].
        
        Returns
        -------
        S_a, S_v, S_d : np.ndarray
            Spectral acceleration, velocity, displacement per mode.
        """
        S_a = np.array([spectrum_func(T) for T in self.period], dtype=float)
        S_v = np.where(self.omega > 0, S_a / self.omega, 0.0)
        S_d = np.where(self.omega > 0, S_a / (self.omega ** 2), 0.0)
        return S_a, S_v, S_d

    # ------------------------------------------------------------------ #
    # Modal superposition peak response (SRSS combination)
    # ------------------------------------------------------------------ #
    def peak_response_srss(self, S_d: np.ndarray) -> np.ndarray:
        """
        Compute peak displacement response by SRSS (Square Root of Sum of
        Squares) modal combination:
          u_max = sqrt( sum_i ( Gamma_i * phi_i * S_di )^2 )
        """
        n_dof = self.n_dof
        u_peak_sq = np.zeros(n_dof, dtype=float)
        for i in range(self.n_modes):
            u_mode = self.gamma[i] * self.phi[:, i] * S_d[i]
            u_peak_sq += u_mode ** 2
        return np.sqrt(u_peak_sq)

    def get_modal_matrix(self) -> np.ndarray:
        """Return the mode shape matrix Phi (n_dof x n_modes)."""
        return self.phi.copy()

    def get_natural_frequencies(self) -> np.ndarray:
        """Return natural circular frequencies [rad/s]."""
        return self.omega.copy()

    def get_periods(self) -> np.ndarray:
        """Return natural periods [s]."""
        return self.period.copy()
