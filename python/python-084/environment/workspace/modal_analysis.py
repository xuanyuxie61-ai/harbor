# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional


class ModalAnalysis:

    def __init__(self, M: np.ndarray, K: np.ndarray, n_modes: Optional[int] = None):
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.n_dof = self.M.shape[0]

        if n_modes is None:
            n_modes = self.n_dof
        self.n_modes = min(n_modes, self.n_dof)

        self._solve_eigenproblem()
        self._normalize_modes()
        self._compute_participation()




    def _solve_eigenproblem(self):
        M = self.M
        K = self.K


        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError as e:

            eps = 1e-10 * np.max(np.diag(M))
            M_reg = M + eps * np.eye(self.n_dof)
            L = np.linalg.cholesky(M_reg)


        L_inv = np.linalg.inv(L)
        A = L_inv @ K @ L_inv.T


        eigvals, eigvecs = np.linalg.eigh(A)


        eigvals = np.where(eigvals < 0, 0, eigvals)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]


        self.omega = np.sqrt(eigvals[:self.n_modes])
        self.period = np.where(self.omega > 0, 2.0 * np.pi / self.omega, np.inf)


        self.phi = (L_inv.T @ eigvecs)[:, :self.n_modes]




    def _normalize_modes(self):
        for i in range(self.n_modes):
            phi_i = self.phi[:, i]
            m_i = float(phi_i @ self.M @ phi_i)
            if m_i > 0:
                self.phi[:, i] = phi_i / np.sqrt(m_i)
            else:
                self.phi[:, i] = 0.0




    def _compute_participation(self):
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




    def truncation_error(self, target_ratio: float = 0.9) -> int:
        n_needed = np.searchsorted(self.meff_ratio, target_ratio, side="left") + 1
        return int(min(n_needed, self.n_modes))




    def spectral_quantities(
        self,
        spectrum_func,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S_a = np.array([spectrum_func(T) for T in self.period], dtype=float)
        S_v = np.where(self.omega > 0, S_a / self.omega, 0.0)
        S_d = np.where(self.omega > 0, S_a / (self.omega ** 2), 0.0)
        return S_a, S_v, S_d




    def peak_response_srss(self, S_d: np.ndarray) -> np.ndarray:
        n_dof = self.n_dof
        u_peak_sq = np.zeros(n_dof, dtype=float)
        for i in range(self.n_modes):
            u_mode = self.gamma[i] * self.phi[:, i] * S_d[i]
            u_peak_sq += u_mode ** 2
        return np.sqrt(u_peak_sq)

    def get_modal_matrix(self) -> np.ndarray:
        return self.phi.copy()

    def get_natural_frequencies(self) -> np.ndarray:
        return self.omega.copy()

    def get_periods(self) -> np.ndarray:
        return self.period.copy()
