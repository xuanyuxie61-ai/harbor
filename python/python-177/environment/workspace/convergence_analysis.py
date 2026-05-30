# -*- coding: utf-8 -*-

import numpy as np


class ConvergenceAnalysis:

    @staticmethod
    def l2_error(phi_num, phi_exact, dx, dy):
        diff = phi_num - phi_exact
        return np.sqrt(np.sum(diff ** 2) * dx * dy)

    @staticmethod
    def linf_error(phi_num, phi_exact):
        return np.max(np.abs(phi_num - phi_exact))

    @staticmethod
    def h1_seminorm_error(phi_num, phi_exact, dx, dy):
        nx, ny = phi_num.shape
        diff = phi_num - phi_exact
        grad_sq = np.zeros_like(diff)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dx_diff = (diff[i + 1, j] - diff[i - 1, j]) / (2.0 * dx)
                dy_diff = (diff[i, j + 1] - diff[i, j - 1]) / (2.0 * dy)
                grad_sq[i, j] = dx_diff ** 2 + dy_diff ** 2

        return np.sqrt(np.sum(grad_sq) * dx * dy)

    @staticmethod
    def convergence_order(errors, resolutions):
        orders = []
        for i in range(len(errors) - 1):
            e1, e2 = errors[i], errors[i + 1]
            h1, h2 = resolutions[i], resolutions[i + 1]
            if e1 <= 0 or e2 <= 0 or h1 <= h2:
                orders.append(np.nan)
            else:
                p = np.log(e1 / e2) / np.log(h1 / h2)
                orders.append(p)
        return orders


class ReducedOrderModel:

    def __init__(self, snapshots):
        self.snapshots = np.asarray(snapshots, dtype=np.float64)
        self.mean_vec = None
        self.U = None
        self.S = None
        self.Vt = None
        self.coefficients = None

    def compute_pod_basis(self, energy_threshold=0.99):
        A = self.snapshots.copy()
        self.mean_vec = np.mean(A, axis=1)
        A_centered = A - self.mean_vec[:, np.newaxis]


        U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
        self.U = U
        self.S = S
        self.Vt = Vt


        total_energy = np.sum(S ** 2)
        cumsum = np.cumsum(S ** 2)
        r = np.searchsorted(cumsum / total_energy, energy_threshold) + 1
        r = max(1, min(r, len(S)))
        self.r = r
        self.Ur = U[:, :r]


        self.coefficients = self.Ur.T @ A_centered
        return self.Ur, self.S[:r]

    def reconstruct(self, mode_indices=None):
        if self.Ur is None:
            raise ValueError("Must call compute_pod_basis first")
        if mode_indices is None:
            Ur_use = self.Ur
            coeffs_use = self.coefficients
        else:
            Ur_use = self.Ur[:, mode_indices]
            coeffs_use = self.coefficients[mode_indices, :]

        A_recon = Ur_use @ coeffs_use + self.mean_vec[:, np.newaxis]
        return A_recon

    def project_state(self, phi):
        phi_centered = phi - self.mean_vec
        return self.Ur.T @ phi_centered

    def reconstruct_from_coefficients(self, alpha):
        return self.Ur @ alpha + self.mean_vec

    def get_mode_energy(self):
        if self.S is None:
            return None
        total = np.sum(self.S ** 2)
        return self.S ** 2 / total

    def reduced_galerkin_matrix(self, L_full):
        if self.Ur is None:
            raise ValueError("Must call compute_pod_basis first")
        L_reduced = self.Ur.T @ L_full @ self.Ur
        return L_reduced
