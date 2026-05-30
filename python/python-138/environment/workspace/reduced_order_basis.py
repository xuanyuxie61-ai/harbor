
import numpy as np
from typing import Tuple, List, Optional


class ReducedOrderBasisBuilder:

    def __init__(self, tolerance: float = 1.0e-12):
        self.tol = max(tolerance, 1.0e-15)

    def modified_gram_schmidt(
        self, snapshots: np.ndarray
    ) -> Tuple[np.ndarray, int, float]:
        N, m = snapshots.shape
        if N == 0 or m == 0:
            return np.zeros((N, 0)), 0, 0.0


        U = snapshots.copy().astype(float)
        basis_list = []

        for k in range(m):
            v = U[:, k].copy()
            for j in range(len(basis_list)):
                phi = basis_list[j]
                coeff = np.dot(v, phi)
                v -= coeff * phi
            norm_v = np.linalg.norm(v)
            if norm_v > self.tol:
                phi_k = v / norm_v
                basis_list.append(phi_k)

        rank = len(basis_list)
        if rank == 0:
            return np.zeros((N, 0)), 0, 0.0

        basis = np.column_stack(basis_list)

        G = basis.T @ basis
        I = np.eye(rank)
        orth_error = np.linalg.norm(G - I, "fro")

        return basis, rank, orth_error

    def compute_pod_modes_svd(
        self, snapshots: np.ndarray, n_modes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        N, m = snapshots.shape
        if N == 0 or m == 0:
            return np.zeros((N, 0)), np.array([]), 0.0

        U_mat, s, Vh = np.linalg.svd(snapshots, full_matrices=False)
        total_energy = np.sum(s ** 2)
        if total_energy < 1.0e-20:
            return U_mat[:, :0], s[:0], 0.0

        if n_modes is None:

            cum_energy = np.cumsum(s ** 2) / total_energy
            n_modes = int(np.searchsorted(cum_energy, 0.999)) + 1
            n_modes = min(n_modes, len(s))

        n_modes = min(n_modes, len(s))
        pod_basis = U_mat[:, :n_modes]
        energy_ratio = np.sum(s[:n_modes] ** 2) / total_energy
        return pod_basis, s[:n_modes], energy_ratio

    def project_onto_basis(
        self, field: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        if basis.size == 0:
            return np.array([])
        coeffs = basis.T @ field
        return coeffs

    def reconstruct_from_basis(
        self, coeffs: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        if basis.size == 0 or len(coeffs) == 0:
            return np.zeros(basis.shape[0])
        return basis @ coeffs

    def compute_reduction_error(
        self, snapshots: np.ndarray, basis: np.ndarray
    ) -> float:
        if basis.size == 0:
            return 1.0
        _, m = snapshots.shape
        total_err = 0.0
        for j in range(m):
            u = snapshots[:, j]
            u_proj = self.reconstruct_from_basis(
                self.project_onto_basis(u, basis), basis
            )
            norm_u = np.linalg.norm(u)
            if norm_u > 1.0e-14:
                total_err += np.linalg.norm(u - u_proj) / norm_u
        return total_err / m

    def build_operator_rom(
        self, A_full: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        if basis.size == 0:
            return np.zeros((0, 0))
        A_rom = basis.T @ A_full @ basis
        return A_rom
