
import numpy as np


class SVDRomCompressor:

    def __init__(self, snapshot_matrix):
        self.S = np.asarray(snapshot_matrix, dtype=float)
        if self.S.ndim != 2:
            raise ValueError("snapshot_matrix must be 2D.")
        self.N_space, self.N_time = self.S.shape
        self.U = None
        self.Sigma = None
        self.Vt = None
        self.singular_values = None
        self._decomposed = False

    def decompose(self):
        if self._decomposed:
            return self

        self.U, s, self.Vt = np.linalg.svd(self.S, full_matrices=False)
        self.Sigma = np.diag(s)
        self.singular_values = s
        self._decomposed = True
        return self

    def low_rank_approximation(self, rank):
        if not self._decomposed:
            self.decompose()

        rank = int(rank)
        if rank < 1:
            raise ValueError("rank must be positive.")
        r_eff = min(rank, len(self.singular_values))

        Ur = self.U[:, :r_eff]
        Sr = np.diag(self.singular_values[:r_eff])
        Vtr = self.Vt[:r_eff, :]
        S_approx = Ur @ Sr @ Vtr


        frob_original = np.linalg.norm(self.S, 'fro')
        if frob_original > 0.0:
            rel_error = np.linalg.norm(self.S - S_approx, 'fro') / frob_original
        else:
            rel_error = 0.0


        compression = (self.N_space * r_eff + r_eff + r_eff * self.N_time) / (
            self.N_space * self.N_time)

        return S_approx, rel_error, compression

    def cumulative_energy(self):
        if not self._decomposed:
            self.decompose()
        s2 = self.singular_values ** 2
        total = np.sum(s2)
        if total <= 0.0:
            return np.zeros_like(s2)
        cumsum = np.cumsum(s2) / total
        return cumsum

    def find_optimal_rank(self, threshold=0.99):
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1].")
        cum_energy = self.cumulative_energy()
        rank = np.searchsorted(cum_energy, threshold, side='left') + 1
        return int(min(rank, len(self.singular_values)))

    def pod_modes(self, n_modes):
        if not self._decomposed:
            self.decompose()
        n_modes = min(int(n_modes), self.U.shape[1])
        return self.U[:, :n_modes].copy(), self.singular_values[:n_modes].copy()

    def galerkin_projection_rhs(self, n_modes, full_rhs_func, current_state):
        phi, _ = self.pod_modes(n_modes)

        full_state = phi @ current_state

        rhs_full = full_rhs_func(full_state)

        a_dot = phi.T @ rhs_full
        return a_dot

    def reconstruct_field(self, modal_coefficients):
        n_modes = modal_coefficients.size
        phi, _ = self.pod_modes(n_modes)
        return phi @ modal_coefficients


def svd_blackwhite_approx(m, n, rank, U, Sigma, Vt):
    rank = int(rank)
    if np.ndim(Sigma) == 1:
        s = Sigma[:rank]
        approx = U[:, :rank] @ np.diag(s) @ Vt[:rank, :]
    else:
        approx = U[:, :rank] @ Sigma[:rank, :rank] @ Vt[:rank, :]
    return approx


class DynamicModeDecomposition:

    def __init__(self, snapshot_matrix):
        self.S = np.asarray(snapshot_matrix, dtype=float)
        if self.S.ndim != 2:
            raise ValueError("snapshot_matrix must be 2D.")

    def compute_modes(self, rank=None):
        X1 = self.S[:, :-1]
        X2 = self.S[:, 1:]

        U, s, Vt = np.linalg.svd(X1, full_matrices=False)
        if rank is not None:
            rank = min(int(rank), len(s))
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]


        S_inv = np.diag(1.0 / s)
        A_tilde = U.T @ X2 @ Vt.T @ S_inv

        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

        dmd_modes = X2 @ Vt.T @ S_inv @ eigenvectors

        return eigenvalues, dmd_modes
