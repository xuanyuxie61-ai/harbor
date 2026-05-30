
import numpy as np


def compute_pod_basis(data_matrix, n_modes, subtract_mean=True):
    A = data_matrix.copy()
    m, n = A.shape
    mean_vec = None

    if subtract_mean:
        mean_vec = np.mean(A, axis=1, keepdims=True)
        A = A - mean_vec


    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    n_modes = min(n_modes, len(s))
    modes = U[:, :n_modes]
    singular_values = s[:n_modes]
    coefficients = np.diag(singular_values) @ Vt[:n_modes, :]

    return modes, singular_values, coefficients, mean_vec


def low_rank_approximation(data_matrix, rank):
    U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    A_r = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
    return A_r


def compression_ratio(m, n, rank):
    return (m * rank + rank + rank * n) / (m * n)


def cumulative_energy(singular_values):
    total = np.sum(singular_values ** 2)
    cum = np.cumsum(singular_values ** 2) / total
    return cum


def reconstruct_field(modes, coefficients, mean_vec=None):
    recon = modes @ coefficients
    if mean_vec is not None:
        recon = recon + mean_vec
    return recon


class ReducedOrderModel:
    def __init__(self, modes, singular_values, mean_vec=None):
        self.modes = modes
        self.singular_values = singular_values
        self.mean_vec = mean_vec
        self.n_modes = modes.shape[1]

    def project_to_reduced(self, field):
        if self.mean_vec is not None:
            field = field - self.mean_vec.flatten()
        return self.modes.T @ field

    def reconstruct_from_reduced(self, coeffs):
        recon = self.modes @ coeffs
        if self.mean_vec is not None:
            recon = recon + self.mean_vec.flatten()
        return recon

    def galilean_invariance_error(self, field, shifted_field):
        a1 = self.project_to_reduced(field)
        a2 = self.project_to_reduced(shifted_field)
        return np.linalg.norm(a1 - a2) / (np.linalg.norm(a1) + 1e-30)


def analyze_trajectory_pca(trajectories, n_modes=5):

    all_pos = np.vstack(trajectories)

    mean_pos = np.mean(all_pos, axis=0)
    centered = all_pos - mean_pos


    cov = centered.T @ centered / centered.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)


    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    cumulative_variance = np.cumsum(eigvals) / np.sum(eigvals)
    return eigvecs[:, :n_modes], eigvals[:n_modes], cumulative_variance[:n_modes]
