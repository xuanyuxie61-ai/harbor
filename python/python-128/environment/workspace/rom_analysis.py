
import numpy as np


def compute_pod_basis(data_matrix: np.ndarray, energy_threshold: float = 0.99):
    A = np.asarray(data_matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("compute_pod_basis: data_matrix 必须为二维数组")
    M, N = A.shape


    mean_vec = A.mean(axis=1, keepdims=True)
    A_centered = A - mean_vec


    try:
        U_full, sigma_full, Vt_full = np.linalg.svd(A_centered, full_matrices=False)
    except np.linalg.LinAlgError:

        U_full, sigma_full, Vt_full = np.linalg.svd(A_centered + 1e-12 * np.random.randn(M, N), full_matrices=False)


    sigma_sq = sigma_full ** 2
    total_energy = np.sum(sigma_sq)
    if total_energy < 1e-30:

        return np.zeros((M, 1)), np.array([0.0]), np.zeros((1, N)), 1, np.array([1.0])

    cum_energy = np.cumsum(sigma_sq) / total_energy
    L = int(np.searchsorted(cum_energy, energy_threshold)) + 1
    L = min(L, min(M, N))
    L = max(1, L)

    return U_full[:, :L], sigma_full[:L], Vt_full[:L, :], L, cum_energy[:L]


def reconstruct_from_pod(U, sigma, Vt):
    if sigma.size == 0:
        return np.zeros((U.shape[0], Vt.shape[1]))
    return U @ np.diag(sigma) @ Vt


def pod_galerkin_projection(rhs_func, U, initial_coeff, dt, n_steps):
    L = U.shape[1]
    a = np.asarray(initial_coeff, dtype=float).copy()
    history = np.zeros((L, n_steps + 1))
    history[:, 0] = a
    for n in range(n_steps):
        u_full = U @ a
        f_full = rhs_func(u_full)
        a = a + dt * (U.T @ f_full)
        history[:, n + 1] = a
    return history


class ChemotaxisROM:

    def __init__(self, snapshot_list=None):
        self.U = None
        self.sigma = None
        self.Vt = None
        self.L = 0
        self.mean_vec = None
        self.original_shape = None
        self.energy = None
        if snapshot_list is not None and len(snapshot_list) > 0:
            self.build_basis(snapshot_list)

    def build_basis(self, snapshot_list, energy_threshold=0.99):
        self.original_shape = snapshot_list[0].shape
        M = int(np.prod(self.original_shape))
        N = len(snapshot_list)
        A = np.zeros((M, N), dtype=float)
        for j, snap in enumerate(snapshot_list):
            A[:, j] = snap.flatten()
        self.U, self.sigma, self.Vt, self.L, self.energy = compute_pod_basis(A, energy_threshold)
        self.mean_vec = A.mean(axis=1)

    def reconstruct(self, coeff):
        if self.U is None:
            raise RuntimeError("ChemotaxisROM: 尚未构建 POD 基")
        coeff = np.asarray(coeff, dtype=float)
        flat = self.U @ coeff + self.mean_vec
        return flat.reshape(self.original_shape)

    def project(self, field):
        if self.U is None:
            raise RuntimeError("ChemotaxisROM: 尚未构建 POD 基")
        flat = np.asarray(field, dtype=float).flatten() - self.mean_vec
        return self.U.T @ flat

    def relative_error(self, field):
        a = self.project(field)
        recon = self.reconstruct(a)
        norm_true = np.linalg.norm(field)
        if norm_true < 1e-15:
            return 0.0
        return float(np.linalg.norm(field - recon) / norm_true)

    def summary(self):
        if self.U is None:
            return "ChemotaxisROM: 未构建基"
        lines = []
        lines.append("=" * 60)
        lines.append("Chemotaxis ROM 摘要")
        lines.append("  原始场维度 M = %d" % self.U.shape[0])
        lines.append("  保留模态数 L = %d" % self.L)
        lines.append("  压缩比 M/L = %.2f" % (self.U.shape[0] / max(1, self.L)))
        lines.append("  主导奇异值: " + ", ".join("%.4g" % s for s in self.sigma[:min(5, self.sigma.size)]))
        lines.append("  保留能量: %.4f" % (self.energy[-1] if self.energy.size > 0 else 0.0))
        lines.append("=" * 60)
        return "\n".join(lines)
