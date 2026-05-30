
import numpy as np
from typing import Tuple


def compute_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        raise ValueError("空矩阵")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt


def low_rank_approximation(
    A: np.ndarray, rank: int
) -> Tuple[np.ndarray, float, float]:
    m, n = A.shape
    rank = max(1, min(rank, min(m, n)))
    U, s, Vt = compute_svd(A)

    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]

    A_approx = U_r @ np.diag(s_r) @ Vt_r

    compression = (m * rank + rank + rank * n) / (m * n)
    energy = s[:rank].sum() / s.sum() if s.sum() > 0 else 0.0

    return A_approx, compression, energy


def pod_basis_from_snapshots(
    snapshots: np.ndarray, n_modes: int
) -> Tuple[np.ndarray, np.ndarray]:
    n_dof, n_snap = snapshots.shape
    n_modes = max(1, min(n_modes, min(n_dof, n_snap)))


    mean_state = snapshots.mean(axis=1, keepdims=True)
    centered = snapshots - mean_state

    U, s, _ = compute_svd(centered)
    basis = U[:, :n_modes]
    return basis, s[:n_modes]


def project_jacobian_to_reduced_space(
    J_full: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    if J_full.shape[0] != basis.shape[0] or J_full.shape[1] != basis.shape[0]:
        raise ValueError("Jacobian 维度与基函数维度不匹配")
    return basis.T @ J_full @ basis


def svd_bw(m: int, n: int, r: int, U: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
    if r < 1:
        return np.zeros((m, n))
    U_r = U[:, :r]
    S_r = S[:r, :r] if S.ndim == 2 else np.diag(S[:r])
    V_r = V[:, :r]
    return U_r @ S_r @ V_r.T


def apply_mor_to_drift_diffusion(
    n_spatial: int = 50,
    n_time_snapshots: int = 20,
    n_pod_modes: int = 5,
) -> dict:

    x = np.linspace(0, 1, n_spatial)
    snapshots = np.zeros((n_spatial, n_time_snapshots))
    for k in range(n_time_snapshots):

        t = k * 0.1 + 0.01
        snapshots[:, k] = np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)

    basis, s = pod_basis_from_snapshots(snapshots, n_pod_modes)


    reconstruction = basis @ (basis.T @ snapshots)
    rel_error = np.linalg.norm(reconstruction - snapshots) / np.linalg.norm(snapshots)


    J_full = np.diag(-2 * np.ones(n_spatial)) + np.diag(np.ones(n_spatial - 1), 1) + np.diag(
        np.ones(n_spatial - 1), -1
    )
    J_red = project_jacobian_to_reduced_space(J_full, basis)

    return {
        "n_pod_modes": n_pod_modes,
        "singular_values": s.tolist(),
        "relative_reconstruction_error": float(rel_error),
        "reduced_jacobian_shape": J_red.shape,
        "reduced_jacobian_condition_number": float(np.linalg.cond(J_red)) if J_red.size > 0 else np.inf,
        "compression_ratio": (n_spatial * n_pod_modes + n_pod_modes + n_pod_modes * n_time_snapshots) / (
            n_spatial * n_time_snapshots
        ),
    }


if __name__ == "__main__":

    A = np.random.rand(50, 30)
    U, s, Vt = compute_svd(A)
    A_approx, comp, energy = low_rank_approximation(A, 5)
    err = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
    print(f"秩-5 近似相对误差: {err:.3e}, 压缩比: {comp:.3f}, 能量占比: {energy:.4f}")


    mor_result = apply_mor_to_drift_diffusion()
    print(f"POD 降阶误差: {mor_result['relative_reconstruction_error']:.3e}")
    print(f"降阶 Jacobian 条件数: {mor_result['reduced_jacobian_condition_number']:.3e}")
