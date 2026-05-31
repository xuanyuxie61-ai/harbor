"""
model_reduction.py
基于种子项目 1187_svd_fingerprint (SVD decomposition and low-rank approximation)
改造为钙钛矿太阳能电池多物理场模型的降阶加速模块。

在太阳能电池的全器件模拟中，漂移-扩散方程离散后产生的大型非线性系统
Newton 迭代代价高昂。本模块利用 SVD 对 Jacobian 矩阵或解流形进行
低秩近似，构建投影基（POD/DEIM），实现模型降阶（MOR）。

核心公式：
  1. SVD 分解：A = U Σ V^T
     其中 U ∈ R^{m×m}, Σ ∈ R^{m×n}, V ∈ R^{n×n}
  2. 秩-r 近似：A_r = U_r Σ_r V_r^T
     压缩比：ρ = (m·r + r + r·n) / (m·n)
  3. 奇异值能量占比：η(r) = Σ_{i=1}^r σ_i / Σ_{i=1}^{min(m,n)} σ_i
  4. POD 基提取：从快照矩阵 S = [φ_1, φ_2, ..., φ_N] 中提取主导模态
       S = Φ Λ Ψ^T,  降阶基 B = Φ[:, :r]
  5. Galerkin 投影：
       d a/dt = B^T f(B a)
"""

import numpy as np
from typing import Tuple


def compute_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算矩阵 A 的 SVD 分解 A = U S Vt。
    对应原项目 svd_fingerprint 中的 svd 调用。
    """
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        raise ValueError("空矩阵")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt


def low_rank_approximation(
    A: np.ndarray, rank: int
) -> Tuple[np.ndarray, float, float]:
    """
    计算矩阵 A 的秩-r 低秩近似，返回近似矩阵、压缩比和能量占比。

    Parameters
    ----------
    A : (m, n) array
    rank : int

    Returns
    -------
    A_approx : (m, n) array
    compression_ratio : float
    energy_ratio : float
    """
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
    """
    从快照矩阵提取 POD 基。

    Parameters
    ----------
    snapshots : (n_dof, n_snap) array
        每个列为一个系统状态快照
    n_modes : int
        保留的 POD 模态数

    Returns
    -------
    basis : (n_dof, n_modes) array
        POD 基（列正交）
    singular_values : (n_modes,) array
        对应的奇异值
    """
    n_dof, n_snap = snapshots.shape
    n_modes = max(1, min(n_modes, min(n_dof, n_snap)))

    # 去均值
    mean_state = snapshots.mean(axis=1, keepdims=True)
    centered = snapshots - mean_state

    U, s, _ = compute_svd(centered)
    basis = U[:, :n_modes]
    return basis, s[:n_modes]


def project_jacobian_to_reduced_space(
    J_full: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """
    将全阶 Jacobian 投影到降阶空间：
      J_red = B^T J_full B

    Parameters
    ----------
    J_full : (n, n) array
    basis : (n, r) array

    Returns
    -------
    J_red : (r, r) array
    """
    if J_full.shape[0] != basis.shape[0] or J_full.shape[1] != basis.shape[0]:
        raise ValueError("Jacobian 维度与基函数维度不匹配")
    return basis.T @ J_full @ basis


def svd_bw(m: int, n: int, r: int, U: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    从 SVD 分量构建低秩近似矩阵（对应原项目 svd_bw）。
    """
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
    """
    演示对漂移-扩散方程的 POD 降阶。
    生成模拟快照 -> SVD -> 降阶 -> 验证近似误差。
    """
    # 模拟快照：稳态电势 + 瞬态演化
    x = np.linspace(0, 1, n_spatial)
    snapshots = np.zeros((n_spatial, n_time_snapshots))
    for k in range(n_time_snapshots):
        # 模拟扩散过程的解析近似解
        t = k * 0.1 + 0.01
        snapshots[:, k] = np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)

    basis, s = pod_basis_from_snapshots(snapshots, n_pod_modes)

    # 重建误差
    reconstruction = basis @ (basis.T @ snapshots)
    rel_error = np.linalg.norm(reconstruction - snapshots) / np.linalg.norm(snapshots)

    # Jacobian 投影示例
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
    # SVD 测试
    A = np.random.rand(50, 30)
    U, s, Vt = compute_svd(A)
    A_approx, comp, energy = low_rank_approximation(A, 5)
    err = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
    print(f"秩-5 近似相对误差: {err:.3e}, 压缩比: {comp:.3f}, 能量占比: {energy:.4f}")

    # MOR 测试
    mor_result = apply_mor_to_drift_diffusion()
    print(f"POD 降阶误差: {mor_result['relative_reconstruction_error']:.3e}")
    print(f"降阶 Jacobian 条件数: {mor_result['reduced_jacobian_condition_number']:.3e}")
