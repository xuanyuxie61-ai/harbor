"""
rom_pod.py
降阶模型（POD/SVD）模块

融合种子项目:
- 1184_svd_basis: SVD降阶基提取（POD/ROM）

科学应用: 从大量软体机器人形状快照中提取降阶基，实现实时运动学
"""

import numpy as np
from typing import Tuple, Optional


def compute_svd_basis(snapshot_matrix: np.ndarray,
                      basis_num: int,
                      subtract_mean: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从快照矩阵计算SVD降阶基 — 基于种子项目1184_svd_basis

    快照矩阵 A: (M, N)，每列是一个展平的解向量
    A = U * S * V^T

    返回前 basis_num 个左奇异向量作为POD基

    参数:
        snapshot_matrix: (M, N) 快照矩阵
        basis_num: 保留的基函数个数
        subtract_mean: 是否减去均值

    返回:
        basis: (M, basis_num) POD基向量
        singular_values: (basis_num,) 奇异值
        mean_vector: (M,) 均值向量（若subtract_mean=True）
    """
    if snapshot_matrix.ndim != 2:
        raise ValueError("snapshot_matrix must be 2D")

    M, N = snapshot_matrix.shape
    basis_num = min(basis_num, M, N)

    mean_vector = np.zeros(M)
    A = snapshot_matrix.copy()

    if subtract_mean:
        mean_vector = np.mean(A, axis=1)
        A = A - mean_vector.reshape(-1, 1)

    # 经济型SVD
    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD失败时使用随机化方法
        U, S, Vt = randomized_svd(A, basis_num)

    basis = U[:, :basis_num]
    singular_values = S[:basis_num]

    return basis, singular_values, mean_vector


def randomized_svd(A: np.ndarray, k: int, p: int = 5, q: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    随机化SVD（用于大规模矩阵）

    步骤:
        1. 生成随机矩阵 Omega: (n, k+p)
        2. Y = A @ Omega
        3. QR分解 Y = Q @ R
        4. B = Q^T @ A
        5. 对B进行SVD
    """
    m, n = A.shape
    k = min(k, min(m, n))

    # 随机采样矩阵
    np.random.seed(42)
    Omega = np.random.randn(n, k + p)

    # 幂迭代提高精度
    Y = A @ Omega
    for _ in range(q):
        Y = A @ (A.T @ Y)

    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A

    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    return U[:, :k], S[:k], Vt[:k, :]


def project_onto_basis(field: np.ndarray, basis: np.ndarray,
                       mean_vector: Optional[np.ndarray] = None) -> np.ndarray:
    """
    将场投影到POD基上，得到模态系数

    coefficients = basis^T @ (field - mean)
    """
    if mean_vector is not None:
        field = field - mean_vector
    coeffs = basis.T @ field
    return coeffs


def reconstruct_from_basis(coefficients: np.ndarray, basis: np.ndarray,
                           mean_vector: Optional[np.ndarray] = None) -> np.ndarray:
    """
    从模态系数重构场

    field = basis @ coefficients + mean
    """
    field = basis @ coefficients
    if mean_vector is not None:
        field = field + mean_vector
    return field


def pod_galerkin_rom(M_mass: np.ndarray, K_stiff: np.ndarray, F_force: np.ndarray,
                     basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    POD-Galerkin降阶模型

    将完整阶系统投影到POD基上:
        M_rom = basis^T @ M @ basis
        K_rom = basis^T @ K @ basis
        F_rom = basis^T @ F

    返回降阶质量矩阵、刚度矩阵、力向量
    """
    M_rom = basis.T @ M_mass @ basis
    K_rom = basis.T @ K_stiff @ basis
    F_rom = basis.T @ F_force
    return M_rom, K_rom, F_rom


def generate_snapshots_soft_robot(L: float, Ns: int, n_snapshots: int,
                                  material_params: dict) -> np.ndarray:
    """
    生成软体机器人的形状快照集合

    通过改变驱动曲率分布生成不同的形状
    """
    from cosserat_core import forward_kinematics_cosserat

    n_nodes = Ns + 1
    M = n_nodes * 3  # 每节点3DOF

    snapshots = np.zeros((M, n_snapshots))
    rng = np.random.RandomState(42)

    for i in range(n_snapshots):
        # TODO: Hole 3 — 实现快照生成循环
        # 需要随机生成曲率分布 kappa(s)，调用 forward_kinematics_cosserat
        # 计算中心线 r(s)，并将展平后的结果存入 snapshots[:, i]
        # 注意: 曲率分布应包含至少两个频率分量（弯曲+扭转）
        raise NotImplementedError("Hole 3: 实现快照生成循环")

    return snapshots


def energy_fraction(singular_values: np.ndarray) -> np.ndarray:
    """
    计算累积能量分数

    energy_k = sum_{i=1}^k s_i^2 / sum_{i=1}^N s_i^2
    """
    total = np.sum(singular_values ** 2)
    if total < 1e-14:
        return np.ones(len(singular_values))
    cumsum = np.cumsum(singular_values ** 2)
    return cumsum / total


def optimal_basis_size(singular_values: np.ndarray,
                       threshold: float = 0.99) -> int:
    """
    根据能量阈值确定最优POD基维数
    """
    energy = energy_fraction(singular_values)
    size = np.searchsorted(energy, threshold) + 1
    return min(size, len(singular_values))
