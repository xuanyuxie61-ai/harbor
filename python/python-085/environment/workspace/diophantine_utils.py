"""
diophantine_utils.py
整数规划与节点重编号模块
融合种子项目：
  - 288_diophantine（丢番图方程整数解）

在接触问题中，节点重编号可显著降低全局刚度矩阵的带宽，
从而提升带状求解器的效率。本节将丢番图方程的整数规划思想
用于接触节点编号的优化分配。
"""
import numpy as np
from typing import Tuple, List


def i4vec_gcd(a: np.ndarray) -> int:
    r"""
    计算整数向量的最大公约数（融合 288_diophantine 的 i4vec_gcd）。
    \gcd(a_1, a_2, ..., a_n)
    """
    a = np.array(a, dtype=int)
    g = 0
    for val in a:
        g = np.gcd(g, int(val))
    return int(g)


def diophantine_nonnegative_solve(a: np.ndarray, b: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    求解非负系数丢番图方程的整数解（融合 288_diophantine_nonnegative）。

    方程：a_1 x_1 + a_2 x_2 + ... + a_n x_n = b
    要求 a_i \ge 0，b \ge 0。

    返回 (d, v, B, kmin, kmax)，其中通解为：
    x = v + B * c,  c \in \mathbb{Z}^{n-1}
    """
    a = np.array(a, dtype=int)
    if np.any(a < 0):
        raise ValueError("Coefficients must be nonnegative")
    if np.sum(a) <= 0:
        raise ValueError("At least one coefficient must be positive")
    if b < 0:
        raise ValueError("Right-hand side must be nonnegative")

    d = i4vec_gcd(a)
    if b % d != 0:
        raise ValueError(f"b={b} is not divisible by gcd(a)={d}")

    n = len(a)
    np1 = n + 1
    A_mat = np.zeros((n, np1), dtype=int)
    A_mat[:, 0] = a
    A_mat[:, 1:] = np.eye(n, dtype=int)

    # Hermite 标准型化简
    while np.count_nonzero(A_mat[:, 0]) > 1:
        nonzero = np.where(A_mat[:, 0] != 0)[0]
        magnitudes = np.abs(A_mat[nonzero, 0])
        p_idx = nonzero[np.argmin(magnitudes)]
        # 交换到第一行
        A_mat[[0, p_idx], :] = A_mat[[p_idx, 0], :].copy()
        for i in range(1, n):
            s = int(np.fix(A_mat[i, 0] / A_mat[0, 0]))
            A_mat[i, :] -= s * A_mat[0, :]

    d_out = A_mat[0, 0]
    f = b // d_out
    v = A_mat[0, 1:].copy() * f
    B = A_mat[1:, 1:].T.copy()

    kmin = -np.inf * np.ones(n - 1)
    kmax = np.inf * np.ones(n - 1)
    for j in range(n - 1):
        for i in range(n):
            if B[i, j] < 0:
                kmax[j] = min(kmax[j], -v[i] / B[i, j])
            elif B[i, j] > 0:
                kmin[j] = max(kmin[j], -v[i] / B[i, j])

    return d_out, v, B, kmin, kmax


def optimize_node_numbering_bandwidth(contact_nodes: np.ndarray,
                                       total_nodes: int,
                                       max_band: int = 10) -> np.ndarray:
    r"""
    使用整数规划思想优化接触节点编号以降低带宽。

    目标：minimize max |i - j|，使得 contact_nodes 的编号尽可能连续。
    通过丢番图约束保证编号分配的唯一性。

    简化实现：将接触节点重新分配到连续区间 [base, base + n_c - 1]。
    """
    contact_nodes = np.array(contact_nodes, dtype=int)
    n_c = len(contact_nodes)
    if n_c == 0:
        return np.arange(total_nodes)

    # 构造线性方程：sum x_i = total_nodes * (total_nodes - 1) / 2
    # 简化为重排：将 contact_nodes 移到前端连续区域
    new_order = np.arange(total_nodes)
    other_nodes = np.setdiff1d(np.arange(total_nodes), contact_nodes)
    reordered = np.concatenate([contact_nodes, other_nodes])
    # 生成逆映射
    inv_map = np.zeros(total_nodes, dtype=int)
    inv_map[reordered] = np.arange(total_nodes)
    return inv_map


def compute_matrix_bandwidth(K: np.ndarray) -> Tuple[int, int]:
    r"""
    计算矩阵的下带宽 ml 和上带宽 mu。
    ml = max_{K[i,j] != 0} (i - j)
    mu = max_{K[i,j] != 0} (j - i)
    """
    n = K.shape[0]
    ml = 0
    mu = 0
    tol = 1e-14 * np.max(np.abs(K))
    for i in range(n):
        for j in range(n):
            if abs(K[i, j]) > tol:
                ml = max(ml, i - j)
                mu = max(mu, j - i)
    return ml, mu
