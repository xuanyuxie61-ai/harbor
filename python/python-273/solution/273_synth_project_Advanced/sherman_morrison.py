"""
sherman_morrison.py — Sherman-Morrison 秩1修正求解器
===================================================

融合种子项目:
  - 995_r8sm: Sherman-Morrison 矩阵库
    B = A - u*v', 求解 (A - u*v')x = b
    核心: alpha = 1/(1 - v'*w), x = x0 + alpha*beta*w
    其中 A*w = u, A*x0 = b, beta = w'*b

物理背景:
  在声子计算中，缺陷或杂质原子的引入等价于对完美晶格
  动力矩阵 D_0 的秩1(或低秩)修正:
    D_defect = D_0 - delta_D
  其中 delta_D = u * v^T 描述局域力常数变化。

  Sherman-Morrison 公式避免了重新对角化 N×N 矩阵,
  将计算量从 O(N^3) 降至 O(N^2)。

  在 Green 函数方法中:
    G(omega) = [(omega^2 + i*eps)*I - D]^{-1}
  对缺陷系统: G_defect = G_0 + G_0 * T * G_0
  其中 T 矩阵可用 SM 公式高效计算。

核心公式:
  (A - u*v')^{-1} = A^{-1} + A^{-1}*u*v'*A^{-1} / (1 - v'*A^{-1}*u)
  条件: 1 - v'*A^{-1}*u ≠ 0 (非奇异性条件)
"""

import numpy as np
from typing import Tuple, Optional


def lu_factorize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    LU 分解 (融合 r8sm 的 r8ge_fa)。
    使用部分主元选取的高斯消去。

    返回:
        LU: (n, n) 包含 L 和 U 的紧凑存储
        piv: (n,) 主元索引
    """
    n = len(A)
    LU = A.astype(float).copy()
    piv = np.arange(n)

    for k in range(n - 1):
        # 部分主元
        max_idx = k + np.argmax(np.abs(LU[k:, k]))
        if max_idx != k:
            LU[[k, max_idx]] = LU[[max_idx, k]]
            piv[[k, max_idx]] = piv[[max_idx, k]]

        if abs(LU[k, k]) < 1e-30:
            raise ValueError(f"LU 分解失败: 第 {k} 步主元为零")

        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            LU[i, k + 1:] -= LU[i, k] * LU[k, k + 1:]

    return LU, piv


def lu_solve(LU: np.ndarray, piv: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    LU 分解回代求解 (融合 r8sm 的 r8ge_sl)。
    前代 + 回代: Ly = Pb, Ux = y
    """
    n = len(LU)
    x = b[piv].astype(float).copy()

    # 前代 (L y = Pb)
    for i in range(1, n):
        for j in range(i):
            x[i] -= LU[i, j] * x[j]

    # 回代 (U x = y)
    for i in range(n - 1, -1, -1):
        if abs(LU[i, i]) < 1e-30:
            raise ValueError(f"回代失败: U[{i},{i}] = 0")
        for j in range(i + 1, n):
            x[i] -= LU[i, j] * x[j]
        x[i] /= LU[i, i]

    return x


def sherman_morrison_solve(
    A: np.ndarray, u: np.ndarray, v: np.ndarray, b: np.ndarray,
    lu_cache: Optional[Tuple] = None,
) -> Tuple[np.ndarray, float]:
    """
    Sherman-Morrison 公式求解 (A - u*v') x = b。
    融合 r8sm 的 r8sm_sl 算法。

    步骤:
      1. 对 A 做 LU 分解 (或复用缓存)
      2. 解 A*w = u, A*x0 = b
      3. beta = w'*b (注意: 此处为 w^T * b 而非 w'*A^{-1}*b)
         实际 beta = v' * x0
      4. alpha = 1 / (1 - v' * w)
      5. x = x0 + alpha * beta * w

    参数:
        A: (n, n) 基础矩阵
        u: (n,) 左修正向量
        v: (n,) 右修正向量
        b: (n,) 右端项
        lu_cache: 可选的预计算 LU 缓存

    返回:
        x: (n,) 解向量
        denom: 1 - v'*w (非奇异性度量, 接近零则病态)
    """
    n = len(A)
    assert A.shape == (n, n)
    assert u.shape == (n,) and v.shape == (n,) and b.shape == (n,)

    if lu_cache is not None:
        LU, piv = lu_cache
    else:
        LU, piv = lu_factorize(A)

    # 解 A*w = u 和 A*x0 = b
    w = lu_solve(LU, piv, u)
    x0 = lu_solve(LU, piv, b)

    # Sherman-Morrison 修正
    vw = np.dot(v, w)
    denom = 1.0 - vw
    if abs(denom) < 1e-14:
        raise ValueError(
            f"Sherman-Morrison 分母接近零: |1 - v^T w| = {abs(denom):.2e}\n"
            "修正后矩阵可能奇异。物理含义: 缺陷强度恰好使系统共振。"
        )

    vx0 = np.dot(v, x0)
    alpha = vx0 / denom
    x = x0 + alpha * w

    return x, denom


def sherman_morrison_woodbury_solve(
    A: np.ndarray, U: np.ndarray, V: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """
    Sherman-Morrison-Woodbury 公式 (秩-k 修正)。

    (A - U * V')^{-1} = A^{-1} + A^{-1} * U * (I - V' * A^{-1} * U)^{-1} * V' * A^{-1}

    其中 U 是 (n, k), V 是 (n, k), k << n。

    物理应用: 多缺陷声子散射问题。
    """
    n = len(A)
    k = U.shape[1]
    assert V.shape == (n, k)

    LU, piv = lu_factorize(A)

    # 解 A * W = U 和 A * X0 = b
    W = np.zeros((n, k))
    for j in range(k):
        W[:, j] = lu_solve(LU, piv, U[:, j])
    x0 = lu_solve(LU, piv, b)

    # 计算 (I - V'*W)^{-1}
    M = np.eye(k) - V.T @ W
    M_inv = np.linalg.inv(M)

    # x = x0 + W * M_inv * V' * x0
    Vx0 = V.T @ x0
    correction = W @ (M_inv @ Vx0)
    x = x0 + correction

    return x


def rank1_update_eigenvalues(
    eigenvalues_0: np.ndarray,
    eigenvectors_0: np.ndarray,
    u: np.ndarray, v: np.ndarray,
) -> np.ndarray:
    """
    利用秩1修正近似更新特征值 (一阶微扰)。

    若 A = Q * diag(lambda) * Q^T, 则
    (A - u*v') 的特征值近似为:
      lambda_i' ≈ lambda_i - (q_i^T u) * (v^T q_i)
                 = lambda_i - (Q^T u)_i * (Q^T v)_i

    物理: 缺陷对声子频率的一阶修正。
    """
    Q = eigenvectors_0  # 列向量为特征向量
    Qtu = Q.T @ u
    Qtv = Q.T @ v
    delta_lambda = -Qtu * Qtv
    return eigenvalues_0 + delta_lambda


def compute_defect_green_function(
    omega_sq: float,
    D0_inv: np.ndarray,
    u: np.ndarray, v: np.ndarray,
    eta: float = 1e-6,
) -> np.ndarray:
    """
    缺陷系统的推迟 Green 函数。

    G(omega) = [(omega + i*eta)*I - D_0 + u*v']^{-1}
             = G_0 + G_0 * u * (1 - v'*G_0*u)^{-1} * v' * G_0

    其中 G_0 = [(omega + i*eta)*I - D_0]^{-1}

    物理: 用于计算缺陷态密度和局域声子模式。
    """
    n = len(D0_inv)
    z = omega_sq + 1j * eta
    # G_0 = z * I - D0, 但此处 D0_inv 是预计算的 (z*I - D0)^{-1}
    G0 = z * np.eye(n) - np.linalg.inv(D0_inv + 1e-15 * np.eye(n))
    G0u = G0 @ u
    vG0u = np.dot(v, G0u)
    denom = 1.0 - vG0u
    if abs(denom) < 1e-14:
        denom = 1e-14 * np.exp(1j * np.angle(denom + 1e-30))
    G_defect = G0 + np.outer(G0u, v @ G0) / denom
    return G_defect
