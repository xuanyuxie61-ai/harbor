"""
iterative_solver.py
================================================================================
基于 Jacobi 迭代的线性系统求解器与量子自洽场方程松弛。

融合来源：603_jacobi（Jacobi 迭代）

物理背景：
  在量子退火中，我们经常需要求解有效单粒子方程（类似 Hartree-Fock
  自洽场）。对于平均场近似下的量子 Ising 模型，局域有效场为

      h_i^{eff} = h_i + Σ_j J_{ij} ⟨σ_j^z⟩

  自洽方程为
      ⟨σ_i^z⟩ = tanh( β h_i^{eff} )

  这构成一个非线性不动点问题，可用 Jacobi 型迭代求解：

      m_i^{(k+1)} = tanh( β [ h_i + Σ_j J_{ij} m_j^{(k)} ] )

  收敛条件（压缩映射）：
      max_i Σ_j | ∂m_i^{(k+1)} / ∂m_j^{(k)} | < 1

  对于线性系统 A x = b，Jacobi 迭代格式为：
      x_i^{(k+1)} = ( b_i - Σ_{j≠i} A_{ij} x_j^{(k)} ) / A_{ii}

  收敛的充分条件：A 严格对角占优或对称正定。
"""

import numpy as np
from typing import Tuple, Optional, Callable


def jacobi_iteration_step(A: np.ndarray, b: np.ndarray,
                          x: np.ndarray) -> np.ndarray:
    """
    执行一次 Jacobi 迭代步。

    输入：
        A: n×n 系数矩阵（要求对角元非零）
        b: n 维右端向量
        x: 当前解估计

    输出：
        x_new: 更新后的解估计
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if b.shape != (n,) or x.shape != (n,):
        raise ValueError("b and x must be 1D arrays of length n")
    x_new = np.zeros(n, dtype=float)
    for i in range(n):
        diag = A[i, i]
        if abs(diag) < 1e-15:
            raise RuntimeError(f"Zero diagonal element at index {i}")
        s = b[i]
        for j in range(n):
            if j != i:
                s -= A[i, j] * x[j]
        x_new[i] = s / diag
    return x_new


def jacobi_solve(A: np.ndarray, b: np.ndarray,
                 x0: Optional[np.ndarray] = None,
                 max_iter: int = 10000, tol: float = 1e-10,
                 omega: float = 1.0) -> Tuple[np.ndarray, int, float]:
    """
    带松弛因子 ω 的 Jacobi 迭代求解器（JORM）。

    松弛格式：
        x^{(k+1)} = (1-ω) x^{(k)} + ω * jacobi_update(x^{(k)})

    当 0 < ω < 1 时为低松弛，1 < ω < 2 时为超松弛（SOR 的 Jacobi 版本）。
    """
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)
    if not (0.0 < omega <= 2.0):
        raise ValueError("omega must be in (0, 2]")

    residual_history = []
    for it in range(max_iter):
        x_new = jacobi_iteration_step(A, b, x)
        x = (1.0 - omega) * x + omega * x_new
        r = b - A @ x
        res_norm = float(np.linalg.norm(r, ord=np.inf))
        residual_history.append(res_norm)
        if res_norm < tol:
            return x, it + 1, res_norm
        # 发散检测（宽松阈值）
        if it > 100 and res_norm > 1e12 * max(residual_history[0], 1.0):
            raise RuntimeError("Jacobi iteration appears to diverge")
    return x, max_iter, float(np.linalg.norm(b - A @ x, ord=np.inf))


def self_consistent_mean_field(J: np.ndarray, h: np.ndarray,
                                beta: float = 1.0,
                                max_iter: int = 5000,
                                tol: float = 1e-10,
                                damping: float = 0.5) -> Tuple[np.ndarray, int]:
    """
    求解量子伊辛模型的平均场自洽方程：

        m_i = tanh( β [ h_i + Σ_j J_{ij} m_j ] )

    使用阻尼迭代（Damped Fixed-Point Iteration）提高稳定性：

        m^{new} = (1-α) m^{old} + α * tanh( β h^{eff} )

    其中 α ∈ (0,1] 为阻尼系数。
    """
    # TODO: 实现平均场自洽方程的阻尼迭代求解
    # 提示：需要迭代直至收敛，每一步计算有效场 h_eff = h + J @ m
    #       然后应用双曲正切非线性映射，并加入阻尼因子
    raise NotImplementedError("Hole 2: 请补全平均场自洽场方程的阻尼迭代求解逻辑")


def variational_ground_state_energy(J: np.ndarray, h: np.ndarray,
                                     m: np.ndarray) -> float:
    """
    在平均场近似下计算变分基态能量：

        E_MF = - Σ_{i<j} J_{ij} m_i m_j - Σ_i h_i m_i
               + (1/β) Σ_i [ (1+m_i)/2 ln((1+m_i)/2) + (1-m_i)/2 ln((1-m_i)/2) ]

    最后一项为熵贡献（变分自由能）。
    """
    n = h.size
    m = np.clip(m, -1.0 + 1e-12, 1.0 - 1e-12)
    e_int = -0.5 * float(m @ J @ m) - float(h @ m)
    # 熵项
    p_plus = (1.0 + m) / 2.0
    p_minus = (1.0 - m) / 2.0
    entropy = np.sum(p_plus * np.log(p_plus) + p_minus * np.log(p_minus))
    return float(e_int + entropy)


def power_iteration_eigenvalue(A: np.ndarray, max_iter: int = 1000,
                                tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """
    幂迭代法估算矩阵最大模特征值与特征向量。

    在量子退火中用于估算能隙下界：
        λ_max(A) ≥ ΔE_min
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    lam = 0.0
    for it in range(max_iter):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        lam_new = float(v_new @ A @ v_new)
        if abs(lam_new - lam) < tol:
            return lam_new, v_new
        lam = lam_new
        v = v_new
    return lam, v


def chebyshev_accelerated_jacobi(A: np.ndarray, b: np.ndarray,
                                  x0: Optional[np.ndarray] = None,
                                  max_iter: int = 500,
                                  tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    """
    Chebyshev 加速 Jacobi 迭代。

    对于对称正定矩阵 A，设 D = diag(A)，迭代矩阵 G = I - D^{-1} A，
    其特征值范围为 [μ_min, μ_max] ⊂ (-1,1)。Chebyshev 多项式 T_k(z) 的
    极小极大性质可用于加速收敛：

        x^{(k+1)} = x^{(k)} + α_k (b - A x^{(k)}) + β_k (x^{(k)} - x^{(k-1)})

    其中 α_k, β_k 由 Chebyshev 递推系数确定。
    """
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)
    Dinv = np.diag(1.0 / np.diag(A))
    # 估算特征值范围（Gerschgorin 圆盘近似）
    mu_est = []
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) / abs(A[i, i]) - 1.0
        mu_est.append(row_sum)
    mu_max = min(max(mu_est), 0.99)
    mu_min = -mu_max
    if mu_max <= 0:
        mu_max = 0.5
        mu_min = -0.5

    # Chebyshev 参数
    sigma = (mu_max - mu_min) / 2.0
    delta = (mu_max + mu_min) / 2.0
    alpha = 2.0 / (2.0 - mu_max - mu_min)

    x_prev = x.copy()
    for it in range(max_iter):
        r = b - A @ x
        if it == 0:
            x_new = x + alpha * (Dinv @ r)
            rho = 1.0 / (1.0 - 0.5 * sigma * sigma)
        else:
            rho_prev = rho
            rho = 1.0 / (1.0 - 0.25 * sigma * sigma * rho_prev)
            x_new = x + rho * alpha * (Dinv @ r) + (1.0 - rho) * (x - x_prev)
        x_prev, x = x, x_new
        res_norm = float(np.linalg.norm(b - A @ x, ord=np.inf))
        if res_norm < tol:
            return x, it + 1, res_norm
    return x, max_iter, float(np.linalg.norm(b - A @ x, ord=np.inf))
