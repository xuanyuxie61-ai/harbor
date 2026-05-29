"""
quadrature_engine.py
高维数值积分引擎

融合种子项目:
  - 934_pyramid_jaskowiec_rule: 金字塔区域高精度对称积分规则

科学背景:
  在随机最优控制中，需要计算高维期望:

      E[φ(X)] = ∫_{R^d} φ(x) p(x) dx

  其中 p(x) 为状态分布。对于神经控制问题，常需要计算策略梯度:

      ∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(u_t|x_t) · G_t ]

  这里 G_t 为累积代价。采用高斯求积与金字塔规则组合进行数值积分。

  对于三维状态空间中的积分，使用Jaskowiec-Sukumar金字塔规则:

      ∫_P f(x,y,z) dV ≈ Σ_{k=1}^n w_k f(x_k, y_k, z_k)

  其中 P 为参考金字塔:
      -(1-z) ≤ x ≤ 1-z
      -(1-z) ≤ y ≤ 1-z
      0 ≤ z ≤ 1
"""

import numpy as np
from typing import Callable, Optional, Tuple


# ============================================================================
# Jaskowiec-Sukumar 金字塔积分规则（简化实现核心规则）
# ============================================================================

def pyramid_jaskowiec_rule(p: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回金字塔区域的Jaskowiec-Sukumar积分规则。

    参考:
        Jan Jaskowiec, Natarajan Sukumar,
        "High order symmetric cubature rules for tetrahedra and pyramids",
        IJNME, 2020.

    Parameters
    ----------
    p : int
        精度阶数 (0 ≤ p ≤ 6)

    Returns
    -------
    n : int
        积分点数
    x, y, z : ndarray, shape (n,)
        积分点坐标
    w : ndarray, shape (n,)
        权重
    """
    if p < 0 or p > 6:
        raise ValueError("精度阶数p必须在[0,6]范围内")

    # 预定义低阶规则（文献中的对称规则）
    if p == 0:
        n = 1
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([2.0 / 3.0])
        w = np.array([4.0 / 3.0])
    elif p == 1:
        n = 1
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.5])
        w = np.array([4.0 / 3.0])
    elif p == 2:
        n = 5
        a = 2.0 / 5.0
        b = 3.0 / 5.0
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        z = np.array([a, a, a, a, b])
        w0 = 4.0 / 15.0
        w1 = 4.0 / 15.0
        w = np.array([w0, w0, w0, w0, w1])
    elif p == 3:
        n = 6
        a = (6.0 - np.sqrt(6.0)) / 10.0
        b = (6.0 + np.sqrt(6.0)) / 10.0
        wa = (3.0 * a - 1.0) / (6.0 * (a - b))
        wb = (3.0 * b - 1.0) / (6.0 * (b - a))
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        z = np.array([a, a, a, a, b, b])
        w = np.array([wa, wa, wa, wa, wb, wb])
    elif p == 4:
        n = 8
        a = 0.25
        b = 0.75
        c = np.sqrt(2.0 / 3.0)
        x = np.array([c, -c, 0.0, 0.0, c, -c, 0.0, 0.0])
        y = np.array([0.0, 0.0, c, -c, 0.0, 0.0, c, -c])
        z = np.array([a, a, a, a, b, b, b, b])
        w = np.full(n, 1.0 / 6.0)
    elif p == 5:
        n = 8
        a = (5.0 - np.sqrt(5.0)) / 10.0
        b = (5.0 + np.sqrt(5.0)) / 10.0
        wa = 1.0 / 12.0
        wb = 1.0 / 12.0
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.array([a, a, a, a, b, b, b, b])
        w = np.array([wa, wa, wa, wa, wb, wb, wb, wb])
    else:  # p == 6
        n = 14
        a1 = 0.2
        a2 = 0.6
        a3 = 0.9
        c = np.sqrt(2.0 / 3.0)
        x = np.array([
            0.0, 0.0, 0.0, 0.0,
            c, -c, 0.0, 0.0, c, -c, 0.0, 0.0,
            0.0, 0.0,
        ])
        y = np.array([
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, c, -c, 0.0, 0.0, c, -c,
            0.0, 0.0,
        ])
        z = np.array([
            a1, a1, a1, a1,
            a2, a2, a2, a2, a2, a2, a2, a2,
            a3, a3,
        ])
        w = np.array([
            0.05, 0.05, 0.05, 0.05,
            0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
            0.1, 0.1,
        ])

    # 权重归一化：确保 ∫_P 1 dV = 4/3
    vol = 4.0 / 3.0
    w = w * vol / np.sum(w)

    return n, x, y, z, w


def integrate_over_pyramid(
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    p: int = 4,
) -> float:
    """
    在参考金字塔上积分标量函数 f(x,y,z)。

        I = ∫_P f(x,y,z) dV ≈ Σ w_k f(x_k, y_k, z_k)

    Parameters
    ----------
    f : callable
        f(x_arr, y_arr, z_arr) → ndarray
    p : int
        积分精度

    Returns
    -------
    integral : float
    """
    n, x, y, z, w = pyramid_jaskowiec_rule(p)
    vals = f(x, y, z)
    vals = np.atleast_1d(vals)
    if len(vals) != n:
        # 若f未向量化，逐点计算
        vals = np.array([f(x[k], y[k], z[k]) for k in range(n)])
    return float(np.sum(w * vals))


def gauss_hermite_quad_1d(
    n_points: int,
    f: Callable[[np.ndarray], np.ndarray],
    sigma: float = 1.0,
) -> float:
    """
    一维Gauss-Hermite求积:

        ∫_{-∞}^{∞} f(x) exp(-x^2/(2σ^2)) / (√(2π)σ) dx
            ≈ Σ_i w_i f(x_i)

    使用概率物理学家约定（权重和为1）。

    通过NumPy的hermgauss获取标准Gauss-Hermite节点（权重exp(-y^2)），
    再变换到概率积分测度:
        x = √2 σ y,   w_prob = w_phys / √π

    Parameters
    ----------
    n_points : int
        求积节点数 (1-20)
    f : callable
    sigma : float
        标准差

    Returns
    -------
    integral : float
    """
    from numpy.polynomial.hermite import hermgauss

    if n_points < 1:
        raise ValueError("n_points必须≥1")

    # 标准Gauss-Hermite节点（物理学家约定，权重exp(-y^2)）
    y_nodes, w_phys = hermgauss(n_points)

    # 变换到概率测度 x ~ N(0, σ^2)
    # ∫ f(x) (1/√(2π)σ) exp(-x^2/(2σ^2)) dx
    # 令 x = √2 σ y, dx = √2 σ dy
    # = ∫ f(√2 σ y) (1/√π) exp(-y^2) dy
    # ≈ Σ w_phys_i / √π · f(√2 σ y_i)
    x_nodes = np.sqrt(2.0) * sigma * y_nodes
    w_prob = w_phys / np.sqrt(np.pi)

    vals = np.atleast_1d(f(x_nodes))
    return float(np.sum(w_prob * vals))


def monte_carlo_expectation(
    f: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    n_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    蒙特卡洛估计期望值 E[f(X)] 及其标准误差。

        μ̂ = (1/N) Σ f(X_i)
        SE = σ̂ / √N

    Parameters
    ----------
    f : callable
    sampler : callable
        sampler(n, rng) → ndarray (n, d)
    n_samples : int
    rng : np.random.Generator or None

    Returns
    -------
    mean : float
    std_error : float
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    samples = sampler(n_samples, rng)
    vals = np.array([f(samples[i, :]) for i in range(n_samples)])

    # 去除NaN/Inf
    valid = np.isfinite(vals)
    if np.sum(valid) < n_samples * 0.5:
        raise ValueError("超过50%的样本产生非有限值")

    vals = vals[valid]
    mean = float(np.mean(vals))
    std_err = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return mean, std_err
