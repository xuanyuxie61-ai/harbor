"""
recombination_models.py
基于种子项目 641_laguerre_polynomial (Laguerre polynomials & Gauss-Laguerre quadrature)
改造为钙钛矿太阳能电池中辐射复合与 Auger 复合的高级数值积分模块。

在太阳能电池中，总复合率 R_total = R_rad + R_Auger + R_SRH。
本模块重点处理：
  1. 辐射复合（Radiative recombination）：
       R_rad = B (n p - n_i^2)
     其中 B 为辐射复合系数。
  2. Auger 复合（Auger recombination）：
       R_Auger = (C_n n + C_p p) (n p - n_i^2)
  3. 带尾态辐射复合（Band-to-tail，Urbach 尾）：
       R_tail = ∫_{E_g}^{∞} g(E) f_c(E) f_v(E) dE
     其中 g(E) 为联合态密度，涉及 exp(-E/kT) 权函数。

Gauss-Laguerre 求积（对应原项目 l_quadrature_rule）特别适用于
积分核含 exp(-x) 的半无穷区间积分，如带尾态分布的积分计算。

核心公式：
  1. 广义拉盖尔多项式 L_n^{(α)}(x)：
       (n+1) L_{n+1}^{(α)} = (2n+α+1-x) L_n^{(α)} - (n+α) L_{n-1}^{(α)}
  2. Gauss-Laguerre 求积：
       ∫_0^∞ x^α e^{-x} f(x) dx ≈ Σ_i w_i f(x_i)
  3. 辐射复合系数 B 的温度依赖（van Roosbroeck-Shockley 关系）：
       B(T) = (2π)^{1/2} ħ^{3/2} / (m_r^{3/2} (kT)^{3/2} τ_rad)
       更常用经验公式：B(T) = B_300 * (T/300)^{-3/2}
"""

import numpy as np
from typing import Tuple


def imtqlx(n: int, d: np.ndarray, e: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    隐式 QL 算法对角化对称三对角矩阵。
    对应原项目 imtqlx。
    """
    d = d.copy()
    e = e.copy()
    z = z.copy()
    e[n - 1] = 0.0
    for l in range(1, n + 1):
        j = 0
        while True:
            for m in range(l, n):
                if m == n - 1:
                    break
                if abs(e[m]) <= 1e-14 * (abs(d[m]) + abs(d[m + 1])):
                    break
            if m == l - 1:
                break
            if j >= 30:
                raise RuntimeError("imtqlx 未收敛")
            j += 1
            g = (d[l] - d[l - 1]) / (2.0 * e[l - 1])
            r = np.sqrt(g * g + 1.0)
            g = d[m - 1] - d[l - 1] + e[l - 1] / (g + np.copysign(r, g))
            s, c = 1.0, 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) >= abs(g):
                    c = g / f
                    r = np.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c *= s
                else:
                    s = f / g
                    r = np.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s *= c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                # 更新特征向量
                for k in range(n):
                    temp = z[k + n * (i + 1)]
                    z[k + n * (i + 1)] = s * z[k + n * i] + c * temp
                    z[k + n * i] = c * z[k + n * i] - s * temp
            d[l - 1] -= p
            e[l - 1] = g
            e[m - 1] = 0.0
    return d, z


def laguerre_quadrature_rule(n: int, alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    广义 Gauss-Laguerre 求积规则。
    计算 ∫_0^∞ x^α e^{-x} f(x) dx ≈ Σ w_i f(x_i)。
    对应原项目 l_quadrature_rule 的扩展，使用 numpy.linalg.eigh
    对角化 Jacobi 矩阵（数值更稳定）。

    Parameters
    ----------
    n : int
        求积阶数
    alpha : float
        广义拉盖尔参数（默认 0）

    Returns
    -------
    x : (n,) array
        节点
    w : (n,) array
        权重
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    if alpha < -1.0:
        raise ValueError("alpha 必须 ≥ -1")

    import math
    # 构造对称三对角 Jacobi 矩阵
    diag = np.zeros(n)
    offdiag = np.zeros(n - 1)
    for i in range(n):
        diag[i] = 2.0 * i + 1.0 + alpha
    for i in range(n - 1):
        offdiag[i] = np.sqrt((i + 1.0) * (i + 1.0 + alpha))

    jacobi = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(jacobi)

    x_out = eigenvalues
    v0 = eigenvectors[0, :]
    w_out = math.gamma(alpha + 1.0) * v0 ** 2
    return x_out, w_out


def lm_polynomial_values(n: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    计算关联拉盖尔多项式 L_n^{(m)}(x) 的值。
    对应原项目 lm_polynomial_values。
    """
    x = np.asarray(x)
    v = np.zeros((n + 1, len(x)))
    if n >= 0:
        v[0, :] = 1.0
    if n >= 1:
        v[1, :] = (m + 1.0) - x
    for i in range(2, n + 1):
        v[i, :] = ((m + 2.0 * i - 1.0 - x) * v[i - 1, :] + (1.0 - m - i) * v[i - 2, :]) / i
    return v[n, :]


def radiative_recombination_rate(
    n: float, p: float, n_i: float, T: float = 300.0
) -> float:
    """
    辐射复合率 [cm^{-3}·s^{-1}]。
    R_rad = B(T) * (n p - n_i^2)
    B(T) = B_300 * (T/300)^{-1.5}
    """
    B_300 = 2.0e-10  # cm^3/s (钙钛矿典型值)
    B = B_300 * (T / 300.0) ** (-1.5)
    return B * (n * p - n_i * n_i)


def auger_recombination_rate(
    n: float, p: float, n_i: float, T: float = 300.0
) -> float:
    """
    Auger 复合率 [cm^{-3}·s^{-1}]。
    R_Auger = (C_n n + C_p p) (n p - n_i^2)
    C_n, C_p 典型值 ~ 1e-29 cm^6/s
    """
    C_n = 1.0e-29
    C_p = 1.0e-29
    return (C_n * n + C_p * p) * (n * p - n_i * n_i)


def band_to_tail_recombination_integral(
    E_g: float, T: float, n: float, p: float,
    n_i: float, N_t_tail: float = 1e16, E_u: float = 0.015,
    quadrature_order: int = 16,
) -> float:
    """
    使用 Gauss-Laguerre 求积计算带尾态辐射复合积分。

    模型：
      带尾态密度 g(E) = N_t_tail * exp(-(E - E_g)/E_u) / E_u
      复合率 R_tail = ∫_{E_g}^∞ g(E) * v(E) * f_c(E) * (1-f_v(E)) dE
    通过变量替换 x = (E - E_g)/E_u，积分变为：
      R_tail = N_t_tail * v_th * σ * ∫_0^∞ e^{-x} * f_c(E_g + x E_u) * (1-f_v) * E_u dx
    进一步简化，使用有效复合速率近似：
      R_tail ≈ N_t_tail * C * ∫_0^∞ e^{-x} * (n p - n_i^2) / (n + p + 2 n_i cosh(x)) dx

    这里使用 Gauss-Laguerre 对含 exp(-x) 的积分进行数值求积。
    """
    if E_g <= 0 or T <= 0 or E_u <= 0:
        return 0.0

    kT = 8.617333e-5 * T  # eV
    x_nodes, w_nodes = laguerre_quadrature_rule(quadrature_order, alpha=0.0)

    # 简化模型：带尾态的有效复合截面
    sigma_eff = 1e-14  # cm^2
    v_th = 1e7  # cm/s (热运动速度近似)

    integral = 0.0
    for xi, wi in zip(x_nodes, w_nodes):
        # 能级 E = E_g + xi * E_u
        # 陷阱相对于带边的深度
        trap_depth = xi * E_u
        # 简化的复合概率
        # 使用 Fermi-Dirac 积分的近似
        n1 = n_i * np.exp(trap_depth / kT)
        p1 = n_i * np.exp(-trap_depth / kT)
        denom = p + n1 + n + p1
        if denom > 0:
            integrand = (n * p - n_i * n_i) / denom
            integral += wi * integrand

    # 数值鲁棒性
    if not np.isfinite(integral):
        integral = 0.0

    R_tail = N_t_tail * sigma_eff * v_th * integral
    return max(R_tail, 0.0)


def total_recombination_rate(
    n: float, p: float, n_i: float, T: float = 300.0,
    tau_n: float = 1e-9, tau_p: float = 1e-9,
    E_t: float = 0.0, E_g: float = 1.6,
    N_t_tail: float = 1e16, E_u: float = 0.015,
) -> dict:
    """
    计算总复合率及各项分量。

    Returns
    -------
    rates : dict
        {'SRH', 'radiative', 'auger', 'tail', 'total'}
    """
    # SRH
    # TODO(Hole_1): 实现 Shockley-Read-Hall 复合率计算
    # 需要计算 n1, p1 以及 denom_srh，然后得到 R_srh
    # 公式: R_srh = (n * p - n_i^2) / (tau_p * (n + n1) + tau_n * (p + p1))
    # 其中 n1 = N_c * exp(-E_t / kT), p1 = N_v * exp(E_t / kT)
    # 注意与 main.py 中 collection_efficiency 的耦合关系
    kT = 8.617333e-5 * T
    R_srh = 0.0  # placeholder

    R_rad = radiative_recombination_rate(n, p, n_i, T)
    R_aug = auger_recombination_rate(n, p, n_i, T)
    R_tail = band_to_tail_recombination_integral(E_g, T, n, p, n_i, N_t_tail, E_u)

    return {
        "SRH": max(R_srh, 0.0),
        "radiative": max(R_rad, 0.0),
        "auger": max(R_aug, 0.0),
        "tail": max(R_tail, 0.0),
        "total": max(R_srh + R_rad + R_aug + R_tail, 0.0),
    }


if __name__ == "__main__":
    x, w = laguerre_quadrature_rule(8)
    print(f"Gauss-Laguerre (n=8) 节点: {x}")
    print(f"权重和: {w.sum():.6f} (应≈1)")

    rates = total_recombination_rate(1e15, 1e15, 1e10, 300.0, 1e-9, 1e-9, 0.0, 1.6)
    print("复合率分量:", rates)
