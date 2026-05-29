r"""
mode_analysis.py
================
光子空间-光谱模式分析模块 —— 融合原项目 881_polpak (球谐 / Jacobi / Legendre
多项式) 与 013_approx_bernstein (Bernstein 多项式逼近)。

在纠缠光源中，横向空间模式与频率模式共同决定了纠缠维度与态纯度。
本模块提供：
1. 归一化连带 Legendre 函数与球谐函数，用于横向角向模式分解。
2. Jacobi 多项式，用于非对称横向模式基底。
3. Bernstein 多项式基，用于光滑逼近泵浦光谱包络与相位匹配函数。

核心公式
--------
**连带 Legendre 函数** (Ferrers 函数)

.. math::
    P_\ell^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_\ell(x)

**归一化球谐函数** (Condon-Shortley 相位)

.. math::
    Y_\ell^m(\theta,\phi) = \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}}
    P_\ell^m(\cos\theta) e^{i m \phi}

**Jacobi 多项式** (:math:`P_n^{(\alpha,\beta)}(x)`)

.. math::
    P_n^{(\alpha,\beta)}(x) = \frac{\Gamma(\alpha+n+1)}{n!\,\Gamma(\alpha+\beta+n+1)}
    \sum_{k=0}^{n} \binom{n}{k}
    \frac{\Gamma(\alpha+\beta+n+k+1)}{\Gamma(\alpha+k+1)}
    \left(\frac{x-1}{2}\right)^k

**Bernstein 基多项式**

.. math::
    B_{i,n}(x) = \binom{n}{i} x^i (1-x)^{n-i}, \quad x\in[0,1]

Bernstein 逼近：

.. math::
    f(x) \approx \sum_{i=0}^{n} f\left(\frac{i}{n}\right) B_{i,n}(x)
"""

import numpy as np
from scipy.special import gamma, factorial, comb
from typing import Tuple


def associated_legendre_normalized(l_max: int, m: int, x: np.ndarray) -> np.ndarray:
    r"""
    计算归一化连带 Legendre 函数 :math:`\bar{P}_\ell^m(x)`，
    其中 :math:`\ell = m, \dots, l_{\max}`，
    满足正交归一条件

    .. math::
        \int_{-1}^{1} \bar{P}_\ell^m(x) \bar{P}_{\ell'}^m(x) \,dx = \delta_{\ell\ell'}

    参数
    ----
    l_max : int
        最大阶数，:math:`l_{\max} \ge |m|`。
    m : int
        阶数，:math:`m \ge 0`。
    x : np.ndarray
        自变量，范围 :math:`[-1, 1]`。

    返回
    ----
    P : np.ndarray, shape (len(x), l_max+1)
        第 k 列对应 :math:`\bar{P}_k^m(x)`（k < m 时为零）。
    """
    if l_max < m:
        raise ValueError("l_max 必须不小于 |m|")
    x = np.atleast_1d(x)
    if np.any(np.abs(x) > 1.0 + 1e-12):
        raise ValueError("x 必须位于 [-1, 1] 区间。")
    x = np.clip(x, -1.0, 1.0)

    n_points = x.size
    P = np.zeros((n_points, l_max + 1), dtype=np.float64)

    # 初始值 P_m^m
    if m == 0:
        P[:, 0] = np.sqrt(0.5)  # 归一化常数已融入递推
    else:
        pmm = np.ones(n_points, dtype=np.float64)
        somx2 = np.sqrt(np.maximum(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0
        # 归一化因子 sqrt((2m+1)/(2) * (2m)! / (2m)!!^2) ... 用显式公式
        norm_pm = np.sqrt((2.0 * m + 1.0) / 2.0 * factorial(2 * m) /
                          (2.0 ** (2 * m) * factorial(m) ** 2))
        P[:, m] = norm_pm * pmm

    # 递推计算 P_{m+1}^m, ..., P_{l_max}^m
    if m < l_max:
        if m == 0:
            P[:, 1] = np.sqrt(3.0 / 2.0) * x
        else:
            pll = x * (2.0 * m + 1.0) * P[:, m] / np.sqrt((2.0 * m + 1.0) * (2.0 * m + 3.0))
            # 更稳妥：使用标准递推后补归一化
            # 这里用递推关系 (ell-m) P_ell^m = x(2ell-1) P_{ell-1}^m - (ell+m-1) P_{ell-2}^m
            # 先算未归一化版本
            p_unnorm = np.zeros((n_points, l_max + 1), dtype=np.float64)
            p_unnorm[:, m] = 1.0
            for mm in range(1, m + 1):
                p_unnorm[:, m] *= (-1.0) * (2.0 * mm - 1.0) * somx2
            if m < l_max:
                p_unnorm[:, m + 1] = x * (2.0 * m + 1.0) * p_unnorm[:, m]
            for ell in range(m + 2, l_max + 1):
                p_unnorm[:, ell] = (x * (2.0 * ell - 1.0) * p_unnorm[:, ell - 1]
                                    - (ell + m - 1.0) * p_unnorm[:, ell - 2]) / (ell - m)
            # 逐列归一化
            for ell in range(m, l_max + 1):
                norm = np.sqrt((2.0 * ell + 1.0) / 2.0 * factorial(ell - m) / factorial(ell + m))
                P[:, ell] = norm * p_unnorm[:, ell]
            return P

    for ell in range(m + 1, l_max + 1):
        # 使用归一化递推
        a1 = np.sqrt((4.0 * ell ** 2 - 1.0) / (ell ** 2 - m ** 2))
        a2 = np.sqrt(((2.0 * ell + 1.0) * (ell + m - 1.0) * (ell - m - 1.0)) /
                     ((2.0 * ell - 3.0) * (ell ** 2 - m ** 2)))
        P[:, ell] = a1 * x * P[:, ell - 1] - a2 * P[:, ell - 2]

    return P


def spherical_harmonic_basis(l_max: int, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    计算实球谐函数基底 :math:`Y_{\ell m}(\theta,\phi)` 的实部与虚部。

    参数
    ----
    l_max : int
    theta : np.ndarray
        极角 :math:`\theta \in [0, \pi]`。
    phi : np.ndarray
        方位角 :math:`\phi \in [0, 2\pi]`。

    返回
    ----
    Y_real, Y_imag : np.ndarray, shape (n_points, (l_max+1)^2)
        按 :math:`(\ell, m)` 扁平化排列。
    """
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    n_points = theta.size
    n_modes = (l_max + 1) ** 2
    Y_real = np.zeros((n_points, n_modes), dtype=np.float64)
    Y_imag = np.zeros((n_points, n_modes), dtype=np.float64)

    idx = 0
    for l in range(l_max + 1):
        x = np.cos(theta)
        Plm = associated_legendre_normalized(l, l, x)  #  shape (n, l+1)
        for m in range(-l, l + 1):
            m_abs = abs(m)
            # 取 Plm 的第 m_abs 列（对应 P_l^{m_abs}）
            plm_val = Plm[:, m_abs] if m_abs <= l else np.zeros(n_points)
            if m < 0:
                # Y_l^{-m} = (-1)^m \bar{Y}_l^m
                phase = (-1.0) ** m_abs
                Y_real[:, idx] = phase * plm_val * np.cos(m_abs * phi)
                Y_imag[:, idx] = -phase * plm_val * np.sin(m_abs * phi)
            else:
                Y_real[:, idx] = plm_val * np.cos(m * phi)
                Y_imag[:, idx] = plm_val * np.sin(m * phi)
            idx += 1

    return Y_real, Y_imag


def jacobi_polynomial(n: int, alpha: float, beta: float, x: np.ndarray) -> np.ndarray:
    r"""
    计算 Jacobi 多项式 :math:`P_n^{(\alpha,\beta)}(x)`。

    使用三项递推：

    .. math::
        P_0^{(\alpha,\beta)}(x) &= 1 \\
        P_1^{(\alpha,\beta)}(x) &= \frac{\alpha-\beta}{2} + \frac{\alpha+\beta+2}{2} x \\
        a_n P_{n}^{(\alpha,\beta)}(x) &= (b_n + c_n x) P_{n-1}^{(\alpha,\beta)}(x)
                                         - d_n P_{n-2}^{(\alpha,\beta)}(x)

    其中

    .. math::
        a_n &= 2n(n+\alpha+\beta)(2n+\alpha+\beta-2) \\
        b_n &= (2n+\alpha+\beta-1)(\alpha^2-\beta^2) \\
        c_n &= (2n+\alpha+\beta-2)(2n+\alpha+\beta-1)(2n+\alpha+\beta) \\
        d_n &= 2(n+\alpha-1)(n+\beta-1)(2n+\alpha+\beta)
    """
    x = np.atleast_1d(x)
    if n < 0:
        raise ValueError("n 必须为非负整数。")
    if n == 0:
        return np.ones_like(x, dtype=np.float64)
    if n == 1:
        return 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x

    P_prev2 = np.ones_like(x, dtype=np.float64)
    P_prev1 = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x

    for nn in range(2, n + 1):
        a_n = 2.0 * nn * (nn + alpha + beta) * (2.0 * nn + alpha + beta - 2.0)
        b_n = (2.0 * nn + alpha + beta - 1.0) * (alpha ** 2 - beta ** 2)
        c_n = (2.0 * nn + alpha + beta - 2.0) * (2.0 * nn + alpha + beta - 1.0) * (2.0 * nn + alpha + beta)
        d_n = 2.0 * (nn + alpha - 1.0) * (nn + beta - 1.0) * (2.0 * nn + alpha + beta)

        if abs(a_n) < 1e-15:
            raise RuntimeError(f"Jacobi 递推系数 a_{nn} 过小。")
        P_curr = ((b_n + c_n * x) * P_prev1 - d_n * P_prev2) / a_n
        P_prev2 = P_prev1
        P_prev1 = P_curr

    return P_prev1


def bernstein_basis(n: int, x: np.ndarray) -> np.ndarray:
    r"""
    计算 n 次 Bernstein 基函数 :math:`B_{i,n}(x)` 在 x 处的值。

    参数
    ----
    n : int
        次数，:math:`n \ge 0`。
    x : np.ndarray
        自变量，范围 :math:`[0,1]`。

    返回
    ----
    B : np.ndarray, shape (len(x), n+1)
        B[:, i] = B_{i,n}(x)。
    """
    x = np.atleast_1d(x)
    if np.any((x < -1e-12) | (x > 1.0 + 1e-12)):
        raise ValueError("Bernstein 基仅定义在 [0,1] 上。")
    x = np.clip(x, 0.0, 1.0)

    n_points = x.size
    B = np.zeros((n_points, n + 1), dtype=np.float64)
    if n == 0:
        B[:, 0] = 1.0
        return B

    # 使用递推：B_{i,n}(x) = (1-x) B_{i,n-1}(x) + x B_{i-1,n-1}(x)
    B[:, 0] = 1.0 - x
    B[:, 1] = x
    for j in range(2, n + 1):
        B[:, j] = x * B[:, j - 1]
        for k in range(j - 1, 0, -1):
            B[:, k] = x * B[:, k - 1] + (1.0 - x) * B[:, k]
        B[:, 0] = (1.0 - x) * B[:, 0]

    return B


def bernstein_approximate(f_values: np.ndarray, a: float, b: float,
                          n: int, x_eval: np.ndarray) -> np.ndarray:
    r"""
    利用 Bernstein 多项式在 [a,b] 上逼近函数 f。

    .. math::
        f(x) \approx \sum_{i=0}^{n} f\left(a + \frac{i}{n}(b-a)\right) B_{i,n}\!
        \left(\frac{x-a}{b-a}\right)

    参数
    ----
    f_values : np.ndarray, shape (n+1,)
        在等距节点上的函数值。
    a, b : float
        区间端点。
    n : int
        Bernstein 次数。
    x_eval : np.ndarray
        求值点。

    返回
    ----
    y_eval : np.ndarray
        逼近值。
    """
    if len(f_values) != n + 1:
        raise ValueError("f_values 长度必须等于 n+1。")
    if abs(b - a) < 1e-15:
        raise ValueError("区间长度必须为正。")

    t = (x_eval - a) / (b - a)
    B = bernstein_basis(n, t)
    y_eval = B @ f_values
    return y_eval
