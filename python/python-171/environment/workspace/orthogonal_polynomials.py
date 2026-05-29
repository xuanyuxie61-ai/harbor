# -*- coding: utf-8 -*-
"""
orthogonal_polynomials.py
==========================
正交多项式族与 Jacobi 矩阵构造。

融合种子项目：
- 641_laguerre_polynomial : Laguerre 多项式（标准/广义/位移）及其求积规则
- 525_hermite_rule        : Hermite 多项式与 IQPACK 算法

核心公式：
  标准 Laguerre 多项式 L_n(x) 满足：
      L_0(x) = 1
      L_1(x) = 1 - x
      n L_n(x) = (2n - 1 - x) L_{n-1}(x) - (n - 1) L_{n-2}(x)

  广义 Laguerre 函数 Lf_n^{(α)}(x) 满足：
      Lf_0^{(α)}(x) = 1
      Lf_1^{(α)}(x) = 1 + α - x
      n Lf_n^{(α)}(x) = (2n - 1 + α - x) Lf_{n-1}^{(α)}(x) - (n - 1 + α) Lf_{n-2}^{(α)}(x)

  正交性：
      ∫_0^∞ exp(-x) L_m(x) L_n(x) dx = δ_{mn}
      ∫_0^∞ exp(-x) x^α Lf_m^{(α)}(x) Lf_n^{(α)}(x) dx = Γ(n+α+1) / n!
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# 标准 Laguerre 多项式
# ---------------------------------------------------------------------------

def laguerre_polynomial(m, n, x):
    """
    计算 Laguerre 多项式 L_0(x), ..., L_n(x) 在 m 个点上的值。

    返回
    ----
    v : ndarray, shape (m, n+1)
        v[:, j] = L_j(x)
    """
    x = np.asarray(x, dtype=float).flatten()
    m = x.size
    if n < 0:
        return np.zeros((m, 0), dtype=float)
    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n == 0:
        return v
    v[:, 1] = 1.0 - x
    for j in range(2, n + 1):
        v[:, j] = (((2 * j - 1) - x) * v[:, j - 1] + (-j + 1) * v[:, j - 2]) / j
    return v


def generalized_laguerre_function(m, n, alpha, x):
    """
    广义 Laguerre 函数 Lf_n^{(α)}(x)。
    要求 alpha > -1。
    """
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1 for generalized Laguerre.")
    x = np.asarray(x, dtype=float).flatten()
    m = x.size
    if n < 0:
        return np.zeros((m, 0), dtype=float)
    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n == 0:
        return v
    v[:, 1] = 1.0 + alpha - x
    for i in range(2, n + 1):
        v[:, i] = (((2 * i - 1 + alpha) - x) * v[:, i - 1] + (-i + 1 - alpha) * v[:, i - 2]) / i
    return v


# ---------------------------------------------------------------------------
# Hermite 多项式（概率学家/物理学家约定）
# ---------------------------------------------------------------------------

def hermite_probabilist(n, x):
    """
    概率学家 Hermite 多项式 He_n(x)：
        He_0(x) = 1
        He_1(x) = x
        He_n(x) = x He_{n-1}(x) - (n-1) He_{n-2}(x)
    正交权：exp(-x^2/2) / sqrt(2π)
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    He_prev2 = np.ones_like(x)
    He_prev1 = x.copy()
    for k in range(2, n + 1):
        He_curr = x * He_prev1 - (k - 1) * He_prev2
        He_prev2, He_prev1 = He_prev1, He_curr
    return He_prev1


def hermite_physicist(n, x):
    """
    物理学家 Hermite 多项式 H_n(x)：
        H_0(x) = 1
        H_1(x) = 2x
        H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
    正交权：exp(-x^2)
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    H_prev2 = np.ones_like(x)
    H_prev1 = 2.0 * x
    for k in range(2, n + 1):
        H_curr = 2.0 * x * H_prev1 - 2.0 * (k - 1) * H_prev2
        H_prev2, H_prev1 = H_prev1, H_curr
    return H_prev1


# ---------------------------------------------------------------------------
# IMTQLX：隐式 QL 算法对角化对称三对角 Jacobi 矩阵
# ---------------------------------------------------------------------------

def imtqlx(n, d, e, z):
    """
    隐式 QL 算法对角化对称三对角矩阵。

    输入的 Jacobi 矩阵为：
        J = diag(d) + offdiag(e)
    其中 e 为次对角线（长度 n，e[n-1] 不使用）。

    算法输出：
        d : 特征值（升序排列）
        z : Q^T z（若 z 为初始权重的平方根，则输出 z 的平方为求积权重）

    参考：Martin & Wilkinson, Numerische Mathematik 12 (1968), 377-383.
    """
    d = np.asarray(d, dtype=float).copy().flatten()
    e = np.asarray(e, dtype=float).copy().flatten()
    z = np.asarray(z, dtype=float).copy().flatten()

    if n == 1:
        return d, z

    itn = 30
    prec = np.finfo(float).eps
    e[n - 1] = 0.0

    for l in range(n):
        j = 0
        while True:
            m = l
            while m < n - 1:
                if abs(e[m]) <= prec * (abs(d[m]) + abs(d[m + 1])):
                    break
                m += 1

            p = d[l]
            if m == l:
                break

            if j >= itn:
                raise RuntimeError("IMTQLX: iteration limit exceeded.")

            j += 1
            g = (d[l + 1] - p) / (2.0 * e[l])
            r = math.sqrt(g * g + 1.0)
            g = d[m] - p + e[l] / (g + math.copysign(r, g))
            s = 1.0
            c = 1.0
            p_local = 0.0
            mml = m - l

            for ii in range(1, mml + 1):
                i = m - ii
                f = s * e[i]
                b = c * e[i]

                if abs(f) >= abs(g):
                    c_val = g / f
                    r = math.sqrt(c_val * c_val + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c = c_val * s
                else:
                    s_val = f / g
                    r = math.sqrt(s_val * s_val + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s = s_val * c

                g = d[i + 1] - p_local
                r = (d[i] - g) * s + 2.0 * c * b
                p_local = s * r
                d[i + 1] = g + p_local
                g = c * r - b
                f = z[i + 1]
                z[i + 1] = s * z[i] + c * f
                z[i] = c * z[i] - s * f

            d[l] = d[l] - p_local
            e[l] = g
            e[m] = 0.0

    # 冒泡排序特征值，同时重排 z
    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p

    return d, z


# ---------------------------------------------------------------------------
# Gauss-Laguerre 求积规则
# ---------------------------------------------------------------------------

def gauss_laguerre_rule(n):
    """
    生成 n 点 Gauss-Laguerre 求积规则，用于积分：
        ∫_0^∞ exp(-x) f(x) dx ≈ Σ_{i=1}^n w_i f(x_i)

    算法：
      1) 构造 Jacobi 矩阵（对角元 d_i = 2i - 1，次对角元 e_i = sqrt(i)）
      2) 用 IMTQLX 对角化得到节点 x_i = 特征值，权重 w_i = z_i^2
      3) z 的初值为 [sqrt(zemu), 0, ..., 0]，zemu = 1（零阶矩）
    """
    # TODO: 实现 Gauss-Laguerre 求积规则的 Jacobi 矩阵构造与 IMTQLX 对角化
    #       要求返回节点 x_i（特征值）和权重 w_i，满足 sum(w_i) = 1（零阶矩）
    #       注意：节点的分布和权重的语义必须与 polynomial_spectral_preconditioner 中的映射一致
    pass


def gauss_generalized_laguerre_rule(n, alpha):
    """
    广义 Gauss-Laguerre 求积规则，用于积分：
        ∫_0^∞ exp(-x) x^α f(x) dx ≈ Σ w_i f(x_i)

    要求 alpha >= 0。
    零阶矩 zemu = Γ(α + 1)。
    Jacobi 矩阵：
        d_i = 2i - 1 + α
        e_i = sqrt(i (i + α))
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0 for generalized Laguerre quadrature.")
    zemu = math.gamma(alpha + 1.0)
    bj = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        bj[i - 1] = math.sqrt(i * (i + alpha))
    x = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        x[i - 1] = 2.0 * i - 1.0 + alpha
    w = np.zeros(n, dtype=float)
    w[0] = math.sqrt(zemu)

    x, w = imtqlx(n, x, bj, w)
    w = w ** 2
    return x, w


# ---------------------------------------------------------------------------
# Gauss-Hermite 求积规则
# ---------------------------------------------------------------------------

def gauss_hermite_rule(n):
    """
    n 点 Gauss-Hermite 求积规则（物理学家约定），用于积分：
        ∫_{-∞}^{+∞} exp(-x^2) f(x) dx ≈ Σ w_i f(x_i)

    使用 numpy.polynomial.hermite.hermgauss 获取节点与权重。
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return nodes.astype(float), weights.astype(float)


# ---------------------------------------------------------------------------
# 正交多项式构造的谱预处理算子辅助函数
# ---------------------------------------------------------------------------

def build_polynomial_preconditioner_spectrum(n, poly_type='laguerre', param=0.0):
    """
    基于正交多项式零点构造近似逆谱分布，用于谱预处理。
    返回节点和权重，可用于构造对角缩放或多项式预处理算子。
    """
    if poly_type == 'laguerre':
        return gauss_laguerre_rule(n)
    elif poly_type == 'generalized_laguerre':
        return gauss_generalized_laguerre_rule(n, param)
    elif poly_type == 'hermite':
        return gauss_hermite_rule(n)
    else:
        raise ValueError(f"Unknown poly_type: {poly_type}")
