"""
vandermonde_reconstruction.py
==============================
基于 Vandermonde 插值的分子量分布曲线重构

基于种子项目 1384_vandermonde_interp_1d 融合重构。

科学背景：
---------
在聚合反应工程中，实验测得的分子量分布通常以离散数据点
(M_i, w_i) 的形式给出。为了获得连续分布并计算任意阶矩，
需要进行多项式插值或样条重构。

本模块采用 Vandermonde 矩阵方法求解插值多项式系数，
并对高次插值进行切比雪夫节点重配置以抑制 Runge 现象。

核心数学：
----------
给定 n 个数据点 {(x_i, y_i)}_{i=1}^n，求 n-1 次多项式：

    p(x) = c_1 + c_2 x + c_3 x² + ... + c_n x^{n-1}

满足 p(x_i) = y_i。

Vandermonde 线性系统：

    [ 1   x_1   x_1²  ...  x_1^{n-1} ] [ c_1 ]   [ y_1 ]
    [ 1   x_2   x_2²  ...  x_2^{n-1} ] [ c_2 ] = [ y_2 ]
    [ ...                               ] [ ... ]   [ ... ]
    [ 1   x_n   x_n²  ...  x_n^{n-1} ] [ c_n ]   [ y_n ]

即 V c = y，解为 c = V^{-1} y（通过线性求解器）。

切比雪夫节点配置：
    x_i = cos( (2i-1)π / (2n) ),  i = 1,...,n

在区间 [a,b] 上映射：
    x_i^{mapped} = (a+b)/2 + (b-a)/2 * x_i

数值稳定性：
    条件数 cond(V) 随 n 指数增长，本模块限制 n ≤ 20，
    并采用 scaled Vandermonde 矩阵 V_{ij} = (x_i / s)^{j-1}
    其中 s = max(|x_i|)。
"""

import numpy as np
from typing import Tuple, Optional


def vandermonde_matrix_1d(x: np.ndarray, n: int, scale: Optional[float] = None) -> np.ndarray:
    """
    构造一维 Vandermonde 矩阵 V_{ij} = x_i^{j-1}
    基于 vandermonde_matrix_1d.m

    参数：
        x     : 节点坐标，形状 (n,) 或 (n,1)
        n     : 多项式阶数+1（即未知系数个数）
        scale : 缩放因子（若提供，则 V_{ij} = (x_i/scale)^{j-1}）
    """
    x = np.asarray(x).flatten()
    if scale is None:
        scale = np.max(np.abs(x))
        if scale < 1.0e-12:
            scale = 1.0

    V = np.zeros((x.size, n))
    for j in range(n):
        V[:, j] = (x / scale) ** j
    return V, scale


def vandermonde_interp_coef(x: np.ndarray, y: np.ndarray,
                            use_chebyshev: bool = False,
                            a: float = 0.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    求解插值多项式系数 c，使得 p(x_i) = y_i
    基于 vandermonde_interp_1d_coef.m

    参数：
        x, y          : 数据点
        use_chebyshev : 是否使用切比雪夫节点重配置
        a, b          : 插值区间（仅当 use_chebyshev=True 时有效）

    返回：
        c     : 多项式系数（从常数项到高次项）
        x_use : 实际使用的节点
        scale : 缩放因子
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = x.size

    if use_chebyshev:
        # 切比雪夫节点
        k = np.arange(1, n + 1)
        x_cheb = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n))
        # 映射到 [a,b]
        x_use = 0.5 * (a + b) + 0.5 * (b - a) * x_cheb
        # 通过线性插值获取对应 y 值（若 x 非均匀）
        y_use = np.interp(x_use, np.sort(x), y[np.argsort(x)])
    else:
        x_use = x.copy()
        y_use = y.copy()

    V, scale = vandermonde_matrix_1d(x_use, n)

    # 边界处理：若条件数过大，使用最小二乘或截断
    cond_v = np.linalg.cond(V)
    if cond_v > 1.0e12 or n > 20:
        # 降阶或正则化
        n_reduced = min(n, 15)
        V_red, scale = vandermonde_matrix_1d(x_use, n_reduced)
        c, residuals, rank, s = np.linalg.lstsq(V_red, y_use, rcond=None)
        # 填充高阶系数为零
        c = np.pad(c, (0, n - n_reduced), mode='constant')
    else:
        c = np.linalg.solve(V, y_use)

    return c, x_use, scale


def polyval_horner(c: np.ndarray, x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    使用 Horner 法则求多项式值。
    p(x) = c_1 + c_2*(x/scale) + c_3*(x/scale)^2 + ...

    基于 r8poly_value_horner.m 的思想。
    """
    x = np.asarray(x).flatten()
    z = x / scale
    n = c.size
    p = c[-1]
    for i in range(n - 2, -1, -1):
        p = p * z + c[i]
    return p


def reconstruct_mwd_curve(molecular_weights: np.ndarray,
                          mass_fractions: np.ndarray,
                          n_interp: int = 200,
                          log_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    利用 Vandermonde 插值重构连续的分子量分布曲线。

    参数：
        molecular_weights : 分子量数据点 [g/mol]
        mass_fractions    : 对应质量分数
        n_interp          : 插值后输出点数
        log_scale         : 是否在 log(M) 坐标上进行插值

    返回：
        M_interp : 插值后的分子量网格
        w_interp : 插值后的质量分数
        c        : 多项式系数
    """
    mw = np.asarray(molecular_weights, dtype=float)
    wf = np.asarray(mass_fractions, dtype=float)

    # 排序
    idx = np.argsort(mw)
    mw = mw[idx]
    wf = wf[idx]

    # 非负归一化
    wf = np.maximum(wf, 0.0)
    sum_wf = np.sum(wf)
    if sum_wf > 1.0e-15:
        wf /= sum_wf

    if log_scale:
        x_data = np.log10(mw)
    else:
        x_data = mw.copy()

    # 边界处理
    x_min, x_max = x_data[0], x_data[-1]

    # 使用切比雪夫节点重配置
    c, _, scale = vandermonde_interp_coef(x_data, wf, use_chebyshev=True,
                                          a=x_min, b=x_max)

    # 插值网格
    x_interp = np.linspace(x_min, x_max, n_interp)
    w_interp = polyval_horner(c, x_interp, scale=scale)

    # 非负截断与重归一化
    w_interp = np.maximum(w_interp, 0.0)
    integral = np.trapezoid(w_interp, x_interp)
    if integral > 1.0e-15:
        w_interp /= integral

    if log_scale:
        M_interp = 10.0 ** x_interp
    else:
        M_interp = x_interp

    return M_interp, w_interp, c


def derivative_mwd_curve(c: np.ndarray,
                         molecular_weights: np.ndarray,
                         scale: float = 1.0,
                         log_scale: bool = True) -> np.ndarray:
    """
    计算重构 MWD 曲线的导数 dw/dM。

    若 w(x) = Σ c_k (x/scale)^{k-1}，则
        dw/dx = Σ_{k=2}^{n} (k-1) c_k (x/scale)^{k-2} / scale

    在 log 坐标下：
        dw/dM = (dw/dx) * (1/(M ln 10))
    """
    x = np.log10(molecular_weights) if log_scale else molecular_weights
    x = np.asarray(x)

    n = c.size
    if n <= 1:
        return np.zeros_like(x)

    c_deriv = np.zeros(n - 1)
    for k in range(1, n):
        c_deriv[k - 1] = k * c[k] / scale

    dw_dx = polyval_horner(c_deriv, x, scale=1.0)

    if log_scale:
        M = np.asarray(molecular_weights)
        M = np.maximum(M, 1.0e-12)
        dw_dM = dw_dx / (M * np.log(10.0))
    else:
        dw_dM = dw_dx

    return dw_dM


def monomial_moments_from_coeffs(c: np.ndarray,
                                 scale: float,
                                 max_moment: int = 3,
                                 log_scale: bool = True) -> np.ndarray:
    """
    由插值多项式系数直接计算 log(M) 坐标下的矩。

    在 log 坐标 x = log10(M) 下，M = 10^x，dM = M ln(10) dx
    质量分数 w(M)dM = w(10^x) 10^x ln(10) dx

    若 p(x) = Σ c_k (x/scale)^{k-1} 为 w 的近似，则
    第 m 阶矩（关于 M）：
        μ_m = ∫ M^m w(M) dM ≈ Σ c_k ∫ (x/scale)^{k-1} 10^{m x} ln(10) dx

    采用数值积分近似。
    """
    # 简化为在插值区间上的数值积分
    n = c.size
    x_grid = np.linspace(-2.0, 6.0, 500)  # log10(M) 从 0.01 到 1e6
    w_grid = polyval_horner(c, x_grid, scale=scale)
    w_grid = np.maximum(w_grid, 0.0)

    moments = np.zeros(max_moment + 1)
    for m in range(max_moment + 1):
        M_grid = 10.0 ** x_grid
        integrand = (M_grid ** m) * w_grid * M_grid * np.log(10.0)
        moments[m] = np.trapezoid(integrand, x_grid)

    # 归一化
    if moments[0] > 1.0e-15:
        moments /= moments[0]
        # 恢复零阶矩为1
        moments[0] = 1.0

    return moments
