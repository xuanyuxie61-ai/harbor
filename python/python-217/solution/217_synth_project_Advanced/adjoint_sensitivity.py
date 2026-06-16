"""
adjoint_sensitivity.py
----------------------
伴随灵敏度分析模块 —— 映射自种子项目 280_diff_forward
核心思想：利用前向差分近似计算目标泛函对设计变量的梯度，
并由此构建高阶中心差分、Richardson 外推以及伴随方程的离散梯度。

科学背景：
    在鲁棒优化问题中，目标泛函通常包含对不确定性集合的极值或期望：
        Phi(x) = sup_{w in W} J(x, w)     （worst-case 鲁棒目标）
        Psi(x) = E_w[ J(x, w) ] + beta * Var_w[ J(x, w) ]   （均值-方差目标）
    其梯度可由伴随方程给出：
        dPhi/dx = partial J / partial x + (partial c / partial x)^T lambda
    其中 lambda 满足伴随方程 (partial c / partial u)^T lambda = -partial J / partial u.

本模块提供的功能：
    1. 前向差分 gradient_forward
    2. 中心差分 gradient_central
    3. Richardson 外推 gradient_richardson
    4. 伴随方程求解 solve_adjoint (基于 Toeplitz/circulant 结构)
    5. 鲁棒目标的复合梯度 robust_objective_gradient
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

# 数值微分中的机器精度安全下界
_EPS_SQRT = np.sqrt(np.finfo(float).eps)
_EPS_CBRT = np.cbrt(np.finfo(float).eps)


def gradient_forward(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float | None = None,
) -> np.ndarray:
    """
    前向差分梯度:
        df/dx_i  ≈  [ f(x + h e_i) - f(x) ] / h
    截断误差 O(h)，舍入误差 O(eps/h)，最优步长 h* ~ sqrt(eps).

    参数
    ----
    f : 目标函数 R^n -> R
    x : 当前点
    h : 步长，若为 None 则自动选取 sqrt(eps) * max(1, ||x||)

    返回
    ----
    g : 梯度向量
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if h is None:
        h = _EPS_SQRT * max(1.0, float(np.linalg.norm(x)))
    if h <= 0.0:
        raise ValueError("gradient_forward: 步长 h 必须为正")
    f0 = float(f(x))
    g = np.zeros(n)
    for i in range(n):
        x_p = x.copy()
        x_p[i] += h
        g[i] = (float(f(x_p)) - f0) / h
    return g


def gradient_central(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float | None = None,
) -> np.ndarray:
    """
    中心差分梯度:
        df/dx_i  ≈  [ f(x + h e_i) - f(x - h e_i) ] / (2h)
    截断误差 O(h^2)，最优步长 h* ~ eps^{1/3}.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if h is None:
        h = _EPS_CBRT * max(1.0, float(np.linalg.norm(x)))
    if h <= 0.0:
        raise ValueError("gradient_central: 步长 h 必须为正")
    g = np.zeros(n)
    for i in range(n):
        x_p = x.copy()
        x_m = x.copy()
        x_p[i] += h
        x_m[i] -= h
        g[i] = (float(f(x_p)) - float(f(x_m))) / (2.0 * h)
    return g


def gradient_richardson(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-2,
    order: int = 3,
) -> np.ndarray:
    """
    Richardson 外推中心差分：
        D^{(0)}_h = [f(x+h)-f(x-h)]/(2h)
        D^{(k)}_h = [ 4^k D^{(k-1)}_{h/2} - D^{(k-1)}_h ] / (4^k - 1)
    可将精度提升至 O(h^{2*order}).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if h <= 0.0:
        raise ValueError("gradient_richardson: h 必须为正")
    if order < 1:
        raise ValueError("order 必须 >= 1")

    # 构建不同尺度的中心差分
    scales = [h / (2 ** k) for k in range(order)]
    D = np.zeros((order, n))
    for k, hk in enumerate(scales):
        for i in range(n):
            x_p = x.copy()
            x_m = x.copy()
            x_p[i] += hk
            x_m[i] -= hk
            D[k, i] = (float(f(x_p)) - float(f(x_m))) / (2.0 * hk)

    # Richardson 外推表
    for k in range(1, order):
        factor = 4.0 ** k
        D[k:] = (factor * D[k:] - D[k - 1 : -1 if k > 1 else None]) / (factor - 1.0)
    return D[-1]


def solve_adjoint(
    K: np.ndarray,
    rhs: np.ndarray,
    reg: float = 1e-8,
) -> np.ndarray:
    """
    求解伴随方程  K^T lambda = -rhs,
    其中 K 为 PDE 离散后的刚度/对流-扩散矩阵。
    使用 Tikhonov 正则化保证正定性：
        (K^T K + reg I) lambda = -K^T rhs
    对应于最小二乘伴随问题 min ||K lambda + rhs||^2 + reg ||lambda||^2.

    参数
    ----
    K : (m, n) 离散 PDE 算子
    rhs : (m,) 目标泛函对状态的偏导
    reg : Tikhonov 正则参数

    返回
    ----
    lam : (n,) 伴随变量
    """
    K = np.atleast_2d(K)
    rhs = np.asarray(rhs, dtype=float).ravel()
    if K.shape[0] != rhs.size:
        raise ValueError("solve_adjoint: K 的行数与 rhs 不匹配")
    KtK = K.T @ K
    Ktr = K.T @ rhs
    A = KtK + reg * np.eye(K.shape[1])
    # 使用 Cholesky 分解保证数值稳定
    try:
        L = np.linalg.cholesky(A)
        z = np.linalg.solve(L, -Ktr)
        lam = np.linalg.solve(L.T, z)
    except np.linalg.LinAlgError:
        # 回退到 lstsq
        lam, *_ = np.linalg.lstsq(A, -Ktr, rcond=None)
    return lam


def robust_objective_gradient(
    x: np.ndarray,
    mean_fn: Callable[[np.ndarray], float],
    worst_case_fn: Callable[[np.ndarray], float],
    risk_weight: float,
    h: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """
    计算均值-方差-最坏-case 复合鲁棒目标的梯度：
        F(x) = (1 - risk_weight) * E[J] + risk_weight * sup_w J(x,w)
    采用中心差分并加权混合。

    返回 (梯度, 目标值估计)
    """
    if not 0.0 <= risk_weight <= 1.0:
        raise ValueError("risk_weight 必须位于 [0, 1]")
    g_mean = gradient_central(mean_fn, x, h=h)
    g_wc = gradient_central(worst_case_fn, x, h=h)
    g = (1.0 - risk_weight) * g_mean + risk_weight * g_wc
    fval = (1.0 - risk_weight) * float(mean_fn(x)) + risk_weight * float(worst_case_fn(x))
    return g, fval


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    def rosenbrock(z):
        return sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2)

    x0 = np.array([-1.0, 1.0, -1.0])
    gf = gradient_forward(rosenbrock, x0)
    gc = gradient_central(rosenbrock, x0)
    gr = gradient_richardson(rosenbrock, x0, h=1e-2, order=3)
    print("forward :", gf)
    print("central :", gc)
    print("richard :", gr)
