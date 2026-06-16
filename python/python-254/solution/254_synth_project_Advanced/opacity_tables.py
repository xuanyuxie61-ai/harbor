# -*- coding: utf-8 -*-
"""
opacity_tables.py
=================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

用 Chebyshev 插值构造波长相关的不透明度表  kappa(lambda, T, rho).

Lanthanide 不透明度
-------------------
镧系元素的束缚-束缚跃迁在光学/近红外产生大量谱线, 导致
kappa ~ 1--100 cm^2/g (比铁族元素高 1-2 个数量级)::

    kappa(lambda) = kappa_bound(lambda) + kappa_free(lambda) + kappa_scatter

其中::

    kappa_bound ~ sum_lines (pi e^2 / m_e c) * f_ij * phi(lambda - lambda_ij)
    kappa_free ~ 0.2 * (1 + X_H) * rho * T^{-3.5}  (Kramers)
    kappa_scatter ~ sigma_T / (mu_e m_u)

Chebyshev 插值
--------------
在 [a, b] 上用 n 个 Chebyshev 节点::

    x_k = (a + b)/2 + (b - a)/2 * cos((2k+1) pi / (2n))

构造 Newton 形式的插值多项式, 再求最大误差.

映射种子项目
-----------
- 014 (approx_chebyshev)  → Chebyshev 节点生成 (chebyspace),
  差商计算 (divdif), Newton 形式求值 (interp)  的完整实现
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable


# ---------------------------------------------------------------------------
# Chebyshev 节点
# ---------------------------------------------------------------------------
def chebyspace(a: float, b: float, n: int) -> List[float]:
    """生成 [a, b] 上的 n 个 Chebyshev 节点.

    使用 "第一类" Chebyshev 节点 (零点)::

        x_k = (a+b)/2 + (b-a)/2 * cos((2k+1) pi / (2n))

    k = 0..n-1. 这些节点保证互不相同.
    """
    if n < 1:
        raise ValueError("n >= 1")
    if n == 1:
        return [(a + b) / 2.0]
    nodes = []
    for k in range(n):
        theta = (2 * k + 1) * math.pi / (2 * n)
        c = math.cos(theta)
        x = 0.5 * ((a + b) + (b - a) * c)
        nodes.append(x)
    # 按升序排列
    nodes.sort()
    return nodes


# ---------------------------------------------------------------------------
# 差商表
# ---------------------------------------------------------------------------
def divided_differences(x: List[float], y: List[float]) -> List[float]:
    """计算差商表, 返回最高阶差商组成的数组 (Newton 形式系数).

    算法::

        dd[j] = y[j]
        for i in 2..n:
            for j in n..i (降序):
                dd[j] = (dd[j] - dd[j-1]) / (x[j] - x[j - i + 1])
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have same length")
    dd = y[:]
    for i in range(2, n + 1):
        for j in range(n - 1, i - 2, -1):
            denom = x[j] - x[j - i + 1]
            if abs(denom) < 1.0e-30:
                # 跳过几乎重合的节点对
                dd[j] = 0.0
            else:
                dd[j] = (dd[j] - dd[j - 1]) / denom
    return dd


def evaluate_newton(xd: List[float], dd: List[float],
                    xp: List[float]) -> List[float]:
    """用 Newton 形式在 xp 处求插值多项式.

    P(x) = dd[0] + (x - xd[0]) * (dd[1] + (x - xd[1]) * (...))
    """
    nd = len(xd)
    result = []
    for x in xp:
        yp = dd[nd - 1]
        for i in range(nd - 2, -1, -1):
            yp = dd[i] + (x - xd[i]) * yp
        result.append(yp)
    return result


def chebyshev_approximation(f: Callable[[float], float],
                            a: float, b: float, n: int
                            ) -> Tuple[List[float], List[float], float]:
    """用 n 个 Chebyshev 节点逼近函数 f 在 [a, b] 上的行为.

    Returns
    -------
    (xd, yp, maxerr)
        xd     : Chebyshev 节点
        yp     : 在 101 个均匀点上的插值
        maxerr : 与真实 f 的最大绝对误差
    """
    xd = chebyspace(a, b, n)
    yd = [f(x) for x in xd]
    dd = divided_differences(xd, yd)
    # 评估误差
    ne = 1001
    xe = [a + (b - a) * i / (ne - 1) for i in range(ne)]
    ye = evaluate_newton(xd, dd, xe)
    fe = [f(x) for x in xe]
    maxerr = max(abs(ye[i] - fe[i]) for i in range(ne))
    # 返回 101 个点
    xp = [a + (b - a) * i / 100 for i in range(101)]
    yp = evaluate_newton(xd, dd, xp)
    return xd, yp, maxerr


# ---------------------------------------------------------------------------
# Kilonova 不透明度模型
# ---------------------------------------------------------------------------
def kappa_bound_lanthanide(lambda_angstrom: float,
                           T_K: float,
                           Ye: float) -> float:
    """镧系束缚-束缚不透明度 (简化 Tanaka et al. 2017 模型).

    kappa_bound ~ kappa_0 * (T / T_0)^alpha * (Ye_c / Ye)^beta
                * (lambda / lambda_0)^gamma * exp(-lambda / lambda_cut)

    Parameters
    ----------
    lambda_angstrom : float  波长 [Angstrom]
    T_K             : float  温度 [K]
    Ye              : float  电子丰度 (0.1 -- 0.5)
    """
    kappa_0 = 30.0          # cm^2/g (参考值)
    T_0 = 5000.0            # K
    lambda_0 = 1.0e4        # Angstrom
    lambda_cut = 2.0e4      # Angstrom (紫外截断)
    Ye_crit = 0.25
    alpha = 1.5
    beta = 1.2
    gamma = -1.5
    T_safe = max(T_K, 1000.0)
    Ye_safe = max(Ye, 0.05)
    lam_safe = max(lambda_angstrom, 100.0)
    kappa = (kappa_0
             * (T_safe / T_0) ** alpha
             * (Ye_crit / Ye_safe) ** beta
             * (lam_safe / lambda_0) ** gamma
             * math.exp(-lam_safe / lambda_cut))
    return max(kappa, 1.0e-3)


def kappa_free_free(lambda_angstrom: float, T_K: float, rho: float) -> float:
    """自由-自由 (Kramers) 不透明度.

    kappa_ff = 4.0e25 * (1 + X) * rho * T^{-3.5} * lambda^2  [cm^2/g]

    粗略近似 (适用于 T > 3000 K).
    """
    X = 0.0  # 中子星抛射物无氢
    T_safe = max(T_K, 3000.0)
    lam_cm = lambda_angstrom * 1.0e-8
    return 4.0e25 * (1.0 + X) * rho * (T_safe ** -3.5) * (lam_cm * lam_cm)


def kappa_scattering_electron(Ye: float) -> float:
    """电子散射不透明度 (Thomson).

    kappa_es = sigma_T / (mu_e * m_u)
    mu_e = 1 / Ye
    """
    from physical_constants import SIGMA_THOMSON, M_PROTON
    mu_e = 1.0 / max(Ye, 0.05)
    return SIGMA_THOMSON / (mu_e * M_PROTON)


def kappa_total(lambda_angstrom: float, T_K: float, rho: float, Ye: float
                ) -> float:
    """总 Rosseland 平均不透明度.

    1 / kappa_R = integral (1 / kappa_lambda) d B_lambda / dT dlambda
                 / integral d B_lambda / dT dlambda

    此处简化为各机制的算术平均 (定性).
    """
    kb = kappa_bound_lanthanide(lambda_angstrom, T_K, Ye)
    kf = kappa_free_free(lambda_angstrom, T_K, rho)
    ks = kappa_scattering_electron(Ye)
    return max(kb, 1.0e-3) + max(kf, 1.0e-3) + max(ks, 1.0e-3)


# ---------------------------------------------------------------------------
# 用 Chebyshev 拟合 kappa(lambda) 曲线
# ---------------------------------------------------------------------------
def build_chebyshev_opacity_table(T_K: float, rho: float, Ye: float, n_cheb: int
                                  ) -> dict:
    """在波长区间 [3000, 30000] Angstrom 上 Chebyshev 拟合 kappa(lambda).

    Returns
    -------
    dict  包含 nodes, coeffs, maxerr, tabulated values
    """
    a, b = 3000.0, 30000.0

    def f(lam):
        return kappa_total(lam, T_K, rho, Ye)

    xd, yp, maxerr = chebyshev_approximation(f, a, b, n_cheb)
    # 系数
    yd = [f(x) for x in xd]
    dd = divided_differences(xd, yd)
    return {
        "T_K": T_K, "rho": rho, "Ye": Ye,
        "nodes": xd,
        "cheb_coeffs": dd,
        "maxerr": maxerr,
        "values_101": yp,
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """用 sin(x) 验证 Chebyshev 插值精度."""
    f = math.sin
    a, b = 0.0, math.pi
    for n in [4, 8, 16]:
        _, _, maxerr = chebyshev_approximation(f, a, b, n)
        if n == 16 and maxerr > 1.0e-8:
            raise AssertionError(f"Chebyshev n=16: err {maxerr:.3e} unexpectedly large")
    # 物理量级
    kappa = kappa_total(10000.0, 5000.0, 1.0e-13, 0.2)
    assert 0.1 < kappa < 1000.0, f"kappa = {kappa:.3e} out of range"
    return True


if __name__ == "__main__":
    _self_check()
    print("opacity_tables self-check passed.")
    T_K, rho, Ye = 5000.0, 1.0e-13, 0.2
    table = build_chebyshev_opacity_table(T_K, rho, Ye, n_cheb=12)
    print(f"  T = {T_K:.0f} K, rho = {rho:.2e} g/cm^3, Ye = {Ye:.2f}")
    print(f"  Chebyshev max error = {table['maxerr']:.4e} cm^2/g")
    print(f"  nodes : {[f'{x:.0f}' for x in table['nodes']]}")
