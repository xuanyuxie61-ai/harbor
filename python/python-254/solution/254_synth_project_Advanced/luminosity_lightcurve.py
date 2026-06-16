# -*- coding: utf-8 -*-
"""
luminosity_lightcurve.py
========================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

从辐射转移结果计算 kilonova 光度曲线 L(t) 与有效温度 T_eff(t).

Arnett 模型 (解析近似)
-----------------------
对于放射性加热的抛射物, 光度::

    L(t) = Q_dot(t_dep) * exp(-tau_m / t_dep)

其中 t_dep 为沉积时标, tau_m 为有效光学深度.

更精确的半解析模型 (Metzger 2017)::

    L(t) = alpha * M_ej^{a} * v^{b} * kappa^{c} * t^{d} * Q_dot(t)

其中指数通过拟合辐射转移模拟得到.

有效温度::

    L = 4 pi R_ph^2 sigma_SB T_eff^4

光球半径 R_ph::

    integral_{R_in}^{R_ph} kappa rho dr = 2/3

色温度修正 (由于非灰体辐射)::

    T_color ~ 1.5 T_eff  (早期, 蓝色成分)
    T_color ~ 0.8 T_eff  (晚期, 红色成分)

映射种子项目
-----------
- 677 (line_distance_stats) → 距离统计: 光球半径的蒙特卡洛
  估计等价于单位线段上两点距离的统计问题
- 049_asa239 (alnorm)       → 正态 CDF 用于拟合残差的统计检验
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict

from physical_constants import SIGMA_SB, C_LIGHT, M_SUN, DAY_S


# ---------------------------------------------------------------------------
# 光球半径
# ---------------------------------------------------------------------------
def photosphere_radius(
    r_grid_cm: List[float],
    rho_grid: List[float],
    kappa_grid: List[float],
    target_tau: float = 2.0 / 3.0,
) -> float:
    """计算光球半径  R_ph, 使得::

        integral_{R_ph}^{R_out} kappa(rho) dr = 2/3

    从外向内积分, 找到第一个超过 2/3 的位置.
    """
    n = len(r_grid_cm)
    if n < 2:
        return r_grid_cm[-1] if r_grid_cm else 0.0
    tau = 0.0
    for i in range(n - 1, 0, -1):
        dr = r_grid_cm[i] - r_grid_cm[i - 1]
        kappa_avg = 0.5 * (kappa_grid[i] + kappa_grid[i - 1])
        rho_avg = 0.5 * (rho_grid[i] + rho_grid[i - 1])
        tau += kappa_avg * rho_avg * dr
        if tau >= target_tau:
            return r_grid_cm[i]
    return r_grid_cm[0]


# ---------------------------------------------------------------------------
# 有效温度
# ---------------------------------------------------------------------------
def effective_temperature(L_erg_s: float, R_ph_cm: float) -> float:
    """有效温度  T_eff = (L / (4 pi R_ph^2 sigma))^{1/4}  [K]."""
    if R_ph_cm <= 0 or L_erg_s <= 0:
        return 0.0
    T4 = L_erg_s / (4.0 * math.pi * R_ph_cm ** 2 * SIGMA_SB)
    return T4 ** 0.25


# ---------------------------------------------------------------------------
# 解析光度曲线 (Arnett 型)
# ---------------------------------------------------------------------------
def arnett_lightcurve(
    t_days: List[float],
    M_ej_Msun: float,
    v_c_km_s: float,
    kappa_cgs: float,
    X_r: float = 0.01,
) -> Dict[str, List[float]]:
    """Arnett 型解析光度曲线.

    L(t) = M_ej * Q_dot(t) * (1 - exp(-tau_m / t))

    其中 tau_m = kappa M_ej / (beta c R_0)  为特征光学深度.
    beta = 13.7 (几何因子)
    R_0 = v * t_0  初始半径

    Parameters
    ----------
    t_days     : List[float]  时间 [天]
    M_ej_Msun  : float        抛射物质量 [M_sun]
    v_c_km_s   : float        特征速度 [km/s]
    kappa_cgs  : float        不透明度 [cm^2/g]
    X_r        : float        r-process 质量分数
    """
    M_ej = M_ej_Msun * M_SUN
    v_c = v_c_km_s * 1.0e5
    beta = 13.7
    t_0 = 1.0 * DAY_S
    R_0 = v_c * t_0
    tau_m = kappa_cgs * M_ej / (beta * C_LIGHT * R_0)

    L_list = []
    T_eff_list = []
    for t_day in t_days:
        t_s = t_day * DAY_S
        # 加热率
        t_day_eff = max(t_day, 0.01)
        Q_dot = 2.0e10 * (t_day_eff ** -1.3) * X_r  # erg/s/g
        # 沉积因子
        if tau_m > 0:
            f_dep = 1.0 - math.exp(-tau_m / max(t_day, 0.001))
        else:
            f_dep = 1.0
        L = M_ej * Q_dot * f_dep
        L_list.append(L)
        # 光球半径 (homologous expansion)
        R_ph = v_c * t_s
        T_eff = effective_temperature(L, R_ph)
        T_eff_list.append(T_eff)

    return {
        "L_erg_s": L_list,
        "T_eff_K": T_eff_list,
        "tau_m": tau_m,
    }


# ---------------------------------------------------------------------------
# 峰值光度与时间
# ---------------------------------------------------------------------------
def peak_luminosity(L_array: List[float], t_days: List[float]
                    ) -> Tuple[float, float]:
    """找出峰值光度 L_peak 及其对应时间 t_peak."""
    if not L_array:
        return 0.0, 0.0
    L_max = max(L_array)
    i_max = L_array.index(L_max)
    return L_max, t_days[i_max]


# ---------------------------------------------------------------------------
# 统计残差检验 (映射 677/049)
# ---------------------------------------------------------------------------
def residuals_normality_check(residuals: List[float]) -> Dict[str, float]:
    """检验残差是否近似正态 (粗略).

    计算均值、标准差、偏度.
    """
    n = len(residuals)
    if n < 3:
        return {"mean": 0.0, "std": 0.0, "skewness": 0.0}
    mean = sum(residuals) / n
    var = sum((r - mean) ** 2 for r in residuals) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std > 0:
        skew = sum(((r - mean) / std) ** 3 for r in residuals) / n
    else:
        skew = 0.0
    return {"mean": mean, "std": std, "skewness": skew}


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """验证光度曲线解析预期."""
    t_days = [1.0, 3.0, 7.0, 14.0, 30.0]
    result = arnett_lightcurve(t_days, M_ej_Msun=0.05, v_c_km_s=10000,
                               kappa_cgs=10.0, X_r=0.01)
    L = result["L_erg_s"]
    # 光度应随时间下降
    if L[0] <= L[-1]:
        raise AssertionError("Luminosity should decrease with time")
    # 有效温度为正
    for T in result["T_eff_K"]:
        if T <= 0:
            raise AssertionError("T_eff must be positive")
    return True


if __name__ == "__main__":
    _self_check()
    print("luminosity_lightcurve self-check passed.")
    t_days = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    result = arnett_lightcurve(t_days, M_ej_Msun=0.05, v_c_km_s=10000,
                               kappa_cgs=10.0)
    L_peak, t_peak = peak_luminosity(result["L_erg_s"], t_days)
    print(f"  L_peak = {L_peak:.3e} erg/s  at t = {t_peak:.2f} days")
    print(f"  T_eff at t=1d = {result['T_eff_K'][1]:.0f} K")
    print(f"  tau_m = {result['tau_m']:.2f}")
