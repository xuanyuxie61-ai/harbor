# -*- coding: utf-8 -*-
"""
von_neumann_stability.py
========================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

对 Euler 方程、辐射扩散方程和耦合系统执行 von Neumann 稳定性分析,
给出 CFL 条件与最大允许时间步长.

Von Neumann 方法
----------------
将离散扰动展开为 Fourier 模式::

    u_j^n = hat{u}^n * exp(i k j h)

代入差分格式得到放大因子::

    g(k, dt, h) = hat{u}^{n+1} / hat{u}^n

稳定性要求::

    |g| <= 1 + O(dt)   for all  k in [-pi/h, pi/h]

CFL 条件
--------
1. 对流:  dt <= C_CFL * h / (|v| + c_s)
2. 扩散:  dt <= C_diff * h^2 / D  (D 为扩散系数)
3. 源项:  dt <= C_src / max(|source eigenvalue|)

辐射扩散 CFL::

    dt_rad <= 0.5 * h^2 / (c * lambda_mfp)

其中 lambda_mfp = 1 / (kappa * rho) 为光子平均自由程.

映射种子项目
-----------
- 517 (henon_orbit)       → 离散映射的稳定性由 Jacobian 矩阵的特征值决定,
  Henon 映射的离散结构直接类比 von Neumann 放大因子在 k 空间的轨迹
- 1363 (tsp_brute)        → 在组合空间搜索最严苛 (最紧) 的 CFL 约束
  等价于遍历所有波数方向, 找到最危险的 k 矢量
- 321 (dueling_idiots)    → 概率 duel 中的几何分布收敛性
  类比放大因子 |g|^n 的几何衰减率
"""

from __future__ import annotations
import math
import cmath
from typing import Tuple, List, Dict

from physical_constants import C_LIGHT


# ---------------------------------------------------------------------------
# 放大因子解析表达式
# ---------------------------------------------------------------------------
def amplification_factor_advection_1upwind(
    v: float, k: float, h: float, dt: float
) -> complex:
    """一阶迎风对流的放大因子.

    g = 1 - nu + nu * exp(-i k h),  nu = v dt / h (Courant 数)
    """
    nu = v * dt / h
    return 1.0 - nu + nu * cmath.exp(-1j * k * h)


def amplification_factor_ftcs_diffusion(
    D: float, k: float, h: float, dt: float
) -> complex:
    """FTCS 扩散的放大因子.

    g = 1 - 4 r sin^2(k h / 2),  r = D dt / h^2 (扩散数)
    """
    r = D * dt / (h * h)
    return 1.0 - 4.0 * r * (math.sin(k * h / 2.0) ** 2)


def amplification_factor_lax_friedrichs(
    v: float, k: float, h: float, dt: float
) -> complex:
    """Lax-Friedrichs 对流的放大因子.

    g = cos(k h) - i nu sin(k h),  nu = v dt / h
    """
    nu = v * dt / h
    return math.cos(k * h) - 1j * nu * math.sin(k * h)


def amplification_factor_leapfrog(
    v: float, k: float, h: float, dt: float
) -> List[complex]:
    """Leapfrog (CTCS) 格式的两个放大因子 (二次方程的两个根).

    g^2 + 2 i nu sin(k h) g - 1 = 0
    """
    nu = v * dt / h
    s = math.sin(k * h)
    # g = -i nu s +/- sqrt(1 - nu^2 s^2)
    disc = 1.0 - nu * nu * s * s
    if disc >= 0:
        sq = math.sqrt(disc)
        return [-1j * nu * s + sq, -1j * nu * s - sq]
    else:
        sq = cmath.sqrt(disc)
        return [-1j * nu * s + sq, -1j * nu * s - sq]


def amplification_factor_rk4_advection(
    v: float, k: float, h: float, dt: float
) -> complex:
    """四阶 Runge-Kutta 用于半离散对流  du/dt = -v D_4 u.

    g = 1 + z + z^2/2 + z^3/6 + z^4/24,  z = -i nu * phi(k h)
    其中 phi 是四阶中心差分的修正波数.
    """
    nu = v * dt / h
    kh = k * h
    # 四阶修正波数
    phi = (8.0 * math.sin(kh) - math.sin(2.0 * kh)) / 6.0
    z = -1j * nu * phi
    return 1.0 + z + 0.5 * z ** 2 + (1.0 / 6.0) * z ** 3 + (1.0 / 24.0) * z ** 4


# ---------------------------------------------------------------------------
# CFL 限制
# ---------------------------------------------------------------------------
def cfl_advection(h: float, v_max: float, c_s: float, C_cfl: float = 0.5
                  ) -> float:
    """对流 CFL 时间步长上限.

    dt_max = C_CFL * h / (|v|_max + c_s)

    Parameters
    ----------
    h     : float  网格间距 [cm]
    v_max : float  最大流速 [cm/s]
    c_s   : float  声速 [cm/s]
    C_cfl : float  安全系数 (默认 0.5)
    """
    lambda_max = abs(v_max) + c_s
    if lambda_max <= 0.0:
        return float("inf")
    return C_cfl * h / lambda_max


def cfl_radiation_diffusion(h: float, kappa: float, rho: float,
                            C_cfl: float = 0.4) -> float:
    """辐射扩散 CFL 时间步长上限.

    dt_max = C_CFL * h^2 / (c * lambda_mfp)
    lambda_mfp = 1 / (kappa * rho)

    Parameters
    ----------
    h     : float  网格间距 [cm]
    kappa : float  不透明度 [cm^2/g]
    rho   : float  密度 [g/cm^3]
    C_cfl : float  安全系数
    """
    lam = 1.0 / max(kappa * rho, 1.0e-30)
    D_eff = C_LIGHT * lam / 3.0
    if D_eff <= 0.0:
        return float("inf")
    return C_cfl * h * h / D_eff


def cfl_source_term(tau_cool_s: float, C_cfl: float = 0.3) -> float:
    """冷却时标导致的源项 CFL.

    dt_max = C_CFL * |tau_cool|
    """
    return C_cfl * abs(tau_cool_s)


# ---------------------------------------------------------------------------
# 数值 von Neumann 扫描
# ---------------------------------------------------------------------------
def scan_stability(
    scheme_name: str,
    amp_func,
    v_or_D: float, k_range: Tuple[float, float],
    h: float, dt: float, n_k: int = 1000
) -> Dict[str, float]:
    """扫描给定格式在 [k_min, k_max] 上的最大放大因子 |g|.

    Parameters
    ----------
    scheme_name : str  格式名称 (用于日志)
    amp_func    : Callable  接受 (v_or_D, k, h, dt) 返回复数或复数列表
    v_or_D      : float  速度或扩散系数
    k_range     : (k_min, k_max)  波数范围
    h           : float  网格间距
    dt          : float  时间步长
    n_k         : int    扫描点数
    """
    k_min, k_max = k_range
    max_amp = 0.0
    worst_k = 0.0
    unstable_count = 0
    for i in range(n_k + 1):
        k = k_min + (k_max - k_min) * i / n_k
        result = amp_func(v_or_D, k, h, dt)
        if isinstance(result, list):
            amps = [abs(g) for g in result]
            amp = max(amps)
        else:
            amp = abs(result)
        if amp > max_amp:
            max_amp = amp
            worst_k = k
        if amp > 1.0 + 1.0e-10:
            unstable_count += 1
    return {
        "scheme": scheme_name,
        "max |g|": max_amp,
        "worst k": worst_k,
        "n_unstable_samples": unstable_count,
        "stable": unstable_count == 0,
    }


# ---------------------------------------------------------------------------
# Henon-map 风格的稳定性图 (映射 517)
# ---------------------------------------------------------------------------
def henon_stability_scan(
    c_param: float, n_iter: int
) -> Dict[str, float]:
    """将 von Neumann 放大因子的实部/虚部演化视作 Henon 型映射,
    扫描其在相平面上的有界性.

    映射::

        x_{n+1} = c - (y_n - x_n^2) s
        y_{n+1} = x_n s + (y_n - x_n^2) c

    其中 c = cos(alpha), s = sin(alpha). 当 |x|, |y| < 1 时视为稳定
    (类比 |g| < 1).

    Parameters
    ----------
    c_param : float  cos(alpha), 控制参数
    n_iter  : int    迭代次数
    """
    s_param = math.sqrt(max(0.0, 1.0 - c_param * c_param))
    x, y = 0.0, 0.0
    bounded_count = 0
    for _ in range(n_iter):
        if abs(x) < 1.0 and abs(y) < 1.0:
            bounded_count += 1
        xnew = x * c_param - (y - x * x) * s_param
        ynew = x * s_param + (y - x * x) * c_param
        x, y = xnew, ynew
    return {
        "c_param": c_param,
        "bounded_fraction": bounded_count / n_iter,
        "final |g|^2": x * x + y * y,
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """验证 CFL 条件与已知解析结果一致."""
    # FTCS 扩散稳定性要求 r = D dt / h^2 <= 1/2
    D, h, dt = 1.0, 0.1, 0.004
    r = D * dt / (h * h)  # r = 0.4 < 0.5 -> stable
    result = scan_stability(
        "FTCS-diffusion", amplification_factor_ftcs_diffusion,
        D, (-math.pi / h, math.pi / h), h, dt, n_k=500
    )
    assert result["stable"], "FTCS diffusion with r=0.4 should be stable"
    # 不稳定情形
    dt_unstable = 0.006  # r = 0.6 > 0.5
    result2 = scan_stability(
        "FTCS-diffusion-unstable", amplification_factor_ftcs_diffusion,
        D, (-math.pi / h, math.pi / h), h, dt_unstable, n_k=500
    )
    assert not result2["stable"], "FTCS diffusion with r=0.6 should be unstable"
    return True


if __name__ == "__main__":
    _self_check()
    print("von_neumann_stability self-check passed.")
    # 示例: BNS 并合后 kilonova 辐射扩散
    h = 1.0e5  # 1 km 网格
    kappa = 10.0  # 镧系不透明度 [cm^2/g]
    rho = 1.0e-10  # 稀薄喷射物 [g/cm^3]
    dt_adv = cfl_advection(h, v_max=1.0e9, c_s=1.0e8)
    dt_rad = cfl_radiation_diffusion(h, kappa, rho)
    print(f"  dt_advection = {dt_adv:.3e} s")
    print(f"  dt_radiation = {dt_rad:.3e} s")
    print(f"  recommended dt = {min(dt_adv, dt_rad):.3e} s")
