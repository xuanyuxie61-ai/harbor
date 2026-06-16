# -*- coding: utf-8 -*-
"""
parker_wind.py
--------------
Parker 太阳风跨声速解 + 逃逸概率 (committor) 分析.

物理模型
--------
Parker (1958) 等温太阳风模型:

    v dv/dr = - (1/rho) dp/dr - G M_sun / r^2
    质量通量守恒: rho v r^2 = const

导出经典 Parker 方程:

    (v^2 - c_s^2) (1/v) dv/dr
        = (2 c_s^2 / r) - G M_sun / r^2

跨声速临界点 (sonic point):
    r_c = G M_sun / (2 c_s^2),   v(r_c) = c_s

四类解:
- 太阳风解 (v < c_s 内, v > c_s 外, dv/dr > 0 everywhere): 物理正确
- 太阳震 (solar breeze): 全程亚声速
- 双超声速
- 双亚声速

本模块:
1) 数值积分四类解 (RK4)
2) 计算跨声速转变的 "committor 概率" (借鉴 1083_DLCommittor):
   q(r, v) = 系统从状态 (r, v) 出发, 在噪声扰动下达到超声速 (v > 2 c_s)
            而非回落亚声速 (v < 0.5 c_s) 的概率.
   q 满足向后 Fokker-Planck / backward Kolmogorov 方程:
       (1/2) sigma^2 Delta q + b . grad q = 0
   其中 b 为确定性漂移 (Parker RHS), sigma 为随机扰动强度
   (代表光球湍动注入的 Alfvén 波涨落).
"""
from __future__ import annotations
import numpy as np
from solar_constants import (
    GRAVITATIONAL_CONST, SOLAR_MASS, SOLAR_RADIUS, sound_speed,
    parker_critical_radius,
)


def parker_rhs(r: float, v: float, cs: float) -> float:
    """Parker 方程 RHS: dv/dr = [ 2 cs^2/r - GM/r^2 ] / (v - cs^2/v).

    在跨声速点附近正则化: 分母加 epsilon.
    """
    num = 2.0 * cs**2 / r - GRAVITATIONAL_CONST * SOLAR_MASS / r**2
    den = v - cs**2 / (v + 1.0e-12)
    return num / (den + 1.0e-12 * np.sign(den))


def integrate_parker(r_start: float, v_start: float, r_end: float,
                     cs: float, n_step: int = 500) -> tuple:
    """从 r_start 向 r_end 积分 Parker 方程 (RK4)."""
    r_vals = np.linspace(r_start, r_end, n_step + 1)
    v_vals = np.zeros(n_step + 1)
    v_vals[0] = v_start
    dr = (r_end - r_start) / n_step

    for k in range(n_step):
        r = r_vals[k]
        v = v_vals[k]
        k1 = parker_rhs(r, v, cs)
        k2 = parker_rhs(r + 0.5 * dr, v + 0.5 * dr * k1, cs)
        k3 = parker_rhs(r + 0.5 * dr, v + 0.5 * dr * k2, cs)
        k4 = parker_rhs(r + dr, v + dr * k3, cs)
        v_new = v + (dr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # 速度非负约束
        v_vals[k + 1] = max(v_new, 1.0)
    return r_vals, v_vals


def critical_solution(cs: float, r_inner: float = SOLAR_RADIUS,
                      r_outer: float = 20.0 * SOLAR_RADIUS,
                      n_half: int = 300) -> tuple:
    """构造经过临界点的物理太阳风解 (双方向积分)."""
    r_c = parker_critical_radius(cs)
    v_c = cs

    # 从 r_c 向 r_outer (外向, 超声速)
    r_out, v_out = integrate_pander_safe(r_c, r_outer, v_c, cs, n_half,
                                         direction="out")
    # 从 r_c 向 r_inner (内向, 亚声速)
    r_in, v_in = integrate_pander_safe(r_c, r_inner, v_c, cs, n_half,
                                       direction="in")
    # 拼接
    r_full = np.concatenate([r_in[::-1], r_out[1:]])
    v_full = np.concatenate([v_in[::-1], v_out[1:]])
    return r_full, v_full, r_c


def integrate_pander_safe(r_c: float, r_end: float, v_c: float, cs: float,
                          n_step: int, direction: str) -> tuple:
    """从临界点出发积分, 使用 L'Hopital 规则在临界点附近启动."""
    # L'Hopital 在 r_c: dv/dr|_{r_c} 满足二次方程
    #   (dv/dr)^2 + (A) dv/dr + B = 0  => 取正根 (太阳风)
    # 简化: 用 cs/r_c 作为起始斜率
    eps_r = 0.005 * r_c
    if direction == "out":
        r_start = r_c + eps_r
        v_start = v_c + eps_r * (cs / r_c)
        return integrate_parker(r_start, v_start, r_end, cs, n_step)
    else:
        r_start = r_c - eps_r
        v_start = v_c - eps_r * (cs / r_c)
        v_start = max(v_start, 0.1 * cs)
        r_arr, v_arr = integrate_parker(
            r_start, v_start, r_end, cs, n_step
        )
        return r_arr, v_arr


# ============================================================
# Committor 概率 (借鉴 1083_DLCommittor)
# ============================================================
def solve_committor(r_grid: np.ndarray, v_grid: np.ndarray, cs: float,
                    sigma: float = 0.05 * SOLAR_RADIUS) -> np.ndarray:
    """求解 Parker 相空间上的 committor 方程:

    (1/2) sigma^2 ( d^2 q / dr^2 + d^2 q / dv^2 )
        + b_r dq/dr + b_v dq/dv = 0

    边界条件:
    - q = 0 on v = 0.5 cs (亚声速吸收)
    - q = 1 on v = 2.0 cs (超声速吸收)
    - Neumann 在 r 边界.

    使用有限差分 + 直接求解.
    """
    nr = r_grid.size
    nv = v_grid.size
    dr = r_grid[1] - r_grid[0] if nr > 1 else 1.0
    dv = v_grid[1] - v_grid[0] if nv > 1 else 1.0

    q = np.zeros((nr, nv))
    v_lo, v_hi = 0.5 * cs, 2.0 * cs

    # 设置边界: v = v_hi => q = 1, v = v_lo => q = 0
    for i in range(nr):
        for j in range(nv):
            if v_grid[j] >= v_hi:
                q[i, j] = 1.0
            elif v_grid[j] <= v_lo:
                q[i, j] = 0.0
            else:
                q[i, j] = (v_grid[j] - v_lo) / (v_hi - v_lo)

    # 迭代求解 (Gauss-Seidel)
    sigma2 = sigma**2
    for _iter in range(200):
        q_old = q.copy()
        for i in range(1, nr - 1):
            for j in range(1, nv - 1):
                if v_grid[j] <= v_lo or v_grid[j] >= v_hi:
                    continue
                r = r_grid[i]
                v = v_grid[j]
                b_r = parker_rhs(r, v, cs)
                b_v = 0.0  # 简化: 只考虑 v 扩散
                lap_q = (
                    (q[i + 1, j] + q[i - 1, j] - 2 * q[i, j]) / dr**2
                    + (q[i, j + 1] + q[i, j - 1] - 2 * q[i, j]) / dv**2
                )
                dq_dr = (q[i + 1, j] - q[i - 1, j]) / (2 * dr)
                # Jacobi 更新
                rhs = 0.5 * sigma2 * lap_q + b_r * dq_dr
                denom = sigma2 / dr**2 + sigma2 / dv**2 + 1.0e-12
                q[i, j] = (
                    0.5 * sigma2 * (
                        (q[i + 1, j] + q[i - 1, j]) / dr**2
                        + (q[i, j + 1] + q[i, j - 1]) / dv**2
                    )
                    + b_r * dq_dr
                ) / denom
        # 收敛检查
        dq = np.abs(q - q_old).max()
        if dq < 1.0e-5:
            break
    return q


def escape_probability_at_state(r: float, v: float, cs: float,
                                q_table: np.ndarray,
                                r_grid: np.ndarray,
                                v_grid: np.ndarray) -> float:
    """双线性插值查表得到 q(r, v)."""
    ir = np.searchsorted(r_grid, r) - 1
    iv = np.searchsorted(v_grid, v) - 1
    ir = np.clip(ir, 0, r_grid.size - 2)
    iv = np.clip(iv, 0, v_grid.size - 2)
    dr = r_grid[ir + 1] - r_grid[ir]
    dv = v_grid[iv + 1] - v_grid[iv]
    wr = (r - r_grid[ir]) / (dr + 1.0e-12)
    wv = (v - v_grid[iv]) / (dv + 1.0e-12)
    wr = np.clip(wr, 0.0, 1.0)
    wv = np.clip(wv, 0.0, 1.0)
    q00 = q_table[ir, iv]
    q10 = q_table[ir + 1, iv]
    q01 = q_table[ir, iv + 1]
    q11 = q_table[ir + 1, iv + 1]
    return float(
        (1 - wr) * (1 - wv) * q00
        + wr * (1 - wv) * q10
        + (1 - wr) * wv * q01
        + wr * wv * q11
    )
