# -*- coding: utf-8 -*-
"""
phase_space_diagnostics.py
==========================

相空间诊断模块: 互相关分析, 粒子计数, Lyness 积分.

本模块实现辐射带模拟结果的诊断分析:
  - 通量涨落的互相关分析 (DAS 项目 -> 通量关联)
  - 相空间粒子计数 (candy_count 项目 -> 通量统计)
  - 三角形 Lyness 积分规则 (lyness_rule 项目 -> 相空间积分)
  - 分布函数矩计算

物理背景:

1. 互相关分析:
   辐射带电子通量在不同 L 壳层间存在时间延迟关联,
   反映径向扩散过程. 互相关函数:
     C_{12}(tau) = <delta j_1(t) * delta j_2(t+tau)>

2. 相空间粒子计数:
   在 (L, E) 网格中统计粒子数:
     N_{ij} = integral_{cell} f(L, E) dL dE

3. Lyness 积分:
   对于三角形区域上的积分, Lyness 规则提供高精度近似:
     integral_T f(x,y) dA ~ sum_k w_k * f(x_k, y_k)

参考文献:
  [1] Lyness, J.N. & Jespersen, D., "Moderate degree symmetric
      quadrature formulas for the triangle", JIMA (1975)
  [2] Morozova, V. et al., "Cross-correlation of radiation belt fluxes",
      JGR (2018)
"""

import numpy as np
import physical_constants as pc


# =============================================================================
#  互相关分析 (来自 DAS 项目)
# =============================================================================

def cross_correlation(signal1, signal2, n_lags=None):
    """
    计算两个信号的互相关函数.

    物理公式:
      C_{12}(tau) = <s_1(t) * s_2(t+tau)>
                  = (1/N) * sum_t s_1(t) * s_2(t+tau)

    参数
    ----
    signal1, signal2 : ndarray
        输入信号
    n_lags : int, optional
        最大延迟步数

    返回
    -------
    xcorr : ndarray
        互相关函数
    lags : ndarray
        延迟索引
    """
    n = len(signal1)
    if n_lags is None:
        n_lags = n // 4

    # 去均值
    s1 = signal1 - np.mean(signal1)
    s2 = signal2 - np.mean(signal2)

    # 归一化
    std1 = np.std(signal1)
    std2 = np.std(signal2)
    if std1 > pc.EPSILON_NUM:
        s1 /= std1
    if std2 > pc.EPSILON_NUM:
        s2 /= std2

    xcorr = np.zeros(2 * n_lags + 1)
    lags = np.arange(-n_lags, n_lags + 1)

    for i, lag in enumerate(lags):
        if lag >= 0:
            xcorr[i] = np.mean(s1[:n-lag] * s2[lag:]) if lag < n else 0.0
        else:
            xcorr[i] = np.mean(s1[-lag:] * s2[:n+lag]) if -lag < n else 0.0

    return xcorr, lags


def flux_cross_correlation(f_history, L_index1, L_index2, dt):
    """
    计算两个 L 壳层通量的互相关.

    参数
    ----
    f_history : list of ndarray
        分布函数时间序列
    L_index1, L_index2 : int
        L 壳层索引
    dt : float
        时间步长

    返回
    -------
    xcorr : ndarray
        互相关函数
    tau : ndarray
        延迟时间 [s]
    peak_lag : float
        峰值延迟 [s]
    """
    # 提取通量时间序列
    flux1 = np.array([np.sum(f[L_index1, :]) for f in f_history])
    flux2 = np.array([np.sum(f[L_index2, :]) for f in f_history])

    xcorr, lags = cross_correlation(flux1, flux2)
    tau = lags * dt

    # 峰值延迟
    peak_idx = np.argmax(np.abs(xcorr))
    peak_lag = tau[peak_idx]

    return xcorr, tau, peak_lag


# =============================================================================
#  相空间粒子计数 (来自 candy_count 项目)
# =============================================================================

def phase_space_count(f, L_edges, E_edges):
    """
    在 (L, E) 网格中统计粒子数.

    物理公式:
      N_{ij} = f(L_i, E_j) * delta_L * delta_E

    参数
    ----
    f : ndarray, shape (n_L, n_E)
        分布函数
    L_edges : ndarray
        L 边界
    E_edges : ndarray
        E 边界

    返回
    -------
    counts : ndarray, shape (n_L, n_E)
        各网格单元粒子数
    total : float
        总粒子数
    """
    n_L = len(L_edges) - 1
    n_E = len(E_edges) - 1

    counts = np.zeros((n_L, n_E))
    for i in range(n_L):
        dL = L_edges[i+1] - L_edges[i]
        for j in range(n_E):
            dE = E_edges[j+1] - E_edges[j]
            counts[i, j] = f[i, j] * dL * dE

    total = np.sum(counts)
    return counts, total


def phase_space_moments(f, L, E):
    """
    计算分布函数的矩.

    物理量:
      数密度: n = integral f dE
      能通量: F = integral E * f dE
      平均能量: <E> = F / n

    参数
    ----
    f : ndarray, shape (n_L, n_E)
        分布函数
    L : ndarray
        L 参数
    E : ndarray
        能量 [MeV]

    返回
    -------
    moments : dict
        矩: density, energy_flux, mean_energy
    """
    dE = np.diff(E)
    dE = np.append(dE, dE[-1])  # 最后一个间距

    # 数密度 (对 E 积分)
    density = np.sum(f * dE[np.newaxis, :], axis=1)

    # 能通量
    energy_flux = np.sum(f * E[np.newaxis, :] * dE[np.newaxis, :], axis=1)

    # 平均能量
    mean_energy = np.where(density > pc.EPSILON_NUM,
                           energy_flux / density,
                           0.0)

    return {
        'density': density,
        'energy_flux': energy_flux,
        'mean_energy': mean_energy,
        'total_particles': float(np.sum(density) * (L[1] - L[0]) if len(L) > 1 else 0.0),
    }


# =============================================================================
#  Lyness 三角形积分规则 (来自 triangle_lyness_rule 项目)
# =============================================================================

def lyness_triangle_quadrature(order=3):
    """
    三角形区域上的 Lyness 积分规则.

    对于参考三角形 T = {(x,y): 0 <= x, 0 <= y, x+y <= 1},
    Lyness 规则提供高精度的数值积分:
      integral_T f(x,y) dA ~ sum_k w_k * f(x_k, y_k)

    参数
    ----
    order : int
        积分精度阶数 (1, 3, 4, 5, 6, 7)

    返回
    -------
    points : ndarray, shape (n_points, 2)
        积分点坐标
    weights : ndarray
        积分权重
    """
    if order == 1:
        # 1 阶: 重心
        points = np.array([[1.0/3.0, 1.0/3.0]])
        weights = np.array([0.5])

    elif order == 3:
        # 3 阶: 3 点规则
        points = np.array([
            [1.0/6.0, 1.0/6.0],
            [2.0/3.0, 1.0/6.0],
            [1.0/6.0, 2.0/3.0]
        ])
        weights = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])

    elif order == 4:
        # 4 阶: 6 点规则
        a1 = 0.445948490915965
        a2 = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        points = np.array([
            [a1, a1],
            [1-2*a1, a1],
            [a1, 1-2*a1],
            [a2, a2],
            [1-2*a2, a2],
            [a2, 1-2*a2]
        ])
        weights = np.array([w1, w1, w1, w2, w2, w2])

    elif order >= 5:
        # 5 阶: 7 点规则
        a1 = 0.470142064105115
        a2 = 0.101286507323456
        w0 = 0.1125
        w1 = 0.066197076394253
        w2 = 0.062969590272414
        points = np.array([
            [1.0/3.0, 1.0/3.0],
            [a1, a1],
            [1-2*a1, a1],
            [a1, 1-2*a1],
            [a2, a2],
            [1-2*a2, a2],
            [a2, 1-2*a2]
        ])
        weights = np.array([w0, w1, w1, w1, w2, w2, w2])

    else:
        raise ValueError(f"不支持的阶数: {order}")

    return points, weights


def integrate_over_triangle(func, vertices, order=5):
    """
    在任意三角形上积分函数.

    参数
    ----
    func : callable
        被积函数 f(x, y)
    vertices : ndarray, shape (3, 2)
        三角形顶点
    order : int
        积分精度

    返回
    -------
    integral : float
        积分值
    """
    points_ref, weights_ref = lyness_triangle_quadrature(order)

    # 参考三角形 -> 实际三角形的映射
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    # 雅可比行列式
    J = abs((v1[0]-v0[0])*(v2[1]-v0[1]) - (v2[0]-v0[0])*(v1[1]-v0[1]))

    integral = 0.0
    for i in range(len(points_ref)):
        # 重心坐标映射
        lam1, lam2 = points_ref[i]
        lam0 = 1.0 - lam1 - lam2
        x = lam0 * v0[0] + lam1 * v1[0] + lam2 * v2[0]
        y = lam0 * v0[1] + lam1 * v1[1] + lam2 * v2[1]
        integral += weights_ref[i] * func(x, y)

    return integral * J


# =============================================================================
#  综合诊断
# =============================================================================

def comprehensive_diagnostics(f_history, grid, dt):
    """
    综合诊断分析.

    参数
    ----
    f_history : list of ndarray
        分布函数时间序列
    grid : MagnetosphereGrid
        相空间网格
    dt : float
        时间步长

    返回
    -------
    diagnostics : dict
        诊断结果
    """
    if len(f_history) < 2:
        return {'error': '需要至少 2 个时间步'}

    diagnostics = {}

    # 1. 粒子数变化
    N_initial = np.sum(f_history[0]) * grid.dL * grid.dE_avg
    N_final = np.sum(f_history[-1]) * grid.dL * grid.dE_avg
    diagnostics['particle_change'] = {
        'initial': float(N_initial),
        'final': float(N_final),
        'relative_change': float((N_final - N_initial) / max(N_initial, pc.EPSILON_NUM)),
    }

    # 2. 矩分析
    moments_init = phase_space_moments(f_history[0], grid.L, grid.E_MeV)
    moments_final = phase_space_moments(f_history[-1], grid.L, grid.E_MeV)
    diagnostics['moments'] = {
        'initial': moments_init,
        'final': moments_final,
    }

    # 3. 互相关 (如果数据足够)
    if len(f_history) >= 10:
        L_mid = grid.n_L // 2
        L_outer = 3 * grid.n_L // 4
        xcorr, tau, peak_lag = flux_cross_correlation(
            f_history, L_mid, L_outer, dt
        )
        diagnostics['cross_correlation'] = {
            'peak_lag': float(peak_lag),
            'max_correlation': float(np.max(np.abs(xcorr))),
        }

    return diagnostics


if __name__ == "__main__":
    print("相空间诊断模块自检验证")

    # 互相关测试
    t = np.linspace(0, 10, 100)
    s1 = np.sin(2*np.pi*t)
    s2 = np.sin(2*np.pi*(t - 0.5))
    xcorr, lags = cross_correlation(s1, s2, n_lags=20)
    print(f"互相关峰值延迟: {lags[np.argmax(np.abs(xcorr))]}")

    # Lyness 积分测试
    def test_func(x, y):
        return x**2 + y**2
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    integral = integrate_over_triangle(test_func, vertices, order=5)
    exact = 1.0/6.0  # 精确值
    print(f"Lyness 积分: {integral:.6f} (精确: {exact:.6f})")
