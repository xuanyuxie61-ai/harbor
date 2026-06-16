"""
piecewise_constant_phase.py
============================
分段常数相指示函数与 CALPHAD 多相 Gibbs 能曲面构造。

种子项目 923_pwc_plot_1d:
  1D 分段常数函数的表示:
    f(x) = c_i,  x in [x_i, x_{i+1})
  通过构造扩展坐标对 (xp, yp) 表示水平段和垂直跳变。

映射到 CALPHAD:
  在多相区域, 系统的平衡 Gibbs 能是分段定义的:
    G_eq(x,T) = min_phi G_phi(x,T)

  这产生分段常数/分段光滑结构:
  - 在单相区: G_eq = G_phi (光滑)
  - 在两相区: G_eq = 公共切线值 (也是光滑的, 但由不同相组合)
  - 在相边界: dG_eq/dc 可能不连续

本模块实现:
  1. 分段常数相指示函数 phi_i(x)
  2. 全局平衡 Gibbs 能 G_eq(x,T) 的构造
  3. 相边界处的不连续性检测和光滑化处理
  4. 分段常数近似的误差分析
"""

import numpy as np
from calphad_fec_constants import X_C_MIN, X_C_MAX, N_X_GRID
from gibbs_energy_calphad import gibbs_substitutional


def build_phase_indicator(x_grid, T, phases):
    """
    构建分段常数相指示函数:

    phi_i(x) = 1  如果相 i 在 x 处具有最低 Gibbs 能
             = 0  否则

    对于每个 x, 找到 Gibbs 能最低的相:
        phi_winner(x) = argmin_i G_i(x, T)

    Parameters
    ----------
    x_grid : np.ndarray
        成分网格
    T : float
        温度 (K)
    phases : list of str
        相名称列表

    Returns
    -------
    dict
        {
            'phase_index': np.ndarray,   # 每个网格点的胜出相索引
            'phase_names': list,         # 各网格点的胜出相名称
            'G_min': np.ndarray,         # 最低 Gibbs 能
            'G_all': dict,               # 各相的 Gibbs 能
            'boundaries': list,          # 相边界位置
        }
    """
    n_x = len(x_grid)
    n_phases = len(phases)

    G_all = {}
    for phase in phases:
        G_all[phase] = gibbs_substitutional(x_grid, T, phase)

    G_stack = np.array([G_all[p] for p in phases])  # (n_phases, n_x)
    phase_idx = np.argmin(G_stack, axis=0)
    G_min = np.min(G_stack, axis=0)

    phase_names = [phases[idx] for idx in phase_idx]

    # 检测相边界 (相指示函数跳变)
    boundaries = []
    for i in range(n_x - 1):
        if phase_idx[i] != phase_idx[i + 1]:
            # 线性插值精确定位
            p1 = phases[phase_idx[i]]
            p2 = phases[phase_idx[i + 1]]
            dG = G_all[p1] - G_all[p2]
            if abs(dG[i + 1] - dG[i]) > 1e-30:
                x_b = x_grid[i] - dG[i] * (x_grid[i + 1] - x_grid[i]) / (
                    dG[i + 1] - dG[i])
            else:
                x_b = (x_grid[i] + x_grid[i + 1]) / 2
            boundaries.append({
                'x_boundary': float(x_b),
                'phase_left': p1,
                'phase_right': p2,
                'G_value': float(G_min[i]),
            })

    return {
        'phase_index': phase_idx,
        'phase_names': phase_names,
        'G_min': G_min,
        'G_all': G_all,
        'boundaries': boundaries,
    }


def pwc_expanded_coordinates(x_grid, values, phase_indices):
    """
    构建分段常数函数的扩展坐标 (种子项目 923 核心算法):

    对于分段常数函数 f(x) = c_i on [x_i, x_{i+1}),
    生成绘制用的扩展坐标:
      xp = [x_0, x_1, x_1, x_2, x_2, ..., x_N]
      yp = [c_0, c_0, c_1, c_1, c_2, ..., c_{N-1}]

    这使得水平段和垂直跳变都能正确绘制。

    Parameters
    ----------
    x_grid : np.ndarray
        网格点 (N+1 个点定义 N 个区间)
    values : np.ndarray
        各区间的常数值 (N 个值)
    phase_indices : np.ndarray
        各区间对应的相索引

    Returns
    -------
    tuple
        (xp, yp): 扩展坐标
    """
    N = len(values)
    if N == 0:
        return np.array([]), np.array([])

    xp = np.zeros(2 * N)
    yp = np.zeros(2 * N)

    for i in range(N):
        xp[2 * i] = x_grid[i]
        xp[2 * i + 1] = x_grid[i + 1] if i + 1 < len(x_grid) else x_grid[i]
        yp[2 * i] = values[i]
        yp[2 * i + 1] = values[i]

    return xp, yp


def smoothed_gibbs_landscape(x_grid, T, phases, smoothing_width=0.005):
    """
    构造光滑化的平衡 Gibbs 能曲面。

    原始 G_eq = min_i G_i(x,T) 在相边界处不可微。
    使用正则化 (log-sum-exp 近似):

    G_smooth = -beta^{-1} * ln(sum_i exp(-beta * G_i))

    当 beta → infinity 时, G_smooth → min_i G_i。
    有限的 beta 提供光滑近似, 便于梯度-based 优化。

    Parameters
    ----------
    x_grid : np.ndarray
        成分网格
    T : float
        温度 (K)
    phases : list of str
        相名称
    smoothing_width : float
        光滑化宽度 (控制 beta)

    Returns
    -------
    dict
        {
            'G_smooth': np.ndarray,
            'phase_fractions': dict,  # 各相的分数
            'beta': float,
        }
    """
    n_x = len(x_grid)
    n_phases = len(phases)

    # beta 参数: 越大越接近 min
    G_scale = 1000.0  # 典型 Gibbs 能差量级
    beta = 1.0 / (smoothing_width * G_scale)

    G_all = {}
    for phase in phases:
        G_all[phase] = gibbs_substitutional(x_grid, T, phase)

    G_stack = np.array([G_all[p] for p in phases])  # (n_phases, n_x)

    # Log-sum-exp 技巧 (数值稳定)
    G_min_col = np.min(G_stack, axis=0)
    exp_terms = np.exp(-beta * (G_stack - G_min_col))
    sum_exp = np.sum(exp_terms, axis=0)
    sum_exp = np.maximum(sum_exp, 1e-30)

    G_smooth = G_min_col - np.log(sum_exp) / beta

    # 相分数 (softmax)
    phase_fractions = {}
    for i, phase in enumerate(phases):
        phase_fractions[phase] = exp_terms[i] / sum_exp

    return {
        'G_smooth': G_smooth,
        'phase_fractions': phase_fractions,
        'beta': float(beta),
        'G_all': G_all,
    }


def pwc_approximation_error(x_grid, T, phases, n_refinements=3):
    """
    分段常数近似的误差分析。

    比较不同网格密度下分段常数近似与真实 Gibbs 能的偏差:
        err(N) = max_x |G_pwc(x) - G_exact(x)|

    理论上, 对于光滑函数, err ~ O(h) = O(1/N)。
    在相边界处, 误差可能更大。

    Parameters
    ----------
    x_grid : np.ndarray
        基准网格
    T : float
        温度 (K)
    phases : list of str
        相名称
    n_refinements : int
        细化次数

    Returns
    -------
    dict
        {
            'n_grid': list,
            'max_errors': list,
            'convergence_rates': list,
        }
    """
    results = {'n_grid': [], 'max_errors': [], 'convergence_rates': []}

    fine_x = np.linspace(X_C_MIN, X_C_MAX, 2000)
    # 精细参考解
    fine_indicator = build_phase_indicator(fine_x, T, phases)
    G_exact = fine_indicator['G_min']

    for ref in range(n_refinements + 1):
        n_pts = 20 * (2 ** ref)
        coarse_x = np.linspace(X_C_MIN, X_C_MAX, n_pts)
        coarse_indicator = build_phase_indicator(coarse_x, T, phases)

        # 插值到精细网格进行比较
        G_coarse_interp = np.interp(
            fine_x, coarse_x, coarse_indicator['G_min']
        )
        err = np.max(np.abs(G_coarse_interp - G_exact))

        results['n_grid'].append(n_pts)
        results['max_errors'].append(float(err))

        if len(results['max_errors']) > 1:
            rate = np.log(results['max_errors'][-2] / max(err, 1e-30)) / np.log(2.0)
            results['convergence_rates'].append(float(rate))

    return results
