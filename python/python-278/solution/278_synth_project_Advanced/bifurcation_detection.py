"""
bifurcation_detection.py
========================
相边界分岔检测: 追踪 CALPHAD 相图中的相变边界。

种子项目 1085_PierceRyan_border-collision-bifurcations:
  - 分段光滑系统中的边界碰撞分岔
  - 事件驱动的时间延迟系统模拟
  - 不连续面的精确追踪

映射到 CALPHAD:
  相图本质上是 Gibbs 能曲面的分岔图:
  - 两相平衡线 → 切线分岔 (tangent bifurcation)
  - 共晶/包晶点 → 鞍结分岔 (saddle-node)
  - Spinodal 线 → 跨临界分岔 (transcritical)
  - 有序-无序转变 → 叉式分岔 (pitchfork)

检测方法:
  1. 追踪 Jacobian 行列式的零点:
     det(J(x,T)) = 0 → 分岔点
  2. 检测 Gibbs 能曲面的公共切线消失
  3. 追踪序参量的突变 (一级相变)
  4. 追踪 d²G/dc² 的符号变化 (二级相变)

分岔类型判定:
  - 若 d²G/dc² 连续过零 → 二级 (spinodal)
  - 若 d²G/dc² 不连续跳跃 → 一级 (相界)
  - 若两个平衡解碰撞湮灭 → 鞍结
"""

import numpy as np
from calphad_fec_constants import BF_TOL, BF_DJUMP_THRESH
from gibbs_energy_calphad import (
    gibbs_substitutional,
    chemical_potential_C,
    chemical_potential_Fe,
    second_derivative_G,
)


def jacobian_determinant_2phase(x_a, x_b, T, phase_a, phase_b):
    """
    计算两相平衡方程组的 Jacobian 行列式。

    对于 F(x_a, x_b; T) = 0:
    det(J) = d(mu_Fe^a)/d(x_a) * (-d(mu_C^b)/d(x_b))
           - (-d(mu_Fe^b)/d(x_b)) * d(mu_C^a)/d(x_a)

    det(J) = 0 意味着平衡解的退化 (分岔点)。

    Parameters
    ----------
    x_a : float
        alpha 相成分
    x_b : float
        beta 相成分
    T : float
        温度 (K)
    phase_a : str
        alpha 相
    phase_b : str
        beta 相

    Returns
    -------
    float
        det(J)
    """
    eps = 1e-7

    # 数值 Jacobian
    mu_Fe_a_p = chemical_potential_Fe(min(x_a + eps, 0.3), T, phase_a)
    mu_Fe_a_m = chemical_potential_Fe(max(x_a - eps, 1e-10), T, phase_a)
    mu_C_a_p = chemical_potential_C(min(x_a + eps, 0.3), T, phase_a)
    mu_C_a_m = chemical_potential_C(max(x_a - eps, 1e-10), T, phase_a)

    mu_Fe_b_p = chemical_potential_Fe(min(x_b + eps, 0.3), T, phase_b)
    mu_Fe_b_m = chemical_potential_Fe(max(x_b - eps, 1e-10), T, phase_b)
    mu_C_b_p = chemical_potential_C(min(x_b + eps, 0.3), T, phase_b)
    mu_C_b_m = chemical_potential_C(max(x_b - eps, 1e-10), T, phase_b)

    dxa = min(x_a + eps, 0.3) - max(x_a - eps, 1e-10)
    dxb = min(x_b + eps, 0.3) - max(x_b - eps, 1e-10)

    J11 = (mu_Fe_a_p - mu_Fe_a_m) / dxa
    J12 = -(mu_Fe_b_p - mu_Fe_b_m) / dxb
    J21 = (mu_C_a_p - mu_C_a_m) / dxa
    J22 = -(mu_C_b_p - mu_C_b_m) / dxb

    return J11 * J22 - J12 * J21


def detect_tangent_bifurcation(T_range, phase_a, phase_b,
                               x_init=None, n_T=50):
    """
    追踪切线分岔 (相界) 沿温度轴。

    对于每个温度, 求解相平衡, 并追踪
    det(J(T)) 的符号变化。

    Parameters
    ----------
    T_range : tuple
        (T_min, T_max) 温度范围 (K)
    phase_a : str
        alpha 相
    phase_b : str
        beta 相
    x_init : tuple
        初始成分猜测
    n_T : int
        温度扫描点数

    Returns
    -------
    dict
        {
            'T_values': np.ndarray,
            'x_a_values': np.ndarray,
            'x_b_values': np.ndarray,
            'detJ_values': np.ndarray,
            'bifurcation_points': list,
            'phase_boundary_type': str,
        }
    """
    from newton_maehly_equilibrium import newton_maehly_solve

    T_vals = np.linspace(T_range[0], T_range[1], n_T)
    x_a_vals = np.zeros(n_T)
    x_b_vals = np.zeros(n_T)
    detJ_vals = np.zeros(n_T)

    bifurcations = []

    prev_detJ = None
    prev_xa = None

    for i, T in enumerate(T_vals):
        result = newton_maehly_solve(T, phase_a, phase_b, x_init=x_init)

        if result['converged']:
            xa = result['x_C_alpha']
            xb = result['x_C_beta']
            x_a_vals[i] = xa
            x_b_vals[i] = xb

            detJ = jacobian_determinant_2phase(xa, xb, T, phase_a, phase_b)
            detJ_vals[i] = detJ

            # 分岔检测: det(J) 过零
            if prev_detJ is not None and prev_detJ * detJ < 0:
                # 线性插值精确定位
                T_bif = T_vals[i - 1] + (T - T_vals[i - 1]) * abs(prev_detJ) / (
                    abs(prev_detJ) + abs(detJ))
                bifurcations.append({
                    'T_bifurcation': float(T_bif),
                    'type': 'detJ_zero',
                    'x_a_approx': float((prev_xa + xa) / 2),
                })

            # 跳跃检测 (一级相变特征)
            if prev_xa is not None and abs(xa - prev_xa) > BF_DJUMP_THRESH:
                bifurcations.append({
                    'T_bifurcation': float(T),
                    'type': 'composition_jump',
                    'jump_magnitude': float(abs(xa - prev_xa)),
                })

            prev_detJ = detJ
            prev_xa = xa
            x_init = (xa, xb)
        else:
            x_a_vals[i] = np.nan
            x_b_vals[i] = np.nan
            detJ_vals[i] = np.nan

    # 判定相界类型
    has_detJ_zero = any(b['type'] == 'detJ_zero' for b in bifurcations)
    has_jump = any(b['type'] == 'composition_jump' for b in bifurcations)

    if has_jump and has_detJ_zero:
        boundary_type = 'first_order_with_critical'
    elif has_jump:
        boundary_type = 'first_order'
    elif has_detJ_zero:
        boundary_type = 'second_order'
    else:
        boundary_type = 'continuous'

    return {
        'T_values': T_vals,
        'x_a_values': x_a_vals,
        'x_b_values': x_b_vals,
        'detJ_values': detJ_vals,
        'bifurcation_points': bifurcations,
        'phase_boundary_type': boundary_type,
    }


def border_collision_map(T, phase_a, phase_b, x_range, n_scan=100):
    """
    边界碰撞分岔图: 追踪 Gibbs 能曲面的不连续性。

    对于固定温度, 扫描成分空间, 检测:
  1. G_alpha(x) = G_beta(x) 的点 (等能点)
  2. dG_alpha/dc = dG_beta/dc 的点 (等化学势点)
  3. 两个条件同时满足 → 公共切线/两相平衡

  Parameters
  ----------
  T : float
      温度 (K)
  phase_a : str
      alpha 相
  phase_b : str
      beta 相
  x_range : tuple
      成分范围
  n_scan : int
      扫描点数

  Returns
  -------
  dict
      {
          'x_grid': np.ndarray,
          'G_alpha': np.ndarray,
          'G_beta': np.ndarray,
          'dG_diff': np.ndarray,
          'equilibrium_points': list,
      }
    """
    x_grid = np.linspace(x_range[0], x_range[1], n_scan)

    G_alpha = gibbs_substitutional(x_grid, T, phase_a)
    G_beta = gibbs_substitutional(x_grid, T, phase_b)

    # 化学势差
    mu_C_a = chemical_potential_C(x_grid, T, phase_a)
    mu_C_b = chemical_potential_C(x_grid, T, phase_b)
    dG_diff = mu_C_a - mu_C_b

    # 找等能点
    G_diff = G_alpha - G_beta
    eq_points = []
    for i in range(n_scan - 1):
        if G_diff[i] * G_diff[i + 1] < 0:
            # 等能点
            x_eq = x_grid[i] + (x_grid[i + 1] - x_grid[i]) * abs(G_diff[i]) / (
                abs(G_diff[i]) + abs(G_diff[i + 1]))
            eq_points.append({'x': float(x_eq), 'type': 'equal_G'})

        if dG_diff[i] * dG_diff[i + 1] < 0:
            # 等化学势点
            x_eq = x_grid[i] + (x_grid[i + 1] - x_grid[i]) * abs(dG_diff[i]) / (
                abs(dG_diff[i]) + abs(dG_diff[i + 1]))
            eq_points.append({'x': float(x_eq), 'type': 'equal_mu_C'})

    return {
        'x_grid': x_grid,
        'G_alpha': G_alpha,
        'G_beta': G_beta,
        'dG_diff': dG_diff,
        'equilibrium_points': eq_points,
    }


def eutectic_point_search(T_range, phases, n_T=30):
    """
    共晶点搜索: 找三相平衡 (L + alpha + beta) 的不变点。

    共晶条件:
    mu_i^L = mu_i^alpha = mu_i^beta,  for all components i

    在二元系中, 共晶点是零自由度的不变点 (Gibbs 相律)。

    Parameters
    ----------
    T_range : tuple
        (T_min, T_max)
    phases : list of str
        三个相的名称
    n_T : int
        温度扫描点数

    Returns
    -------
    dict
        共晶点的温度和成分
    """
    if len(phases) < 3:
        return {'found': False, 'message': '需要至少3个相'}

    phase_L = phases[0]
    phase_a = phases[1]
    phase_b = phases[2]

    T_vals = np.linspace(T_range[0], T_range[1], n_T)

    best_T = None
    best_x = None
    best_err = np.inf

    for T in T_vals:
        # L + alpha 平衡
        from newton_maehly_equilibrium import newton_maehly_solve
        res_La = newton_maehly_solve(T, phase_L, phase_a)
        res_Lb = newton_maehly_solve(T, phase_L, phase_b)

        if res_La['converged'] and res_Lb['converged']:
            # 检查 alpha 和 beta 的化学势是否也相等
            x_La = res_La['x_C_alpha']  # L 相在 L+alpha 中的成分
            x_Lb = res_Lb['x_C_alpha']  # L 相在 L+beta 中的成分

            err = abs(x_La - x_Lb)
            if err < best_err:
                best_err = err
                best_T = T
                best_x = (x_La + x_Lb) / 2

    if best_T is not None and best_err < 0.05:
        return {
            'found': True,
            'T_eutectic': float(best_T),
            'x_eutectic': float(best_x),
            'residual': float(best_err),
        }
    else:
        return {
            'found': False,
            'T_eutectic': float(best_T) if best_T else 0.0,
            'x_eutectic': float(best_x) if best_x else 0.0,
            'residual': float(best_err),
        }
