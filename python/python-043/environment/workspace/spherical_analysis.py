"""
spherical_analysis.py — 球面分析模块

融合以下种子项目：
- 1192_svd_sphere : SVD 球面映射分析
- 1196_task_division : 处理器任务划分
- 1342_triangulation_order3_contour : 三阶三角化上的标量场分析

功能：
1. 对球面上的磁场、速度场进行 SVD 模态分解
2. 球谐展开与 Gaunt 系数计算
3. 球面任务划分（按谐波阶数并行分解）
4. 标量场在三角化球面上的积分与梯度估计

核心数学模型：
-------------
SVD 模态分解：
  给定球面数据矩阵 X (n_points × n_components)，
  进行 SVD:  X = U Σ V^T
  其中 U 的列为空间模态，V 的列为时间模态，
  Σ 的对角元为奇异值（能量分布）。

球谐展开：
  f(θ,φ) = Σ_{l=0}^L Σ_{m=-l}^l f_l^m Y_l^m(θ,φ)
  Y_l^m(θ,φ) = N_l^m P_l^m(cosθ) e^{imφ}
  N_l^m = sqrt((2l+1)(l-m)! / (4π(l+m)!))

任务划分：
  将 l=0,...,L 的谐波阶数均匀分配给 P 个处理器，
  每个处理器处理连续的一段 l 范围。
"""

import numpy as np
from special_functions import associated_legendre


def svd_sphere_decomposition(data_matrix, n_modes=10):
    """
    对球面数据进行 SVD 模态分解（源自 1192_svd_sphere）。

    参数：
      data_matrix : (n_points, n_times) 球面标量场时间序列
      n_modes     : 保留的模态数

    返回：
      U : (n_points, n_modes) 空间模态
      S : (n_modes,) 奇异值
      Vt: (n_modes, n_times) 时间模态转置
      energy_ratio : 前 n_modes 模态的能量占比
    """
    U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)

    total_energy = np.sum(S ** 2)
    retained_energy = np.sum(S[:n_modes] ** 2)
    energy_ratio = retained_energy / (total_energy + 1e-30)

    return U[:, :n_modes], S[:n_modes], Vt[:n_modes, :], energy_ratio


def spherical_harmonic_transform(theta, phi, values, L_max=8):
    """
    球谐变换：将球面上的标量场展开为球谐系数。

    参数：
      theta  : (N,) 极角数组
      phi    : (N,) 方位角数组
      values : (N,) 标量场值
      L_max  : 最大谐波阶数

    返回：
      coeffs : (L_max+1, 2*L_max+1) 复系数数组 f_l^m
    """
    N = len(values)
    coeffs = np.zeros((L_max + 1, 2 * L_max + 1), dtype=complex)

    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # 计算归一化连带 Legendre
            P_lm = associated_legendre(l, abs(m), np.cos(theta))
            N_lm = np.sqrt((2 * l + 1) * np.math.factorial(l - abs(m))
                           / (4 * np.pi * np.math.factorial(l + abs(m))))
            Y_lm = N_lm * P_lm * np.exp(1j * m * phi)

            # 数值积分（蒙特卡洛/梯形）
            integrand = values * np.conj(Y_lm) * np.sin(theta)
            # 假设点大致均匀分布，用简单平均近似积分
            f_lm = np.mean(integrand) * 4 * np.pi
            coeffs[l, m + L_max] = f_lm

    return coeffs


def inverse_spherical_harmonic_transform(coeffs, theta, phi, L_max=None):
    """
    逆球谐变换：由系数重构球面标量场。
    """
    if L_max is None:
        L_max = coeffs.shape[0] - 1

    N = len(theta)
    values = np.zeros(N, dtype=complex)

    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            P_lm = associated_legendre(l, abs(m), np.cos(theta))
            N_lm = np.sqrt((2 * l + 1) * np.math.factorial(l - abs(m))
                           / (4 * np.pi * np.math.factorial(l + abs(m))))
            Y_lm = N_lm * P_lm * np.exp(1j * m * phi)
            values += coeffs[l, m + L_max] * Y_lm

    return np.real(values)


def task_division_spherical_harmonics(L_max, proc_first, proc_last):
    """
    将球谐阶数 l=0,...,L_max 分配给多个处理器（源自 1196_task_division）。

    参数：
      L_max      : 最大谐波阶数
      proc_first : 首个处理器编号
      proc_last  : 末个处理器编号

    返回：
      divisions : 列表 [(proc, n_tasks, l_lo, l_hi), ...]
    """
    task_number = L_max + 1
    p = proc_last + 1 - proc_first

    divisions = []
    i_hi = -1
    task_remain = task_number
    proc_remain = p

    for proc in range(proc_first, proc_last + 1):
        task_proc = _div_rounded(task_remain, proc_remain)
        proc_remain -= 1
        task_remain -= task_proc

        i_lo = i_hi + 1
        i_hi = i_hi + task_proc

        divisions.append((proc, task_proc, i_lo, i_hi))

    return divisions


def _div_rounded(a, b):
    """
    四舍五入除法（源自 1196_task_division/i4_div_rounded.m）。
    """
    if b == 0:
        return 0
    value = a / b
    if value < 0:
        return int(value - 0.5)
    else:
        return int(value + 0.5)


def compute_gauss_spectral_coefficients(B_field, theta, phi, nodes, r,
                                         L_max=6, R_surface=1.0):
    """
    计算球面磁场的高斯谱系数 g_l^m 和 h_l^m。
    地磁场通常表示为：
      B_r = -∂Φ/∂r
      B_θ = -(1/r) ∂Φ/∂θ
      B_φ = -(1/(r sinθ)) ∂Φ/∂φ
    其中 Φ 为磁标势：
      Φ = R Σ_{l=1}^L Σ_{m=0}^l (R/r)^{l+1} P_l^m(cosθ)
          · [g_l^m cos(mφ) + h_l^m sin(mφ)]
    """
    n_points = len(theta)
    g_coeffs = np.zeros((L_max + 1, L_max + 1))
    h_coeffs = np.zeros((L_max + 1, L_max + 1))

    # 取核幔边界附近的数据
    mask = np.abs(r - R_surface) < 0.1
    if not np.any(mask):
        mask = np.ones(n_points, dtype=bool)

    theta_s = theta[mask]
    phi_s = phi[mask]
    Br = B_field[mask, 0] if B_field.ndim > 1 else B_field[mask]

    for l in range(1, L_max + 1):
        for m in range(0, l + 1):
            P_lm = associated_legendre(l, m, np.cos(theta_s))
            N_lm = np.sqrt((2 * l + 1) * np.math.factorial(l - m)
                           / (4 * np.pi * np.math.factorial(l + m)))
            integrand_g = Br * P_lm * np.cos(m * phi_s) * np.sin(theta_s)
            integrand_h = Br * P_lm * np.sin(m * phi_s) * np.sin(theta_s)
            # 归一化因子
            norm = (l + 1) / (R_surface * (2 - (m == 0)))  # 修正归一化
            g_coeffs[l, m] = np.mean(integrand_g) * 4 * np.pi * norm
            h_coeffs[l, m] = np.mean(integrand_h) * 4 * np.pi * norm

    return g_coeffs, h_coeffs


def spectral_dipole_tilt(g_coeffs, h_coeffs):
    """
    由谱系数计算偶极倾角。
    偶极矩分量：
      g₁⁰, g₁¹, h₁¹
    倾角：
      θ_tilt = arctan(√((g₁¹)² + (h₁¹)²) / |g₁⁰|)
    """
    g10 = g_coeffs[1, 0]
    g11 = g_coeffs[1, 1]
    h11 = h_coeffs[1, 1]
    tilt = np.arctan2(np.sqrt(g11 ** 2 + h11 ** 2), abs(g10))
    return np.degrees(tilt)


def field_gradient_on_triangulation(nodes, elements, values):
    """
    在三角化网格上计算标量场的梯度（源自 1342_triangulation_order3_contour）。
    对每个三角形单元，梯度为常数。
    """
    if elements.size == 0:
        return np.zeros_like(nodes)

    n_nodes = len(nodes)
    grad = np.zeros((n_nodes, 3))
    count = np.zeros(n_nodes)

    for elem in elements:
        p0, p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = p3 - p0

        # 体积
        vol = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        if vol < 1e-15:
            continue

        val = values[elem]
        # 简化：仅计算径向梯度估计
        grad_elem = np.array([
            (val[1] - val[0]) / (np.linalg.norm(v0) + 1e-15),
            (val[2] - val[0]) / (np.linalg.norm(v1) + 1e-15),
            (val[3] - val[0]) / (np.linalg.norm(v2) + 1e-15),
        ])

        for idx in elem:
            grad[idx] += grad_elem
            count[idx] += 1

    for i in range(n_nodes):
        if count[i] > 0:
            grad[i] /= count[i]

    return grad
