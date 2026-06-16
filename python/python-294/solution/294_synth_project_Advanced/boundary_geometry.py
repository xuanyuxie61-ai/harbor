"""
boundary_geometry.py - 等离子体边界几何处理

本模块处理激光等离子体相互作用仿真中的边界几何计算。

在 1D 仿真中，等离子体边界的几何特性影响:
  1. 激光入射角度与反射特性
  2. 临界面 (critical surface) 的位置和形状
  3. 吸收层的几何配置
  4. 粒子注入/提取边界条件

核心几何计算:

  1. 有符号距离函数 (Signed Distance Function):
     对于由两点 p1, p2 定义的边界线段, 点 p 的有符号距离为:
         d = n_hat · (p - p1)
     其中 n_hat = [-dy, dx] / |[-dy, dx]| 为单位法向量,
     (dx, dy) = p2 - p1 为切线方向。

     正值表示在法线正方向侧 (等离子体内部),
     负值表示在法线负方向侧 (真空侧)。

  2. 临界面几何:
     临界面是等离子体密度等于临界密度的等值面:
         n_e(x_c) = n_c = m_e * eps_0 * omega_L^2 / e^2
     在 1D 中为点, 在 2D 中为曲线。

  3. 吸收层 (PML) 几何:
     完美匹配层的衰减剖面:
         sigma(x) = sigma_max * ((x - x_start) / d_pml)^m
     其中 m 通常为 3 或 4 (多项式阶数)。

  4. 密度梯度尺度长度:
     L_n = |n_e / (dn_e/dx)|^{-1}
     决定激光能量沉积的共振吸收效率。
"""

import numpy as np


def signed_distance_to_line(p1, p2, p):
    """计算点 p 到线段 (p1, p2) 的有符号距离。

    有符号距离公式:
        d = n_hat · (p - p1)

    其中法向量为切线方向的 90 度逆时针旋转:
        tangent = (p2 - p1)
        normal = (-tangent_y, tangent_x) / |tangent|

    这个计算源自计算几何中点到直线距离的基本公式,
    在等离子体仿真中用于判断粒子相对于边界的位置。

    Parameters
    ----------
    p1 : ndarray (2,)
        线段起点
    p2 : ndarray (2,)
        线段终点
    p : ndarray (2,) or (N, 2)
        待计算距离的点 (可以是单个点或点集)

    Returns
    -------
    distance : float or ndarray
        有符号距离 (正值在法线正方向侧)
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p = np.asarray(p, dtype=float)

    # 切线方向
    l_dv = p2 - p1
    length = np.linalg.norm(l_dv)

    if length < 1.0e-30:
        # 退化为点
        if p.ndim == 1:
            return np.linalg.norm(p - p1)
        return np.linalg.norm(p - p1, axis=-1)

    # 单位法向量 (逆时针旋转 90 度)
    l_nv = np.array([-l_dv[1], l_dv[0]]) / length

    # 有符号距离
    if p.ndim == 1:
        return np.dot(l_nv, p - p1)
    else:
        return (p - p1) @ l_nv


def plasma_boundary_normal(x, n_profile, dx):
    """计算等离子体密度梯度的法线方向。

    密度梯度方向:
        grad_n = dn/dx
    法线方向 (归一化):
        n_hat = grad_n / |grad_n|

    在 1D 中法线只有正负两个方向,
    但在分析边界几何和入射角时很重要。

    Parameters
    ----------
    x : ndarray
        空间坐标
    n_profile : ndarray
        密度剖面
    dx : float
        网格间距

    Returns
    -------
    normal : ndarray
        归一化密度梯度方向
    gradient_mag : ndarray
        密度梯度幅值
    """
    # 中心差分计算梯度
    grad = np.zeros_like(n_profile)
    grad[1:-1] = (n_profile[2:] - n_profile[:-2]) / (2.0 * dx)
    grad[0] = (n_profile[1] - n_profile[0]) / dx
    grad[-1] = (n_profile[-1] - n_profile[-2]) / dx

    gradient_mag = np.abs(grad)
    # 避免除零
    safe_mag = np.where(gradient_mag > 1.0e-30, gradient_mag, 1.0)
    normal = grad / safe_mag

    return normal, gradient_mag


def density_gradient_scale_length(x, n_profile, dx):
    """计算密度梯度尺度长度 L_n。

    密度梯度尺度长度定义为:
        L_n = |n_e / (dn_e/dx)|

    这是一个关键的等离子体参数，决定:
      - 共振吸收效率: eta_res ~ (k_0 * L_n)^{2/3}
      - 受激拉曼散射增益: gamma_SRS ~ a_0 * omega_p * sqrt(k_0 * L_n)
      - 激光穿透深度

    Parameters
    ----------
    x : ndarray
        空间坐标
    n_profile : ndarray
        密度剖面
    dx : float
        网格间距

    Returns
    -------
    L_n : ndarray
        密度梯度尺度长度
    """
    grad = np.zeros_like(n_profile)
    grad[1:-1] = (n_profile[2:] - n_profile[:-2]) / (2.0 * dx)
    grad[0] = (n_profile[1] - n_profile[0]) / dx
    grad[-1] = (n_profile[-1] - n_profile[-2]) / dx

    safe_n = np.where(np.abs(n_profile) > 1.0e-30, n_profile, 1.0e-30)
    safe_grad = np.where(np.abs(grad) > 1.0e-30, grad, 1.0e-30)

    L_n = np.abs(safe_n / safe_grad)
    return L_n


def pml_profile(x_pml, sigma_max, m=3):
    """计算完美匹配层 (PML) 的吸收剖面。

    PML 电导率剖面:
        sigma(x) = sigma_max * ((x - x_start) / d_pml)^m

    其中 m 为多项式阶数 (通常 3-4), sigma_max 为最大电导率。

    最优 sigma_max 由阻抗匹配条件确定:
        sigma_max = -(m+1) * ln(R) / (2 * d_pml * eta_0)
    其中 R 为目标反射系数 (如 1e-6), eta_0 为真空阻抗。

    Parameters
    ----------
    x_pml : ndarray
        PML 区域内的局部坐标 (0 到 d_pml)
    sigma_max : float
        最大电导率
    m : int
        多项式阶数

    Returns
    -------
    sigma : ndarray
        电导率分布
    """
    x_pml = np.asarray(x_pml)
    # 确保局部坐标非负
    x_local = np.maximum(x_pml, 0.0)
    # 归一化 (假设 x_pml 已归一化到 [0, 1])
    sigma = sigma_max * np.power(x_local, m)
    return sigma


def find_critical_surface(x, n_profile, n_critical):
    """查找临界面位置 (n_e = n_critical)。

    临界面是激光等离子体相互作用中最关键的几何特征。
    在临界面处:
      - 激光频率等于局部等离子体频率: omega_L = omega_p(x_c)
      - 激光群速度趋于零
      - 激光能量通过共振吸收转化为等离子体波

    通过线性插值在离散网格上找到临界面位置。

    Parameters
    ----------
    x : ndarray
        空间坐标
    n_profile : ndarray
        密度剖面
    n_critical : float
        临界密度值

    Returns
    -------
    x_critical : list of float
        临界面位置列表 (可能有多个)
    """
    # 找到密度穿越临界值的位置
    shifted = n_profile - n_critical
    sign_changes = np.where(np.diff(np.sign(shifted)))[0]

    x_critical = []
    for idx in sign_changes:
        # 线性插值
        x1, x2 = x[idx], x[idx + 1]
        n1, n2 = shifted[idx], shifted[idx + 1]
        if abs(n2 - n1) < 1.0e-30:
            xc = 0.5 * (x1 + x2)
        else:
            xc = x1 - n1 * (x2 - x1) / (n2 - n1)
        x_critical.append(xc)

    return x_critical


def boundary_geometry_analysis(config):
    """执行完整的边界几何分析。

    综合计算等离子体边界的所有几何特征,
    为后续仿真提供几何信息。

    Parameters
    ----------
    config : SimulationConfig
        仿真配置

    Returns
    -------
    geometry : dict
        边界几何信息字典
    """
    x = config.x
    n_profile = config.plasma_density_profile(x)
    dx = config.dx

    # 密度梯度分析
    normal, grad_mag = plasma_boundary_normal(x, n_profile, dx)
    L_n = density_gradient_scale_length(x, n_profile, dx)

    # 临界面
    n_critical = config.omega_0 ** 2  # 归一化临界密度
    x_critical = find_critical_surface(x, n_profile, n_critical)

    # 峰值密度位置和梯度
    idx_peak = np.argmax(n_profile)
    x_peak = x[idx_peak]
    n_peak = n_profile[idx_peak]
    L_n_peak = L_n[idx_peak] if L_n[idx_peak] < 1.0e10 else float('inf')

    # 边界定义线 (用于有符号距离计算)
    # 在 1D 中, 边界为两个端点
    p_left = np.array([x[0], 0.0])
    p_right = np.array([x[-1], 0.0])

    geometry = {
        'normal': normal,
        'gradient_magnitude': grad_mag,
        'scale_length': L_n,
        'critical_positions': x_critical,
        'critical_density': n_critical,
        'peak_position': x_peak,
        'peak_density': n_peak,
        'scale_length_at_peak': L_n_peak,
        'boundary_left': p_left,
        'boundary_right': p_right,
        'has_critical_surface': len(x_critical) > 0,
    }

    return geometry
