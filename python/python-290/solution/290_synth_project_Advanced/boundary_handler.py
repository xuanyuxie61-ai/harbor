"""
boundary_handler.py - 周期性边界与阿尔芬波特征边界处理模块

本模块为环形等离子体中阿尔芬波的高阶有限差分模拟提供边界条件处理。
核心算法源自 Zeller 同余法中的模运算循环思想 (1412_weekday_zeller)，
将其从日期计算推广到物理场的周期性缠绕处理。

物理背景:
  在托卡马克环形几何中，环向角 φ ∈ [0, 2π) 和极向角 θ ∈ [0, 2π)
  均为周期性坐标。对于沿磁力线方向的有限差分离散，需要使用周期性
  边界条件来处理越过域边界的场值索引。

  阿尔芬波在环向的色散关系要求波函数满足:
    f(φ + 2π) = f(φ) * exp(i * n * 2π) = f(φ)
  其中 n 为环向模数。

核心公式:
  模运算缠绕 (改编自 Zeller 同余):
    wrap(i, N) = i mod N,  对于 i ∈ [0, N-1]
    wrap(i, N) = i + N,    对于 i < 0
    wrap(i, N) = i - N,    对于 i ≥ N

  螺旋磁场线的准周期性:
    f(θ + 2πq) = f(θ)
  其中 q 为安全因子 (safety factor), 在有理面 q = m/n 处
  阿尔芬波共振产生奇异层。

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np


def cyclic_wrap(value, lo, hi):
    """
    将整数值缠绕到 [lo, hi] 区间内 (周期性边界)。

    算法直接改编自 Zeller 同余法中的 i4_wrap 函数，
    用于处理托卡马克环向/极向坐标的周期性。

    数学表达:
      wrap(v, lo, hi) = lo + mod(v - lo, hi - lo + 1)

    在阿尔芬波模拟中的应用:
      对于 N 个网格点的环向域，索引 i 超出 [0, N-1] 时
      自动缠绕到有效范围内，实现 f(φ+2π) = f(φ)。

    参数:
      value: int or array_like, 待缠绕的值
      lo: int, 区间下界
      hi: int, 区间上界

    返回:
      int or ndarray, 缠绕后的值
    """
    lo = int(lo)
    hi = int(hi)
    if isinstance(value, np.ndarray):
        value = value.astype(int)
    else:
        value = int(value)

    if lo > hi:
        lo, hi = hi, lo

    period = hi - lo + 1
    result = lo + (value - lo) % period
    return result


def periodic_index(i, n_grid):
    """
    计算周期边界条件下的有效网格索引。

    对于环向网格 [0, n_grid-1]，超出范围的索引自动缠绕。
    这是阿尔芬波沿环向传播时的基本边界操作。

    参数:
      i: int or array_like, 原始索引
      n_grid: int, 网格点数

    返回:
      int or ndarray, 有效索引 ∈ [0, n_grid-1]
    """
    return cyclic_wrap(i, 0, n_grid - 1)


def periodic_roll(field, shift, axis=0):
    """
    对场数据施加周期性移位 (环形平移算子)。

    在阿尔芬波传播中，经过一个时间步 Δt 后，波前沿磁力线
    移动距离 v_A * Δt，对应的网格移位为 shift = v_A * Δt / Δx。
    周期性移位算子 T_shift 作用在场 f 上:
      [T_shift f](i) = f(i - shift mod N)

    参数:
      field: ndarray, 场数据
      shift: int, 移位量
      axis: int, 移位轴

    返回:
      ndarray, 移位后的场
    """
    return np.roll(field, shift, axis=axis)


def helical_boundary_map(theta_idx, phi_idx, q_safety, n_theta, n_phi):
    """
    螺旋磁力线在 (θ, φ) 平面上的准周期性映射。

    磁力线方程: dθ/dφ = 1/q(ψ)
    积分得到: θ(φ) = θ₀ + φ/q

    对于有理面 q = m/n，磁力线闭合，周期为 n 圈环向。
    对于无理面，磁力线遍历整个磁面 (KAM 理论)。

    本函数计算从 (θ_idx, phi_idx) 出发，沿磁力线走一步
    到达的网格位置，用于沿场线的有限差分插值。

    参数:
      theta_idx: int, 极向角索引
      phi_idx: int, 环向角索引
      q_safety: float, 安全因子 (q = rB_φ / (RB_θ))
      n_theta: int, 极向网格数
      n_phi: int, 环向网格数

    返回:
      tuple (theta_new, phi_new), 映射后的索引
    """
    if abs(q_safety) < 1e-14:
        raise ValueError("安全因子 q 不能为零 (物理上对应纯极向磁场)")

    # 沿磁力线走 Δφ = 2π/n_phi，对应的 Δθ = 2π/(q * n_phi)
    # 在离散网格上:
    delta_theta_idx = int(round(1.0 / (q_safety * n_phi) * n_theta))

    theta_new = periodic_index(theta_idx + delta_theta_idx, n_theta)
    phi_new = periodic_index(phi_idx + 1, n_phi)

    return theta_new, phi_new


def field_line_tracing(q_profile, r_norm, theta0, n_steps, n_theta, n_phi):
    """
    追踪一条磁力线在 (θ, φ) 截面上的轨迹。

    给定安全因子剖面 q(r)，从初始极向角 θ₀ 出发，
    沿环向逐步追踪磁力线。

    物理意义:
      磁力线方程在柱坐标 (r, θ, φ) 下为:
        dr/dφ = 0  (在对称平衡中，磁力线在磁面上)
        dθ/dφ = 1/q(r)

    参数:
      q_profile: float or callable, 安全因子 (常数或 q(r) 函数)
      r_norm: float, 归一化半径 r/a
      theta0: float, 初始极向角 [rad]
      n_steps: int, 追踪步数
      n_theta: int, 极向网格数
      n_phi: int, 环向网格数

    返回:
      theta_array: ndarray, 极向角序列 [rad]
      phi_array: ndarray, 环向角序列 [rad]
      theta_idx_array: ndarray, 极向角索引序列
      phi_idx_array: ndarray, 环向角索引序列
    """
    if callable(q_profile):
        q = q_profile(r_norm)
    else:
        q = float(q_profile)

    if abs(q) < 1e-14:
        raise ValueError("安全因子 q 不能为零")

    theta_array = np.zeros(n_steps + 1)
    phi_array = np.zeros(n_steps + 1)
    theta_idx_array = np.zeros(n_steps + 1, dtype=int)
    phi_idx_array = np.zeros(n_steps + 1, dtype=int)

    theta_array[0] = theta0
    phi_array[0] = 0.0

    dphi = 2.0 * np.pi / n_phi
    dtheta_per_dphi = 1.0 / q

    for step in range(1, n_steps + 1):
        phi_array[step] = step * dphi
        theta_array[step] = theta0 + step * dphi * dtheta_per_dphi

        # 将连续角度映射到离散网格索引
        theta_idx = int(np.floor(theta_array[step] / (2.0 * np.pi) * n_theta))
        phi_idx = int(np.floor(phi_array[step] / (2.0 * np.pi) * n_phi))

        theta_idx_array[step] = periodic_index(theta_idx, n_theta)
        phi_idx_array[step] = periodic_index(phi_idx, n_phi)

    return theta_array, phi_array, theta_idx_array, phi_idx_array


def rational_surface_check(q_value, m_mode, n_mode, tolerance=1e-3):
    """
    检查给定安全因子是否对应 (m, n) 模的有理共振面。

    阿尔芬波连续谱的共振条件:
      q(r_res) = m/n

    在共振面处，阿尔芬波连续谱出现奇异 (连续阻尼)，
    高能粒子的漂移共振可以激发阿尔芬本征模 (TAE, EAE 等)。

    参数:
      q_value: float, 当地安全因子
      m_mode: int, 极向模数
      n_mode: int, 环向模数
      tolerance: float, 判断容差

    返回:
      is_rational: bool, 是否接近有理面
      q_rational: float, 有理面值 m/n
      shear_parameter: float, 磁剪切 s = (r/q)(dq/dr) 的近似
    """
    if n_mode == 0:
        return False, float('inf'), 0.0

    q_rational = float(m_mode) / float(n_mode)
    is_rational = abs(q_value - q_rational) < tolerance

    # 简化的磁剪切估计 (假设 q 剖面为 q(r) = q₀ + (q_a - q₀)(r/a)²)
    # s = (r/q) * dq/dr = 2r²(q_a - q₀)/(a²q)
    # 此处仅返回局部 q 值与有理面的偏差
    shear_parameter = (q_value - q_rational) / q_rational if abs(q_rational) > 1e-14 else 0.0

    return is_rational, q_rational, shear_parameter


def boundary_layer_profile(n_points, delta_boundary):
    """
    生成边界层中的场值衰减剖面。

    在阿尔芬波模拟中，为避免边界反射，通常在计算域边缘
    设置吸收层 (类似 PML)。场的振幅在边界层内按指数衰减:
      f(r) = f_interior * exp(-(r - r_boundary)² / δ²)
    其中 δ 为边界层宽度参数。

    参数:
      n_points: int, 边界层中的网格点数
      delta_boundary: float, 边界层宽度参数 δ

    返回:
      profile: ndarray, 衰减因子数组 ∈ (0, 1]
      x_boundary: ndarray, 边界层局部坐标
    """
    if delta_boundary <= 0.0:
        raise ValueError("边界层宽度 delta_boundary 必须为正")

    x_boundary = np.linspace(0.0, 3.0 * delta_boundary, n_points)
    profile = np.exp(-(x_boundary ** 2) / (delta_boundary ** 2))

    # 确保数值下界
    profile = np.maximum(profile, 1e-15)

    return profile, x_boundary
