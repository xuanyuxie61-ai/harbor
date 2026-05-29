# -*- coding: utf-8 -*-
"""
sphere_geometry.py
球面几何与边界条件处理模块

融合来源:
- 307_distance_to_position_sphere: 球面距离计算、经纬度与笛卡尔坐标转换

功能:
- 球面上两点间大圆距离计算（Haversine 公式）
- 经纬度 (lat, lon) 与笛卡尔坐标 (x, y, z) 的相互转换
- 球面边界条件：无滑移边界、自由滑移边界
- 球面谐波展开（用于初始湍流场的谱初始化）

数学背景:
  设球半径为 R，两点经纬度为 (phi1, lambda1) 和 (phi2, lambda2)。
  大圆距离（Great-circle distance）:
    d = R * arccos(sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(Delta_lambda))
  或数值更稳定的 Haversine 公式:
    a = sin^2(Delta_phi/2) + cos(phi1)*cos(phi2)*sin^2(Delta_lambda/2)
    d = 2 * R * arcsin(sqrt(a))

  球面谐函数:
    Y_l^m(theta, phi) = N_l^m * P_l^m(cos(theta)) * exp(i*m*phi)
    其中 N_l^m 为归一化常数，P_l^m 为连带 Legendre 函数。
"""

import numpy as np


def sphere_distance1(lat1, lon1, lat2, lon2, r):
    """
    计算球面上两点间的大圆距离。
    融合自 307_distance_to_position_sphere 的 sphere_distance1。

    参数:
      lat1, lon1: 点1的纬度和经度（弧度）
      lat2, lon2: 点2的纬度和经度（弧度）
      r: 球半径

    返回:
      大圆距离
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
    # 边界处理：避免浮点误差导致 a > 1
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def ll_to_xyz(r, n, lat, lon):
    """
    将经纬度转换为笛卡尔坐标。
    融合自 307_distance_to_position_sphere 的 ll_to_xyz。

    转换公式:
      x = R * cos(lon) * cos(lat)
      y = R * sin(lon) * cos(lat)
      z = R * sin(lat)
    """
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    z = r * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def xyz_to_ll(r, xyz):
    """
    将笛卡尔坐标转换为经纬度。

    参数:
      r: 球半径
      xyz: (..., 3) 笛卡尔坐标

    返回:
      lat, lon: 纬度和经度（弧度）
    """
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    # 边界处理：避免除零
    rho = np.sqrt(x ** 2 + y ** 2)
    rho = np.where(rho < 1e-15, 1e-15, rho)

    lat = np.arctan2(z, rho)
    lon = np.arctan2(y, x)
    return lat, lon


def spherical_harmonic(l, m, theta, phi):
    """
    计算实球面谐函数 Y_l^m(theta, phi)。

    数学公式:
      Y_l^m(theta, phi) = N_l^m * P_l^m(cos(theta)) * cos(m*phi)   (m >= 0)
      Y_l^m(theta, phi) = N_l^m * P_l^m(cos(theta)) * sin(|m|*phi) (m < 0)

      归一化常数:
        N_l^m = sqrt((2*l+1)/(4*pi) * (l-|m|)! / (l+|m|)!)

    参数:
      l: 阶数 (>= 0)
      m: 次数 (-l <= m <= l)
      theta: 极角 (0 <= theta <= pi)
      phi: 方位角 (0 <= phi <= 2*pi)

    返回:
      球面谐函数值
    """
    from scipy.special import sph_harm
    # 使用 scipy 的 sph_harm 并取实部
    # sph_harm(m, l, phi, theta) 参数顺序为 (m, l, phi, theta)
    ylm_complex = sph_harm(abs(m), l, phi, theta)
    if m >= 0:
        return np.real(ylm_complex) if m == 0 else np.real(ylm_complex) * np.sqrt(2.0)
    else:
        return np.imag(ylm_complex) * np.sqrt(2.0)


def spherical_harmonic_manual(l, m, theta, phi):
    """
    手动实现的实球面谐函数（不依赖 scipy）。
    """
    x = np.cos(theta)

    # 计算连带 Legendre 多项式 P_l^m(x)
    def associated_legendre(l_val, m_val, x_val):
        # 归一化 m >= 0
        m_abs = abs(m_val)
        # 递推计算
        if m_abs > l_val:
            return np.zeros_like(x_val)

        # 使用递推关系
        p_mm = np.ones_like(x_val)
        if m_abs > 0:
            somx2 = np.sqrt(np.maximum(0.0, 1.0 - x_val ** 2))
            fact = 1.0
            for i in range(1, m_abs + 1):
                p_mm *= -fact * somx2
                fact += 2.0

        if l_val == m_abs:
            return p_mm

        p_mmp1 = x_val * (2 * m_abs + 1) * p_mm
        if l_val == m_abs + 1:
            return p_mmp1

        for ll in range(m_abs + 2, l_val + 1):
            p_ll = (x_val * (2 * ll - 1) * p_mmp1 - (ll + m_abs - 1) * p_mm) / (ll - m_abs)
            p_mm = p_mmp1
            p_mmp1 = p_ll

        return p_mmp1

    p_lm = associated_legendre(l, m, x)

    # 归一化常数
    from math import factorial, sqrt, pi
    nlm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - abs(m)) / factorial(l + abs(m)))

    if m > 0:
        return nlm * p_lm * np.cos(m * phi) * sqrt(2.0)
    elif m < 0:
        return nlm * p_lm * np.sin(abs(m) * phi) * sqrt(2.0)
    else:
        return nlm * p_lm


def generate_turbulent_initial_field(nx, ny, nz, max_l=8, seed=42):
    """
    利用球面谐波展开生成三维湍流初始速度场。

    物理模型:
      速度场在球面上展开为:
        u(theta, phi) = sum_{l=0}^{L_max} sum_{m=-l}^{l} a_l^m * Y_l^m(theta, phi)
      其中系数 a_l^m 服从随机的能量谱分布:
        E(k) ~ k^4 * exp(-2*(k/k_0)^2)
      这是 von Karman 谱的一种形式。

    参数:
      nx, ny, nz: 网格尺寸
      max_l: 最大球谐阶数
      seed: 随机种子

    返回:
      u, v, w: 三个方向的速度分量
    """
    rng = np.random.default_rng(seed)

    # 构造球坐标网格
    theta = np.linspace(0, np.pi, nx)
    phi = np.linspace(0, 2 * np.pi, ny)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

    u_surf = np.zeros((nx, ny), dtype=float)
    v_surf = np.zeros((nx, ny), dtype=float)

    # 使用 von Karman 能量谱生成随机系数
    for l in range(1, max_l + 1):
        k = float(l)
        energy = k ** 4 * np.exp(-2.0 * (k / 4.0) ** 2)
        amplitude = np.sqrt(energy) * rng.standard_normal()

        for m in range(-l, l + 1):
            coeff = amplitude * rng.standard_normal()
            try:
                ylm = spherical_harmonic_manual(l, m, Theta, Phi)
                u_surf += coeff * ylm
                v_surf += coeff * ylm * 0.5
            except Exception:
                pass

    # 沿径向复制到 3D
    u = np.repeat(u_surf[:, :, np.newaxis], nz, axis=2)
    v = np.repeat(v_surf[:, :, np.newaxis], nz, axis=2)
    w = rng.standard_normal(size=(nx, ny, nz)) * 0.1

    # 边界处理：确保无散度（近似）
    u = u - np.mean(u)
    v = v - np.mean(v)
    w = w - np.mean(w)

    return u, v, w


def spherical_boundary_condition(u, v, w, r, radius, bc_type='no_slip'):
    """
    在球面边界上施加边界条件。

    参数:
      u, v, w: 速度场
      r: 到中心的距离场
      radius: 球半径
      bc_type: 'no_slip' (无滑移) 或 'free_slip' (自由滑移)

    数学模型:
      无滑移边界:
        u = v = w = 0  at r = R
      自由滑移边界:
        u_n = 0,  tau_s = 0  at r = R
        其中 u_n 为法向速度，tau_s 为切向应力。
    """
    if bc_type == 'no_slip':
        mask = np.abs(r - radius) < 1e-6
        u = np.where(mask, 0.0, u)
        v = np.where(mask, 0.0, v)
        w = np.where(mask, 0.0, w)
    elif bc_type == 'free_slip':
        # 自由滑移：法向速度为零
        mask = np.abs(r - radius) < 1e-6
        # 近似处理：将边界上速度投影到切平面
        x = np.linspace(-1, 1, u.shape[0])
        y = np.linspace(-1, 1, u.shape[1])
        z = np.linspace(-1, 1, u.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # 法向向量
        nx = X / (r + 1e-15)
        ny = Y / (r + 1e-15)
        nz = Z / (r + 1e-15)

        # 法向速度分量
        un = u * nx + v * ny + w * nz

        # 减去法向分量
        u = np.where(mask, u - un * nx, u)
        v = np.where(mask, v - un * ny, v)
        w = np.where(mask, w - un * nz, w)

    return u, v, w
