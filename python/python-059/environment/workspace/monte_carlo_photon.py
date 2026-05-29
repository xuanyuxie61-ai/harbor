"""
monte_carlo_photon.py
三维蒙特卡洛光子传输模块

整合原项目:
  - 1011_random_walk_3d_simulation: 三维随机游走

功能:
  使用蒙特卡洛方法模拟光子在大气气溶胶-云体系中的三维随机行走，
  计算多次散射后的出射辐射强度和光路长度分布。

核心物理:
  - 光子自由程: l = -ln(ξ) / β_e,  ξ ~ Uniform(0,1)
  - 散射方向: 由 Henyey-Greenstein 相函数抽样
  - 边界条件: 大气层顶逃逸、地表吸收/反射

数学公式:
  - HG 相函数抽样 (Rejection method 或解析反演):
      cos Θ = (1 + g^2 - ((1 - g^2) / (1 - g + 2gξ))^2) / (2g)   (g ≠ 0)
  - 步进更新:
      x_{n+1} = x_n + l * sin θ cos φ
      y_{n+1} = y_n + l * sin θ sin φ
      z_{n+1} = z_n + l * cos θ
"""

import numpy as np
from math import sqrt, log, cos, sin, pi, acos


class MonteCarloError(Exception):
    pass


def sample_hg_scattering_angle(g):
    """
    从 Henyey-Greenstein 相函数中抽样散射角余弦 cos Θ。

    解析反演公式:
      若 g ≠ 0:
        cos Θ = (1 + g^2 - ((1 - g^2)/(1 - g + 2gξ))^2) / (2g)
      若 g = 0 (各向同性):
        cos Θ = 2ξ - 1

    参数:
      g: 不对称因子

    返回:
      cos_theta: 散射角余弦
    """
    xi = np.random.rand()
    if abs(g) < 1e-8:
        return 2.0 * xi - 1.0

    # 边界处理: 避免除以零
    denom = 1.0 - g + 2.0 * g * xi
    if abs(denom) < 1e-15:
        denom = 1e-15

    t = (1.0 - g ** 2) / denom
    cos_theta = (1.0 + g ** 2 - t ** 2) / (2.0 * g)
    # 截断到 [-1, 1] 防止浮点溢出
    return float(np.clip(cos_theta, -1.0, 1.0))


def rotate_direction(u, v, w, cos_theta):
    """
    根据散射角 cos_theta 更新光子传播方向 (u, v, w)。
    方位角 φ 在 [0, 2π] 均匀随机抽样。

    公式 (基于球坐标旋转):
      sin Θ = sqrt(1 - cos^2 Θ)
      φ = 2π ξ_φ
      u' = sin Θ cos φ
      v' = sin Θ sin φ
      w' = cos Θ
    然后旋转到原方向坐标系中。
    """
    sin_theta = sqrt(max(0.0, 1.0 - cos_theta ** 2))
    phi = 2.0 * pi * np.random.rand()

    # 若原方向接近 z 轴，简化旋转
    if abs(w) > 0.99999:
        u_new = sin_theta * cos(phi)
        v_new = sin_theta * sin(phi)
        w_new = np.sign(w) * cos_theta
    else:
        temp = sqrt(1.0 - w ** 2)
        u_new = sin_theta * (u * w * cos(phi) - v * sin(phi)) / temp + u * cos_theta
        v_new = sin_theta * (v * w * cos(phi) + u * sin(phi)) / temp + v * cos_theta
        w_new = -sin_theta * cos(phi) * temp + w * cos_theta

    # 归一化
    norm = sqrt(u_new ** 2 + v_new ** 2 + w_new ** 2)
    if norm < 1e-15:
        return 0.0, 0.0, 1.0
    return u_new / norm, v_new / norm, w_new / norm


def photon_random_walk_3d(
    num_photons=1000,
    max_steps=200,
    extinction_coeff=1.0,  # km^-1
    layer_height=10.0,     # km
    g_asymmetry=0.6,
    albedo=0.9,
    surface_albedo=0.2,
):
    """
    模拟大量光子在大气层中的三维随机游走。

    参数:
      num_photons: 光子数
      max_steps: 每光子最大步数
      extinction_coeff: 消光系数 β_e (km^-1)
      layer_height: 大气层高度 (km)
      g_asymmetry: HG 不对称因子
      albedo: 单次散射反照率
      surface_albedo: 地表反照率

    返回:
      escaped_up: 层顶逃逸光子数
      absorbed_surface: 地表吸收光子数
      absorbed_atm: 大气吸收光子数
      path_lengths: 光子总光路长度列表
    """
    if extinction_coeff <= 0 or layer_height <= 0:
        raise MonteCarloError("photon_random_walk_3d: 物理参数必须为正")

    escaped_up = 0
    absorbed_surface = 0
    absorbed_atm = 0
    path_lengths = []

    for _ in range(num_photons):
        # 初始位置: 层顶入射
        x, y, z = 0.0, 0.0, layer_height
        # 初始方向: 向下 (towards surface)
        u, v, w = 0.0, 0.0, -1.0
        path_len = 0.0

        for _ in range(max_steps):
            # 抽样自由程
            xi = np.random.rand()
            if xi < 1e-15:
                xi = 1e-15
            free_path = -log(xi) / extinction_coeff

            # 新位置
            x_new = x + free_path * u
            y_new = y + free_path * v
            z_new = z + free_path * w

            path_len += free_path

            # 检查边界
            if z_new > layer_height:
                # 逃逸出层顶
                escaped_up += 1
                break
            if z_new < 0:
                # 到达地表
                if np.random.rand() < surface_albedo:
                    # 反射，继续游走
                    z_new = 0.0
                    w = abs(w)
                    x, y, z = x_new, y_new, z_new
                    continue
                else:
                    absorbed_surface += 1
                    break

            # 在大气内部: 散射或吸收
            x, y, z = x_new, y_new, z_new

            if np.random.rand() > albedo:
                absorbed_atm += 1
                break

            # 散射，更新方向
            cos_theta = sample_hg_scattering_angle(g_asymmetry)
            u, v, w = rotate_direction(u, v, w, cos_theta)
        else:
            # 达到最大步数，视为吸收
            absorbed_atm += 1

        path_lengths.append(path_len)

    return escaped_up, absorbed_surface, absorbed_atm, np.array(path_lengths)


def estimate_optical_depth_monte_carlo(path_lengths, layer_height):
    """
    由光路长度分布估算有效光学厚度。

    理论关系:
      平均自由程 <l> = 1/β_e
      总光路 L = Σ l_i
      有效 τ ≈ β_e * <z> = layer_height / <l>
    """
    if len(path_lengths) == 0:
        return 0.0
    mean_path = np.mean(path_lengths)
    if mean_path <= 0:
        return 0.0
    return float(layer_height / mean_path)
