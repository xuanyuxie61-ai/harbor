# -*- coding: utf-8 -*-
"""
field_solver.py
===============

磁场求解器模块: 偶极场 + Chirikov 混沌映射.

本模块实现地球磁场的建模, 包括:
  1. 偶极磁场 (解析解)
  2. Chirikov 标准映射 (磁力线混沌)
  3. 磁场线追踪
  4. 磁场拓扑分析

物理背景:

1. 偶极磁场:
   在球坐标 (r, theta, phi) 中, 偶极磁场的分量为:
     B_r = -2*M*cos(theta) / r^3
     B_theta = -M*sin(theta) / r^3
     B_phi = 0

   其中 M 为偶极矩, theta 为磁余纬.

2. Chirikov 标准映射 (Standard Map):
   描述磁力线在扰动磁场中的混沌行为:
     p_{n+1} = p_n + K * sin(theta_n)   (mod 2*pi)
     theta_{n+1} = theta_n + p_{n+1}     (mod 2*pi)

   其中 K 为随机性参数:
     K < 1: 规则运动 (KAM 环面)
     K > K_c ~ 0.9716: 全局混沌
     K >> 1: 完全扩散

   物理对应:
     theta: 环向角 (方位角)
     p: 极向角 (磁纬)
     K: 扰动幅度 (非偶极场成分)

3. 磁场线方程:
   dr/B_r = r*dtheta/B_theta = r*sin(theta)*dphi/B_phi

   在偶极场中, 磁场线方程为:
     r = L * R_E * sin^2(theta)

参考文献:
  [1] Chirikov, B.V., "A universal instability of multi-dimensional
      oscillator systems", Phys. Rep. 52, 263-379 (1979)
  [2] Roederer, J.G., "Dynamics of Geomagnetically Trapped Radiation",
      Springer (1970)
"""

import numpy as np
import physical_constants as pc


# =============================================================================
#  偶极磁场
# =============================================================================

class DipoleField:
    """
    地球偶极磁场.

    参数
    ----
    moment : float
        偶极矩 [A*m^2]
    tilt : float
        偶极倾角 [rad]
    """

    def __init__(self, moment=None, tilt=None):
        self.moment = pc.DIPOLE_MOMENT if moment is None else moment
        self.tilt = pc.DIPOLE_TILT if tilt is None else tilt

    def field_spherical(self, r, theta, phi=0.0):
        """
        计算球坐标下的磁场分量.

        物理公式:
          B_r = -2 * B_eq * (R_E/r)^3 * cos(theta)
          B_theta = -B_eq * (R_E/r)^3 * sin(theta)
          B_phi = 0

        参数
        ----
        r : ndarray
            径向距离 [m]
        theta : ndarray
            磁余纬 [rad] (0 = 北极, pi/2 = 赤道)
        phi : ndarray
            方位角 [rad]

        返回
        -------
        B_r, B_theta, B_phi : ndarray
            磁场分量 [T]
        """
        r = np.asarray(r, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)

        r = np.maximum(r, pc.R_EARTH * 0.5)  # 防止 r=0

        # B_eq * (R_E/r)^3 给出赤道面磁场强度
        factor = pc.B_EQUATORIAL * (pc.R_EARTH / r)**3

        B_r = -2.0 * factor * np.cos(theta)
        B_theta = -factor * np.sin(theta)
        B_phi = np.zeros_like(B_r)

        return B_r, B_theta, B_phi

    def field_cartesian(self, x, y, z):
        """
        计算笛卡尔坐标下的磁场.

        转换关系:
          r = sqrt(x^2 + y^2 + z^2)
          theta = arccos(z/r)
          phi = arctan2(y, x)

        参数
        ----
        x, y, z : ndarray
            笛卡尔坐标 [m]

        返回
        -------
        B_x, B_y, B_z : ndarray
            磁场分量 [T]
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, pc.R_EARTH * 0.5)
        theta = np.arccos(np.clip(z / r, -1, 1))
        phi = np.arctan2(y, x)

        B_r, B_theta, B_phi = self.field_spherical(r, theta, phi)

        # 球坐标 -> 笛卡尔
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)

        B_x = B_r * sin_t * cos_p + B_theta * cos_t * cos_p - B_phi * sin_p
        B_y = B_r * sin_t * sin_p + B_theta * cos_t * sin_p + B_phi * cos_p
        B_z = B_r * cos_t - B_theta * sin_t

        return B_x, B_y, B_z

    def field_magnitude(self, r, theta):
        """
        计算磁场模长.

        |B| = mu_0*M/(4*pi*r^3) * sqrt(1 + 3*cos^2(theta))
            = B_eq * (R_E/r)^3 * sqrt(1 + 3*cos^2(theta))

        参数
        ----
        r : ndarray
            径向距离 [m]
        theta : ndarray
            磁余纬 [rad]

        返回
        -------
        B_mag : ndarray
            磁场模长 [T]
        """
        r = np.asarray(r, dtype=np.float64)
        r = np.maximum(r, pc.R_EARTH * 0.5)
        # 使用赤道磁场和 L = r/R_E 来缩放
        # B_eq * (R_E/r)^3 = B_eq / L^3
        return pc.B_EQUATORIAL * (pc.R_EARTH / r)**3 * np.sqrt(1.0 + 3.0 * np.cos(theta)**2)

    def field_line(self, L, theta_range=None):
        """
        计算偶极场磁力线形状.

        偶极场磁力线方程:
          r(theta) = L * R_E * sin^2(theta)

        参数
        ----
        L : float
            McIlwain L 参数
        theta_range : ndarray, optional
            磁余纬范围

        返回
        -------
        r, theta : ndarray
            磁力线坐标
        """
        if theta_range is None:
            theta_range = np.linspace(0.1, np.pi - 0.1, 100)

        r = L * pc.R_EARTH * np.sin(theta_range)**2
        return r, theta_range


# =============================================================================
#  Chirikov 标准映射
# =============================================================================

class ChirikovMap:
    """
    Chirikov 标准映射 (Standard Map).

    映射规则:
      p_{n+1} = p_n + K * sin(theta_n)   (mod 2*pi)
      theta_{n+1} = theta_n + p_{n+1}     (mod 2*pi)

    物理对应:
      - theta: 环向角 (方位角, 磁力线环绕方向)
      - p: 共轭动量 (极向角, 磁纬方向)
      - K: 随机参数 (非偶极场扰动幅度)

    对于磁层物理:
      K ~ delta_B / B_0 (非偶极场扰动相对幅度)
      K > K_c ~ 0.9716 时, 磁力线全局混沌, 粒子径向扩散增强.

    参数
    ----
    K : float
        随机参数 (默认 1.5, 超临界混沌)
    """

    def __init__(self, K=1.5):
        self.K = K

    def iterate(self, theta, p):
        """
        执行一次映射迭代.

        参数
        ----
        theta : float 或 ndarray
            当前角度
        p : float 或 ndarray
            当前动量

        返回
        -------
        theta_new, p_new : float 或 ndarray
            映射后的值
        """
        p_new = p + self.K * np.sin(theta)
        theta_new = theta + p_new

        # 取模到 [0, 2*pi]
        p_new = p_new % (2.0 * np.pi)
        theta_new = theta_new % (2.0 * np.pi)

        return theta_new, p_new

    def orbit(self, theta0, p0, n_steps):
        """
        计算轨道.

        参数
        ----
        theta0, p0 : float
            初始条件
        n_steps : int
            迭代步数

        返回
        -------
        theta_orbit, p_orbit : ndarray
            轨道
        """
        theta_orbit = np.zeros(n_steps)
        p_orbit = np.zeros(n_steps)

        theta, p = theta0, p0
        for i in range(n_steps):
            theta_orbit[i] = theta
            p_orbit[i] = p
            theta, p = self.iterate(theta, p)

        return theta_orbit, p_orbit

    def lyapunov_exponent(self, theta0, p0, n_steps=1000):
        """
        计算最大 Lyapunov 指数.

        物理意义:
          lambda > 0: 混沌轨道 (指数分离)
          lambda = 0: 规则轨道 (KAM 环面)

        算法:
          跟踪两个初始相距 delta_0 的轨道,
          lambda ~ lim_{n->inf} (1/n) * sum log(delta_i / delta_0)

        参数
        ----
        theta0, p0 : float
            初始条件
        n_steps : int
            迭代步数

        返回
        -------
        lambda_max : float
            最大 Lyapunov 指数
        """
        delta = 1.0e-8
        theta1, p1 = theta0, p0
        theta2, p2 = theta0 + delta, p0

        lyap_sum = 0.0
        n_renorm = 0

        for i in range(n_steps):
            theta1, p1 = self.iterate(theta1, p1)
            theta2, p2 = self.iterate(theta2, p2)

            # 距离
            d_theta = theta2 - theta1
            d_p = p2 - p1
            d = np.sqrt(d_theta**2 + d_p**2)

            if d > 0:
                lyap_sum += np.log(d / delta)
                n_renorm += 1

                # 重正化
                theta2 = theta1 + delta * d_theta / d
                p2 = p1 + delta * d_p / d

        return lyap_sum / max(n_renorm, 1)

    def diffusion_coefficient(self, K_range=None, n_samples=100, n_steps=500):
        """
        计算准线性扩散系数 D(K).

        物理公式 (准线性理论):
          D_QL = K^2 / 4  (对于 K << 1)

        对于大 K, 扩散系数趋于饱和:
          D ~ pi^2 / 3  (随机行走极限)

        参数
        ----
        K_range : ndarray
            K 值范围
        n_samples : int
            每个 K 的样本数
        n_steps : int
            每个样本的迭代步数

        返回
        -------
        K_values : ndarray
            K 值
        D_values : ndarray
            扩散系数
        """
        if K_range is None:
            K_range = np.linspace(0.1, 5.0, 20)

        D_values = np.zeros_like(K_range)

        for i, K in enumerate(K_range):
            self.K = K
            p_diff_sum = 0.0
            n_valid = 0

            for j in range(n_samples):
                theta0 = np.random.uniform(0, 2*np.pi)
                p0 = np.random.uniform(0, 2*np.pi)

                theta_orb, p_orb = self.orbit(theta0, p0, n_steps)

                # 计算均方位移
                dp = p_orb - p0
                # 处理绕回 (unwrap)
                dp_unwrapped = np.unwrap(dp - p0, period=2*np.pi)
                msd = np.mean(dp_unwrapped**2)

                # D = <(delta p)^2> / (2 * n_steps)
                D = msd / (2.0 * n_steps)
                p_diff_sum += D
                n_valid += 1

            D_values[i] = p_diff_sum / max(n_valid, 1)

        return K_range, D_values


# =============================================================================
#  磁场线追踪
# =============================================================================

def trace_field_line(field, r0, theta0, phi0, n_steps=1000, ds=0.01*pc.R_EARTH):
    """
    追踪磁场线.

    磁场线方程:
      dr/ds = B_r / |B|
      r*dtheta/ds = B_theta / |B|
      r*sin(theta)*dphi/ds = B_phi / |B|

    使用 4 阶 Runge-Kutta 方法.

    参数
    ----
    field : DipoleField
        磁场对象
    r0, theta0, phi0 : float
        初始位置
    n_steps : int
        追踪步数
    ds : float
        步长 [m]

    返回
    -------
    r, theta, phi : ndarray
        磁场线轨迹
    """
    r_arr = np.zeros(n_steps)
    theta_arr = np.zeros(n_steps)
    phi_arr = np.zeros(n_steps)

    r, theta, phi = r0, theta0, phi0

    for i in range(n_steps):
        r_arr[i] = r
        theta_arr[i] = theta
        phi_arr[i] = phi

        # RK4
        k1_r, k1_t, k1_p = _field_line_rhs(field, r, theta, phi)
        k2_r, k2_t, k2_p = _field_line_rhs(field, r + 0.5*ds*k1_r,
                                              theta + 0.5*ds*k1_t,
                                              phi + 0.5*ds*k1_p)
        k3_r, k3_t, k3_p = _field_line_rhs(field, r + 0.5*ds*k2_r,
                                              theta + 0.5*ds*k2_t,
                                              phi + 0.5*ds*k2_p)
        k4_r, k4_t, k4_p = _field_line_rhs(field, r + ds*k3_r,
                                              theta + ds*k3_t,
                                              phi + ds*k3_p)

        r += ds * (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6.0
        theta += ds * (k1_t + 2*k2_t + 2*k3_t + k4_t) / 6.0
        phi += ds * (k1_p + 2*k2_p + 2*k3_p + 6.0) / 6.0

        # 边界检查
        if r < pc.R_EARTH * 0.9 or r > 20 * pc.R_EARTH:
            r_arr[i+1:] = r
            theta_arr[i+1:] = theta
            phi_arr[i+1:] = phi
            break

        # 角度范围
        theta = np.clip(theta, 0.01, np.pi - 0.01)
        phi = phi % (2.0 * np.pi)

    return r_arr, theta_arr, phi_arr


def _field_line_rhs(field, r, theta, phi):
    """磁场线方程右端函数."""
    B_r, B_theta, B_phi = field.field_spherical(r, theta, phi)
    B_mag = np.sqrt(B_r**2 + B_theta**2 + B_phi**2)
    B_mag = max(B_mag, pc.EPSILON_NUM)

    dr_ds = B_r / B_mag
    dtheta_ds = B_theta / (r * B_mag)
    dphi_ds = B_phi / (max(r, pc.EPSILON_NUM) * max(np.sin(theta), pc.EPSILON_NUM) * B_mag)

    return dr_ds, dtheta_ds, dphi_ds


# =============================================================================
#  自检验证
# =============================================================================

def self_test():
    """自检验证."""
    print("=" * 60)
    print("磁场求解器自检验证")
    print("=" * 60)

    # 偶极场
    print("\n--- 偶极磁场 ---")
    dipole = DipoleField()
    B_eq = dipole.field_magnitude(pc.R_EARTH, np.pi/2)
    print(f"  地球表面赤道磁场: {B_eq:.4e} T = {B_eq*1e9:.1f} nT")

    B_L4 = dipole.field_magnitude(4*pc.R_EARTH, np.pi/2)
    print(f"  L=4 赤道磁场: {B_L4:.4e} T")

    # Chirikov 映射
    print("\n--- Chirikov 标准映射 ---")
    for K in [0.5, 1.0, 1.5, 3.0]:
        cmap = ChirikovMap(K=K)
        lyap = cmap.lyapunov_exponent(1.0, 1.0, n_steps=500)
        print(f"  K = {K:.1f}: Lyapunov = {lyap:.4f}")

    # 磁场线追踪
    print("\n--- 磁场线追踪 ---")
    r0 = 4 * pc.R_EARTH
    theta0 = np.pi / 4
    phi0 = 0.0
    r, theta, phi = trace_field_line(dipole, r0, theta0, phi0, n_steps=200)
    print(f"  起点: r = {r0/pc.R_EARTH:.1f} R_E, theta = {np.degrees(theta0):.1f} deg")
    print(f"  终点: r = {r[-1]/pc.R_EARTH:.1f} R_E, theta = {np.degrees(theta[-1]):.1f} deg")

    return True


if __name__ == "__main__":
    self_test()
