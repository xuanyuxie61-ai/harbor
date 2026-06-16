# -*- coding: utf-8 -*-
"""
test_particle_orbit.py
======================

测试粒子轨道积分模块.

本模块实现辐射带电子的测试粒子轨道追踪:
  - 引导中心近似
  - 相对论运动方程
  - 绝热不变量计算
  - 漂移壳层追踪

物理背景:

对于相对论电子在偶极场中的运动, 引导中心近似下:
  - 回旋运动: omega_ce = qB/(gamma*m)
  - 弹跳运动: tau_b ~ 2*pi*R_E / v * L
  - 漂移运动: tau_d ~ 2*pi*R_E / v_drift * L^2

三个绝热不变量:
  mu = p_perp^2 / (2*m*B)       (磁矩, 回旋)
  J = integral p_parallel dl    (弹跳)
  Phi = integral B * dA         (漂移通量)

运动方程 (引导中心):
  dr/dt = v_parallel * b + v_E + v_gradB + v_curv

其中:
  b = B / |B|
  v_E = (E x B) / B^2  (E x B 漂移)
  v_gradB = -mu/(q*B) * (b x grad B)
  v_curv = -m*v_par^2/(q*B^2) * (b x (b . grad) b)

参考文献:
  [1] Northrop, T.G., "The Adiabatic Motion of Charged Particles",
      Interscience (1963)
  [2] Bourdarie, S. et al., "A new three-dimensional radiation belt
      model", JGR (2005)
"""

import numpy as np
import physical_constants as pc
from field_solver import DipoleField


class TestParticle:
    """
    测试粒子.

    参数
    ----
    position : ndarray, shape (3,)
        初始位置 [m] (笛卡尔)
    velocity : ndarray, shape (3,)
        初始速度 [m/s]
    charge : float
        电荷 [C] (默认 -e)
    mass : float
        质量 [kg] (默认 m_e)
    """

    def __init__(self, position, velocity, charge=None, mass=None):
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.charge = -pc.Q_E if charge is None else charge
        self.mass = pc.M_ELECTRON if mass is None else mass

        # 计算初始能量
        v_mag = np.linalg.norm(self.velocity)
        gamma = 1.0 / np.sqrt(1.0 - (v_mag / pc.C_LIGHT)**2) if v_mag < pc.C_LIGHT else 1.0
        self.gamma = gamma
        self.E_kin_MeV = (gamma - 1.0) * pc.E_REST_MeV

    @classmethod
    def from_energy_pitch_angle(cls, L, E_MeV, alpha_eq, phi=0.0, theta_eq=np.pi/2):
        """
        从能量和投掷角创建粒子.

        参数
        ----
        L : float
            McIlwain L 参数
        E_MeV : float
            动能 [MeV]
        alpha_eq : float
            赤道投掷角 [rad]
        phi : float
            方位角 [rad]
        theta_eq : float
            磁余纬 (默认 pi/2, 赤道面)

        返回
        -------
        particle : TestParticle
        """
        # 位置 (赤道面)
        r = L * pc.R_EARTH
        x = r * np.sin(theta_eq) * np.cos(phi)
        y = r * np.sin(theta_eq) * np.sin(phi)
        z = r * np.cos(theta_eq)
        position = np.array([x, y, z])

        # 速度
        gamma = pc.kinetic_to_lorentz(E_MeV)
        v = pc.C_LIGHT * np.sqrt(1.0 - 1.0/gamma**2)

        # 偶极场方向 (赤道面: 主要沿 -z 方向)
        # 垂直分量 (垂直于 B)
        v_perp = v * np.sin(alpha_eq)
        v_parallel = v * np.cos(alpha_eq)

        # 在赤道面, B 沿 z 方向, 垂直速度在 x-y 平面
        v_x = v_perp * np.cos(phi)
        v_y = v_perp * np.sin(phi)
        v_z = v_parallel

        velocity = np.array([v_x, v_y, v_z])

        return cls(position, velocity)


class GuidingCenterIntegrator:
    """
    引导中心轨道积分器.

    参数
    ----
    field : DipoleField
        磁场对象
    dt : float
        时间步长 [s]
    """

    def __init__(self, field=None, dt=1.0):
        self.field = field if field is not None else DipoleField()
        self.dt = dt

    def compute_adiabatic_invariants(self, particle):
        """
        计算绝热不变量.

        参数
        ----
        particle : TestParticle
            测试粒子

        返回
        -------
        mu : float
            第一绝热不变量 [J/T]
        J : float
            第二绝热不变量 [kg*m]
        """
        r = np.linalg.norm(particle.position)
        if r < pc.EPSILON_NUM:
            return 0.0, 0.0

        theta = np.arccos(np.clip(particle.position[2] / r, -1, 1))
        B_mag = self.field.field_magnitude(r, theta)

        v_mag = np.linalg.norm(particle.velocity)
        gamma = particle.gamma

        # 动量
        p = gamma * particle.mass * v_mag

        # 投掷角 (简化: 假设赤道面)
        # 更精确需要计算 b . v
        B_x, B_y, B_z = self.field.field_spherical(r, theta)
        B_vec = np.array([
            B_mag * np.sin(theta) * np.cos(np.arctan2(particle.position[1], particle.position[0])),
            B_mag * np.sin(theta) * np.sin(np.arctan2(particle.position[1], particle.position[0])),
            B_mag * np.cos(theta)
        ])
        B_hat = B_vec / max(np.linalg.norm(B_vec), pc.EPSILON_NUM)

        v_parallel = np.dot(particle.velocity, B_hat)
        v_perp = np.sqrt(max(v_mag**2 - v_parallel**2, 0.0))

        p_perp = gamma * particle.mass * v_perp

        # 第一绝热不变量
        mu = p_perp**2 / (2.0 * particle.mass * max(B_mag, pc.EPSILON_NUM))

        # 第二绝热不变量 (简化: 弹跳长度估算)
        # J ~ p_parallel * L_bounce
        L_bounce = 2.0 * r * np.cos(theta) if theta < np.pi/2 else 0.0
        p_parallel = gamma * particle.mass * v_parallel
        J = p_parallel * L_bounce

        return mu, J

    def lorentz_force(self, position, velocity):
        """
        计算洛伦兹力.

        F = q * (v x B)

        参数
        ----
        position : ndarray
            位置 [m]
        velocity : ndarray
            速度 [m/s]

        返回
        -------
        force : ndarray
            力 [N]
        """
        x, y, z = position
        B_x, B_y, B_z = self.field.field_cartesian(x, y, z)
        B = np.array([B_x, B_y, B_z])

        v_cross_B = np.cross(velocity, B)
        return -pc.Q_E * v_cross_B  # 电子电荷为 -e

    def step_boris(self, particle):
        """
        Boris 推进器 (一步).

        Boris 算法保持相空间体积, 适合长时间积分.

        算法:
          1. v^- = v^n (无电场)
          2. t = q*B*dt / (2*m*gamma)
          3. s = 2*t / (1 + |t|^2)
          4. v' = v^- + v^- × t
          5. v^+ = v^- + v' × s
          6. x^{n+1} = x^n + v^+ * dt

        参数
        ----
        particle : TestParticle
            测试粒子

        返回
        -------
        new_position, new_velocity : ndarray
        """
        x = particle.position.copy()
        v = particle.velocity.copy()
        q = particle.charge
        m = particle.mass
        gamma = particle.gamma
        dt = self.dt

        # 半步位置更新 (用当前速度)
        x_half = x + v * dt * 0.5

        # 在半步位置计算磁场
        B_x, B_y, B_z = self.field.field_cartesian(x_half[0], x_half[1], x_half[2])
        B = np.array([B_x, B_y, B_z])

        # Boris 旋转
        # t = q*B*dt / (2*m*gamma)
        t_vec = q * B * dt / (2.0 * m * gamma)
        t_mag_sq = np.dot(t_vec, t_vec)

        # s = 2*t / (1 + |t|^2)
        s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)

        # v' = v + v × t
        v_prime = v + np.cross(v, t_vec)

        # v^+ = v + v' × s
        v_plus = v + np.cross(v_prime, s_vec)

        # 完整位置更新
        x_new = x + v_plus * dt

        return x_new, v_plus

    def integrate_orbit(self, particle, n_steps, record_interval=1):
        """
        积分轨道.

        参数
        ----
        particle : TestParticle
            初始粒子
        n_steps : int
            积分步数
        record_interval : int
            记录间隔

        返回
        -------
        orbit : dict
            轨道数据
        """
        n_record = n_steps // record_interval + 1
        positions = np.zeros((n_record, 3))
        velocities = np.zeros((n_record, 3))
        gammas = np.zeros(n_record)
        mus = np.zeros(n_record)

        current_p = TestParticle(particle.position.copy(),
                                  particle.velocity.copy(),
                                  particle.charge, particle.mass)

        idx = 0
        positions[0] = current_p.position
        velocities[0] = current_p.velocity
        gammas[0] = current_p.gamma
        mus[0], _ = self.compute_adiabatic_invariants(current_p)

        for step in range(n_steps):
            x_new, v_new = self.step_boris(current_p)
            current_p.position = x_new
            current_p.velocity = v_new

            # 更新 gamma
            v_mag = np.linalg.norm(v_new)
            if v_mag < pc.C_LIGHT:
                current_p.gamma = 1.0 / np.sqrt(1.0 - (v_mag/pc.C_LIGHT)**2)
            current_p.E_kin_MeV = (current_p.gamma - 1.0) * pc.E_REST_MeV

            if (step + 1) % record_interval == 0:
                idx += 1
                if idx < n_record:
                    positions[idx] = current_p.position
                    velocities[idx] = current_p.velocity
                    gammas[idx] = current_p.gamma
                    mus[idx], _ = self.compute_adiabatic_invariants(current_p)

        # 截断
        positions = positions[:idx+1]
        velocities = velocities[:idx+1]
        gammas = gammas[:idx+1]
        mus = mus[:idx+1]

        return {
            'positions': positions,
            'velocities': velocities,
            'gammas': gammas,
            'mu': mus,
            'time': np.arange(len(positions)) * self.dt * record_interval,
        }


def self_test():
    """自检验证."""
    print("=" * 60)
    print("测试粒子轨道模块自检验证")
    print("=" * 60)

    # 创建粒子
    particle = TestParticle.from_energy_pitch_angle(
        L=4.0, E_MeV=1.0, alpha_eq=np.pi/4
    )
    print(f"  初始位置: r = {np.linalg.norm(particle.position)/pc.R_EARTH:.2f} R_E")
    print(f"  初始能量: {particle.E_kin_MeV:.3f} MeV")
    print(f"  初始 gamma: {particle.gamma:.4f}")

    # 积分轨道
    integrator = GuidingCenterIntegrator(dt=0.1)
    orbit = integrator.integrate_orbit(particle, n_steps=100, record_interval=10)

    print(f"  轨道点数: {len(orbit['positions'])}")
    print(f"  最终位置: r = {np.linalg.norm(orbit['positions'][-1])/pc.R_EARTH:.2f} R_E")
    print(f"  mu 变化: {orbit['mu'][0]:.4e} -> {orbit['mu'][-1]:.4e}")

    return True


if __name__ == "__main__":
    self_test()
