r"""
particle_acceleration.py
=======================
高能粒子加速的蒙特卡洛模拟与非线性轨道追踪模块。
在太阳耀斑磁重联中，重联电场 E_z 可将带电粒子加速到
相对论能量（MeV-GeV 级别），产生硬 X 射线和伽马射线辐射。

核心物理模型
------------
1. 测试粒子在电磁场中的运动（Lorentz 方程）:
    dp/dt = q (E + v x B)
    dr/dt = v

    其中相对论动量 p = gamma_r * m * v,
    gamma_r = 1 / sqrt(1 - v^2/c^2) 为洛伦兹因子。

2. 扩散近似下的 Fokker-Planck 方程:
    partial f/partial t = nabla . (D . nabla f) + Q(p)

    其中 D 为扩散张量，Q(p) 为加速源项。

3. 蒙特卡洛方法:
    在速度空间（或动量空间）中随机采样粒子，
    通过随机游走模拟多次散射过程。

圆环上的随机采样（对应 circle_monte_carlo）:
    在垂直于 B 的速度平面中，粒子回旋运动构成圆环，
    方位角 theta 均匀分布于 [0, 2*pi)。

非线性摆类比（对应 pendulum_comparison_ode）:
    在磁场零点（X-point）附近，磁场梯度极大，
    粒子运动方程退化为类似非线性摆的形式:
        d^2 theta/dt^2 + (q B'/m) sin(theta) = q E/m

    这与非线性摆方程 d^2 theta/dt^2 + (g/l) sin(theta) = 0 同构。

融入原项目:
- 181_circle_monte_carlo: 圆周上的蒙特卡洛采样
- 857_pendulum_comparison_ode: 非线性 ODE 轨道积分
"""

import numpy as np
from typing import Tuple, List, Optional

# 物理常数
Q_E = 1.602176634e-19       # 元电荷 [C]
M_E = 9.10938356e-31        # 电子质量 [kg]
M_P = 1.6726219e-27         # 质子质量 [kg]
C_LIGHT = 2.99792458e8      # 光速 [m/s]


class MonteCarloParticleAccelerator:
    """
    蒙特卡洛模拟被重联电场加速的高能粒子。
    """

    def __init__(self,
                 n_particles: int = 10000,
                 particle_charge: float = Q_E,
                 particle_mass: float = M_P,
                 B_strength: float = 1.0e-2,    # 背景磁场 [T]
                 E_reconnection: float = 1.0e3, # 重联电场 [V/m]
                 lambda_acc: float = 1.0e4):    # 加速区长度 [m]
        if n_particles <= 0:
            raise ValueError("n_particles 必须为正")
        if particle_mass <= 0:
            raise ValueError("particle_mass 必须为正")
        self.n = n_particles
        self.q = particle_charge
        self.m = particle_mass
        self.B = B_strength
        self.E = E_reconnection
        self.L = lambda_acc

    def sample_initial_velocities(self,
                                   T_thermal: float = 1.0e6,
                                   seed: Optional[int] = None) -> np.ndarray:
        """
        从麦克斯韦-玻尔兹曼分布采样初始速度（非相对论近似）。
        圆环采样：速度空间中的方位角 theta 均匀分布，
        对应 circle_monte_carlo 的圆周采样思想。
        """
        if seed is not None:
            np.random.seed(seed)

        # 热速度 v_th = sqrt(k_B T / m)
        k_B = 1.380649e-23
        v_th = np.sqrt(k_B * T_thermal / self.m)

        # 平行于 B 的速度分量（高斯分布）
        v_parallel = np.random.normal(0.0, v_th, size=self.n)
        # 垂直于 B 的速度大小（瑞利分布）
        v_perp_mag = np.random.rayleigh(v_th / np.sqrt(2.0), size=self.n)
        # 垂直平面内的方位角（均匀分布，即圆周采样）
        theta = 2.0 * np.pi * np.random.rand(self.n)
        v_perp_x = v_perp_mag * np.cos(theta)
        v_perp_y = v_perp_mag * np.sin(theta)

        # 速度数组: [v_parallel, v_perp_x, v_perp_y]
        v = np.column_stack([v_parallel, v_perp_x, v_perp_y])
        return v

    def accelerate_stochastic(self,
                               v0: np.ndarray,
                               dt: float = 1.0e-6,
                               n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用随机游走模型模拟粒子在重联电场中的加速。
        每次步进加入随机散射（模拟湍动加速）。
        """
        if dt <= 0:
            raise ValueError("dt 必须为正")
        v = np.copy(v0)
        n = len(v)
        energies = np.zeros((n_steps + 1, n))

        # 计算初始能量
        v_sq = np.sum(v ** 2, axis=1)
        gamma = 1.0 / np.sqrt(1.0 - np.minimum(v_sq / C_LIGHT ** 2, 0.9999))
        energies[0] = (gamma - 1.0) * self.m * C_LIGHT ** 2 / Q_E  # eV

        for step in range(n_steps):
            # 确定性加速: dv_parallel/dt = qE/m
            v[:, 0] += dt * self.q * self.E / self.m

            # 随机散射（模拟湍动）
            scatter_angle = np.random.normal(0.0, 0.05, size=n)
            v_perp_mag = np.sqrt(v[:, 1] ** 2 + v[:, 2] ** 2)
            theta_old = np.arctan2(v[:, 2], v[:, 1])
            theta_new = theta_old + scatter_angle
            v[:, 1] = v_perp_mag * np.cos(theta_new)
            v[:, 2] = v_perp_mag * np.sin(theta_new)

            # 速度限制（光速上限）
            v_mag = np.sqrt(np.sum(v ** 2, axis=1))
            mask = v_mag > 0.99 * C_LIGHT
            if np.any(mask):
                v[mask] = 0.99 * C_LIGHT * v[mask] / v_mag[mask, None]

            # 记录能量
            v_sq = np.sum(v ** 2, axis=1)
            gamma = 1.0 / np.sqrt(1.0 - np.minimum(v_sq / C_LIGHT ** 2, 0.9999))
            energies[step + 1] = (gamma - 1.0) * self.m * C_LIGHT ** 2 / Q_E

        return v, energies

    def energy_spectrum(self, energies_ev: np.ndarray, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算能谱 dN/dE。
        """
        e_flat = energies_ev.ravel()
        e_min = max(np.min(e_flat), 1.0)
        e_max = np.max(e_flat)
        if e_max <= e_min:
            e_max = e_min * 10.0
        bins = np.logspace(np.log10(e_min), np.log10(e_max), n_bins)
        hist, edges = np.histogram(e_flat, bins=bins)
        bin_centers = np.sqrt(edges[:-1] * edges[1:])
        dE = edges[1:] - edges[:-1]
        spectrum = hist / (dE * self.n)
        return bin_centers, spectrum


class NonlinearOrbitTracker:
    """
    非线性轨道追踪器，类比非线性摆的 ODE 结构。
    在 X-point 附近，粒子运动方程为:
        d^2 x/dt^2 = (q/m) E_x + (q/m) v_y B_z(x)
        d^2 y/dt^2 = (q/m) E_y - (q/m) v_x B_z(x)

    其中 B_z(x) ~ B' * x（磁场零点附近的线性近似）。
    这导致类似非线性摆的恢复力形式。
    """

    def __init__(self,
                 q: float = Q_E,
                 m: float = M_P,
                 B_prime: float = 1.0e-7,   # 磁场梯度 [T/m]
                 E_field: float = 1.0e3,    # 电场 [V/m]
                 length: float = 1.0):      # 特征长度 [m]
        self.q = q
        self.m = m
        self.Bp = B_prime
        self.E = E_field
        self.L = length
        # 等效"摆频率" omega_0^2 = q B' / m
        self.omega0_sq = q * B_prime / m
        if self.omega0_sq < 0:
            raise ValueError("omega0_sq 为负，检查参数符号")
        self.omega0 = np.sqrt(abs(self.omega0_sq))

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        状态 y = [x, y_pos, vx, vy]。
        dx/dt = vx
        dy/dt = vy
        dvx/dt = (q/m) E + (q/m) vy * B'(y_pos) * x
        dvy/dt = -(q/m) vx * B'(y_pos) * x

        这里简化为 B 仅依赖于 x 的梯度。
        """
        x, yp, vx, vy = y
        # 边界处理：x 限制在 [-L, L]
        x_clip = np.clip(x, -self.L, self.L)
        ax = self.q / self.m * self.E + self.q / self.m * vy * self.Bp * x_clip
        ay = -self.q / self.m * vx * self.Bp * x_clip
        return np.array([vx, vy, ax, ay])

    def integrate_rk4(self,
                      y0: np.ndarray,
                      t_span: Tuple[float, float],
                      n_steps: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 积分。
        """
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 4))
        states[0] = y0
        y = np.copy(y0)

        for i in range(n_steps):
            k1 = self.deriv(t[i], y)
            k2 = self.deriv(t[i] + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.deriv(t[i] + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.deriv(t[i] + dt, y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            states[i + 1] = y
        return t, states

    def compute_lyapunov_exponent(self,
                                   y0: np.ndarray,
                                   t_span: Tuple[float, float],
                                   n_steps: int = 5000,
                                   delta0: float = 1e-8) -> float:
        """
        计算最大 Lyapunov 指数，判断轨道是否混沌。
        lambda = lim_{t->inf} (1/t) ln(|delta(t)| / |delta(0)|)
        """
        t, states = self.integrate_rk4(y0, t_span, n_steps)
        # 扰动初始条件
        y0_pert = y0.copy()
        y0_pert[0] += delta0
        t2, states_pert = self.integrate_rk4(y0_pert, t_span, n_steps)

        delta = states_pert - states
        norms = np.sqrt(np.sum(delta ** 2, axis=1))
        # 取后半段时间的指数增长斜率
        mid = n_steps // 2
        if norms[mid] > 0 and norms[-1] > 0:
            lambda_max = np.log(norms[-1] / norms[mid]) / (t[-1] - t[mid])
        else:
            lambda_max = -np.inf
        return lambda_max


def demo_particles():
    """
    演示粒子加速与轨道追踪。
    """
    print("\n[Particles] 演示: 蒙特卡洛粒子加速")
    acc = MonteCarloParticleAccelerator(n_particles=5000)
    v0 = acc.sample_initial_velocities(T_thermal=1.0e6, seed=42)
    v_final, energies = acc.accelerate_stochastic(v0, dt=1.0e-7, n_steps=500)
    E_centers, spectrum = acc.energy_spectrum(energies[-100:])
    print(f"  初始平均能量: {np.mean(energies[0]):.3e} eV")
    print(f"  最终平均能量: {np.mean(energies[-1]):.3e} eV")
    print(f"  最大能量: {np.max(energies[-1]):.3e} eV")

    print("\n[Particles] 演示: 非线性轨道追踪")
    tracker = NonlinearOrbitTracker()
    y0 = np.array([0.1, 0.0, 1.0e5, 0.0])
    t, states = tracker.integrate_rk4(y0, (0.0, 1.0e-3), n_steps=2000)
    print(f"  初始位置: ({y0[0]:.3e}, {y0[1]:.3e})")
    print(f"  最终位置: ({states[-1, 0]:.3e}, {states[-1, 1]:.3e})")
    lyap = tracker.compute_lyapunov_exponent(y0, (0.0, 5.0e-3), n_steps=5000)
    print(f"  最大 Lyapunov 指数: {lyap:.3e} (正值表示混沌)")


if __name__ == "__main__":
    demo_particles()
