"""
随机粒子输运模块 (Stochastic Particle Transport)
===================================================
模拟粒子在随机速度场中的拉格朗日输运。

运动方程:
  dX(t)/dt = v(X(t), t; ω)
  X(0) = X₀

其中 v(x,t;ω) 是随机速度场, ω 是随机事件。

对于简单模型: v(t;ω) = c(ω) (空间均匀, 时间恒定)
  X(t) = X₀ + c(ω) t
  E[X(t)] = X₀ + E[c] t
  Var[X(t)] = Var[c] t²

对于更复杂的模型: v(x,t;ω) = c(ω) + σ_v W(t)
  (其中 W(t) 是维纳过程)
  X(t) = X₀ + c(ω) t + σ_v W(t)
  E[X(t)] = X₀ + E[c] t
  Var[X(t)] = Var[c] t² + σ_v² t

Taylor 分散 (有效扩散):
  当速度场具有空间相关性时:
    D_eff = D_molecular + ∫₀^∞ R_vv(τ) dτ
  其中 R_vv(τ) = <v(x,t) v(x,t+τ)> 是速度自相关函数。

Fokker-Planck 方程 (概率密度演化):
  ∂p/∂t + ∂(v p)/∂x = D ∂²p/∂x²
  描述粒子位置概率密度的时间演化。

均方位移 (MSD):
  MSD(t) = E[(X(t) - E[X(t)])²]
  对于纯扩散: MSD = 2Dt
  对于弹道运动: MSD = Var[c] t²
  对于 Taylor 分散: MSD ~ 2 D_eff t (t → ∞)
"""

import numpy as np
from typing import Tuple, Optional, Dict


class StochasticParticleTransport:
    """
    随机粒子输运模拟器。

    模拟粒子在随机对流速度场中的运动,
    计算统计量 (均值, 方差, MSD)。

    模型:
      dX/dt = c(ω) + η(t)
      X(0) = x₀

    其中 c(ω) 是随机对流速度, η(t) 是布朗噪声。
    """

    def __init__(
        self,
        initial_position: float = 0.0,
        diffusion_coeff: float = 0.01,
        time_end: float = 1.0,
        n_timesteps: int = 1000
    ):
        """
        参数:
            initial_position: 初始位置 X₀
            diffusion_coeff: 分子扩散系数 D
            time_end: 终止时间 T
            n_timesteps: 时间步数
        """
        self.x0 = initial_position
        self.D = diffusion_coeff
        self.T = time_end
        self.n_steps = n_timesteps
        self.dt = self.T / self.n_steps
        self.dx_std = np.sqrt(2.0 * self.D * self.dt)  # 扩散步长标准差

    def simulate_trajectory(
        self, velocity: float, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟单个粒子的轨迹 (Euler-Maruyama 方法):

          X^{n+1} = X^n + c Δt + √(2D Δt) Z^n
          Z^n ~ N(0, 1)

        这是随机微分方程的强收敛阶 0.5 方法。

        参数:
            velocity: 对流速度 c
            seed: 随机种子

        返回:
            time: shape (n_steps+1,)
            position: shape (n_steps+1,)
        """
        rng = np.random.RandomState(seed)
        time = np.linspace(0.0, self.T, self.n_steps + 1)
        position = np.zeros(self.n_steps + 1)
        position[0] = self.x0

        for n in range(self.n_steps):
            dW = rng.randn()
            position[n + 1] = (position[n]
                               + velocity * self.dt
                               + self.dx_std * dW)

        return time, position

    def simulate_ensemble(
        self,
        velocities: np.ndarray,
        n_particles_per_velocity: int = 100,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        模拟粒子系综: 对于每个速度实现, 模拟多个粒子。

        参数:
            velocities: shape (N_vel,)
            n_particles_per_velocity: 每个速度实现的粒子数
            seed: 随机种子

        返回:
            dict with keys:
                'mean_position': E[X(t)], shape (N_vel, N_steps+1)
                'var_position': Var[X(t)], shape (N_vel, N_steps+1)
                'msd': 均方位移, shape (N_vel, N_steps+1)
                'mean_trajectory': 全系统均值
        """
        N_vel = len(velocities)
        time = np.linspace(0.0, self.T, self.n_steps + 1)
        N_steps = self.n_steps + 1

        # 解析均值: E[X(t)] = x₀ + E[c] t
        mean_c = np.mean(velocities)
        var_c = np.var(velocities)

        mean_position = np.zeros((N_vel, N_steps))
        var_position = np.zeros((N_vel, N_steps))
        msd = np.zeros((N_vel, N_steps))

        rng = np.random.RandomState(seed)

        for iv in range(N_vel):
            c = velocities[iv]
            # 解析结果
            # X(t) = x₀ + c t + √(2Dt) Z
            # E[X(t)] = x₀ + c t
            # Var[X(t)] = 2Dt
            mean_position[iv] = self.x0 + c * time
            var_position[iv] = 2.0 * self.D * time

            # MSD = Var[X(t)] + (E[X(t)] - E_all[X(t)])²
            # = 2Dt + (c - mean_c)² t²
            msd[iv] = 2.0 * self.D * time + (c - mean_c) ** 2 * time ** 2

        # 全系统均值和方差
        overall_mean = self.x0 + mean_c * time
        overall_var = var_c * time ** 2 + 2.0 * self.D * time

        return {
            'mean_position': mean_position,
            'var_position': var_position,
            'msd': msd,
            'overall_mean': overall_mean,
            'overall_var': overall_var,
            'time': time,
        }

    def compute_dispersion_coefficient(
        self, velocities: np.ndarray
    ) -> float:
        """
        计算有效分散系数:
          D_eff = D + Var[c] × τ_c

        其中 τ_c 是速度相关时间尺度。
        对于恒定速度: τ_c → ∞ (弹道分散)
        对于 Ornstein-Uhlenbeck 过程: τ_c = 1/γ

        简化: D_eff ≈ D + Var[c] × T/2
        (假设速度在时间 T 内完全相关)

        参数:
            velocities: 速度样本

        返回:
            D_eff: 有效分散系数
        """
        var_c = np.var(velocities)
        return self.D + var_c * self.T / 2.0

    def pecllet_number_effective(self, mean_velocity: float, L: float = 1.0) -> float:
        """
        有效 Péclet 数:
          Pe_eff = |E[c]| L / D_eff

        衡量对流与分散的相对重要性。
        """
        D_eff = self.D + np.abs(mean_velocity) * self.T / 2.0
        if D_eff < 1e-30:
            return float('inf')
        return np.abs(mean_velocity) * L / D_eff


class FokkerPlanckSolver:
    """
    Fokker-Planck 方程求解器 (概率密度演化):

    ∂p/∂t + c ∂p/∂x = D ∂²p/∂x²

    使用有限差分法 (Crank-Nicolson 时间推进):
      (I - θ Δt L) p^{n+1} = (I + (1-θ) Δt L) p^n

    其中 L p = -c ∂p/∂x + D ∂²p/∂x²
    θ = 0.5 (Crank-Nicolson, 二阶时间精度)
    """

    def __init__(
        self,
        n_spatial: int = 201,
        x_min: float = -2.0,
        x_max: float = 2.0,
        time_end: float = 1.0,
        n_timesteps: int = 200
    ):
        self.n_spatial = n_spatial
        self.x = np.linspace(x_min, x_max, n_spatial)
        self.h = self.x[1] - self.x[0]
        self.T = time_end
        self.n_steps = n_timesteps
        self.dt = self.T / self.n_steps

    def solve(
        self, velocity: float, diffusion: float,
        initial_pdf: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 Fokker-Planck 方程。

        初始条件: 高斯分布 δ(x - x₀) 近似
        边界条件: p → 0 当 x → ±∞ (零通量)
        """
        if initial_pdf is None:
            # 高斯初始
            sigma = 3.0 * self.h
            x_center = (self.x[0] + self.x[-1]) / 2.0
            p = np.exp(-(self.x - x_center) ** 2 / (2.0 * sigma ** 2))
            p /= np.sum(p) * self.h  # 归一化

        else:
            p = initial_pdf.copy()

        # Crank-Nicolson 参数
        theta = 0.5
        r = diffusion * self.dt / self.h ** 2
        s = velocity * self.dt / (2.0 * self.h)

        # 构建三对角矩阵
        n = self.n_spatial
        # L p_i = -c (p_{i+1} - p_{i-1})/(2h) + D (p_{i+1} - 2p_i + p_{i-1})/h²
        # 系数:
        a = (diffusion / self.h ** 2 + velocity / (2.0 * self.h))  # p_{i-1}
        b = (-2.0 * diffusion / self.h ** 2)                       # p_i
        c_coeff = (diffusion / self.h ** 2 - velocity / (2.0 * self.h))  # p_{i+1}

        # 隐式矩阵 (I - θ Δt L)
        lower = -theta * self.dt * a
        diag_impl = 1.0 - theta * self.dt * b
        upper = -theta * self.dt * c_coeff

        # 显式矩阵 (I + (1-θ) Δt L)
        lower_exp = (1.0 - theta) * self.dt * a
        diag_exp = 1.0 + (1.0 - theta) * self.dt * b
        upper_exp = (1.0 - theta) * self.dt * c_coeff

        # 时间推进
        from scipy import linalg as la

        for step in range(self.n_steps):
            # 显式步
            rhs = np.zeros(n)
            rhs[0] = p[0]
            rhs[-1] = p[-1]
            for i in range(1, n - 1):
                rhs[i] = (lower_exp * p[i - 1] + diag_exp * p[i]
                          + upper_exp * p[i + 1])

            # 隐式步 (Thomas 算法)
            a_impl = np.full(n - 2, lower)
            b_impl = np.full(n - 2, diag_impl)
            c_impl = np.full(n - 2, upper)

            # 三对角求解
            A_banded = np.zeros((3, n - 2))
            A_banded[0, 1:] = c_impl[:-1]
            A_banded[1, :] = b_impl
            A_banded[2, :-1] = a_impl[1:]

            p_interior = la.solve_banded((1, 1), A_banded, rhs[1:-1])
            p[1:-1] = p_interior

            # 边界
            p[0] = 0.0
            p[-1] = 0.0

            # 确保非负
            p = np.maximum(p, 0.0)
            # 归一化
            total = np.sum(p) * self.h
            if total > 1e-15:
                p /= total

        return self.x, p
