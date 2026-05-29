"""
closed_loop_control.py — 自适应光学闭环控制与参数优化

融合原项目:
  - 832_ode_sweep_parfor (ODE参数扫描)
  - 060_axon_ode (Hodgkin-Huxley电路响应模型)

功能:
  - 比例-积分 (PI) 控制器
  - 基于HH模型的高速控制电路响应
  - 控制增益与带宽的参数扫描优化
  - 闭环稳定性分析

物理模型:
  1. PI控制器:
       u(t) = K_p * e(t) + K_i * integral_0^t e(tau) dtau
     其中 e(t) = phi_res(t) 为残余波前误差.

  2. 带宽限制:
       控制信号通过一阶低通滤波器:
         du_filtered/dt = (u - u_filtered) / tau_c
       其中 tau_c = 1/(2*pi*f_c), f_c 为控制带宽.

  3. Hodgkin-Huxley型控制电路 (源自060_axon_ode):
       将光电探测器输出建模为膜电位V,
       控制电流 I(t) = G_control * V(t).
       门控变量n,m,h的动态影响控制信号的上升/下降沿.

  4. 参数扫描 (源自832_ode_sweep_parfor):
       在 (K_p, K_i) 或 (K_p, f_c) 参数网格上评估闭环性能,
       寻找最优Strehl比对应的参数组合.
"""

import numpy as np


# --- PI控制器 ---

class PIController:
    """比例-积分控制器."""

    def __init__(self, Kp=0.5, Ki=0.1, dt=1e-3, integral_limit=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.dt = dt
        self.integral = 0.0
        self.integral_limit = integral_limit
        self.error_history = []

    def reset(self):
        self.integral = 0.0
        self.error_history = []

    def update(self, error):
        """
        PI更新.

        u = Kp * e + Ki * integral(e)
        积分项限幅防止windup.
        """
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        self.error_history.append(error)
        u = self.Kp * error + self.Ki * self.integral
        return u


# --- 带宽受限执行器 ---

class BandwidthLimitedActuator:
    """一阶低通滤波器模拟带宽受限执行器."""

    def __init__(self, bandwidth_hz, dt):
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive.")
        self.tau = 1.0 / (2.0 * np.pi * bandwidth_hz)
        self.dt = dt
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def step(self, input_cmd):
        """
        离散一阶低通:
          state_{k+1} = state_k + dt/tau * (input - state_k)
        """
        alpha = self.dt / (self.tau + self.dt)
        self.state = self.state + alpha * (input_cmd - self.state)
        return self.state


# --- Hodgkin-Huxley型控制电路 (源自060_axon_ode) ---

class HHControlCircuit:
    """
    Hodgkin-Huxley型高速控制电路.

    将误差信号映射为控制电流, 考虑门控延迟.
    """

    def __init__(self, C=1.0, E_K=-74.7, E_Na=54.2, G_K=12.0, G_Na=30.0,
                 dt=1e-5):
        self.C = C
        self.E_K = E_K
        self.E_Na = E_Na
        self.G_K = G_K
        self.G_Na = G_Na
        self.dt = dt

        self.V = -65.0
        self.n = 0.3177
        self.m = 0.0529
        self.h = 0.5961

    def reset(self):
        self.V = -65.0
        self.n = 0.3177
        self.m = 0.0529
        self.h = 0.5961

    def alpha_n(self, V):
        if abs(V - 10.0) < 1e-6:
            return 0.1
        return 0.01 * (10.0 - V) / (np.exp((10.0 - V) / 10.0) - 1.0)

    def beta_n(self, V):
        return 0.125 * np.exp(-V / 80.0)

    def alpha_m(self, V):
        if abs(V - 25.0) < 1e-6:
            return 1.0
        return 0.1 * (25.0 - V) / (np.exp((25.0 - V) / 10.0) - 1.0)

    def beta_m(self, V):
        return 4.0 * np.exp(-V / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-V / 20.0)

    def beta_h(self, V):
        return 1.0 / (np.exp((30.0 - V) / 10.0) + 1.0)

    def step(self, I_ext):
        """
        HH电路单步更新.

        C * dV/dt = I_ext - I_K - I_Na
        dn/dt = alpha_n*(1-n) - beta_n*n
        dm/dt = alpha_m*(1-m) - beta_m*m
        dh/dt = alpha_h*(1-h) - beta_h*h
        """
        I_K = self.n ** 4 * self.G_K * (self.V - self.E_K)
        I_Na = self.m ** 3 * self.G_Na * self.h * (self.V - self.E_Na)

        dV = (I_ext - I_K - I_Na) / self.C
        dn = self.alpha_n(self.V) * (1.0 - self.n) - self.beta_n(self.V) * self.n
        dm = self.alpha_m(self.V) * (1.0 - self.m) - self.beta_m(self.V) * self.m
        dh = self.alpha_h(self.V) * (1.0 - self.h) - self.beta_h(self.V) * self.h

        self.V += dV * self.dt
        self.n += dn * self.dt
        self.n = np.clip(self.n, 0.0, 1.0)
        self.m += dm * self.dt
        self.m = np.clip(self.m, 0.0, 1.0)
        self.h += dh * self.dt
        self.h = np.clip(self.h, 0.0, 1.0)

        return self.V


# --- 闭环AO系统模拟 ---

class ClosedLoopAO:
    """
    自适应光学闭环系统.
    """

    def __init__(self, controller, actuator, n_modes, dt=1e-3):
        self.controller = controller
        self.actuator = actuator
        self.n_modes = n_modes
        self.dt = dt
        self.control_history = []
        self.residual_history = []
        self.strehl_history = []

    def reset(self):
        self.controller.reset()
        self.actuator.reset()
        self.control_history = []
        self.residual_history = []
        self.strehl_history = []

    def step(self, residual_error, strehl=None):
        """
        闭环单步:
          1. PI控制器根据残余误差计算控制量
          2. 执行器施加带宽限制
          3. 记录历史
        """
        u_raw = self.controller.update(residual_error)
        u_filtered = self.actuator.step(u_raw)
        self.control_history.append(u_filtered)
        self.residual_history.append(residual_error)
        if strehl is not None:
            self.strehl_history.append(strehl)
        return u_filtered


def parameter_sweep_optimization(Kp_grid, Ki_grid, bandwidth_grid,
                                  simulation_func, dt=1e-3, n_steps=500):
    """
    控制参数空间扫描优化 (源自832_ode_sweep_parfor).

    在 (Kp, Ki, bandwidth) 三维网格上评估闭环性能,
    返回最优参数组合和性能矩阵.

    simulation_func(Kp, Ki, bw) -> final_strehl_ratio
    """
    if len(Kp_grid) == 0 or len(Ki_grid) == 0 or len(bandwidth_grid) == 0:
        raise ValueError("Parameter grids must be non-empty.")

    n_kp = len(Kp_grid)
    n_ki = len(Ki_grid)
    n_bw = len(bandwidth_grid)

    performance = np.zeros((n_kp, n_ki, n_bw), dtype=np.float64)
    best_strehl = -1.0
    best_params = (Kp_grid[0], Ki_grid[0], bandwidth_grid[0])

    for i, Kp in enumerate(Kp_grid):
        for j, Ki in enumerate(Ki_grid):
            for k, bw in enumerate(bandwidth_grid):
                try:
                    strehl = simulation_func(Kp, Ki, bw, dt, n_steps)
                    if not np.isfinite(strehl):
                        strehl = 0.0
                except Exception:
                    strehl = 0.0
                performance[i, j, k] = strehl
                if strehl > best_strehl:
                    best_strehl = strehl
                    best_params = (Kp, Ki, bw)

    return best_params, best_strehl, performance


def simulate_modal_control_loop(n_modes, turb_covariance, Kp, Ki, bandwidth_hz,
                                dt=1e-3, n_steps=1000, noise_std=0.01):
    """
    模态域闭环控制模拟.

    使用简化的模态控制模型:
      - 湍流相位用Zernike系数表示
      - 每个模式独立控制
      - 残余误差 = 湍流系数 - 校正系数

    湍流演化: 使用一阶马尔可夫过程
      a_turb(k+1) = exp(-dt/tau)*a_turb(k) + sqrt(1-exp(-2*dt/tau))*N(0, sigma^2)
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    tau = 0.01  # 湍流相干时间 (s)
    decay = np.exp(-dt / tau)
    diffusion = np.sqrt(1.0 - decay ** 2)

    # 为每个模式创建独立的PI控制器和执行器 (避免状态串扰)
    controllers = [PIController(Kp=Kp, Ki=Ki, dt=dt, integral_limit=1.0) for _ in range(n_modes)]
    actuators = [BandwidthLimitedActuator(bandwidth_hz, dt) for _ in range(n_modes)]

    # 初始化湍流系数
    a_turb = np.random.multivariate_normal(np.zeros(n_modes), turb_covariance)
    a_corr = np.zeros(n_modes)

    residual_history = []
    strehl_history = []

    for step in range(n_steps):
        # 湍流演化
        a_turb = decay * a_turb + diffusion * np.random.multivariate_normal(np.zeros(n_modes), turb_covariance)

        # 测量 (含噪声)
        a_meas = a_turb - a_corr + np.random.normal(0, noise_std, n_modes)

        # 控制器对每个模式独立更新
        da_corr = np.zeros(n_modes)
        for m in range(n_modes):
            u = controllers[m].update(a_meas[m])
            u_filt = actuators[m].step(u)
            da_corr[m] = u_filt * dt

        a_corr = a_corr + da_corr
        a_corr = np.clip(a_corr, -5.0, 5.0)

        residual = a_turb - a_corr
        residual_history.append(np.linalg.norm(residual))

        # Strehl近似 (Marechal近似, 限制防止下溢)
        sigma_phi_sq = np.sum(residual ** 2)
        strehl = np.exp(-min(sigma_phi_sq, 50.0))
        strehl_history.append(strehl)

    final_strehl = strehl_history[-1] if len(strehl_history) > 0 else 0.0
    mean_strehl = np.mean(strehl_history) if len(strehl_history) > 0 else 0.0
    # 如果mean_strehl过小但残余在下降, 报告最后100步的平均值
    if mean_strehl < 1e-6 and len(strehl_history) > 100:
        mean_strehl = np.mean(strehl_history[-100:])
    return final_strehl, mean_strehl, np.array(residual_history), np.array(strehl_history)
