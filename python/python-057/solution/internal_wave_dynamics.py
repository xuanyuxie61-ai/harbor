"""
internal_wave_dynamics.py
海洋内波非线性动力学模型

融合项目:
- 322_duffing_ode: 非线性Duffing振子 → 非线性内波振荡方程
- 702_logistic_ode: Logistic增长模型 → 内波能量衰减与饱和

核心物理:
内波在密度分层海洋中的非线性传播可用修正的Duffing方程描述:
    d²ξ/dt² + δ · dξ/dt + α · ξ + β · ξ³ = γ · cos(ωt) + F_buoyancy

其中 ξ 为等密度面位移，δ 为阻尼系数，α 为线性恢复力系数，
β 为非线性系数，γ 为强迫振幅，F_buoyancy 为浮力修正项。

能量衰减采用Logistic-like模型:
    dE/dt = r · E · (1 - E/E_max) - ε_diss
"""

import numpy as np
from scipy.integrate import solve_ivp


class NonlinearInternalWave:
    """
    非线性内波动力学模型
    
    将Duffing型非线性振荡器应用于海洋内波传播，
    考虑密度分层、科里奥利效应和湍流阻尼。
    """
    
    def __init__(self, alpha=1.0, beta=5.0, gamma=8.0, delta=0.02,
                 omega=0.5, N=0.01, f=1.0e-4, depth=200.0):
        """
        初始化内波参数
        
        参数:
            alpha: 线性恢复力系数 [1/s²]
            beta: 非线性系数 [1/(m²·s²)]
            gamma: 强迫振幅 [m/s²]
            delta: 阻尼系数 [1/s]
            omega: 强迫频率 [rad/s]
            N: 浮力频率 [rad/s]
            f: 科里奥利参数 [rad/s]
            depth: 水深 [m]
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.N = N
        self.f = f
        self.depth = depth
        
        # 内波修正的线性频率 (考虑浮力频率)
        self.alpha_eff = alpha + N**2
        
        # 状态变量: [位移, 速度, 能量]
        self.state = np.array([1.0, 0.0, 0.5])
    
    def rhs(self, t, y):
        """
        内波非线性ODE右端项
        
        状态变量: y = [ξ, ξ', E]
        
        dξ/dt = ξ'
        dξ'/dt = -δ·ξ' - α_eff·ξ - β·ξ³ + γ·cos(ωt) + F_coriolis
        dE/dt = r·E·(1 - E/E_max) - ε_diss
        
        科里奥利修正: F_coriolis = -f · ξ'
        耗散项: ε_diss = δ · (ξ')²
        """
        xi, xi_dot, E = y
        
        # 位移边界处理
        xi = np.clip(xi, -50.0, 50.0)
        xi_dot = np.clip(xi_dot, -10.0, 10.0)
        
        # 非线性恢复力
        nonlinear_force = -self.beta * xi**3
        
        # 强迫项
        forcing = self.gamma * np.cos(self.omega * t)
        
        # 科里奥利效应 (f-平面近似)
        coriolis = -self.f * xi_dot
        
        # 阻尼
        damping = -self.delta * xi_dot
        
        # 等密度面位移方程
        dxi_dt = xi_dot
        dxi_dot_dt = damping - self.alpha_eff * xi + nonlinear_force + forcing + coriolis
        
        # 能量方程 (Logistic-like衰减)
        r = 2.0 * self.delta  # 能量增长率
        E_max = 0.5 * self.gamma**2 / self.alpha_eff  # 最大能量
        epsilon_diss = self.delta * xi_dot**2  # 机械能耗散
        
        dE_dt = r * E * (1.0 - E / E_max) - epsilon_diss
        
        # 能量边界处理
        E = np.clip(E, 0.0, E_max * 2.0)
        
        return np.array([dxi_dt, dxi_dot_dt, dE_dt])
    
    def solve(self, t_span=(0, 100), dt=0.1, method='RK45'):
        """
        数值求解内波非线性ODE
        
        参数:
            t_span: 时间区间 [s]
            dt: 输出时间步长 [s]
            method: 积分方法
        
        返回:
            t: 时间数组
            xi: 位移数组 [m]
            xi_dot: 速度数组 [m/s]
            E: 能量数组 [J/kg]
        """
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        
        sol = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=self.state,
            t_eval=t_eval,
            method=method,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        t = sol.t
        xi = sol.y[0, :]
        xi_dot = sol.y[1, :]
        E = sol.y[2, :]
        
        # 后处理边界
        xi = np.clip(xi, -100.0, 100.0)
        xi_dot = np.clip(xi_dot, -20.0, 20.0)
        E = np.clip(E, 0.0, None)
        
        return t, xi, xi_dot, E
    
    def compute_wave_action(self, t, xi, xi_dot):
        """
        计算波作用量 (Wave Action)
        
        A = E / ω
        
        参数:
            t: 时间数组
            xi: 位移数组
            xi_dot: 速度数组
        
        返回:
            action: 波作用量
        """
        E_kin = 0.5 * xi_dot**2
        E_pot = 0.5 * self.alpha_eff * xi**2 + 0.25 * self.beta * xi**4
        E_total = E_kin + E_pot
        
        action = E_total / (self.omega + 1.0e-12)
        action = np.clip(action, 0.0, 1.0e6)
        return action


def kdv_internal_wave(xi0, c, alpha_kdv, beta_kdv, t_span=(0, 50), nx=256):
    """
    Korteweg-de Vries (KdV) 内波方程数值解
    
    KdV方程描述弱非线性内波的传播:
        ∂η/∂t + c · ∂η/∂x + α · η · ∂η/∂x + β · ∂³η/∂x³ = 0
    
    参数:
        xi0: 初始波剖面振幅 [m]
        c: 线性波速 [m/s]
        alpha_kdv: 非线性系数
        beta_kdv: 色散系数 [m³/s]
        t_span: 时间区间 [s]
        nx: 空间网格数
    
    返回:
        x: 空间坐标 [m]
        t: 时间数组 [s]
        eta: 波高数组 [m]
    """
    L = 2000.0  # 计算域长度 [m]
    dx = L / nx
    dt = 0.5 * dx / (abs(c) + 1.0)
    nt = int((t_span[1] - t_span[0]) / dt) + 1
    
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(t_span[0], t_span[1], nt)
    
    # 初始条件: 孤立波
    eta = np.zeros((nt, nx))
    eta[0, :] = xi0 / np.cosh((x - L/2) / 100.0)**2
    
    # 伪谱法求解 (FFT-based)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    ik = 1j * k
    ik3 = (1j * k)**3
    
    for n in range(nt - 1):
        eta_hat = np.fft.fft(eta[n, :])
        
        # 线性步 (FFT)
        eta_hat = eta_hat * np.exp(-1j * k * c * dt)
        
        # 非线性步 (伪谱)
        eta_nl = np.real(np.fft.ifft(eta_hat))
        
        # 非线性对流项
        d_eta_dx = np.real(np.fft.ifft(ik * np.fft.fft(eta_nl)))
        eta_nl = eta_nl - alpha_kdv * eta_nl * d_eta_dx * dt
        
        # 色散项
        eta_hat = np.fft.fft(eta_nl)
        eta_hat = eta_hat * np.exp(-beta_kdv * ik3 * dt)
        
        eta[n+1, :] = np.real(np.fft.ifft(eta_hat))
        
        # 边界处理
        eta[n+1, :] = np.clip(eta[n+1, :], -50.0, 50.0)
    
    return x, t, eta
