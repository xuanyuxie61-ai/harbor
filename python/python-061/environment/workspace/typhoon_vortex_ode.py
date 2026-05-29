"""
台风涡旋中心运动与强度演变ODE系统
====================================
基于种子项目:
  - 100_blood_pressure_ode: 周期性脉冲ODE参数管理思想
  - 1374_unstable_ode: 不稳定/刚性ODE分析
  - 1032_rk2_implicit: 隐式RK2时间积分

核心科学问题：
    台风中心（TC center）的运动遵循一个受环境引导流、beta漂移和涡旋
    自身动力学耦合的ODE系统。强度演变则受海表温度(SST)、风切变、
    和边界层抽吸反馈控制，呈现典型的不稳定增长-衰减特征。

数学模型：

=== 1. 台风中心运动ODE（引导流+Beta漂移）===

    dX/dt = U_env(X, Y, t) + U_beta(h, R_max) + U_wobble(t)
    dY/dt = V_env(X, Y, t) + V_beta(h, R_max) + V_wobble(t)

其中环境引导流由浅水方程背景场插值得到：
    U_env = α * u_background(X, Y)
    V_env = β * v_background(X, Y)

Beta漂移速度（Holland, 1983）：
    U_beta = - (β * R_max²) / (2 * f) * (1 - 2*ln(R_max/R_0))
    V_beta =   (β * R_max²) / (2 * f) * ...

其中 β = df/dy = (2Ω/R) * cos(θ) 为Rossby参数。

=== 2. 台风强度ODE（快速增强不稳定性模型）===

    dP_min/dt = -λ1 * (SST - T_threshold) * (P_env - P_min) / P_env
                + λ2 * |V_wind shear| * (P_env - P_min) / P_env
                + λ3 * ε(t) * (P_min - P_c) / P_c

    dR_max/dt = μ1 * (R_max_eq - R_max) - μ2 * |dP_min/dt| * R_max / P_min

其中：
    P_min: 台风中心最低气压 (hPa)
    P_env: 环境气压 (~1010 hPa)
    P_c:   理论最小气压 (~870 hPa)
    R_max: 最大风速半径 (km)
    SST:   海表温度 (K)
    ε(t):  随机环境噪声（基于 189_clock_solitaire 的随机性思想）

=== 3. 隐式RK2积分器（基于 1032_rk2_implicit）===

采用隐式中点规则，每步需解非线性方程：
    Y_{n+1/2} = Y_n + (Δt/2) * F(t_{n+1/2}, Y_{n+1/2})
    Y_{n+1}   = 2*Y_{n+1/2} - Y_n

使用Newton-Raphson迭代求解隐式方程。
"""

import numpy as np
from scipy.optimize import fsolve

# 物理常数
EARTH_RADIUS = 6.371e6      # m
OMEGA = 7.2921159e-5        # rad/s
GRAVITY = 9.81              # m/s^2
RHO_AIR = 1.225             # kg/m^3


def rossby_parameter(latitude):
    """
    Rossby参数 β = (2Ω/R) * cos(φ)。
    
    参数:
        latitude: 纬度（度）
    
    返回:
        beta: Rossby参数，单位 1/(m·s)
    """
    phi = np.deg2rad(latitude)
    return 2.0 * OMEGA * np.cos(phi) / EARTH_RADIUS


def coriolis_f(latitude):
    """
    科里奥利参数 f = 2Ω*sin(φ)。
    """
    phi = np.deg2rad(latitude)
    return 2.0 * OMEGA * np.sin(phi)


class TyphoonVortexParameters:
    """
    台风涡旋参数管理类（基于 100_blood_pressure_ode 的参数管理思想）。
    """
    def __init__(self):
        # 气压参数
        self.p_env = 1010.0         # hPa，环境气压
        self.p_min_initial = 990.0  # hPa，初始最低气压
        self.p_c = 870.0            # hPa，理论最低气压极限
        
        # 尺度参数
        self.r_max_initial = 50.0   # km，初始最大风速半径
        self.r_max_eq = 30.0        # km，平衡态最大风速半径
        
        # 强度演变系数
        self.lambda_1 = 0.15        # SST增强系数
        self.lambda_2 = 0.08        # 风切变抑制系数
        self.lambda_3 = 0.05        # 噪声系数
        
        # 尺度演变系数
        self.mu_1 = 0.02            # 弛豫系数
        self.mu_2 = 0.5             # 强度-尺度耦合系数
        
        # 海表温度阈值
        self.sst_threshold = 299.15 # K (~26°C)
        
        # 初始位置
        self.x0 = 125.0             # °E
        self.y0 = 18.0              # °N
        
        # 环境场参数
        self.env_flow_factor = 0.7  # 引导流因子
        self.beta_drift_factor = 0.3


class TyphoonVortexODE:
    """
    台风涡旋ODE系统求解器。
    """
    def __init__(self, params=None):
        if params is None:
            params = TyphoonVortexParameters()
        self.params = params
        self.time_history = []
        self.state_history = []
    
    def environment_flow(self, x, y, t):
        """
        模拟环境引导流场（简化模型）。
        
        采用时间调制的正弦流场模拟季风槽/副高引导：
            U_env = U0 + A1*sin(ω1*t) + A2*sin(k*x)
            V_env = V0 + B1*cos(ω2*t) + B2*cos(l*y)
        """
        # 背景东风流
        U0 = -3.0  # m/s，平均向西
        V0 = 1.0   # m/s，平均向北
        
        # 时间调制（季节内振荡，周期约30-60天）
        omega_mjo = 2.0 * np.pi / (45.0 * 86400.0)
        u_t = 2.0 * np.sin(omega_mjo * t)
        v_t = 1.5 * np.cos(omega_mjo * t)
        
        # 空间调制（大尺度波状结构）
        k_wave = 2.0 * np.pi / 30.0  # 1/degree
        l_wave = 2.0 * np.pi / 20.0
        u_x = 1.5 * np.sin(k_wave * (x - 120.0))
        v_y = 1.0 * np.cos(l_wave * (y - 15.0))
        
        u_env = U0 + u_t + u_x
        v_env = V0 + v_t + v_y
        
        return u_env, v_env
    
    def beta_drift_velocity(self, p_min, r_max, latitude):
        """
        计算Beta漂移速度（基于 Holland 涡旋模型）。
        
        理论公式（Chan & Williams, 1987）：
            U_beta ≈ - (β / f²) * (V_max² / 2)
            V_beta ≈   (β / f²) * (V_max² / 2)  *  (某些修正)
        
        其中 V_max 为最大风速，与气压梯度相关：
            V_max ≈ √( (P_env - P_min) / ρ * ln(R_0/R_max) )
        """
        f = coriolis_f(latitude)
        beta = rossby_parameter(latitude)
        
        # 避免f过小
        f_safe = max(abs(f), 1e-8)
        
        # 计算最大风速（简化Rankine涡旋）
        dp = self.params.p_env - p_min  # hPa
        dp = max(dp, 0.0)               # 保证非负
        dp_pa = dp * 100.0              # Pa
        
        # V_max ~ sqrt(ΔP / ρ) * scale_factor
        v_max = np.sqrt(dp_pa / RHO_AIR) * 0.5  # m/s
        
        # Beta漂移（典型值 1-3 m/s）
        u_beta = -beta * v_max**2 / (2.0 * f_safe**2)
        v_beta = beta * v_max**2 / (2.0 * f_safe**2) * 0.3
        
        # 限制在合理范围
        u_beta = np.clip(u_beta, -5.0, 5.0)
        v_beta = np.clip(v_beta, -5.0, 5.0)
        
        return u_beta, v_beta
    
    def rhs(self, t, state):
        """
        计算ODE右端项。
        
        状态向量: state = [x, y, p_min, r_max]
            x:     经度 (°E)
            y:     纬度 (°N)
            p_min: 中心最低气压 (hPa)
            r_max: 最大风速半径 (km)
        
        TODO HOLE 1: 实现台风涡旋ODE右端项
        需要计算:
        1. 位置变化率 (dxdt, dydt): 环境引导流 + Beta漂移，单位转换为 degree/s
        2. 强度变化率 (dpdt): SST增强 - 风切变抑制 + 环境噪声
        3. 尺度变化率 (drdt): 向平衡态弛豫 - 强度-尺度耦合
        
        关键科学公式:
        - Beta漂移速度: U_beta ≈ - (β / f²) * (V_max² / 2)
                       V_beta ≈   (β / f²) * (V_max² / 2) * 0.3
          其中 f = 2Ω*sin(φ), β = (2Ω/R)*cos(φ), V_max ≈ sqrt(ΔP/ρ)*0.5
        - 强度方程: dP_min/dt = -λ1*max(SST-T_thresh,0)*(P_env-P_min)/P_env
                               + λ2*|V_shear|*(P_env-P_min)/P_env
                               + λ3*ε*(P_min-P_c)/P_c
        - 尺度方程: dR_max/dt = μ1*(R_max_eq - R_max) - μ2*|dP_min/dt|*R_max/P_min
        - 坐标转换: dxdt_deg/s = rad2deg( (env*u_env + beta*u_beta) / (R*cos(lat)) )
                    dydt_deg/s = rad2deg( (env*v_env + beta*v_beta) / R )
        
        注意: 返回数组的维度与顺序必须与 ensemble_perturbation.py 和 main.py 一致
        """
        # HOLE 1 BEGIN: 请补全ODE右端项实现
        x, y, p_min, r_max = state
        params = self.params
        
        # TODO: 计算环境引导流与Beta漂移，并转换为 degree/s
        dxdt = 0.0
        dydt = 0.0
        
        # TODO: 计算强度变化率 dpdt (hPa/hour 量级)
        dpdt = 0.0
        
        # TODO: 计算尺度变化率 drdt (km/hour 量级)
        drdt = 0.0
        # HOLE 1 END
        
        return np.array([dxdt, dydt, dpdt, drdt])
    
    def implicit_rk2_step(self, t, state, dt):
        """
        隐式RK2单步推进（基于 1032_rk2_implicit）。
        
        采用隐式中点规则：
            Y_mid = Y_n + (dt/2) * F(t + dt/2, Y_mid)
            Y_{n+1} = 2*Y_mid - Y_n
        
        使用Newton-Raphson / fsolve 求解隐式方程。
        若fsolve不收敛，则回退到显式中点法。
        """
        # 显式Euler猜测
        rhs_val = self.rhs(t, state)
        y_guess = state + 0.5 * dt * rhs_val
        
        # 对于本ODE系统，显式中点法通常足够稳定
        # 先尝试显式中点法（改进Euler）作为基准
        rhs_mid = self.rhs(t + 0.5 * dt, y_guess)
        state_explicit = state + dt * rhs_mid
        
        # 尝试隐式校正（可选）
        def residual(y_mid):
            return y_mid - state - 0.5 * dt * self.rhs(t + 0.5 * dt, y_mid)
        
        try:
            y_mid_imp, infodict, ier, mesg = fsolve(
                residual, y_guess, full_output=True, xtol=1e-8, maxfev=100
            )
            if ier == 1:
                # fsolve收敛成功
                state_new = 2.0 * y_mid_imp - state
            else:
                state_new = state_explicit
        except Exception:
            state_new = state_explicit
        
        # 边界处理
        state_new[2] = np.clip(state_new[2], self.params.p_c, self.params.p_env)
        state_new[3] = np.clip(state_new[3], 5.0, 200.0)
        
        return state_new
    
    def solve(self, t_span=(0.0, 72.0), n_steps=720):
        """
        求解台风涡旋ODE系统。
        
        参数:
            t_span: 时间范围（小时）
            n_steps: 步数
        
        返回:
            t_array: 时间序列（小时）
            states:  状态历史，形状 (n_steps+1, 4)
        """
        t0, tf = t_span
        dt_hours = (tf - t0) / n_steps
        dt_seconds = dt_hours * 3600.0  # rhs 使用 SI 单位 (s)
        
        t_array = np.zeros(n_steps + 1)
        states = np.zeros((n_steps + 1, 4))
        
        # 初始状态
        states[0, 0] = self.params.x0
        states[0, 1] = self.params.y0
        states[0, 2] = self.params.p_min_initial
        states[0, 3] = self.params.r_max_initial
        
        t_array[0] = t0
        
        for i in range(n_steps):
            # t 传给 rhs 时使用秒，t_array 保持小时用于输出
            t_sec = t_array[i] * 3600.0
            states[i + 1] = self.implicit_rk2_step(t_sec, states[i], dt_seconds)
            t_array[i + 1] = t_array[i] + dt_hours
        
        self.time_history = t_array
        self.state_history = states
        
        return t_array, states


# 模块级默认参数实例
params = TyphoonVortexParameters()
