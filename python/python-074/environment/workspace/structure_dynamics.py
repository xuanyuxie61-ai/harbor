r"""
structure_dynamics.py
=====================
弹性支撑圆柱的结构动力学求解器，处理流固耦合中的结构响应。

科学背景
--------
在涡激振动（VIV）问题中，圆柱通常简化为两自由度（2-DOF）弹簧-质量-阻尼系统：

    M \ddot{X} + C \dot{X} + K X = F_{fluid}(t)

其中 X = [x, y]^T 为圆柱中心位移，M 为质量矩阵，C 为阻尼矩阵，K 为刚度矩阵，
F_{fluid} = [F_D, F_L]^T 为流体作用力。

无量纲化后，引入质量比 m^* = m / (\rho_f D^2)、阻尼比 \zeta、
折合频率 f^* = f_n D / U_\infty，结构方程可写为：

    \ddot{X} + 2\zeta (2\pi f^*) \dot{X} + (2\pi f^*)^2 X
    = \frac{1}{2 m^*} \frac{U_\infty^2}{D} C_{force}

非线性效应
----------
当振幅较大时，结构恢复力呈现非线性硬化或软化特性，引入 Duffing 型非线性项：

    M \ddot{X} + C \dot{X} + K X + K_{nl} X^3 = F_{fluid}(t)

同时，干摩擦与滞回效应可用 Bouc-Wen 模型描述，其本构关系为：

    F_{BW} = \alpha_{BW} k_{BW} X + (1-\alpha_{BW}) k_{BW} z_{BW}
    \dot{z}_{BW} = \dot{X}
    - \gamma_{BW} |\dot{X}| z_{BW} |z_{BW}|^{n_{BW}-1}
    - \beta_{BW} \dot{X} |z_{BW}|^{n_{BW}}

本模块采用时间分裂策略处理刚性（stiff）与非线性项：
- 线性 stiff 部分采用隐式梯形法（Crank-Nicolson）
- 非线性项采用显式 Runge-Kutta 子步迭代

对应原种子项目：
- 006_anishchenko_ode（非线性振荡器：引入惯性非线性项与分段非线性）
- 1041_robertson_ode（stiff ODE：学习其多时间尺度处理与守恒量检查思想）
r"""

import numpy as np


class StructureDynamics:
    r"""
    2-DOF 弹性支撑圆柱结构动力学求解器。
    """

    def __init__(self, mass, damping, stiffness, mass_ratio,
                 u_inf, diameter, fn, nonlinear_params=None,
                 time_integrator='cn_rk2'):
        r"""
        参数
        ----
        mass : float
            圆柱质量（kg）。
        damping : float
            阻尼系数 c（N·s/m）。
        stiffness : float
            刚度系数 k（N/m）。
        mass_ratio : float
            质量比 m^* = m / (\rho_f D^2)。
        u_inf : float
            来流速度（m/s）。
        diameter : float
            圆柱直径 D（m）。
        fn : float
            结构固有频率（Hz）。
        nonlinear_params : dict or None
            非线性参数，含 'k_nl'（三次硬化刚度）、
            'bw_params'（Bouc-Wen 参数字典）。
        time_integrator : str
            'cn_rk2' 或 'rk4'。
        """
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.mass_ratio = mass_ratio
        self.u_inf = u_inf
        self.diameter = diameter
        self.fn = fn
        self.omega_n = 2.0 * np.pi * fn
        self.time_integrator = time_integrator

        # 无量纲参数
        self.zeta = damping / (2.0 * np.sqrt(mass * stiffness))
        self.reduced_mass = mass_ratio

        # 状态向量: [x, y, dxdt, dydt, (可选 bw_z_x, bw_z_y)]
        self.state = np.zeros(4)
        self.prev_state = np.zeros(4)
        self.prev_force = np.zeros(2)

        # 非线性参数
        self.has_nonlinear = False
        self.k_nl = 0.0
        self.bw_params = None
        self.bw_state = None

        if nonlinear_params is not None:
            self.has_nonlinear = True
            self.k_nl = nonlinear_params.get('k_nl', 0.0)
            if 'bw_params' in nonlinear_params:
                self.bw_params = nonlinear_params['bw_params']
                self.bw_state = np.zeros(2)  # [z_x, z_y]

        # 时间步（动态设置）
        self.dt = None

    def set_time_step(self, dt):
        r"""设置结构时间步长。"""
        if dt <= 0:
            raise ValueError("dt 必须为正。")
        self.dt = dt

    def _linear_force(self, disp, vel):
        """
        计算线性恢复力与阻尼力：
        F_spring = -K X, F_damping = -C \dot{X}
        r"""
        return -self.stiffness * disp - self.damping * vel

    def _nonlinear_force(self, disp, vel):
        """
        计算非线性恢复力：
        F_nl = -k_{nl} X^3
        若启用 Bouc-Wen，则追加滞回力。
        r"""
        f_nl = -self.k_nl * (disp ** 3)
        if self.bw_params is not None and self.bw_state is not None:
            alpha_bw = self.bw_params.get('alpha', 0.5)
            k_bw = self.bw_params.get('k', self.stiffness)
            f_bw = (1.0 - alpha_bw) * k_bw * self.bw_state
            f_nl += -f_bw
        return f_nl

    def _bouc_wen_derivative(self, vel, z_bw):
        """
        Bouc-Wen 滞回变量演化方程：

        \dot{z} = \dot{x}
        - \gamma |\dot{x}| z |z|^{n-1}
        - \beta \dot{x} |z|^n
        r"""
        if self.bw_params is None:
            return np.zeros(2)
        gamma_bw = self.bw_params.get('gamma', 0.5)
        beta_bw = self.bw_params.get('beta', 0.5)
        n_bw = self.bw_params.get('n', 1.0)

        dz = vel.copy()
        abs_vel = np.abs(vel)
        abs_z = np.abs(z_bw)
        abs_z_pow = np.where(abs_z < 1e-15, 1e-15, abs_z) ** n_bw

        dz -= gamma_bw * abs_vel * z_bw * abs_z_pow / np.where(abs_z < 1e-15, 1e-15, abs_z)
        dz -= beta_bw * vel * abs_z_pow
        return dz

    def _rhs(self, state, force_ext):
        """
        状态空间右端项：
        d/dt [x, y, vx, vy]^T = [vx, vy, (F_spring + F_damping + F_nl + F_ext)/m]^T
        r"""
        disp = state[0:2]
        vel = state[2:4]

        f_lin = self._linear_force(disp, vel)
        f_nl = self._nonlinear_force(disp, vel)
        acc = (f_lin + f_nl + force_ext) / self.mass

        rhs = np.zeros_like(state)
        rhs[0:2] = vel
        rhs[2:4] = acc
        return rhs

    def step_cn_rk2(self, force_ext):
        """
        时间推进：隐式梯形法处理线性 stiff 部分 + RK2 处理非线性/外力。

        对线性系统 M \ddot{X} + C \dot{X} + K X = F，令 Y = [X; V]，则
        \dot{Y} = A Y + B F，其中
        A = [[0, I], [-M^{-1}K, -M^{-1}C]]

        隐式梯形：
        (I - 0.5*dt*A) Y^{n+1} = (I + 0.5*dt*A) Y^n + dt*B*F^{n+1/2}

        对非线性项，采用预测-校正：
        1. 预测步：用显式 Euler 预测 Y^{p}
        2. 计算非线性力 F_nl^p
        3. 校正步：将 F_nl^p 加入右端，隐式求解线性部分
        r"""
        if self.dt is None:
            raise RuntimeError("必须先调用 set_time_step(dt)。")

        dt = self.dt
        state_n = self.state.copy()
        disp_n = state_n[0:2]
        vel_n = state_n[2:4]

        # 线性系统矩阵（2-DOF 解耦，每个方向独立）
        # 对每个方向：
        # [x_{n+1}]   [1 + 0.5*dt^2*k/m,       0.5*dt*(1 - dt*c/m) ] [x_n]   [0.5*dt^2/m (F_n + F_{n+1})]
        # [v_{n+1}] = [-0.5*dt*k/m,            1 - 0.5*dt*c/m      ] [v_n] + [0.5*dt/m   (F_n + F_{n+1})]
        #
        # 实际采用更简洁的矩阵形式：
        m = self.mass
        c = self.damping
        k = self.stiffness

        # 预测步（显式 Euler，仅用于非线性力估计）
        f_lin_n = self._linear_force(disp_n, vel_n)
        f_nl_n = self._nonlinear_force(disp_n, vel_n)
        acc_n = (f_lin_n + f_nl_n + force_ext) / m

        disp_p = disp_n + dt * vel_n
        vel_p = vel_n + dt * acc_n

        # 预测非线性力
        f_nl_p = self._nonlinear_force(disp_p, vel_p)
        f_nl_avg = 0.5 * (f_nl_n + f_nl_p)

        # 隐式梯形求解线性部分 + 平均外力 + 平均非线性力
        # 对每个方向解 2x2 系统：
        # (1 + 0.25*dt^2*k/m) * x_new + (0.5*dt - 0.25*dt^2*c/m) * v_new
        #   = x_n + 0.5*dt*v_n + 0.25*dt^2/m * (F_ext_avg + F_nl_avg)
        # (-0.5*dt*k/m) * x_new + (1 + 0.5*dt*c/m) * v_new
        #   = v_n + 0.5*dt/m * (F_ext_avg + F_nl_avg)

        # TODO: Hole 3 — 请实现隐式梯形法（Crank-Nicolson）的线性系统求解
        # 科学背景：
        #   对每个自由度 d = 0, 1（流向和横向）：
        #   已知：m（质量）、c（阻尼）、k（刚度）、dt（时间步长）
        #   已知：disp_n[d]（当前位移）、vel_n[d]（当前速度）
        #   已知：force_ext[d]（外力）、f_nl_avg[d]（平均非线性力）
        #   令 F_avg = force_ext + f_nl_avg
        #
        #   隐式梯形法要求解 2x2 线性系统：
        #     A_mat * [x_new; v_new] = b_vec
        #   其中：
        #     A_mat[0,0] = 1 + 0.25*dt^2*k/m
        #     A_mat[0,1] = 0.5*dt - 0.25*dt^2*c/m
        #     A_mat[1,0] = -0.5*dt*k/m
        #     A_mat[1,1] = 1 + 0.5*dt*c/m
        #     b_vec[0]   = disp_n + 0.5*dt*vel_n + 0.25*dt^2*F_avg/m
        #     b_vec[1]   = vel_n + 0.5*dt*F_avg/m
        #
        #   对两个方向分别求解，将结果存入 new_state[d] 和 new_state[d+2]。
        #   注意：force_ext 是二维数组，来自 main.py 中 Hole 2 的转换结果。
        raise NotImplementedError("Hole 3: 隐式梯形法矩阵求解尚未实现")

        # Bouc-Wen 状态更新（显式 Euler）
        if self.bw_state is not None:
            vel_avg = 0.5 * (vel_n + new_state[2:4])
            dz = self._bouc_wen_derivative(vel_avg, self.bw_state)
            self.bw_state += dt * dz

        self.prev_state = state_n.copy()
        self.prev_force = force_ext.copy()
        self.state = new_state

    def step_rk4(self, force_ext):
        """
        标准四阶 Runge-Kutta（用于非 stiff 情形或验证）。
        r"""
        if self.dt is None:
            raise RuntimeError("必须先调用 set_time_step(dt)。")

        dt = self.dt
        y = self.state.copy()

        k1 = self._rhs(y, force_ext)
        k2 = self._rhs(y + 0.5 * dt * k1, force_ext)
        k3 = self._rhs(y + 0.5 * dt * k2, force_ext)
        k4 = self._rhs(y + dt * k3, force_ext)

        self.prev_state = y.copy()
        self.prev_force = force_ext.copy()
        self.state = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if self.bw_state is not None:
            vel = self.state[2:4]
            dz = self._bouc_wen_derivative(vel, self.bw_state)
            self.bw_state += dt * dz

    def step(self, force_ext):
        """根据设置选择积分器前进一步。"""
        if self.time_integrator == 'cn_rk2':
            self.step_cn_rk2(force_ext)
        elif self.time_integrator == 'rk4':
            self.step_rk4(force_ext)
        else:
            raise ValueError(f"未知积分器: {self.time_integrator}")

    def get_displacement(self):
        r"""返回当前位移 [x, y]。"""
        return self.state[0:2].copy()

    def get_velocity(self):
        """返回当前速度 [vx, vy]。"""
        return self.state[2:4].copy()

    def get_acceleration(self, force_ext):
        r"""由当前状态计算加速度。"""
        disp = self.state[0:2]
        vel = self.state[2:4]
        f_lin = self._linear_force(disp, vel)
        f_nl = self._nonlinear_force(disp, vel)
        return (f_lin + f_nl + force_ext) / self.mass

    def get_amplitude_envelope(self, history_disp, window=50):
        """
        由位移历史计算振幅包络（Hilbert 变换近似，用滑动极值）。

        参数
        ----
        history_disp : ndarray, shape (N, 2)
            位移时间序列。
        window : int
            滑动窗口长度。

        返回
        ----
        amp : ndarray, shape (N, 2)
            振幅包络。
        r"""
        N = len(history_disp)
        amp = np.zeros_like(history_disp)
        half = window // 2
        for i in range(N):
            i0 = max(0, i - half)
            i1 = min(N, i + half)
            local_max = np.max(history_disp[i0:i1], axis=0)
            local_min = np.min(history_disp[i0:i1], axis=0)
            amp[i] = 0.5 * (local_max - local_min)
        return amp
