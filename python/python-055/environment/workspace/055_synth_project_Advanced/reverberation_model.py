"""
reverberation_model.py
基于种子项目 707_mackey_glass_dde（Mackey-Glass 延迟微分方程），
构建海洋混响（reverberation）动力学模型。

科学背景：多波束声纳接收到的回波信号不仅包含目标海底反射，
还包含水体散射、海面反射、多次海底反射等混响成分。
这些多途效应在时域上表现为具有记忆特性的非线性衰减振荡。

Mackey-Glass 延迟微分方程是描述具有记忆效应的生理控制系统的
经典模型，其形式为：

    dx(t)/dt = β · x(t - τ) / (1 + x(t - τ)^n) - γ · x(t)

其中：
    x(t)    — t 时刻的混响信号强度（归一化）
    β       — 反馈增益系数（控制能量注入速率）
    γ       — 衰减系数（控制能量耗散速率）
    τ       — 延迟时间（对应多途传播时延）
    n       — 非线性指数（控制饱和特性）

在声纳混响应用中：
    τ 对应海底-海面-海底多途传播的额外时间延迟；
    β/γ 比值决定混响的 Q 值；
    n 控制散射体的非线性响应。

本模块采用四阶 Runge-Kutta 方法结合线性插值历史值，
数值求解该 DDE，并输出混响时间序列。
"""

import numpy as np


class MackeyGlassReverberation:
    """
    Mackey-Glass 混响模型求解器。
    """

    def __init__(
        self,
        gamma: float = 0.1,
        beta: float = 0.2,
        n: float = 9.65,
        tau: float = 5.0,
        dt: float = 0.01
    ):
        """
        参数:
            gamma: 衰减系数 (s⁻¹)
            beta:  反馈增益 (s⁻¹)
            n:     非线性指数
            tau:   延迟时间 (s)
            dt:    数值积分步长 (s)
        """
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.n = float(n)
        self.tau = float(tau)
        self.dt = float(dt)

        # 历史缓冲区（用于延迟插值）
        self._history_t = None
        self._history_x = None

    def _dde_rhs(self, t: float, x: float, x_delayed: float) -> float:
        """
        计算 Mackey-Glass DDE 的右端项。

        公式:
            dx/dt = β · x_τ / (1 + x_τ^n) - γ · x
        """
        if x_delayed < 0.0:
            x_delayed = 0.0
        # 饱和非线性项
        denom = 1.0 + x_delayed ** self.n
        if denom < 1e-15:
            denom = 1e-15
        dxdt = self.beta * x_delayed / denom - self.gamma * x
        return dxdt

    def _get_delayed(self, t: float) -> float:
        """通过线性插值获取延迟时刻的历史值。"""
        if self._history_t is None or len(self._history_t) == 0:
            return 0.0
        t_delayed = t - self.tau
        if t_delayed <= self._history_t[0]:
            return self._history_x[0]
        if t_delayed >= self._history_t[-1]:
            return self._history_x[-1]
        # 线性插值
        idx = np.searchsorted(self._history_t, t_delayed)
        if idx == 0:
            return self._history_x[0]
        t0, t1 = self._history_t[idx - 1], self._history_t[idx]
        x0, x1 = self._history_x[idx - 1], self._history_x[idx]
        if abs(t1 - t0) < 1e-15:
            return x0
        alpha = (t_delayed - t0) / (t1 - t0)
        return x0 + alpha * (x1 - x0)

    def solve(
        self,
        t_span: tuple,
        x0: float = 0.5,
        history_const: float = 0.0
    ) -> tuple:
        """
        数值求解 Mackey-Glass DDE。

        参数:
            t_span: (t_start, t_stop)
            x0: 初始时刻的 x 值
            history_const: t < t_start 时的历史常数值
        返回:
            (t_array, x_array)
        """
        t_start, t_stop = t_span
        n_steps = int(np.ceil((t_stop - t_start) / self.dt)) + 1
        t_arr = np.linspace(t_start, t_stop, n_steps)
        x_arr = np.zeros(n_steps, dtype=np.float64)

        # 初始化历史
        self._history_t = [t_start - self.tau]
        self._history_x = [history_const]

        x_arr[0] = x0

        for i in range(n_steps - 1):
            t = t_arr[i]
            x = x_arr[i]
            x_delayed = self._get_delayed(t)

            # RK4 单步（对 DDE 采用显式处理延迟项）
            k1 = self._dde_rhs(t, x, x_delayed)

            x_delayed_k2 = self._get_delayed(t + 0.5 * self.dt)
            k2 = self._dde_rhs(t + 0.5 * self.dt, x + 0.5 * self.dt * k1, x_delayed_k2)

            x_delayed_k3 = self._get_delayed(t + 0.5 * self.dt)
            k3 = self._dde_rhs(t + 0.5 * self.dt, x + 0.5 * self.dt * k2, x_delayed_k3)

            x_delayed_k4 = self._get_delayed(t + self.dt)
            k4 = self._dde_rhs(t + self.dt, x + self.dt * k3, x_delayed_k4)

            x_new = x + self.dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # 非负保护
            if x_new < 0.0:
                x_new = 0.0

            x_arr[i + 1] = x_new
            self._history_t.append(t_arr[i + 1])
            self._history_x.append(x_new)

        return t_arr, x_arr

    def compute_reverberation_envelope(
        self,
        ttw_base: float,
        amplitude: float = 1.0,
        duration_factor: float = 3.0
    ) -> tuple:
        """
        计算以双程传播时间为基准的混响包络。

        参数:
            ttw_base: 主回波双程时间 (s)
            amplitude: 混响幅值系数
            duration_factor: 混响持续时间 = duration_factor * τ
        返回:
            (t_arr, envelope_arr)
        """
        t_start = 0.0
        t_stop = ttw_base + duration_factor * self.tau
        # 初始激励脉冲
        x0 = amplitude
        t_arr, x_arr = self.solve((t_start, t_stop), x0=x0, history_const=0.0)
        # 混响包络：取平滑后的绝对值
        envelope = np.abs(x_arr)
        # 简单平滑（3 点移动平均）
        if len(envelope) >= 3:
            smoothed = np.convolve(envelope, np.ones(3) / 3.0, mode='same')
            envelope = smoothed
        return t_arr, envelope


class StochasticReverberationField:
    """
    随机混响场模型：多个独立 Mackey-Glass 振子的叠加，
    模拟不同散射机制（海底粗糙散射、体积散射、海面反射）的混响叠加。
    """

    def __init__(self, n_modes: int = 5, seed: int = 55):
        self.n_modes = n_modes
        self.rng = np.random.default_rng(seed)
        self.modes = []
        for _ in range(n_modes):
            gamma = self.rng.uniform(0.05, 0.2)
            beta = self.rng.uniform(0.15, 0.35)
            n_exp = self.rng.uniform(7.0, 12.0)
            tau = self.rng.uniform(2.0, 10.0)
            self.modes.append(MackeyGlassReverberation(gamma, beta, n_exp, tau))

    def generate_composite_envelope(
        self,
        ttw_base: float,
        base_amplitude: float = 1.0
    ) -> tuple:
        """
        生成复合混响包络。

        数学模型:
            R(t) = Σ_{k=1}^{K} a_k · r_k(t)
        其中 a_k 为各散射模式的权重，r_k 为对应 Mackey-Glass 解。
        """
        # 统一时间网格
        t_stop = ttw_base + 30.0
        n_points = 2000
        t_common = np.linspace(0.0, t_stop, n_points)
        composite = np.zeros(n_points, dtype=np.float64)

        for i, mode in enumerate(self.modes):
            # 各模式不同权重（海底散射最强，体积散射次之，海面最弱）
            weight = base_amplitude * (0.5 ** i)
            t_mode, env_mode = mode.compute_reverberation_envelope(ttw_base, amplitude=weight)
            # 插值到统一网格
            env_interp = np.interp(t_common, t_mode, env_mode, left=0.0, right=0.0)
            composite += env_interp

        return t_common, composite
