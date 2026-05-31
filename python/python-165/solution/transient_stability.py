"""
transient_stability.py
电力系统暂态稳定性分析
融合种子项目：spring_ode（弹簧阻尼振荡）, humps_ode（非线性ODE）, control_bio_homework（最优控制框架）
"""

import numpy as np
from typing import Callable, Tuple, Optional
from utils import rk4_step


class SwingEquation:
    """
    单机无穷大母线系统的摇摆方程（Swing Equation）：

        M·d²δ/dt² + D·dδ/dt = P_m - P_e

    其中：
        M = 2H/ω_s   为惯性常数（s）
        H 为惯性时间常数（s）
        ω_s = 2πf    为同步角速度（rad/s），f=50/60 Hz
        D 为阻尼系数（p.u.）
        P_m 为机械功率（p.u.）
        P_e = (E'·V_∞ / X) · sin(δ)   为电磁功率
        δ 为转子角（rad）
        ω = dδ/dt  为转子角速度偏差

    化为状态空间形式（一阶ODE组）：
        dδ/dt = ω - ω_s
        dω/dt = (P_m - P_e - D·(ω - ω_s)) / M

    在智能电网中，该方程描述同步发电机在受到大扰动（如短路故障、
    线路跳闸）后的转子动态行为。若 δ(t) 在扰动清除后趋于稳定值，
    则系统暂态稳定；否则将发生失步（loss of synchronism）。
    """

    def __init__(self, H: float, D: float, f_base: float = 50.0,
                 E_prime: float = 1.1, V_inf: float = 1.0, X: float = 0.5):
        self.H = H
        self.D = D
        self.omega_s = 2.0 * np.pi * f_base
        self.M = 2.0 * H / self.omega_s
        self.E_prime = E_prime
        self.V_inf = V_inf
        self.X = X

    def electrical_power(self, delta: float) -> float:
        """
        电磁功率：
            P_e(δ) = (E'·V_∞ / X) · sin(δ)
        """
        if self.X <= 1e-12:
            raise ValueError("reactance X must be positive")
        return (self.E_prime * self.V_inf / self.X) * np.sin(delta)

    def derivative(self, t: float, state: np.ndarray,
                   P_m: float, fault_active: bool = False,
                   X_fault: float = 1.0) -> np.ndarray:
        """
        状态导数函数，用于 RK4 积分。

        参数：
            state = [δ, ω]
            fault_active: 若 True，则在故障期间采用故障电抗 X_fault
        """
        delta, omega = state[0], state[1]
        X_eff = self.X + X_fault if fault_active else self.X
        X_eff = max(X_eff, 1e-12)
        Pe = (self.E_prime * self.V_inf / X_eff) * np.sin(delta)
        d_delta = omega - self.omega_s
        d_omega = (P_m - Pe - self.D * (omega - self.omega_s)) / self.M
        return np.array([d_delta, d_omega])

    def critical_clearing_angle(self, P_m: float) -> Optional[float]:
        """
        计算临界切除角 δ_cr（Equal-Area Criterion，等面积法则）。

        稳定条件：加速面积 A_acc 等于最大减速面积 A_dec。

        设故障前稳态角 δ_0 满足 P_m = P_{e0}(δ_0) = P_{max,0}·sin(δ_0)。
        故障期间 P_e = 0（三相短路极限情形），转子加速。
        故障切除后，P_e = P_{max,post}·sin(δ)。

        临界角 δ_cr 满足：
            ∫_{δ_0}^{δ_cr} (P_m - 0) dδ = ∫_{δ_cr}^{δ_max} (P_{max,post}·sin(δ) - P_m) dδ

        解析解：
            P_m·(δ_cr - δ_0) = P_{max,post}·(cos(δ_cr) - cos(δ_max))
                                 - P_m·(δ_max - δ_cr)
        其中 δ_max = π - arcsin(P_m / P_{max,post})。

        该值是判断保护装置动作时间是否足够的理论上限。
        """
        P_max_post = self.E_prime * self.V_inf / self.X
        if P_m >= P_max_post:
            return None  # 机械功率超过最大传输功率，无法稳定
        delta_0 = np.arcsin(P_m / P_max_post)
        delta_max = np.pi - delta_0
        # 等面积方程的数值求解
        def area_balance(delta_cr):
            A_acc = P_m * (delta_cr - delta_0)
            A_dec = P_max_post * (np.cos(delta_cr) - np.cos(delta_max)) \
                    - P_m * (delta_max - delta_cr)
            return A_acc - A_dec

        from scipy.optimize import brentq
        try:
            delta_cr = brentq(area_balance, delta_0, delta_max)
            return delta_cr
        except Exception:
            # 无 scipy 时的简单二分法
            lo, hi = delta_0, delta_max
            for _ in range(100):
                mid = (lo + hi) * 0.5
                if area_balance(lo) * area_balance(mid) < 0:
                    hi = mid
                else:
                    lo = mid
            return (lo + hi) * 0.5

    def simulate(self, t_span: Tuple[float, float], dt: float,
                 P_m: float, delta0: float, omega0: float,
                 fault_time: Optional[Tuple[float, float]] = None,
                 X_fault: float = 1.0) -> dict:
        """
        时域仿真（RK4 积分）。

        参数：
            fault_time: (t_fault_on, t_fault_off)，若为 None 则无故障
        """
        t0, tf = t_span
        if dt <= 0:
            raise ValueError("dt must be positive")
        n_steps = int(np.ceil((tf - t0) / dt))
        times = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 2), dtype=np.float64)
        states[0] = [delta0, omega0]

        for k in range(n_steps):
            t = times[k]
            y = states[k]
            fault_active = False
            if fault_time is not None:
                t_on, t_off = fault_time
                fault_active = (t >= t_on and t < t_off)

            def f(t_local, y_local):
                return self.derivative(t_local, y_local, P_m,
                                       fault_active=fault_active, X_fault=X_fault)

            states[k + 1] = rk4_step(f, t, y, dt)

        return {
            "t": times,
            "delta": states[:, 0],
            "omega": states[:, 1],
            "stable": self._assess_stability(times, states[:, 0])
        }

    def _assess_stability(self, t: np.ndarray, delta: np.ndarray) -> bool:
        """
        稳定性判据：若最后 20% 时间内转子角振荡幅值单调递减且不超过 π，
        则认为暂态稳定。
        """
        n = len(delta)
        tail = delta[int(0.8 * n):]
        if np.max(np.abs(tail)) > np.pi:
            return False
        # 检查振幅是否衰减
        peaks = []
        for i in range(1, len(tail) - 1):
            if tail[i] > tail[i - 1] and tail[i] > tail[i + 1]:
                peaks.append(tail[i])
        if len(peaks) >= 3:
            return peaks[-1] < peaks[-2] < peaks[0]
        return True


class MultiMachineStability:
    """
    多机系统暂态稳定性（融合 spring_ode 的多体耦合思想）。

    n 台发电机组成的系统，第 i 台机的摇摆方程：
        M_i·d²δ_i/dt² + D_i·dδ_i/dt = P_{m,i} - P_{e,i}

    其中电磁功率由网络导纳矩阵决定：
        P_{e,i} = Σ_{j=1}^{n} |E_i||E_j|·(G_{ij} cos(δ_i-δ_j) + B_{ij} sin(δ_i-δ_j))

    这是经典的多机电力系统暂态模型，用于分析区域间振荡模式
    （inter-area oscillation）和低频振荡稳定性。
    """

    def __init__(self, n_gen: int, H: np.ndarray, D: np.ndarray,
                 E_prime: np.ndarray, Y_reduced: np.ndarray):
        self.n_gen = n_gen
        self.H = np.array(H, dtype=np.float64)
        self.D = np.array(D, dtype=np.float64)
        self.E_prime = np.array(E_prime, dtype=np.float64)
        self.Y_reduced = np.array(Y_reduced, dtype=np.complex128)
        self.G = self.Y_reduced.real
        self.B = self.Y_reduced.imag
        self.omega_s = 2.0 * np.pi * 50.0

    def derivative(self, t: float, state: np.ndarray, P_m: np.ndarray) -> np.ndarray:
        """
        state = [δ_1, ..., δ_n, ω_1, ..., ω_n]
        """
        n = self.n_gen
        delta = state[:n]
        omega = state[n:]
        d_delta = omega - self.omega_s
        d_omega = np.zeros(n, dtype=np.float64)
        for i in range(n):
            Pe = 0.0
            for j in range(n):
                angle_diff = delta[i] - delta[j]
                Pe += self.E_prime[i] * self.E_prime[j] * (
                    self.G[i, j] * np.cos(angle_diff)
                    + self.B[i, j] * np.sin(angle_diff)
                )
            M_i = 2.0 * self.H[i] / self.omega_s
            d_omega[i] = (P_m[i] - Pe - self.D[i] * (omega[i] - self.omega_s)) / M_i
        return np.concatenate([d_delta, d_omega])

    def simulate(self, t_span: Tuple[float, float], dt: float,
                 P_m: np.ndarray, delta0: np.ndarray, omega0: np.ndarray) -> dict:
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        times = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 2 * self.n_gen), dtype=np.float64)
        states[0, :self.n_gen] = delta0
        states[0, self.n_gen:] = omega0

        for k in range(n_steps):
            t = times[k]
            y = states[k]

            def f(_, y_local):
                return self.derivative(t, y_local, P_m)

            states[k + 1] = rk4_step(f, t, y, dt)

        return {
            "t": times,
            "delta": states[:, :self.n_gen],
            "omega": states[:, self.n_gen:]
        }
