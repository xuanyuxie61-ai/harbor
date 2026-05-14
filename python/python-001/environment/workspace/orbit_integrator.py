"""
orbit_integrator.py

基于 line_ncc_rule (Newton-Cotes Closed 数值积分) 与
stochastic_rk (随机 Runge-Kutta 方法，SDE 求解) 核心算法，
实现小行星附近轨道的数值积分器。

科学背景：
近小行星轨道的运动方程：
    d²r/dt² = ∇U(r) + a_perturbation

其中摄动项包括：
1. 太阳引力第三体效应
2. 太阳辐射压 (SRP):  a_SRP = β (GM_sun / c²) (A/m) (r_sun / |r_sun|³)
3. Yarkovsky 效应（热辐射漂移，建模为随机噪声）

本项目同时提供：
- 确定性高阶 Newton-Cotes 积分（用于高精度长期轨道预报）
- 随机 Runge-Kutta (SRK4) 积分（用于含随机摄动的轨道演化）

核心公式：
确定性方程组（一阶化）：
    dr/dt = v
    dv/dt = a_grav(r) + a_SRP + a_3body

随机方程（Yarkovsky 热噪声建模为加性白噪声）：
    dX = f(X) dt + g(X) dW
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


class OrbitIntegratorError(Exception):
    pass


def line_ncc_rule(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Newton-Cotes Closed (NCC) 求积公式。
    基于 line_ncc_rule.m 的 Lagrange 插值 + 反导数算法。

    对积分区间 [a, b]，取 n 个等距节点：
        x_i = a + (i-1)*(b-a)/(n-1),  i=1..n
    权重 w_i 通过构造 Lagrange 基多项式的反导数得到：
        w_i = ∫_a^b L_i(x) dx

    返回:
        x: (n,) 节点
        w: (n,) 权重
    """
    if n < 2:
        raise OrbitIntegratorError("NCC 规则需要至少 2 个节点")
    x = np.linspace(a, b, n)
    w = np.zeros(n)

    for i in range(n):
        d = np.zeros(n)
        d[i] = 1.0

        # 牛顿前向差分形式的 Lagrange 基
        for j in range(1, n):
            for k in range(j, n):
                d[n + j - k - 1] = (d[n + j - k - 2] - d[n + j - k - 1]) / (x[n - k - 1] - x[n + j - k - 1])

        # 转换为标准多项式系数
        for j in range(1, n):
            for k in range(1, n - j + 1):
                d[n - k - 1] = d[n - k - 1] - x[n - k - j] * d[n - k]

        # 计算反导数在端点的值
        yvala = d[n - 1] / n
        yvalb = d[n - 1] / n
        for j in range(n - 2, -1, -1):
            yvala = yvala * a + d[j] / (j + 1)
            yvalb = yvalb * b + d[j] / (j + 1)
        yvala *= a
        yvalb *= b
        w[i] = yvalb - yvala

    return x, w


def newton_cotes_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 9,
    n_sub: int = 100
) -> float:
    """
    使用复合 Newton-Cotes Closed 规则积分 f 在 [a,b] 上的值。
    将区间分为 n_sub 个子区间，每个子区间用 n 点 NCC 规则。
    """
    if n_sub < 1:
        raise OrbitIntegratorError("子区间数必须 ≥ 1")
    h = (b - a) / n_sub
    total = 0.0
    for k in range(n_sub):
        sub_a = a + k * h
        sub_b = sub_a + h
        x, w = line_ncc_rule(n, sub_a, sub_b)
        total += np.sum(w * np.array([f(xi) for xi in x]))
    return total


def srk4_ti_step(
    x: np.ndarray,
    t: float,
    h: float,
    q: float,
    fi: Callable[[np.ndarray], np.ndarray],
    gi: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    四阶随机 Runge-Kutta 单步推进（时间不变系统）。
    基于 rk4_ti_step.m 的 Kasdin 系数。

    SDE 形式:  dX = f(X) dt + g(X) dW
    其中 dW ~ N(0, q/h)。

    Kasdin (1995) 的 Butcher 表系数:
        a21 =  2.71644396264860
        a31 = -6.95653259006152,  a32 = 0.78313689457981
        a41 =  0.0,               a42 = 0.48257353309214, a43 = 0.26171080165848
        a51 =  0.47012396888046,  a52 = 0.36597075368373,
        a53 =  0.08906615686702,  a54 = 0.07483912056879
        q1 =   2.12709852335625,  q2 =  2.73245878238737,
        q3 =  11.22760917474960,  q4 = 13.36199560336697
    """
    a21 = 2.71644396264860
    a31 = -6.95653259006152
    a32 = 0.78313689457981
    a42 = 0.48257353309214
    a43 = 0.26171080165848
    a51 = 0.47012396888046
    a52 = 0.36597075368373
    a53 = 0.08906615686702
    a54 = 0.07483912056879

    q1 = 2.12709852335625
    q2 = 2.73245878238737
    q3 = 11.22760917474960
    q4 = 13.36199560336697

    n1 = np.random.randn(x.shape[0])
    w1 = n1 * np.sqrt(q1 * q / h)
    k1 = h * fi(x) + h * gi(x) * w1

    x2 = x + a21 * k1
    n2 = np.random.randn(x.shape[0])
    w2 = n2 * np.sqrt(q2 * q / h)
    k2 = h * fi(x2) + h * gi(x2) * w2

    x3 = x + a31 * k1 + a32 * k2
    n3 = np.random.randn(x.shape[0])
    w3 = n3 * np.sqrt(q3 * q / h)
    k3 = h * fi(x3) + h * gi(x3) * w3

    x4 = x + a42 * k2 + a43 * k3
    n4 = np.random.randn(x.shape[0])
    w4 = n4 * np.sqrt(q4 * q / h)
    k4 = h * fi(x4) + h * gi(x4) * w4

    xstar = x + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4
    return xstar


def rk4_step(
    x: np.ndarray,
    t: float,
    h: float,
    f: Callable[[np.ndarray, float], np.ndarray]
) -> np.ndarray:
    """
    经典四阶 Runge-Kutta 单步。
    用于确定性轨道积分。
    """
    k1 = f(x, t)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(x + h * k3, t + h)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class OrbitalDynamics:
    """
    封装小行星附近轨道动力学，支持确定性 RK4 和随机 SRK4 积分。
    """

    def __init__(
        self,
        grav_accel_func: Callable[[np.ndarray], np.ndarray],
        gm_sun: float = 1.32712440018e11,  # km³/s²
        solar_distance: float = 1.496e8,    # km (1 AU)
        beta_srp: float = 0.0,              # SRP 系数 (A/m in m²/kg scaled)
        perturbation_std: float = 0.0       # Yarkovsky 随机摄动标准差 (km/s²)
    ):
        self.grav = grav_accel_func
        self.gm_sun = gm_sun
        self.solar_distance = solar_distance
        self.beta_srp = beta_srp
        self.perturbation_std = perturbation_std

    def _solar_direction(self) -> np.ndarray:
        """假设太阳始终位于 +x 方向（简化）"""
        return np.array([1.0, 0.0, 0.0])

    def _solar_radiation_pressure(self, pos: np.ndarray) -> np.ndarray:
        """
        太阳辐射压加速度模型：
            a_SRP = β * (GM_sun / c²) * (1 / r_sun²) * n_sun
        其中 c = 299792.458 km/s
        """
        if self.beta_srp <= 0.0:
            return np.zeros(3)
        c_light = 299792.458  # km/s
        factor = self.beta_srp * self.gm_sun / (c_light ** 2) / (self.solar_distance ** 2)
        return factor * self._solar_direction()

    def _third_body_sun(self, pos: np.ndarray) -> np.ndarray:
        """
        太阳第三体摄动（简化，假设小行星位于原点，太阳在远处固定）：
            a_3b = GM_sun * (r_sun / |r_sun|³ − r_rel / |r_rel|³)
        这里进一步简化：只保留线性潮汐项 ≈ (GM_sun / d³) * pos
        """
        d = self.solar_distance
        return self.gm_sun / (d ** 3) * pos

    def deterministic_rhs(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        确定性右端项。
        state = [r_x, r_y, r_z, v_x, v_y, v_z]
        """
        pos = state[:3]
        vel = state[3:]
        a_grav = self.grav(pos)
        a_srp = self._solar_radiation_pressure(pos)
        a_3b = self._third_body_sun(pos)
        acc = a_grav + a_srp + a_3b
        return np.concatenate([vel, acc])

    def stochastic_drift(self, state: np.ndarray) -> np.ndarray:
        """
        随机扩散项 g(X)，用于 Yarkovsky 热噪声建模。
        返回 6 维向量（位置分量噪声为 0，速度分量噪声为常数）。
        """
        if self.perturbation_std <= 0.0:
            return np.zeros(6)
        g = np.zeros(6)
        g[3:] = self.perturbation_std
        return g

    def stochastic_rhs(self, state: np.ndarray) -> np.ndarray:
        """
        确定性漂移项 f(X)，用于随机 RK。
        """
        return self.deterministic_rhs(state, 0.0)

    def integrate_deterministic(
        self,
        state0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 RK4 积分确定性轨道。

        返回:
            t_array: (n_steps+1,)
            states: (n_steps+1, 6)
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t_array = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 6))
        states[0] = state0.copy()

        for i in range(n_steps):
            states[i + 1] = rk4_step(states[i], t_array[i], h, self.deterministic_rhs)

        return t_array, states

    def integrate_stochastic(
        self,
        state0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000,
        q_spectral: float = 1e-12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 SRK4 积分含随机摄动的轨道。
        q_spectral: 输入白噪声的谱密度 (km²/s³)。
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t_array = np.linspace(t0, tf, n_steps + 1)
        states = np.zeros((n_steps + 1, 6))
        states[0] = state0.copy()

        for i in range(n_steps):
            states[i + 1] = srk4_ti_step(
                states[i],
                t_array[i],
                h,
                q_spectral,
                self.stochastic_rhs,
                self.stochastic_drift
            )

        return t_array, states

    def integrate_adaptive_rk4(
        self,
        state0: np.ndarray,
        t_span: Tuple[float, float],
        atol: float = 1e-9,
        rtol: float = 1e-6,
        h0: float = 1.0,
        h_min: float = 1e-6,
        h_max: float = 1e4
    ) -> Tuple[List[float], List[np.ndarray]]:
        """
        嵌入式 RK4(5) 自适应步长积分（简化实现，使用步长折半法）。
        保证局部截断误差满足给定容差。
        """
        t0, tf = t_span
        t = t0
        state = state0.copy()
        h = h0
        t_list = [t]
        state_list = [state.copy()]

        while t < tf:
            h = min(h, tf - t)
            # 全步
            s1 = rk4_step(state, t, h, self.deterministic_rhs)
            # 两步半
            s_half = rk4_step(state, t, h / 2.0, self.deterministic_rhs)
            s2 = rk4_step(s_half, t + h / 2.0, h / 2.0, self.deterministic_rhs)

            err = np.linalg.norm(s1 - s2)
            scale = atol + rtol * max(np.linalg.norm(s1), np.linalg.norm(s2))

            if err <= scale or h <= h_min:
                t += h
                state = s2.copy()
                t_list.append(t)
                state_list.append(state.copy())
                # 增大步长
                if err > 0:
                    h = min(h_max, h * 0.9 * (scale / err) ** 0.2)
                else:
                    h = min(h_max, 2.0 * h)
            else:
                h = max(h_min, h * 0.9 * (scale / err) ** 0.25)

        return np.array(t_list), np.array(state_list)
