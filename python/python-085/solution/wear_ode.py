"""
wear_ode.py
磨损演化动力学模块
融合种子项目：
  - 828_ode_midpoint（中点法 ODE 积分）

核心科学模型：Archard 磨损定律
"""
import numpy as np
from typing import Tuple, Callable, Optional
from utils import safe_divide


class ArchardWearModel:
    r"""
    Archard 磨损定律的 ODE 表述：

    \frac{d h}{d t} = k_w \cdot p_n(t) \cdot \| v_t(t) \|

    其中：
    - h(t) 为磨损深度
    - k_w 为无量纲磨损系数
    - p_n(t) 为接触法向压力
    - v_t(t) 为相对切向速度

    对于往复滑动，v_t(t) = v_0 \sin(\omega t)。
    """

    def __init__(self, wear_coeff: float = 1e-6, omega: float = 2.0 * np.pi,
                 v0: float = 0.01):
        self.k_w = wear_coeff
        self.omega = omega
        self.v0 = v0

    def tangential_velocity(self, t: float) -> float:
        r"""
        简谐切向速度：v_t(t) = v_0 \sin(\omega t)
        """
        return self.v0 * np.sin(self.omega * t)

    def wear_rate(self, t: float, h: float, pressure_func: Callable[[float], float]) -> float:
        r"""
        磨损率 ODE 右端项：
        f(t, h) = k_w \cdot p_n(t) \cdot |v_t(t)|
        注意：磨损率与 h 本身无关（简化模型），但保留接口以支持更复杂模型。
        """
        p_n = pressure_func(t)
        v_t = self.tangential_velocity(t)
        return self.k_w * p_n * abs(v_t)

    def integrate_midpoint(self, h0: float, t_span: Tuple[float, float], n_steps: int,
                           pressure_func: Callable[[float], float]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        中点法（隐式，融合 828_ode_midpoint）积分磨损演化方程。

        算法：
        对 i = 1, ..., n：
          t_m = t_i + 0.5 * h
          y_m 通过不动点迭代求解：
            y_m^{(k+1)} = y_i + 0.5 * h * f(t_m, y_m^{(k)})
          y_{i+1} = 2 * y_m - y_i

        参数 theta = 0.5 对应中点法。
        """
        a, b = t_span
        h = (b - a) / n_steps
        theta = 0.5
        it_max = 10

        t = np.linspace(a, b, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = h0

        for i in range(n_steps):
            t_m = t[i] + theta * h
            ym = y[i]
            # 不动点迭代
            for _ in range(it_max):
                ym_new = y[i] + theta * h * self.wear_rate(t_m, ym, pressure_func)
                if abs(ym_new - ym) < 1e-14:
                    ym = ym_new
                    break
                ym = ym_new
            y[i + 1] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * y[i]

        return t, y

    def integrate_rk4(self, h0: float, t_span: Tuple[float, float], n_steps: int,
                      pressure_func: Callable[[float], float]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        经典四阶 Runge-Kutta 方法（用于对比验证）：
        y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
        """
        a, b = t_span
        h = (b - a) / n_steps
        t = np.linspace(a, b, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = h0

        for i in range(n_steps):
            k1 = self.wear_rate(t[i], y[i], pressure_func)
            k2 = self.wear_rate(t[i] + 0.5 * h, y[i] + 0.5 * h * k1, pressure_func)
            k3 = self.wear_rate(t[i] + 0.5 * h, y[i] + 0.5 * h * k2, pressure_func)
            k4 = self.wear_rate(t[i] + h, y[i] + h * k3, pressure_func)
            y[i + 1] = y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return t, y

    def compute_wear_volume(self, t: np.ndarray, h: np.ndarray,
                            contact_area: float) -> float:
        r"""
        累积磨损体积（简化模型）：
        V_w = A_c \cdot h_{max}(t)
        """
        return contact_area * float(np.max(h))


def coupled_wear_contact_step(wear_model: ArchardWearModel,
                               current_wear_depth: np.ndarray,
                               pressure: np.ndarray,
                               dt: float) -> np.ndarray:
    r"""
    耦合磨损-接触的单步更新。
    对每个接触节点 i：
    h_i^{new} = h_i + dt * k_w * p_i * |v_t|
    """
    v_t = abs(wear_model.tangential_velocity(0.0))
    rate = wear_model.k_w * pressure * v_t
    return current_wear_depth + dt * rate
