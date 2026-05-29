"""
reliability.py
电网可靠性分析与线路故障概率模型
融合种子项目：circle_distance（圆上点距离概率）, triangle_analyze（三角形分析）
"""

import numpy as np
from typing import List, Tuple
from utils import triangle_area_2d, triangle_angles


class LineReliability:
    """
    输电线路可靠性模型。

    线路故障率建模基于几何随机因素（融合 circle_distance 思想）：
    在环形配电网中，线路沿圆弧分布，故障概率与线路长度和
    环境扰动强度相关。

    假设线路 i 的长度为 L_i，单位长度年故障率为 λ_0，则
        λ_i = λ_0 · L_i

    线路可用率（Availability）：
        A_i = MTTF / (MTTF + MTTR) = μ_i / (λ_i + μ_i)
    其中 MTTF = 1/λ_i 为平均故障间隔时间，MTTR = 1/μ_i 为平均修复时间。

    串联系统可靠性：
        R_series = Π_i A_i

    并联系统可靠性：
        R_parallel = 1 - Π_i (1 - A_i)
    """

    def __init__(self, line_lengths: np.ndarray, lambda0: float = 0.1,
                 mu: float = 8760.0):
        self.line_lengths = np.array(line_lengths, dtype=np.float64)
        self.lambda0 = lambda0
        self.mu = mu
        self.failure_rates = lambda0 * self.line_lengths
        self.availability = self.mu / (self.failure_rates + self.mu)

    def series_reliability(self, indices: List[int]) -> float:
        """串联路径可靠性。"""
        r = 1.0
        for idx in indices:
            r *= self.availability[idx]
        return r

    def parallel_reliability(self, indices: List[int]) -> float:
        """并联路径可靠性。"""
        r = 1.0
        for idx in indices:
            r *= (1.0 - self.availability[idx])
        return 1.0 - r

    def expected_energy_not_supplied(self, load_mw: np.ndarray,
                                     path_indices: List[List[int]]) -> float:
        """
        期望缺供电量 EENS（Expected Energy Not Supplied）。

        对每条负荷路径，计算其失效概率乘以该路径负荷：
            EENS = Σ_k P(path_k 失效) · Load_k · T
        """
        eens = 0.0
        for k, path in enumerate(path_indices):
            prob_fail = 1.0 - self.series_reliability(path)
            eens += prob_fail * load_mw[k]
        return eens


class VoltageStabilityMargin:
    """
    电压稳定性裕度分析（融合 triangle_analyze 的相量三角形思想）。

    在单负荷无穷大母线系统中，电压稳定性由 PV 曲线的鼻点（nose point）决定。
    传输功率：
        P = (E·V / X) · sin(δ)
        Q = (E·V / X) · cos(δ) - V^2 / X

    消去 δ 得到 V 与 P, Q 的关系：
        V^4 + (2QX - E^2)·V^2 + X^2·(P^2 + Q^2) = 0

    令 x = V^2，得到关于 x 的二次方程：
        x^2 + (2QX - E^2)·x + X^2·(P^2 + Q^2) = 0

    鼻点判据：判别式 Δ = 0 时，系统达到最大传输功率极限。
        Δ = (2QX - E^2)^2 - 4·X^2·(P^2 + Q^2) = 0

    电压裕度：
        Margin = (P_max - P_0) / P_max × 100%
    """

    def __init__(self, E: float, X: float, Q: float):
        self.E = E
        self.X = X
        self.Q = Q

    def voltage_solution(self, P: float) -> Tuple[float, float]:
        """
        求解给定 P 时的电压幅值（两个解，高电压解稳定，低电压解不稳定）。
        """
        a = 1.0
        b = 2.0 * self.Q * self.X - self.E**2
        c = self.X**2 * (P**2 + self.Q**2)
        delta = b**2 - 4.0 * a * c
        if delta < 0:
            return (0.0, 0.0)
        x1 = (-b + np.sqrt(delta)) / (2.0 * a)
        x2 = (-b - np.sqrt(delta)) / (2.0 * a)
        v1 = np.sqrt(max(x1, 0.0))
        v2 = np.sqrt(max(x2, 0.0))
        return (v1, v2)

    def max_power_limit(self) -> float:
        """
        计算最大功率传输极限（鼻点功率）。

        由 Δ = 0 的条件：
            P_max = sqrt( (E^2 - 2QX)^2 / (4X^2) - Q^2 )
        """
        val = (self.E**2 - 2.0 * self.Q * self.X)**2 / (4.0 * self.X**2) - self.Q**2
        return np.sqrt(max(val, 0.0))

    def voltage_margin(self, P_operating: float) -> float:
        """
        计算当前运行点到电压崩溃点的有功裕度。
        """
        p_max = self.max_power_limit()
        if p_max < 1e-12:
            return 0.0
        return max(0.0, (p_max - P_operating) / p_max)

    def pv_curve(self, n_points: int = 50) -> dict:
        """
        生成 PV 曲线数据。
        """
        p_max = self.max_power_limit()
        ps = np.linspace(0.0, p_max * 0.99, n_points)
        v_high = []
        v_low = []
        for p in ps:
            v1, v2 = self.voltage_solution(p)
            v_high.append(v1)
            v_low.append(v2)
        return {
            "P": ps,
            "V_high": np.array(v_high),
            "V_low": np.array(v_low),
            "P_max": p_max
        }


class ThreePhaseUnbalance:
    """
    三相不平衡度分析（融合 triangle_analyze 的三角形几何分析）。

    三相电压相量 V_a, V_b, V_c 构成复平面上的三角形。
    对称分量法：
        [V_0]   1 [1  1   1  ][V_a]
        [V_+] = - [1  α   α^2][V_b]
        [V_-]   3 [1  α^2 α  ][V_c]
    其中 α = e^{j·2π/3}。

    三相不平衡度：
        ε = |V_-| / |V_+| × 100%

    几何解释：三相电压相量三角形的面积与边长比值反映了不平衡程度。
    面积越大、最小角越小，不平衡越严重。
    """

    def __init__(self, Va: complex, Vb: complex, Vc: complex):
        self.Va = Va
        self.Vb = Vb
        self.Vc = Vc

    def symmetric_components(self) -> Tuple[complex, complex, complex]:
        """
        计算对称分量 V0, V+, V-。
        """
        alpha = np.exp(1j * 2.0 * np.pi / 3.0)
        V0 = (self.Va + self.Vb + self.Vc) / 3.0
        V1 = (self.Va + alpha * self.Vb + alpha**2 * self.Vc) / 3.0
        V2 = (self.Va + alpha**2 * self.Vb + alpha * self.Vc) / 3.0
        return V0, V1, V2

    def unbalance_factor(self) -> float:
        """电压负序不平衡度（%）。"""
        _, V1, V2 = self.symmetric_components()
        if abs(V1) < 1e-12:
            return 100.0
        return abs(V2) / abs(V1) * 100.0

    def phasor_triangle_analysis(self) -> dict:
        """
        将相量视为复平面上的点，分析其几何三角形性质。
        """
        a = np.array([self.Va.real, self.Va.imag])
        b = np.array([self.Vb.real, self.Vb.imag])
        c = np.array([self.Vc.real, self.Vc.imag])
        area = triangle_area_2d(a, b, c)
        angles = triangle_angles(a, b, c)
        sides = [np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)]
        return {
            "area": area,
            "angles_deg": np.degrees(angles),
            "sides": sides,
            "min_angle_deg": np.degrees(np.min(angles))
        }
