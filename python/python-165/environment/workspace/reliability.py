
import numpy as np
from typing import List, Tuple
from utils import triangle_area_2d, triangle_angles


class LineReliability:

    def __init__(self, line_lengths: np.ndarray, lambda0: float = 0.1,
                 mu: float = 8760.0):
        self.line_lengths = np.array(line_lengths, dtype=np.float64)
        self.lambda0 = lambda0
        self.mu = mu
        self.failure_rates = lambda0 * self.line_lengths
        self.availability = self.mu / (self.failure_rates + self.mu)

    def series_reliability(self, indices: List[int]) -> float:
        r = 1.0
        for idx in indices:
            r *= self.availability[idx]
        return r

    def parallel_reliability(self, indices: List[int]) -> float:
        r = 1.0
        for idx in indices:
            r *= (1.0 - self.availability[idx])
        return 1.0 - r

    def expected_energy_not_supplied(self, load_mw: np.ndarray,
                                     path_indices: List[List[int]]) -> float:
        eens = 0.0
        for k, path in enumerate(path_indices):
            prob_fail = 1.0 - self.series_reliability(path)
            eens += prob_fail * load_mw[k]
        return eens


class VoltageStabilityMargin:

    def __init__(self, E: float, X: float, Q: float):
        self.E = E
        self.X = X
        self.Q = Q

    def voltage_solution(self, P: float) -> Tuple[float, float]:
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
        val = (self.E**2 - 2.0 * self.Q * self.X)**2 / (4.0 * self.X**2) - self.Q**2
        return np.sqrt(max(val, 0.0))

    def voltage_margin(self, P_operating: float) -> float:
        p_max = self.max_power_limit()
        if p_max < 1e-12:
            return 0.0
        return max(0.0, (p_max - P_operating) / p_max)

    def pv_curve(self, n_points: int = 50) -> dict:
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

    def __init__(self, Va: complex, Vb: complex, Vc: complex):
        self.Va = Va
        self.Vb = Vb
        self.Vc = Vc

    def symmetric_components(self) -> Tuple[complex, complex, complex]:
        alpha = np.exp(1j * 2.0 * np.pi / 3.0)
        V0 = (self.Va + self.Vb + self.Vc) / 3.0
        V1 = (self.Va + alpha * self.Vb + alpha**2 * self.Vc) / 3.0
        V2 = (self.Va + alpha**2 * self.Vb + alpha * self.Vc) / 3.0
        return V0, V1, V2

    def unbalance_factor(self) -> float:
        _, V1, V2 = self.symmetric_components()
        if abs(V1) < 1e-12:
            return 100.0
        return abs(V2) / abs(V1) * 100.0

    def phasor_triangle_analysis(self) -> dict:
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
