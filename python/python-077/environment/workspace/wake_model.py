
import numpy as np
from typing import List, Tuple, Optional


class WakeModel:

    def __init__(self, k_wake: float = 0.05, a: float = 0.293, D: float = 126.0):




        raise NotImplementedError("Hole 1-A: 待修复")

    def wake_radius(self, x: float) -> float:
        if x < 0:
            return self.R
        return self.R + self.k_wake * x

    def wake_deficit(self, x: float) -> float:
        if x <= 0:
            return 0.0
        denom = 1.0 + self.k_wake * x / self.D


        raise NotImplementedError("Hole 1-B: 待修复")

    def velocity_deficit_ratio(self, x: float, r: float) -> float:
        if x <= 0:
            return 0.0
        Rw = self.wake_radius(x)
        if r > Rw:
            return 0.0
        return self.wake_deficit(x)

    def local_velocity(self, u0: float, x: float, r: float) -> float:
        deficit = self.velocity_deficit_ratio(x, r)
        return u0 * (1.0 - deficit)

    @staticmethod
    def combine_deficits(deficits: List[float]) -> float:
        s = 0.0
        for d in deficits:
            dd = max(0.0, min(d, 1.0))
            s += dd ** 2
        return np.sqrt(s)

    def swept_area_average_deficit(self, x: float, y_offset: float = 0.0) -> float:
        if x <= 0:
            return 0.0
        Rw = self.wake_radius(x)
        d = abs(y_offset)
        R = self.R


        if d >= R + Rw:
            return 0.0

        if d <= abs(Rw - R) and Rw >= R:
            return self.wake_deficit(x)



        a1 = (-d + R + Rw)
        a2 = (d + R - Rw)
        a3 = (d - R + Rw)
        a4 = (d + R + Rw)
        if a1 <= 0 or a2 <= 0 or a3 <= 0 or a4 <= 0:
            return 0.0

        term_sqrt = 0.5 * np.sqrt(a1 * a2 * a3 * a4)


        cos1 = np.clip((d**2 + R**2 - Rw**2) / (2.0 * d * R + 1e-14), -1.0, 1.0)
        cos2 = np.clip((d**2 + Rw**2 - R**2) / (2.0 * d * Rw + 1e-14), -1.0, 1.0)

        A_overlap = R**2 * np.arccos(cos1) + Rw**2 * np.arccos(cos2) - term_sqrt
        A_swept = np.pi * R**2

        if A_swept < 1e-14:
            return 0.0


        return self.wake_deficit(x) * (A_overlap / A_swept)


class WakeFarm:

    def __init__(self, wake_model: WakeModel):
        self.wm = wake_model

    def compute_effective_velocity(self, turbines: List[Tuple[float, float]],
                                    i: int, u0: float, wind_dir: float) -> float:
        if i < 0 or i >= len(turbines):
            raise ValueError("风机索引越界")
        if u0 <= 0:
            return 0.0

        theta = np.radians(wind_dir)

        wx = np.cos(theta)
        wy = np.sin(theta)

        deficits = []
        xi, yi = turbines[i]

        for j, (xj, yj) in enumerate(turbines):
            if j == i:
                continue

            dx = xi - xj
            dy = yi - yj

            proj = dx * wx + dy * wy
            if proj <= 1e-6:
                continue



            cross = abs(dx * wy - dy * wx)

            deficit = self.wm.swept_area_average_deficit(proj, cross)
            if deficit > 1e-6:
                deficits.append(deficit)

        if not deficits:
            return u0

        total_deficit = WakeModel.combine_deficits(deficits)
        total_deficit = min(total_deficit, 0.99)
        return u0 * (1.0 - total_deficit)

    def compute_farm_power(self, turbines: List[Tuple[float, float]],
                           u0: float, wind_dir: float,
                           power_curve: callable) -> Tuple[float, List[float]]:
        powers = []
        for i in range(len(turbines)):
            u_eff = self.compute_effective_velocity(turbines, i, u0, wind_dir)
            p = power_curve(u_eff)
            powers.append(p)
        return sum(powers), powers
