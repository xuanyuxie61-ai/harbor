
import numpy as np
from typing import Callable, Optional, Tuple
from numerical_utils import pwl_approx_1d


class WindTurbine:

    def __init__(self, D: float = 126.0, hub_height: float = 90.0,
                 rated_power: float = 5.0, u_cut_in: float = 3.0,
                 u_rated: float = 12.0, u_cut_out: float = 25.0,
                 cp_max: float = 0.45, ct_at_rated: float = 0.8):
        if D <= 0:
            raise ValueError("转子直径必须为正")
        if hub_height <= 0:
            raise ValueError("轮毂高度必须为正")
        if not (0 < u_cut_in < u_rated < u_cut_out):
            raise ValueError("风速参数必须满足 u_cut_in < u_rated < u_cut_out")
        if cp_max <= 0 or cp_max > 16.0 / 27.0:
            raise ValueError(f"功率系数必须在 (0, 16/27] 范围内，给定 {cp_max}")
        if ct_at_rated <= 0 or ct_at_rated >= 2.0:
            raise ValueError("推力系数必须在 (0, 2) 范围内")

        self.D = D
        self.R = D / 2.0
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.u_cut_in = u_cut_in
        self.u_rated = u_rated
        self.u_cut_out = u_cut_out
        self.cp_max = cp_max
        self.ct_at_rated = ct_at_rated
        self.rho = 1.225


        self.swept_area = np.pi * self.R ** 2


        self._build_pwl_power_curve()

    def _build_pwl_power_curve(self, n_control: int = 10):

        nd = 50
        xd = np.linspace(self.u_cut_in, self.u_rated, nd)


        ratio = (xd - self.u_cut_in) / (self.u_rated - self.u_cut_in)
        cp_curve = self.cp_max * np.sin(np.pi / 2.0 * ratio) ** 2
        yd = 0.5 * self.rho * self.swept_area * cp_curve * xd**3 / 1e6


        yd = np.minimum(yd, self.rated_power)


        nc = n_control
        xc = np.linspace(self.u_cut_in, self.u_rated, nc)
        self.pwl_yc = pwl_approx_1d(nd, xd, yd, nc, xc)
        self.pwl_xc = xc


        self._power_curve_func = self._make_pwl_power_func()

    def _make_pwl_power_func(self) -> Callable[[float], float]:
        xc = self.pwl_xc
        yc = self.pwl_yc

        def power_func(u: float) -> float:
            if u < self.u_cut_in or u >= self.u_cut_out:
                return 0.0
            if u >= self.u_rated:
                return self.rated_power


            if u <= xc[0]:
                return max(0.0, yc[0])
            if u >= xc[-1]:
                return min(self.rated_power, yc[-1])

            j = np.searchsorted(xc, u, side='right') - 1
            j = max(0, min(j, len(xc) - 2))
            dx = xc[j + 1] - xc[j]
            if dx < 1e-14:
                return float(yc[j])
            t = (u - xc[j]) / dx
            val = yc[j] * (1.0 - t) + yc[j + 1] * t
            return float(max(0.0, min(self.rated_power, val)))

        return power_func

    def power(self, u: float) -> float:
        if u < 0:
            return 0.0
        return self._power_curve_func(u)

    def thrust_coefficient(self, u: float) -> float:
        if u < self.u_cut_in or u >= self.u_cut_out:
            return 0.0
        if u < self.u_rated:
            ratio = (u - self.u_cut_in) / (self.u_rated - self.u_cut_in)
            return self.ct_at_rated * ratio
        else:


            return self.ct_at_rated * (self.u_rated / u) ** 2

    def axial_induction_factor(self, u: float) -> float:

        raise NotImplementedError("Hole 2: 待修复")

    def tip_speed_ratio(self, u: float, omega_rpm: float = 15.0) -> float:
        if u < 1e-6:
            return 0.0
        omega_rad = 2.0 * np.pi * omega_rpm / 60.0
        return (omega_rad * self.R) / u

    def capacity_factor(self, wind_field) -> float:
        from numerical_utils import weibull_pdf

        n = 500
        u = np.linspace(0.0, self.u_cut_out * 1.5, n)
        du = u[1] - u[0]
        pdf = weibull_pdf(u, wind_field.A, wind_field.k)
        powers = np.array([self.power(float(ui)) for ui in u])
        aep = 8760.0 * np.trapezoid(powers * pdf, u)
        max_annual = self.rated_power * 8760.0
        if max_annual < 1e-10:
            return 0.0
        return float(aep / max_annual)


class TurbineFarm:

    def __init__(self, turbine: WindTurbine, positions: Optional[np.ndarray] = None):
        self.turbine = turbine
        self.positions = positions if positions is not None else np.zeros((0, 2))

    def add_turbine(self, x: float, y: float):
        self.positions = np.vstack([self.positions, [x, y]])

    def n_turbines(self) -> int:
        return len(self.positions)

    def pairwise_distances(self) -> np.ndarray:
        n = self.n_turbines()
        if n == 0:
            return np.zeros((0, 0))
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def min_spacing(self) -> float:
        dist = self.pairwise_distances()
        n = self.n_turbines()
        if n <= 1:
            return float('inf')

        np.fill_diagonal(dist, float('inf'))
        return float(np.min(dist))
