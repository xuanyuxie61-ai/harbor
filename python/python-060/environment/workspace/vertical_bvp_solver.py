
import numpy as np
from typing import Tuple, Optional, Callable


class IllConditionedBVPSolver:

    def __init__(self, z_min: float = 10000.0, z_max: float = 50000.0,
                 n_points: int = 200):
        if z_min >= z_max:
            raise ValueError("z_min 必须小于 z_max")
        if n_points < 10:
            raise ValueError("网格点数必须 >= 10")

        self.z_min = z_min
        self.z_max = z_max
        self.n_points = n_points
        self.z = np.linspace(z_min, z_max, n_points)
        self.dz = self.z[1] - self.z[0]

    def _coefficient_V(self, z: np.ndarray) -> np.ndarray:
        z_km = z / 1000.0
        w = 0.001 * np.sin(np.pi * (z_km - 10.0) / 40.0)
        w = np.clip(w, -0.01, 0.01)
        return w

    def _coefficient_kp(self, z: np.ndarray, T: np.ndarray) -> np.ndarray:
        kp = 1e-5 * np.exp(-z / 15000.0)
        return np.clip(kp, 1e-10, 1e-2)

    def _coefficient_kl(self, z: np.ndarray, T: np.ndarray) -> np.ndarray:
        kl = 1e-4 * (1.0 + 0.5 * np.exp(-(z - 25000.0) ** 2 / 5e7))
        return np.clip(kl, 1e-8, 1e-2)

    def _reference_concentration(self, z: np.ndarray) -> np.ndarray:
        z_km = z / 1000.0
        o3_max = 5e12
        o3 = o3_max * np.exp(-((z_km - 25.0) / 8.0) ** 2)
        return np.clip(o3, 1e8, 1e15)

    def solve_finite_difference(self, epsilon: float = 1e-4,
                                 T_profile: Optional[np.ndarray] = None,
                                 max_iter: int = 100,
                                 tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        if epsilon <= 0:
            raise ValueError("epsilon 必须为正")

        z = self.z
        nz = len(z)
        dz = self.dz

        if T_profile is None:
            T = 220.0 * np.ones(nz)
        else:
            T = T_profile
            if len(T) != nz:
                raise ValueError("T_profile 长度与网格不匹配")

        V = self._coefficient_V(z)
        kp = self._coefficient_kp(z, T)
        kl = self._coefficient_kl(z, T)


        y_bottom = self._reference_concentration(z)[0]





        a = np.zeros(nz)
        b = np.zeros(nz)
        c = np.zeros(nz)
        d = np.zeros(nz)


        for i in range(1, nz - 1):
            a[i] = epsilon / dz ** 2 + V[i] / (2.0 * dz)
            b[i] = -2.0 * epsilon / dz ** 2 - kl[i]
            c[i] = epsilon / dz ** 2 - V[i] / (2.0 * dz)
            d[i] = -kp[i]


        b[0] = 1.0
        c[0] = 0.0
        d[0] = y_bottom



        a[-1] = -1.0
        b[-1] = 1.0
        c[-1] = 0.0
        d[-1] = 0.0


        y = self._thomas_algorithm(a, b, c, d)


        epsilon_current = max(epsilon, 1e-2)
        epsilon_target = epsilon
        y_prev = y.copy()

        while epsilon_current > epsilon_target * 0.99:
            epsilon_current = max(epsilon_current * 0.5, epsilon_target)

            for _ in range(max_iter):

                for i in range(1, nz - 1):
                    a[i] = epsilon_current / dz ** 2 + V[i] / (2.0 * dz)
                    b[i] = -2.0 * epsilon_current / dz ** 2 - kl[i]
                    c[i] = epsilon_current / dz ** 2 - V[i] / (2.0 * dz)
                    d[i] = -kp[i]

                y = self._thomas_algorithm(a, b, c, d)
                y = np.clip(y, 0.0, 1e20)

                err = np.max(np.abs(y - y_prev)) / (np.max(np.abs(y_prev)) + 1e-30)
                y_prev = y.copy()

                if err < tol:
                    break

        return z, y

    def _thomas_algorithm(self, a: np.ndarray, b: np.ndarray,
                          c: np.ndarray, d: np.ndarray) -> np.ndarray:
        nz = len(b)
        cp = np.zeros(nz)
        dp = np.zeros(nz)
        y = np.zeros(nz)


        cp[0] = c[0] / (b[0] + 1e-30)
        dp[0] = d[0] / (b[0] + 1e-30)

        for i in range(1, nz):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < 1e-30:
                denom = 1e-30 * np.sign(denom) if denom != 0 else 1e-30
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom


        y[-1] = dp[-1]
        for i in range(nz - 2, -1, -1):
            y[i] = dp[i] - cp[i] * y[i + 1]

        return y

    def solve_shooting(self, epsilon: float = 1e-4,
                       T_profile: Optional[np.ndarray] = None,
                       n_subintervals: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        z = self.z
        nz = len(z)
        dz = self.dz

        if T_profile is None:
            T = 220.0 * np.ones(nz)
        else:
            T = T_profile

        V = self._coefficient_V(z)
        kp = self._coefficient_kp(z, T)
        kl = self._coefficient_kl(z, T)

        y_bottom = self._reference_concentration(z)[0]


        n_sub = max(n_subintervals, 2)
        idx_sub = np.array_split(np.arange(nz), n_sub)



        z, y = self.solve_finite_difference(epsilon, T)


        for sub_idx in idx_sub:
            if len(sub_idx) < 3:
                continue
            i_start = sub_idx[0]
            i_end = sub_idx[-1]

            if i_start > 0 and i_end < nz - 1:
                dy = np.gradient(y[sub_idx], dz)
                d2y = np.gradient(dy, dz)

                resid = epsilon * d2y - V[sub_idx] * dy + kp[sub_idx] - kl[sub_idx] * y[sub_idx]
                max_resid = np.max(np.abs(resid))
                if max_resid > 1e-3:

                    y[sub_idx] = self._local_smooth(y[sub_idx])

        return z, np.clip(y, 0.0, 1e20)

    def _local_smooth(self, y: np.ndarray, n_iter: int = 3) -> np.ndarray:
        y_smooth = y.copy()
        for _ in range(n_iter):
            y_new = y_smooth.copy()
            for i in range(1, len(y_smooth) - 1):
                y_new[i] = 0.25 * y_smooth[i - 1] + 0.5 * y_smooth[i] + 0.25 * y_smooth[i + 1]
            y_smooth = y_new
        return y_smooth

    def compute_ozone_layer_thickness(self, y: np.ndarray,
                                       threshold: float = 1e11) -> float:
        mask = y > threshold
        if not np.any(mask):
            return 0.0
        z_low = np.min(self.z[mask])
        z_high = np.max(self.z[mask])
        return (z_high - z_low) / 1000.0

    def boundary_layer_analysis(self, z: np.ndarray, y: np.ndarray,
                                 epsilon: float) -> dict:

        L = (self.z_max - self.z_min) / 1000.0
        delta = np.sqrt(epsilon) * L


        dy = np.abs(np.gradient(y, z))
        i_max = np.argmax(dy)
        z_bl = z[i_max] / 1000.0

        return {
            'boundary_layer_thickness_km': delta,
            'boundary_layer_position_km': z_bl,
            'max_gradient': dy[i_max],
            'condition_number_estimate': 1.0 / epsilon
        }
