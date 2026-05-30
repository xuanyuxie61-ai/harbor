
import numpy as np
from seafloor_geometry import point_to_triangle_distance


class SoundSpeedProfile:

    def __init__(
        self,
        c0: float = 1500.0,
        g: float = 0.015,
        delta_c: float = 30.0,
        z1: float = 1000.0,
        sigma: float = 500.0
    ):
        self.c0 = float(c0)
        self.g = float(g)
        self.delta_c = float(delta_c)
        self.z1 = float(z1)
        self.sigma = float(sigma)

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if np.any(z < 0.0):

            z = np.where(z < 0.0, 0.0, z)
        c = self.c0 + self.g * z + self.delta_c * np.exp(-((z - self.z1) ** 2) / (2.0 * self.sigma ** 2))
        return c

    def gradient(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        z = np.where(z < 0.0, 0.0, z)
        dz = z - self.z1
        exp_term = np.exp(-(dz ** 2) / (2.0 * self.sigma ** 2))
        grad = self.g - self.delta_c * dz / (self.sigma ** 2) * exp_term
        return grad


class BrentZeroRC:

    def __init__(self, a: float, b: float, tol: float = 1e-10):
        self.a = float(a)
        self.b = float(b)
        self.tol = float(tol)
        self._status = 0
        self._arg = 0.0

        self._c = 0.0
        self._d = 0.0
        self._e = 0.0
        self._fa = 0.0
        self._fb = 0.0
        self._fc = 0.0
        self._sa = 0.0
        self._sb = 0.0

    def start(self) -> float:
        self._status = 1
        self._sa = self.a
        self._sb = self.b
        self._e = self._sb - self._sa
        self._d = self._e
        self._arg = self._sa
        return self._arg

    def iterate(self, value: float) -> tuple:
        if self._status == 1:
            self._fa = value
            self._status = 2
            self._arg = self._sb
            return self._status, self._arg

        if self._status == 2:
            self._fb = value
            if self._fa * self._fb > 0.0:
                self._status = -1
                return self._status, self._arg
            self._c = self._sa
            self._fc = self._fa
        else:
            self._fb = value
            if (self._fb > 0.0 and self._fc > 0.0) or (self._fb <= 0.0 and self._fc <= 0.0):
                self._c = self._sa
                self._fc = self._fa
                self._e = self._sb - self._sa
                self._d = self._e


        if abs(self._fc) < abs(self._fb):
            self._sa = self._sb
            self._sb = self._c
            self._c = self._sa
            self._fa = self._fb
            self._fb = self._fc
            self._fc = self._fa

        tol = 2.0 * np.finfo(float).eps * abs(self._sb) + self.tol
        m = 0.5 * (self._c - self._sb)

        if abs(m) <= tol or self._fb == 0.0:
            self._status = 0
            self._arg = self._sb
            return self._status, self._arg

        if abs(self._e) < tol or abs(self._fa) <= abs(self._fb):
            self._e = m
            self._d = self._e
        else:
            s = self._fb / self._fa
            if self._sa == self._c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = self._fa / self._fc
                r = self._fb / self._fc
                p = s * (2.0 * m * q * (q - r) - (self._sb - self._sa) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            s = self._e
            self._e = self._d

            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                self._d = p / q
            else:
                self._e = m
                self._d = self._e

        self._sa = self._sb
        self._fa = self._fb

        if abs(self._d) > tol:
            self._sb = self._sb + self._d
        elif m > 0.0:
            self._sb = self._sb + tol
        else:
            self._sb = self._sb - tol

        self._arg = self._sb
        self._status += 1
        return self._status, self._arg

    def solve(self, func) -> float:
        arg = self.start()
        max_iter = 100
        for _ in range(max_iter):
            val = func(arg)
            status, arg = self.iterate(val)
            if status <= 0:
                break
        return arg


class AcousticRayTracer:

    def __init__(self, ssp: SoundSpeedProfile):
        self.ssp = ssp

    def trace_ray(
        self,
        x0: float,
        z0: float,
        theta0_deg: float,
        z_bottom_func,
        dt: float = 0.01,
        max_steps: int = 50000
    ) -> dict:
        theta = np.radians(float(theta0_deg))
        x = float(x0)
        z = float(z0)

        traj_x = [x]
        traj_z = [z]
        travel_time = 0.0


        c0 = float(self.ssp.evaluate(np.array([z]))[0])
        ray_param = np.sin(theta) / c0

        for step in range(max_steps):

            k1_x, k1_z, k1_th = self._ray_derivatives(x, z, theta)
            k2_x, k2_z, k2_th = self._ray_derivatives(
                x + 0.5 * dt * k1_x, z + 0.5 * dt * k1_z, theta + 0.5 * dt * k1_th
            )
            k3_x, k3_z, k3_th = self._ray_derivatives(
                x + 0.5 * dt * k2_x, z + 0.5 * dt * k2_z, theta + 0.5 * dt * k2_th
            )
            k4_x, k4_z, k4_th = self._ray_derivatives(
                x + dt * k3_x, z + dt * k3_z, theta + dt * k3_th
            )

            dx = dt / 6.0 * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            dz = dt / 6.0 * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
            dth = dt / 6.0 * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)

            x_new = x + dx
            z_new = z + dz
            theta_new = theta + dth


            if z_new < 0.0:
                z_new = 0.0
                theta_new = -theta_new

            travel_time += dt


            z_bot = float(z_bottom_func(x_new))
            if z_new >= z_bot:

                try:
                    x_hit = self._find_intersection_brent(x, z, x_new, z_new, z_bottom_func)
                    z_hit = float(z_bottom_func(x_hit))

                    ratio = (x_hit - x) / (x_new - x + 1e-15)
                    t_hit = travel_time - dt + ratio * dt
                    traj_x.append(x_hit)
                    traj_z.append(z_hit)
                    return {
                        'hit': True,
                        'x_hit': x_hit,
                        'z_hit': z_hit,
                        'travel_time': t_hit,
                        'ray_param': ray_param,
                        'trajectory_x': np.array(traj_x),
                        'trajectory_z': np.array(traj_z),
                        'n_steps': step + 1,
                    }
                except Exception:

                    ratio = (z_bot - z) / (z_new - z + 1e-15)
                    x_hit = x + ratio * (x_new - x)
                    z_hit = z_bot
                    t_hit = travel_time - dt + ratio * dt
                    traj_x.append(x_hit)
                    traj_z.append(z_hit)
                    return {
                        'hit': True,
                        'x_hit': x_hit,
                        'z_hit': z_hit,
                        'travel_time': t_hit,
                        'ray_param': ray_param,
                        'trajectory_x': np.array(traj_x),
                        'trajectory_z': np.array(traj_z),
                        'n_steps': step + 1,
                    }

            x, z, theta = x_new, z_new, theta_new
            traj_x.append(x)
            traj_z.append(z)

        return {
            'hit': False,
            'x_hit': None,
            'z_hit': None,
            'travel_time': travel_time,
            'ray_param': ray_param,
            'trajectory_x': np.array(traj_x),
            'trajectory_z': np.array(traj_z),
            'n_steps': max_steps,
        }

    def _ray_derivatives(self, x: float, z: float, theta: float) -> tuple:








        raise NotImplementedError("Hole_1: 请实现 _ray_derivatives 方法体")

    def _find_intersection_brent(self, x1, z1, x2, z2, z_bottom_func):
        def f(x):
            z_ray = z1 + (z2 - z1) * (x - x1) / (x2 - x1 + 1e-15)
            z_bot = float(z_bottom_func(x))
            return z_ray - z_bot


        f1 = f(x1)
        f2 = f(x2)
        if f1 * f2 > 0:

            return (x1 + x2) / 2.0

        solver = BrentZeroRC(x1, x2, tol=1e-8)
        arg = solver.start()
        for _ in range(80):
            val = f(arg)
            status, arg = solver.iterate(val)
            if status <= 0:
                break
        return arg

    def compute_ttw_depth(
        self,
        ttw: float,
        theta0_deg: float,
        z_bottom_func,
        x0: float = 0.0,
        z0: float = 0.0
    ) -> float:
        result = self.trace_ray(x0, z0, theta0_deg, z_bottom_func)
        if not result['hit']:
            return -1.0
        one_way = result['travel_time']

        expected_one_way = ttw / 2.0
        if abs(one_way - expected_one_way) < 0.1:
            return result['z_hit']

        c_avg = float(self.ssp.evaluate(np.array([result['z_hit'] / 2.0]))[0])
        depth_est = expected_one_way * c_avg * np.cos(np.radians(theta0_deg))
        return float(depth_est)
