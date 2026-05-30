
import numpy as np
from typing import Callable, Optional, Tuple, List


class GyroscopeDynamics:
    def __init__(self, A1: float = 1.0, A2: float = 1.0, A3: float = 3.0,
                 m_ext: float = 1.0):
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.m_ext = m_ext
        self.y0 = np.array([0.25, 0.4, 0.1, 1.0, 2.0, 3.0])
        self.t0 = 0.0
        self.tstop = 5.0

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        psi, theta, phi, omega1, omega2, omega3 = y

        M1 = -self.m_ext * self.A1 * np.sin(theta) * np.cos(phi)
        M2 = self.m_ext * self.A2 * np.sin(theta) * np.sin(phi)
        M3 = 0.0



        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-10:
            sin_theta = np.sign(sin_theta) * 1e-10

        dpsi = (omega1 * np.sin(phi) + omega2 * np.cos(phi)) / sin_theta
        dtheta = omega1 * np.cos(phi) - omega2 * np.sin(phi)
        dphi = omega3 - np.cos(theta) * dpsi


        domega1 = ((self.A2 - self.A3) * omega2 * omega3 + M1) / self.A1
        domega2 = ((self.A3 - self.A1) * omega3 * omega1 + M2) / self.A2
        domega3 = ((self.A1 - self.A2) * omega1 * omega2 + M3) / self.A3

        return np.array([dpsi, dtheta, dphi, domega1, domega2, domega3])

    def simulate(self, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int((self.tstop - self.t0) / dt) + 1
        t_vals = np.linspace(self.t0, self.tstop, n_steps)
        y_vals = np.zeros((n_steps, 6))
        y_vals[0] = self.y0
        for i in range(n_steps - 1):
            y = y_vals[i]
            t = t_vals[i]
            k1 = self.deriv(t, y)
            k2 = self.deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.deriv(t + dt, y + dt * k3)
            y_vals[i + 1] = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return t_vals, y_vals


class HermiteCubicSpline:
    def __init__(self, xn: np.ndarray, fn: np.ndarray, dn: np.ndarray):
        self.xn = np.asarray(xn, dtype=float).reshape(-1)
        self.fn = np.asarray(fn, dtype=float).reshape(-1)
        self.dn = np.asarray(dn, dtype=float).reshape(-1)
        self.nn = self.xn.shape[0]
        if self.nn < 2:
            raise ValueError("至少需要2个节点")
        if not np.all(np.diff(self.xn) > 0):
            raise ValueError("节点必须严格递增")
        if self.fn.shape[0] != self.nn or self.dn.shape[0] != self.nn:
            raise ValueError("节点、函数值、导数值维度不匹配")

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        n = x.shape[0]
        f = np.zeros(n)
        d = np.zeros(n)
        s = np.zeros(n)
        t = np.zeros(n)



        x_clipped = np.clip(x, self.xn[0], self.xn[-1])

        idx = np.searchsorted(self.xn, x_clipped, side='right') - 1
        idx = np.clip(idx, 0, self.nn - 2)

        for i in range(n):
            j = idx[i]
            x1 = self.xn[j]
            x2 = self.xn[j + 1]
            f1 = self.fn[j]
            f2 = self.fn[j + 1]
            d1 = self.dn[j]
            d2 = self.dn[j + 1]

            h = x2 - x1
            if abs(h) < 1e-14:
                h = 1e-14
            df = (f2 - f1) / h
            c2 = -(2.0 * d1 - 3.0 * df + d2) / h
            c3 = (d1 - 2.0 * df + d2) / (h * h)

            dx = x[i] - x1
            f[i] = f1 + dx * (d1 + dx * (c2 + dx * c3))
            d[i] = d1 + dx * (2.0 * c2 + dx * 3.0 * c3)
            s[i] = 2.0 * c2 + dx * 6.0 * c3
            t[i] = 6.0 * c3
        return f, d, s, t

    def integrate(self, a: float, b: float) -> float:
        a = max(a, self.xn[0])
        b = min(b, self.xn[-1])
        if a >= b:
            return 0.0
        total = 0.0

        for j in range(self.nn - 1):
            x1 = self.xn[j]
            x2 = self.xn[j + 1]
            seg_a = max(a, x1)
            seg_b = min(b, x2)
            if seg_a >= seg_b:
                continue
            f1 = self.fn[j]
            f2 = self.fn[j + 1]
            d1 = self.dn[j]
            d2 = self.dn[j + 1]
            h_full = x2 - x1
            if abs(h_full) < 1e-14:
                continue
            df = (f2 - f1) / h_full
            c2 = -(2.0 * d1 - 3.0 * df + d2) / h_full
            c3 = (d1 - 2.0 * df + d2) / (h_full * h_full)


            def _F(z):
                dz = z - x1
                return f1 * dz + d1 * dz ** 2 / 2.0 + c2 * dz ** 3 / 3.0 + c3 * dz ** 4 / 4.0

            total += _F(seg_b) - _F(seg_a)
        return total


class GeodesicVQEOptimizer:
    def __init__(self, max_iter: int = 200, tol: float = 1e-6,
                 learning_rate: float = 0.05, momentum: float = 0.9):
        self.max_iter = max_iter
        self.tol = tol
        self.lr = learning_rate
        self.momentum = momentum
        self.history: List[float] = []
        self.param_history: List[np.ndarray] = []

    def optimize(self, energy_func: Callable[[np.ndarray], float],
                 gradient_func: Callable[[np.ndarray], np.ndarray],
                 x0: np.ndarray) -> Tuple[np.ndarray, float]:
        x = np.array(x0, dtype=float)
        v = np.zeros_like(x)
        best_x = x.copy()
        best_E = energy_func(x)
        self.history = [best_E]
        self.param_history = [x.copy()]

        for it in range(self.max_iter):
            E = energy_func(x)
            g = gradient_func(x)
            g_norm = np.linalg.norm(g)
            if g_norm < self.tol:
                break


            v = self.momentum * v - self.lr * g
            x_new = x + v


            if it > 0 and it % 10 == 0 and len(self.param_history) >= 3:

                xs = np.array([np.linalg.norm(p) for p in self.param_history[-3:]])
                Es = np.array(self.history[-3:])

                ds = np.zeros(3)
                for i in range(1, 2):
                    dx = xs[i] - xs[i - 1]
                    if abs(dx) > 1e-10:
                        ds[i - 1] = (Es[i] - Es[i - 1]) / dx
                ds[2] = ds[1]
                ds[0] = ds[1]
                try:
                    spline = HermiteCubicSpline(xs, Es, ds)

                    test_xs = np.linspace(xs[0], xs[-1] + 2.0 * self.lr * g_norm, 20)
                    fs, _, _, _ = spline.evaluate(test_xs)
                    min_idx = int(np.argmin(fs))
                    scale = test_xs[min_idx] / (np.linalg.norm(x) + 1e-10)
                    if 0.5 < scale < 2.0:
                        x_new = x * scale
                except ValueError:
                    pass


            x_new = np.clip(x_new, -4 * np.pi, 4 * np.pi)

            E_new = energy_func(x_new)
            if E_new < best_E:
                best_E = E_new
                best_x = x_new.copy()

            x = x_new
            self.history.append(E_new)
            self.param_history.append(x.copy())


            if it > 20 and abs(self.history[-1] - self.history[-20]) < self.tol * 10:
                break

        return best_x, best_E

    def get_convergence_rate(self) -> float:
        if len(self.history) < 20:
            return 0.0
        E_end = self.history[-1]
        errs = np.abs(np.array(self.history[-20:]) - E_end) + 1e-14
        rates = np.log(errs[1:] / errs[:-1])
        return float(np.mean(rates))
