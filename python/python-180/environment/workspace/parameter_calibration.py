
import numpy as np
from typing import Callable, Tuple, Optional
from numerical_utils import r8_hypot


class PraxisOptimizer:

    def __init__(self,
                 tol: float = 1e-8,
                 max_iter: int = 500,
                 h0: float = 1.0):
        self.tol = tol
        self.max_iter = max_iter
        self.h0 = h0

    def _line_minimize(self,
                       f: Callable[[np.ndarray], float],
                       x: np.ndarray,
                       d: np.ndarray,
                       f_x: float) -> Tuple[np.ndarray, float, float]:
        alpha = 0.0
        fa = f_x

        h = self.h0 / (np.linalg.norm(d) + 1e-12)
        fb = f(x + h * d)

        if fb > fa:
            h = -h
            fb = f(x + h * d)
            if fb > fa:
                return x, fa, 0.0


        alpha_c = 2.0 * h
        fc = f(x + alpha_c * d)

        denom = 2.0 * ((fa - fc) / h - (fa - fb) / h)
        if abs(denom) < 1e-16:
            alpha_star = h if fb < fa else 0.0
        else:
            alpha_star = h - ((fa - fb) / h) / denom

            alpha_star = max(min(alpha_star, 2.0 * h), -abs(h))
        x_new = x + alpha_star * d
        f_new = f(x_new)
        if f_new < fa:
            return x_new, f_new, alpha_star
        elif fb < fa:
            return x + h * d, fb, h
        else:
            return x, fa, 0.0

    def minimize(self,
                 f: Callable[[np.ndarray], float],
                 x0: np.ndarray) -> Tuple[np.ndarray, float, int]:
        x = x0.copy().astype(np.float64)
        n = len(x)
        if n < 1:
            raise ValueError("x0 must have length >= 1")


        V = np.eye(n, dtype=np.float64)
        fx = f(x)

        for iteration in range(self.max_iter):
            x_old = x.copy()

            for i in range(n):
                d = V[:, i]
                x, fx, _ = self._line_minimize(f, x, d, fx)


            s = x - x_old
            if np.linalg.norm(s) < self.tol:
                break


            s_norm = np.linalg.norm(s)
            if s_norm > 1e-14:
                d_new = s / s_norm
                x, fx, _ = self._line_minimize(f, x, d_new, fx)

                V[:, 1:] = V[:, :-1]
                V[:, 0] = d_new


            if (iteration + 1) % 5 == 0 and n > 1:
                try:
                    U_svd, S_svd, _ = np.linalg.svd(V)

                    sort_idx = np.argsort(-S_svd)
                    V = U_svd[:, sort_idx]
                except np.linalg.LinAlgError:
                    pass

            if np.linalg.norm(x - x_old) < self.tol + np.sqrt(np.finfo(float).eps) * np.linalg.norm(x_old):
                break

        return x, fx, iteration + 1


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("Rosenbrock requires n >= 2")
    a = x[1:] - x[:-1] ** 2
    b = 1.0 - x[:-1]
    return float(np.sum(100.0 * a ** 2 + b ** 2))


def camel_back(x: np.ndarray) -> float:
    if len(x) != 2:
        raise ValueError("Camel back is 2D only")
    xx, yy = x[0], x[1]
    return float(2.0 * xx ** 2 - 1.05 * xx ** 4 + xx ** 6 / 6.0 + xx * yy + yy ** 2)


class SPDEParameterCalibration:

    def __init__(self,
                 reference_data: np.ndarray,
                 spatial_grid: np.ndarray,
                 solver_factory: Callable,
                 param_bounds: Optional[np.ndarray] = None):
        self.reference = reference_data
        self.spatial_grid = spatial_grid
        self.solver_factory = solver_factory
        self.param_bounds = param_bounds

    def misfit(self, theta: np.ndarray) -> float:
        theta = np.clip(theta, 1e-6, 10.0)
        try:
            u_pred = self.solver_factory(theta)
            diff = u_pred - self.reference
            mse = 0.5 * np.mean(diff ** 2)

            reg = 1e-4 * np.sum(theta ** 2)
            return float(mse + reg)
        except Exception:
            return 1e6

    def calibrate(self, theta0: np.ndarray) -> Tuple[np.ndarray, float]:
        opt = PraxisOptimizer(tol=1e-6, max_iter=100, h0=0.1)
        theta_opt, f_opt, _ = opt.minimize(self.misfit, theta0)
        if self.param_bounds is not None:
            theta_opt = np.clip(theta_opt, self.param_bounds[:, 0], self.param_bounds[:, 1])
        return theta_opt, f_opt
