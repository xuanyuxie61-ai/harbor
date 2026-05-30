
import numpy as np
from typing import Tuple, List, Callable, Optional


class StreamingStatistics:

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = -float('inf')

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    def std(self) -> float:
        return np.sqrt(self.variance())

    def get_stats(self) -> dict:
        return {
            'n': self.n,
            'mean': self.mean,
            'variance': self.variance(),
            'std': self.std(),
            'min': self.min_val,
            'max': self.max_val
        }


def bisection_root_finder(f: Callable[[float], float],
                          a: float, b: float,
                          tol: float = 1e-8,
                          max_iter: int = 100) -> Tuple[float, int, bool]:
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        return (a + b) / 2.0, 0, False

    iterations = 0
    c = a

    while iterations < max_iter and abs(b - a) > tol:
        c = (a + b) / 2.0
        fc = f(c)

        if abs(fc) < tol:
            return c, iterations, True

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iterations += 1

    return c, iterations, True


def calibrate_beta_target(target_r0: float,
                          params_template: dict,
                          ode_solver: Callable,
                          tol: float = 1e-4) -> float:




    raise NotImplementedError("Hole_3: calibrate_beta_target 尚未实现")


def maximum_likelihood_estimation(observed_data: np.ndarray,
                                  model_func: Callable,
                                  param_bounds: List[Tuple[float, float]],
                                  n_grid: int = 50) -> Tuple[np.ndarray, float]:
    n_params = len(param_bounds)


    grids = [np.linspace(b[0], b[1], n_grid) for b in param_bounds]

    best_params = None
    best_ll = -float('inf')

    if n_params == 1:
        for p0 in grids[0]:
            try:
                pred = model_func(p0)
                residuals = observed_data - pred
                ll = -0.5 * np.sum(residuals**2)
                if ll > best_ll:
                    best_ll = ll
                    best_params = np.array([p0])
            except Exception:
                continue
    elif n_params == 2:
        for p0 in grids[0]:
            for p1 in grids[1]:
                try:
                    pred = model_func(p0, p1)
                    residuals = observed_data - pred
                    ll = -0.5 * np.sum(residuals**2)
                    if ll > best_ll:
                        best_ll = ll
                        best_params = np.array([p0, p1])
                except Exception:
                    continue

    return best_params, best_ll


def akaike_information_criterion(log_likelihood: float, k: int, n: int) -> float:
    aic = 2.0 * k - 2.0 * log_likelihood
    if n > k + 1:
        aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        return aicc
    return aic
