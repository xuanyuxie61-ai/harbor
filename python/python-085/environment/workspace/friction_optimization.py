import numpy as np
from typing import Callable, Tuple, Optional


def peaks_function(x: float, y: float) -> float:
    term1 = 3.0 * (1.0 - x) ** 2 * np.exp(-(x ** 2) - (y + 1.0) ** 2)
    term2 = -10.0 * (x / 5.0 - x ** 3 - y ** 5) * np.exp(-(x ** 2) - y ** 2)
    term3 = -(1.0 / 3.0) * np.exp(-((x + 1.0) ** 2) - y ** 2)
    return term1 + term2 + term3


def peaks_gradient(x: float, y: float) -> Tuple[float, float]:
    h = 1e-6
    fx = (peaks_function(x + h, y) - peaks_function(x - h, y)) / (2.0 * h)
    fy = (peaks_function(x, y + h) - peaks_function(x, y - h)) / (2.0 * h)
    return fx, fy


class HookeJeevesOptimizer:

    def __init__(self, rho: float = 0.5, eps: float = 1e-6, itermax: int = 500):
        self.rho = rho
        self.eps = eps
        self.itermax = itermax

    def optimize(self, f: Callable[[np.ndarray], float],
                 x0: np.ndarray) -> Tuple[np.ndarray, int, dict]:
        nvars = len(x0)
        xbefore = np.array(x0, dtype=float)
        newx = xbefore.copy()
        delta = np.array([self.rho if xi == 0.0 else self.rho * abs(xi) for xi in xbefore])
        steplength = self.rho
        iters = 0
        fbefore = f(xbefore)
        funevals = 1
        history = {"fvals": [fbefore], "xvals": [xbefore.copy()]}

        while iters < self.itermax and self.eps < steplength:
            iters += 1
            newx = xbefore.copy()
            newf, newx, funevals = self._best_nearby(delta, newx, fbefore, nvars, f, funevals)
            keep = True
            while newf < fbefore and keep:
                for i in range(nvars):
                    if newx[i] <= xbefore[i]:
                        delta[i] = -abs(delta[i])
                    else:
                        delta[i] = abs(delta[i])
                    tmp = xbefore[i]
                    xbefore[i] = newx[i]
                    newx[i] = newx[i] + newx[i] - tmp
                fbefore = newf
                newf, newx, funevals = self._best_nearby(delta, newx, fbefore, nvars, f, funevals)
                if fbefore <= newf:
                    break
                keep = False
                for i in range(nvars):
                    if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                        keep = True
                        break
            if self.eps <= steplength and fbefore <= newf:
                steplength *= self.rho
                delta *= self.rho
            history["fvals"].append(fbefore)
            history["xvals"].append(xbefore.copy())

        return xbefore, iters, history

    def _best_nearby(self, delta: np.ndarray, x: np.ndarray, fbefore: float,
                     nvars: int, f: Callable, funevals: int) -> Tuple[float, np.ndarray, int]:
        z = x.copy()
        fnow = fbefore
        for i in range(nvars):
            z[i] = x[i] + delta[i]
            ftmp = f(z)
            funevals += 1
            if ftmp < fnow:
                fnow = ftmp
            else:
                z[i] = x[i] - delta[i]
                ftmp = f(z)
                funevals += 1
                if ftmp < fnow:
                    fnow = ftmp
                else:
                    z[i] = x[i]
        return fnow, z, funevals


def friction_coefficient_calibration(
    simulated_func: Callable[[float], float],
    target_value: float,
    mu_bounds: Tuple[float, float] = (0.05, 1.0)
) -> Tuple[float, dict]:
    def objective(mu_vec: np.ndarray) -> float:
        mu = float(np.clip(mu_vec[0], mu_bounds[0], mu_bounds[1]))
        try:
            q = simulated_func(mu)
        except Exception:
            q = 1e10
        return (q - target_value) ** 2

    optimizer = HookeJeevesOptimizer(rho=0.5, eps=1e-7, itermax=200)
    mu0 = np.array([0.3])
    mu_opt, iters, hist = optimizer.optimize(objective, mu0)
    mu_opt_clipped = float(np.clip(mu_opt[0], mu_bounds[0], mu_bounds[1]))
    info = {
        "iterations": iters,
        "final_objective": hist["fvals"][-1],
        "history": hist
    }
    return mu_opt_clipped, info


def peaks_surface_contact_potential(x: np.ndarray, y: np.ndarray,
                                     amplitude: float = 1e8) -> float:
    return amplitude * peaks_function(x, y)
