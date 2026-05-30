
import numpy as np
from typing import Callable, Tuple, Optional, List
from scipy.optimize import least_squares





def golden_section_search(f: Callable[[float], float],
                          a: float, b: float,
                          max_iter: int = 100,
                          x_tol: float = 1e-7) -> Tuple[float, float, int, int]:
    if a >= b:
        raise ValueError("Must have a < b")

    g = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = g * a + (1.0 - g) * b
    x2 = (1.0 - g) * a + g * b
    f1 = f(x1)
    f2 = f(x2)
    nf = 2

    for it in range(max_iter):
        if abs(b - a) <= x_tol:
            return a, b, it, nf

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = g * a + (1.0 - g) * b
            f1 = f(x1)
            nf += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - g) * a + g * b
            f2 = f(x2)
            nf += 1

    return a, b, max_iter, nf





def gradient_descent(fp: Callable[[float], float], x0: float,
                     gamma: float = 0.01,
                     precision: float = 1e-7,
                     max_iter: int = 10000) -> Tuple[float, int]:
    x = x0
    for it in range(1, max_iter + 1):
        x_old = x
        grad = fp(x_old)
        if not np.isfinite(grad):
            raise RuntimeError(f"Non-finite gradient at x={x_old}")
        x = x_old - gamma * grad
        if abs(x - x_old) <= precision:
            return x, it
    return x, max_iter





class ParameterIdentification:

    def __init__(self, forward_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 measured_data: np.ndarray,
                 measurement_points: np.ndarray,
                 param_bounds: Optional[List[Tuple[float, float]]] = None):
        self.forward_model = forward_model
        self.measured_data = np.asarray(measured_data)
        self.measurement_points = np.asarray(measurement_points)
        self.param_bounds = param_bounds
        self.n_params = len(param_bounds) if param_bounds else 3

    def residual(self, params: np.ndarray) -> np.ndarray:
        pred = self.forward_model(params, self.measurement_points)
        return pred - self.measured_data

    def objective(self, params: np.ndarray) -> float:
        r = self.residual(params)
        return 0.5 * np.dot(r, r)

    def jacobian_finite_difference(self, params: np.ndarray,
                                   h: float = 1e-7) -> np.ndarray:
        n = len(params)
        m = len(self.measured_data)
        J = np.zeros((m, n))
        r0 = self.residual(params)

        for j in range(n):
            params_plus = params.copy()
            params_plus[j] += h
            r_plus = self.residual(params_plus)
            J[:, j] = (r_plus - r0) / h

        return J

    def optimize(self, x0: np.ndarray, method: str = 'lm',
                 max_iter: int = 100) -> dict:
        x0 = np.asarray(x0)
        if len(x0) != self.n_params:
            raise ValueError(f"Initial guess length {len(x0)} != n_params {self.n_params}")

        bounds = None
        if self.param_bounds is not None:
            lb = [b[0] for b in self.param_bounds]
            ub = [b[1] for b in self.param_bounds]
            bounds = (lb, ub)


        actual_method = method
        if bounds is not None and method == 'lm':
            actual_method = 'trf'

        result = least_squares(
            self.residual, x0, method=actual_method,
            max_nfev=max_iter * len(x0) * 10,
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            bounds=bounds
        )

        return {
            'params': result.x,
            'cost': result.cost,
            'nfev': result.nfev,
            'njev': result.njev,
            'success': result.success,
            'message': result.message,
            'jacobian': result.jac
        }

    def optimize_golden_section_1d(self, idx: int, fixed_params: np.ndarray,
                                   bracket: Tuple[float, float]) -> Tuple[float, float]:
        def f_1d(x: float) -> float:
            p = fixed_params.copy()
            p[idx] = x
            return self.objective(p)

        a, b, it, nf = golden_section_search(f_1d, bracket[0], bracket[1])
        x_opt = (a + b) / 2.0
        f_opt = f_1d(x_opt)
        return x_opt, f_opt





def bone_remodeling_forward_model(params: np.ndarray,
                                  x_points: np.ndarray,
                                  U_field: Optional[np.ndarray] = None,
                                  t_final: float = 365.0) -> np.ndarray:
    if len(params) < 3:
        raise ValueError("params must contain at least [k_form, k_res, U_ref]")

    k_form, k_res, U_ref = params[0], params[1], params[2]

    if U_ref <= 0:
        return np.full(len(x_points), 0.01)

    if U_field is None:

        U_field = 1.0 * np.exp(-0.01 * (x_points - np.mean(x_points)) ** 2)


    ratio = k_form / max(k_res, 1e-10)
    rho = np.zeros(len(x_points))
    for i, U in enumerate(U_field):
        if U > U_ref:
            rho[i] = min(1.8, 0.5 + ratio * (U - U_ref))
        else:
            rho[i] = max(0.01, 0.5 - (U_ref - U) / U_ref)

    return rho
