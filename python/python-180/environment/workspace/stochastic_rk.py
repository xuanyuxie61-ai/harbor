
import numpy as np
from typing import Callable, Tuple, Optional


def sde_euler_maruyama_step(y: np.ndarray,
                            f: Callable[[np.ndarray], np.ndarray],
                            g: Callable[[np.ndarray], np.ndarray],
                            h: float,
                            dW: np.ndarray) -> np.ndarray:
    if h <= 0:
        raise ValueError("Step size h must be positive")
    return y + f(y) * h + g(y) * dW


def sde_srk_platen_step(y: np.ndarray,
                        f: Callable[[np.ndarray], np.ndarray],
                        g: Callable[[np.ndarray], np.ndarray],
                        h: float,
                        dW: np.ndarray) -> np.ndarray:
    if h <= 0:
        raise ValueError("Step size h must be positive")
    sqrt_h = np.sqrt(h)

    if sqrt_h < 1e-14:
        return y + f(y) * h + g(y) * dW

    fy = f(y)
    gy = g(y)
    H1 = y + fy * h + gy * sqrt_h
    H2 = y + fy * h - gy * sqrt_h
    gH1 = g(H1)
    gH2 = g(H2)

    y_new = y + fy * h + 0.5 * (gH1 + gH2) * dW
    y_new += 0.5 * (gH1 - gH2) * (dW ** 2 - h) / sqrt_h
    return y_new


def sde_milstein_step(y: np.ndarray,
                      f: Callable[[np.ndarray], np.ndarray],
                      g: Callable[[np.ndarray], np.ndarray],
                      dg: Callable[[np.ndarray], np.ndarray],
                      h: float,
                      dW: np.ndarray) -> np.ndarray:
    if h <= 0:
        raise ValueError("Step size h must be positive")

    pass


def stiff_sde_semiimplicit_step(y: np.ndarray,
                                f_lin: np.ndarray,
                                f_nonlin: Callable[[np.ndarray], np.ndarray],
                                g: Callable[[np.ndarray], np.ndarray],
                                h: float,
                                dW: np.ndarray) -> np.ndarray:
    if h <= 0:
        raise ValueError("Step size h must be positive")
    n = len(y)
    I = np.eye(n, dtype=np.float64)
    lhs = I - h * f_lin
    rhs = y + h * f_nonlin(y) + g(y) * dW


    cond_est = np.linalg.cond(lhs)
    if cond_est > 1e14:
        y_new = np.linalg.lstsq(lhs, rhs, rcond=1e-14)[0]
    else:
        y_new = np.linalg.solve(lhs, rhs)
    return y_new


def adaptive_rk12_sde_step(y: np.ndarray,
                           f: Callable[[np.ndarray], np.ndarray],
                           g: Callable[[np.ndarray], np.ndarray],
                           h: float,
                           dW: np.ndarray,
                           tol: float = 1e-4) -> Tuple[np.ndarray, float, bool]:
    if h <= 0:
        raise ValueError("Step size h must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    fy = f(y)
    gy = g(y)
    Y1 = y + fy * h + gy * dW

    fY1 = f(Y1)
    gY1 = g(Y1)
    Y2 = y + 0.5 * (fy + fY1) * h + 0.5 * (gy + gY1) * dW

    err = np.linalg.norm(Y2 - Y1) / (np.linalg.norm(y) + 1e-12)

    factor = 0.9 * np.sqrt(tol / (err + 1e-16))
    factor = min(5.0, max(0.2, factor))
    h_new = h * factor

    accepted = err <= tol
    return Y2, h_new, accepted


class StochasticIntegrator:

    METHODS = ["em", "srk_platen", "milstein", "semiimplicit", "adaptive_rk12"]

    def __init__(self,
                 method: str = "srk_platen",
                 dt: float = 1e-3,
                 tol: float = 1e-4,
                 f_lin: Optional[np.ndarray] = None):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method {method}, must be one of {self.METHODS}")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.method = method
        self.dt = dt
        self.tol = tol
        self.f_lin = f_lin

    def step(self,
             y: np.ndarray,
             f: Callable[[np.ndarray], np.ndarray],
             g: Callable[[np.ndarray], np.ndarray],
             dW: np.ndarray,
             dg: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Tuple[np.ndarray, float]:
        if self.method == "em":
            return sde_euler_maruyama_step(y, f, g, self.dt, dW), self.dt
        elif self.method == "srk_platen":
            return sde_srk_platen_step(y, f, g, self.dt, dW), self.dt
        elif self.method == "milstein":
            if dg is None:
                raise ValueError("Milstein method requires dg")
            return sde_milstein_step(y, f, g, dg, self.dt, dW), self.dt
        elif self.method == "semiimplicit":
            if self.f_lin is None:
                raise ValueError("semiimplicit method requires f_lin matrix")
            return stiff_sde_semiimplicit_step(y, self.f_lin, f, g, self.dt, dW), self.dt
        elif self.method == "adaptive_rk12":
            y_new, h_new, accepted = adaptive_rk12_sde_step(y, f, g, self.dt, dW, self.tol)
            self.dt = h_new
            return y_new, h_new
        else:
            raise RuntimeError("Unreachable")
