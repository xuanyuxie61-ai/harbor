
import numpy as np
from typing import Callable, Tuple


def solve_ivp_rk4(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    m = len(y0)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i, :])
        k2 = f(t[i] + 0.5 * h, y[i, :] + 0.5 * h * k1)
        k3 = f(t[i] + 0.5 * h, y[i, :] + 0.5 * h * k2)
        k4 = f(t[i] + h, y[i, :] + h * k3)
        y[i + 1, :] = y[i, :] + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t, y


def shooting_method(
    epsilon: float,
    ya: float,
    yb: float,
    a: float = -1.0,
    b: float = 1.0,
    n_shoot: int = 1000,
    tol: float = 1e-8,
    max_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    def ode(t: float, Y: np.ndarray) -> np.ndarray:
        dY = np.zeros(2)
        dY[0] = Y[1]
        if abs(epsilon) < 1e-14:
            dY[1] = 0.0
        else:
            dY[1] = (t * Y[1] - Y[0]) / epsilon
        return dY


    s = (yb - ya) / (b - a)

    for it in range(max_iter):
        Y0 = np.array([ya, s])
        t, Y = solve_ivp_rk4(ode, Y0, (a, b), n_shoot)
        phi = Y[-1, 0] - yb
        if abs(phi) < tol:
            return t, Y[:, 0], True


        ds = 1e-6 * max(abs(s), 1.0)
        Y0p = np.array([ya, s + ds])
        _, Yp = solve_ivp_rk4(ode, Y0p, (a, b), n_shoot)
        dphi_ds = (Yp[-1, 0] - Y[-1, 0]) / ds

        if abs(dphi_ds) < 1e-14:
            break
        s = s - phi / dphi_ds

    Y0 = np.array([ya, s])
    t, Y = solve_ivp_rk4(ode, Y0, (a, b), n_shoot)
    return t, Y[:, 0], False


def finite_difference_bvp(
    epsilon: float,
    ya: float,
    yb: float,
    a: float = -1.0,
    b: float = 1.0,
    n: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    h = (b - a) / (n + 1)
    x = np.linspace(a + h, b - h, n)




    main = np.full(n, -2.0 * epsilon / (h * h) + 1.0)
    lower = epsilon / (h * h) + x / (2.0 * h)
    upper = epsilon / (h * h) - x / (2.0 * h)


    rhs = np.zeros(n)
    rhs[0] -= lower[0] * ya
    rhs[-1] -= upper[-1] * yb


    y = _thomas_algorithm(lower, main, upper, rhs)

    x_full = np.concatenate(([a], x, [b]))
    y_full = np.concatenate(([ya], y, [yb]))
    return x_full, y_full


def _thomas_algorithm(
    lower: np.ndarray,
    main: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    n = len(main)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    c_prime[0] = upper[0] / main[0]
    d_prime[0] = rhs[0] / main[0]

    for i in range(1, n):
        denom = main[i] - lower[i] * c_prime[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def boundary_layer_thickness(epsilon: float, U_ref: float = 1.0, L_ref: float = 1.0) -> float:
    return L_ref * np.sqrt(epsilon)


def compute_blade_boundary_layer(
    Re: float,
    chord_length: float = 2.0,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    epsilon = 1.0 / max(Re, 1.0)
    x, y, _ = shooting_method(epsilon, ya=0.0, yb=1.0, a=-1.0, b=1.0, n_shoot=n_points)

    x_phys = chord_length * 0.5 * (x + 1.0)
    u_norm = np.clip(y, 0.0, 1.0)
    return x_phys, u_norm
