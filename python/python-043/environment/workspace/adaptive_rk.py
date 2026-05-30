
import numpy as np
from typing import Callable, Tuple, List






def rk12_adaptive(f: Callable[[float, np.ndarray], np.ndarray],
                  tspan: Tuple[float, float],
                  y0: np.ndarray,
                  dt_init: float,
                  tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0, t1 = tspan
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)
    dt = float(dt_init)

    ts = [t]
    ys = [y.copy()]
    es = [0.0]

    while t < t1:
        dt = min(dt, t1 - t)
        k1 = dt * f(t, y)
        y1 = y + k1
        k2 = dt * f(t + dt, y1)
        y2 = y + 0.5 * k1 + 0.5 * k2

        err = float(np.linalg.norm(y2 - y1))
        threshold = tol * abs(dt)

        if err > threshold and dt > 1e-15:
            dt *= 0.5
            continue


        y = y2.copy()
        t += dt
        ts.append(t)
        ys.append(y.copy())
        es.append(err)

        if err < threshold / 16.0:
            dt *= 2.0

    t_array = np.array(ts, dtype=float)
    y_array = np.array(ys, dtype=float)
    e_array = np.array(es, dtype=float)
    return t_array, y_array, e_array






def rk45_adaptive(f: Callable[[float, np.ndarray], np.ndarray],
                  tspan: Tuple[float, float],
                  y0: np.ndarray,
                  dt_init: float,
                  tol: float = 1e-8,
                  safety: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0, t1 = tspan
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)
    dt = float(dt_init)

    ts = [t]
    ys = [y.copy()]
    es = [0.0]


    a2 = np.array([1.0 / 5.0])
    a3 = np.array([3.0 / 40.0, 9.0 / 40.0])
    a4 = np.array([44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0])
    a5 = np.array([19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0])
    a6 = np.array([9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0])
    a7 = np.array([35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0])

    b5 = a7
    b4 = np.array([5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
                   -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0])








    raise NotImplementedError("Hole_2: RK45 自适应步进循环待实现")

    t_array = np.array(ts, dtype=float)
    y_array = np.array(ys, dtype=float)
    e_array = np.array(es, dtype=float)
    return t_array, y_array, e_array








def implicit_trapezoidal_linear(J: np.ndarray,
                                 y0: np.ndarray,
                                 dt: float,
                                 n_steps: int) -> np.ndarray:
    J = np.asarray(J, dtype=float)
    y = np.asarray(y0, dtype=float).copy()
    n = y.size
    I = np.eye(n, dtype=float)
    M = I - 0.5 * dt * J
    N = I + 0.5 * dt * J

    for _ in range(n_steps):
        rhs = N @ y
        y_new = np.linalg.solve(M, rhs)
        y = y_new
    return y






def hybrid_integrator(f_nonstiff: Callable[[float, np.ndarray], np.ndarray],
                      J_stiff: np.ndarray,
                      tspan: Tuple[float, float],
                      y0: np.ndarray,
                      dt_init: float,
                      stiff_fraction: float = 0.5,
                      tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    t0, t1 = tspan
    y = np.asarray(y0, dtype=float).copy()
    n = y.size
    n_stiff = max(1, int(n * stiff_fraction))

    t = float(t0)
    dt = float(dt_init)
    ts = [t]
    ys = [y.copy()]

    I_full = np.eye(n, dtype=float)
    M = I_full - 0.5 * dt * J_stiff
    N = I_full + 0.5 * dt * J_stiff

    while t < t1:
        dt = min(dt, t1 - t)

        rhs = N @ y
        y = np.linalg.solve(M, rhs)

        y = y + dt * f_nonstiff(t, y)
        t += dt
        ts.append(t)
        ys.append(y.copy())

    return np.array(ts, dtype=float), np.array(ys, dtype=float)





def _self_test():

    f = lambda t, y: -y
    t, y, e = rk12_adaptive(f, (0.0, 1.0), np.array([1.0]), 0.1, tol=1e-5)
    assert abs(y[-1, 0] - np.exp(-1.0)) < 1e-3


    t, y, e = rk45_adaptive(f, (0.0, 1.0), np.array([1.0]), 0.1, tol=1e-8)
    assert abs(y[-1, 0] - np.exp(-1.0)) < 1e-6


    J = np.array([[-10.0]])
    y = implicit_trapezoidal_linear(J, np.array([1.0]), 0.01, 100)

    assert abs(y[0] - np.exp(-10.0)) < 1e-3

    print("adaptive_rk: self-test passed.")


if __name__ == "__main__":
    _self_test()
