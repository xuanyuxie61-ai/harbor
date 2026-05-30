
import numpy as np
from typing import Callable, Tuple, Optional
from utils import check_bounds, EPSILON_MACHINE




RUCKLIDGE_KAPPA = 2.0
RUCKLIDGE_LAMBDA = 1.7




ARNEODO_ALPHA = -5.5
ARNEODO_BETA = 3.5
ARNEODO_DELTA = -1.0


def rucklidge_deriv(t: float, xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[0], xyz[1], xyz[2]
    dxdt = RUCKLIDGE_KAPPA * x - RUCKLIDGE_LAMBDA * y - y * z
    dydt = x
    dzdt = -z + y ** 2
    return np.array([dxdt, dydt, dzdt], dtype=float)


def arneodo_deriv(t: float, xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[0], xyz[1], xyz[2]
    dxdt = y
    dydt = z
    dzdt = -ARNEODO_ALPHA * x - ARNEODO_BETA * y - z + ARNEODO_DELTA * (x ** 3)
    return np.array([dxdt, dydt, dzdt], dtype=float)


def rkf45_step(f: Callable[[float, np.ndarray], np.ndarray],
               t: float, y: np.ndarray, h: float,
               relerr: float = 1e-6, abserr: float = 1e-9) -> Tuple[np.ndarray, float, float, bool]:
    y = np.asarray(y, dtype=float)
    neqn = y.size


    a2, a3, a4, a5, a6 = 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 0.5
    b21 = 1.0 / 4.0
    b31, b32 = 3.0 / 32.0, 9.0 / 32.0
    b41, b42, b43 = 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0
    b51, b52, b53, b54 = 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0
    b61, b62, b63, b64, b65 = -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0
    c1, c3, c4, c5 = 25.0 / 216.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0
    d1, d3, d4, d5, d6 = 16.0 / 135.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0


    k1 = h * f(t, y)
    k2 = h * f(t + a2 * h, y + b21 * k1)
    k3 = h * f(t + a3 * h, y + b31 * k1 + b32 * k2)
    k4 = h * f(t + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3)
    k5 = h * f(t + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
    k6 = h * f(t + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)


    y4 = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    y5 = y + d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6


    scale = 2.0 / relerr if relerr > 0 else 1.0
    ae = scale * abserr
    err_max = 0.0
    for i in range(neqn):
        et = abs(y[i]) + abs(y5[i]) + ae
        if et <= 0.0:

            return y, t, h * 0.5, False
        ee = abs((-2090.0 * k1[i]
                  + (21970.0 * k3[i] - 15048.0 * k4[i])
                  + (22528.0 * k2[i] - 27360.0 * k5[i])))
        err_local = abs(h) * ee * scale / 752400.0
        err_ratio = err_local / et
        err_max = max(err_max, err_ratio)

    esttol = err_max

    if esttol <= 1.0:

        s = 5.0 if esttol <= 0.0001889568 else 0.9 / (esttol ** 0.2)
        h_new = s * abs(h)
        h_new = min(h_new, 5.0 * abs(h))
        h_new = max(h_new, 26.0 * EPSILON_MACHINE * max(abs(t), abs(h)))
        return y5, t + h, h_new if h > 0 else -h_new, True
    else:

        s = 0.9 / (esttol ** 0.2) if esttol < 59049.0 else 0.1
        h_new = s * abs(h)
        h_new = max(h_new, 26.0 * EPSILON_MACHINE * max(abs(t), abs(h)))
        return y, t, h_new if h > 0 else -h_new, False


def integrate_trajectory(f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
                         relerr: float = 1e-6, abserr: float = 1e-9,
                         max_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    if t0 >= t1:
        raise ValueError("t_span must satisfy t0 < t1")

    t = t0
    y = y0.copy()
    h = (t1 - t0) * 0.01
    h = max(h, 26.0 * EPSILON_MACHINE * max(abs(t0), abs(t1 - t0)))

    t_list = [t]
    y_list = [y.copy()]
    nfe = 0

    for _ in range(max_steps):
        if abs(t1 - t) <= 26.0 * EPSILON_MACHINE * abs(t):
            break


        if abs(t1 - t) <= abs(h):
            h = t1 - t

        y_new, t_new, h_new, accepted = rkf45_step(f, t, y, h, relerr, abserr)
        nfe += 6

        if accepted:
            t = t_new
            y = y_new
            t_list.append(t)
            y_list.append(y.copy())
            h = h_new if t < t1 else -abs(h_new)
        else:
            h = h_new

        if nfe > 3000 * y0.size:
            print("[WARNING] RKF45: too many function evaluations, stopping early.")
            break
    else:
        print("[WARNING] RKF45: max_steps reached, integration may be incomplete.")

    return np.array(t_list), np.array(y_list)


def compute_particle_load_field(particles: np.ndarray, domain: Tuple[float, float, float, float],
                                nx: int, ny: int) -> np.ndarray:
    particles = np.asarray(particles, dtype=float)
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    rho = np.zeros((nx, ny), dtype=float)

    for p in range(particles.shape[0]):
        x, y = particles[p, 0], particles[p, 1]

        x = max(xmin + 1e-12, min(xmax - 1e-12, x))
        y = max(ymin + 1e-12, min(ymax - 1e-12, y))

        ix = int((x - xmin) / dx)
        iy = int((y - ymin) / dy)
        ix = min(ix, nx - 1)
        iy = min(iy, ny - 1)


        wx = (x - xmin) / dx - ix
        wy = (y - ymin) / dy - iy
        wx = max(0.0, min(1.0, wx))
        wy = max(0.0, min(1.0, wy))


        ixp1 = min(ix + 1, nx - 1)
        iyp1 = min(iy + 1, ny - 1)

        rho[ix, iy] += (1.0 - wx) * (1.0 - wy)
        rho[ixp1, iy] += wx * (1.0 - wy)
        rho[ix, iyp1] += (1.0 - wx) * wy
        rho[ixp1, iyp1] += wx * wy


    cell_volume = dx * dy
    if cell_volume > 0:
        rho /= cell_volume

    return rho
