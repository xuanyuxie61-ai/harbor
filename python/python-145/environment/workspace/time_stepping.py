
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


def rk2_step(yprime, t, y, dt):
    if dt <= 0.0:
        raise ValueError("rk2_step: dt 必须大于 0")
    y = np.asarray(y, dtype=float)
    k1 = dt * np.asarray(yprime(t, y), dtype=float)
    k2 = dt * np.asarray(yprime(t + dt, y + k1), dtype=float)
    return y + 0.5 * (k1 + k2)


def rk3_step(yprime, t, y, dt):
    if dt <= 0.0:
        raise ValueError("rk3_step: dt 必须大于 0")
    y = np.asarray(y, dtype=float)
    k1 = dt * np.asarray(yprime(t, y), dtype=float)
    k2 = dt * np.asarray(yprime(t + dt, y + k1), dtype=float)
    k3 = dt * np.asarray(yprime(t + 0.5 * dt, y + 0.25 * (k1 + k2)), dtype=float)
    return y + (k1 + k2 + 4.0 * k3) / 6.0


def rk23_integrate(yprime, tspan, y0, n_steps):
    if n_steps <= 0:
        raise ValueError("rk23_integrate: n_steps 必须大于 0")
    t0, t1 = tspan
    if t1 <= t0:
        raise ValueError("rk23_integrate: tspan 必须满足 t0 < t1")

    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    dt = (t1 - t0) / n_steps

    t = np.zeros(n_steps + 1, dtype=float)
    y = np.zeros((n_steps + 1, m), dtype=float)
    e = np.zeros((n_steps + 1, m), dtype=float)

    t[0] = t0
    y[0, :] = y0
    e[0, :] = 0.0

    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]

        k1 = dt * np.asarray(yprime(ti, yi), dtype=float)
        k2 = dt * np.asarray(yprime(ti + dt, yi + k1), dtype=float)
        k3 = dt * np.asarray(yprime(ti + 0.5 * dt, yi + 0.25 * (k1 + k2)), dtype=float)

        y2 = yi + 0.5 * (k1 + k2)
        y3 = yi + (k1 + k2 + 4.0 * k3) / 6.0

        t[i + 1] = ti + dt
        y[i + 1, :] = y3
        e[i + 1, :] = y3 - y2

    return t, y, e


def backward_euler_step(A, M, u_old, dt, f_rhs, bc_indices=None, bc_values=None):
    if dt <= 0.0:
        raise ValueError("backward_euler_step: dt 必须大于 0")
    u_old = np.asarray(u_old, dtype=float)
    f_rhs = np.asarray(f_rhs, dtype=float)
    N = u_old.shape[0]

    lhs = M + dt * A
    rhs = M @ u_old + dt * f_rhs


    if bc_indices is not None and bc_values is not None:
        bc_indices = np.asarray(bc_indices, dtype=int)
        bc_values = np.asarray(bc_values, dtype=float)
        for idx, val in zip(bc_indices, bc_values):
            if 0 <= idx < N:

                row_start = lhs.indptr[idx]
                row_end = lhs.indptr[idx + 1]
                lhs.data[row_start:row_end] = 0.0

                diag_found = False
                for j in range(row_start, row_end):
                    if lhs.indices[j] == idx:
                        lhs.data[j] = 1.0
                        diag_found = True
                        break
                if not diag_found:

                    lhs = lhs.todense()
                    lhs = np.array(lhs)
                    lhs[idx, :] = 0.0
                    lhs[idx, idx] = 1.0
                    rhs[idx] = val
                    u_new = np.linalg.solve(lhs, rhs)
                    return u_new
                rhs[idx] = val


    if sp.isspmatrix(lhs):
        u_new = spsolve(lhs.tocsr(), rhs)
    else:
        u_new = np.linalg.solve(lhs, rhs)
    return u_new


def adaptive_rk23(yprime, tspan, y0, tol=1e-6, h_init=0.01, h_min=1e-6, h_max=1.0):
    t0, t1 = tspan
    if t1 <= t0:
        raise ValueError("adaptive_rk23: tspan 必须满足 t0 < t1")
    y0 = np.asarray(y0, dtype=float)
    t = t0
    y = y0.copy()
    h = h_init

    t_list = [t]
    y_list = [y.copy()]

    max_steps = 100000
    step = 0

    while t < t1 and step < max_steps:
        h = min(h, t1 - t)
        if h < h_min:
            raise RuntimeError(f"adaptive_rk23: 步长降至最小值以下 h={h}")

        k1 = h * np.asarray(yprime(t, y), dtype=float)
        k2 = h * np.asarray(yprime(t + h, y + k1), dtype=float)
        k3 = h * np.asarray(yprime(t + 0.5 * h, y + 0.25 * (k1 + k2)), dtype=float)

        y2 = y + 0.5 * (k1 + k2)
        y3 = y + (k1 + k2 + 4.0 * k3) / 6.0
        e = y3 - y2
        err_norm = np.linalg.norm(e) / max(1.0, np.linalg.norm(y3))

        if err_norm <= tol or h <= h_min * 1.1:
            t = t + h
            y = y3
            t_list.append(t)
            y_list.append(y.copy())
            step += 1

            factor = 0.9 * (tol / max(err_norm, 1e-15)) ** (1.0 / 3.0)
            factor = min(5.0, max(0.2, factor))
            h = min(factor * h, h_max)
        else:

            factor = 0.9 * (tol / max(err_norm, 1e-15)) ** (1.0 / 3.0)
            factor = max(0.1, factor)
            h = max(factor * h, h_min)

    y_array = np.array(y_list)
    return np.array(t_list), y_array
