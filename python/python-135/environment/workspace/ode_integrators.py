
import numpy as np
from scipy.optimize import fsolve
from utils import validate_positive, clip_concentration


def explicit_trapezoidal(f, tspan, y0, n_steps):
    validate_positive(n_steps, "n_steps")
    m = len(y0)
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0

    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]
        fi = np.asarray(f(ti, yi))

        y_star = yi + dt * fi

        f_star = np.asarray(f(ti + dt, y_star))
        y[i + 1, :] = yi + 0.5 * dt * (fi + f_star)

    return t, y


def bdf3_solver(f, tspan, y0, n_steps):
    validate_positive(n_steps, "n_steps")
    m = len(y0)
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0

    def bdf3_residual(y_new, t_new, y_n, y_nm1, y_nm2):
        return (11.0 * y_new - 18.0 * y_n + 9.0 * y_nm1 - 2.0 * y_nm2) / (6.0 * dt) - np.asarray(f(t_new, y_new))

    for i in range(n_steps):
        if i < 2:

            ti = t[i]
            yi = y[i, :]
            k1 = dt * np.asarray(f(ti, yi))
            k2 = dt * np.asarray(f(ti + dt, yi + k1))
            k3 = dt * np.asarray(f(ti + 0.5 * dt, yi + 0.25 * k1 + 0.25 * k2))
            y[i + 1, :] = yi + (k1 + k2 + 4.0 * k3) / 6.0
        else:
            t_new = t[i + 1]
            y_n = y[i, :]
            y_nm1 = y[i - 1, :]
            y_nm2 = y[i - 2, :]
            y_guess = y_n + dt * np.asarray(f(t[i], y_n))
            try:
                y_new = fsolve(bdf3_residual, y_guess, args=(t_new, y_n, y_nm1, y_nm2), full_output=False)
            except Exception:

                y_new = fsolve(lambda yn: (yn - y_n) / dt - np.asarray(f(t_new, yn)), y_guess)
            y[i + 1, :] = y_new

    return t, y


def predator_prey_like_cycles(f, tspan, y0, n_steps, threshold=1e-6):
    t, y = explicit_trapezoidal(f, tspan, y0, n_steps)
    m = y.shape[1]


    peaks = []
    for i in range(1, n_steps):
        if y[i, 0] > y[i - 1, 0] and y[i, 0] > y[i + 1, 0]:
            peaks.append((t[i], y[i, :]))

    if len(peaks) < 2:
        return None, t, y

    periods = []
    for i in range(1, len(peaks)):
        periods.append(peaks[i][0] - peaks[i - 1][0])

    period = np.mean(periods) if periods else None
    amplitude = np.max(y[:, 0]) - np.min(y[:, 0])
    return {"period": period, "amplitude": amplitude, "num_peaks": len(peaks)}, t, y


def solve_stiff_amine_ode(f, tspan, y0, n_steps, stiffness_threshold=100.0):

    h = 1e-8
    f0 = np.asarray(f(tspan[0], y0))
    n = len(y0)
    J = np.zeros((n, n))
    for j in range(n):
        y_pert = y0.copy()
        y_pert[j] += h
        J[:, j] = (np.asarray(f(tspan[0], y_pert)) - f0) / h

    eigenvalues = np.linalg.eigvals(J)
    abs_eig = np.abs(eigenvalues)
    nonzero = abs_eig[abs_eig > 1e-12]
    if len(nonzero) == 0:
        stiffness_ratio = 1.0
    else:
        stiffness_ratio = np.max(abs_eig) / np.min(nonzero)

    if stiffness_ratio > stiffness_threshold:
        print(f"  [Stiffness detected: ratio={stiffness_ratio:.2e}, using BDF3]")
        return bdf3_solver(f, tspan, y0, n_steps)
    else:
        print(f"  [Non-stiff system: ratio={stiffness_ratio:.2e}, using explicit trapezoidal]")
        return explicit_trapezoidal(f, tspan, y0, n_steps)


def gear_bdf2(f, tspan, y0, n_steps):
    validate_positive(n_steps, "n_steps")
    m = len(y0)
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0


    def be_residual(y_new, t_new, y_n):
        return (y_new - y_n) / dt - np.asarray(f(t_new, y_new))

    y_guess = y0 + dt * np.asarray(f(t[0], y0))
    y[1, :] = fsolve(be_residual, y_guess, args=(t[1], y[0, :]))

    for i in range(1, n_steps):
        def bdf2_residual(y_new, t_new, y_n, y_nm1):
            return (3.0 * y_new - 4.0 * y_n + y_nm1) / (2.0 * dt) - np.asarray(f(t_new, y_new))
        y_guess = y[i, :] + dt * np.asarray(f(t[i], y[i, :]))
        y[i + 1, :] = fsolve(bdf2_residual, y_guess, args=(t[i + 1], y[i, :], y[i - 1, :]))

    return t, y
