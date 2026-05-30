
import numpy as np
import cmath
from math import sqrt, log, exp, cos, sin, pi






def grazing_parameters():
    return {
        'a': 1.1,
        'c1': 1.2,
        'c2': 1.5,
        'd1': 0.001,
        'd2': 0.001,
        'k': 3000.0,
        'r1': 0.8,
        't0': 0.0,
        'y0': np.array([3000.0, 5.0]),
        'tstop': 100.0
    }


def volatility_orderflow_deriv(t, y, params):
    u, v = y[0], y[1]
    a = params['a']
    c1 = params['c1']
    c2 = params['c2']
    d1 = params['d1']
    d2 = params['d2']
    k = params['k']
    r1 = params['r1']

    dudt = r1 * u * (1.0 - u / k) - c1 * v * (1.0 - exp(-d1 * u))
    dvdt = -a * v + c2 * v * (1.0 - exp(-d2 * u))
    return np.array([dudt, dvdt], dtype=np.float64)


def rk4_integrate(deriv_func, y0, t_span, h=0.01, args=()):
    y0 = np.asarray(y0, dtype=np.float64)
    t0, tf = t_span
    if t0 >= tf:
        raise ValueError("t0必须小于tf")
    if h <= 0:
        raise ValueError("步长h必须为正")

    steps = int(np.ceil((tf - t0) / h))
    h = (tf - t0) / steps
    trajectory = np.zeros((steps + 1, len(y0)), dtype=np.float64)
    times = np.zeros(steps + 1, dtype=np.float64)
    trajectory[0] = y0
    times[0] = t0
    y = y0.copy()
    t = t0

    for i in range(steps):
        k1 = h * deriv_func(t, y, *args)
        k2 = h * deriv_func(t + 0.5 * h, y + 0.5 * k1, *args)
        k3 = h * deriv_func(t + 0.5 * h, y + 0.5 * k2, *args)
        k4 = h * deriv_func(t + h, y + k3, *args)
        y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += h
        trajectory[i + 1] = y
        times[i + 1] = t

    return times, trajectory


def feller_dynamics_analysis(kappa, theta, sigma):
    feller_ratio = 2.0 * kappa * theta / (sigma * sigma)
    stable = feller_ratio >= 1.0
    absorption_prob = np.exp(-2.0 * kappa * theta / (sigma * sigma)) if sigma > 0 else 0.0
    return {
        'feller_ratio': feller_ratio,
        'feller_satisfied': stable,
        'absorption_probability_estimate': absorption_prob,
        'boundary_classification': 'entrance' if stable else 'regular'
    }






def iterated_function_system(n_iter, maps, probs, x0=None, dim=2):
    if x0 is None:
        x0 = np.random.rand(dim)
    x0 = np.asarray(x0, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.sum(probs)

    trajectory = np.zeros((n_iter, dim), dtype=np.float64)
    x = x0.copy()
    rng = np.random.default_rng()

    for i in range(n_iter):
        j = rng.choice(len(maps), p=probs)
        x = maps[j](x)
        trajectory[i] = x

    return trajectory


def dragon_curve_ifs(n_iter=5000):
    A = np.array([[-0.5, 0.5], [-0.5, -0.5]], dtype=np.float64)
    b1 = np.array([0.0, 0.0], dtype=np.float64)
    b2 = np.array([1.0, 0.0], dtype=np.float64)

    def map1(x):
        return A @ x + b1

    def map2(x):
        return A @ x + b2

    return iterated_function_system(n_iter, [map1, map2], [0.5, 0.5], dim=2)


def multifractal_spectrum(trajectory, q_values=None):
    if q_values is None:
        q_values = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError("trajectory必须为二维数组")
    N, d = traj.shape


    mins = np.min(traj, axis=0)
    maxs = np.max(traj, axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0
    traj_norm = (traj - mins) / ranges


    box_counts = [4, 8, 16, 32]
    results = {}
    for q in q_values:
        log_eps = []
        log_moments = []
        for boxes in box_counts:
            eps = 1.0 / boxes

            indices = np.floor(traj_norm * boxes).astype(np.int64)
            indices = np.clip(indices, 0, boxes - 1)

            flat_idx = np.ravel_multi_index(indices.T, [boxes] * d)
            unique, counts = np.unique(flat_idx, return_counts=True)
            p = counts / N
            if q == 1.0:

                moment = -np.sum(p * np.log(p + 1e-30))
            elif q == 0.0:

                moment = len(unique)
            else:
                moment = np.sum(p ** q)
            log_eps.append(log(eps))
            if q == 0.0:
                log_moments.append(log(moment + 1e-30))
            elif q == 1.0:
                log_moments.append(moment)
            else:
                log_moments.append(log(moment + 1e-30))


        if len(log_eps) >= 2:
            log_eps = np.array(log_eps)
            log_moments = np.array(log_moments)
            if q == 1.0:

                slope = np.polyfit(log_eps, log_moments, 1)[0]
                D_q = slope
            elif q == 0.0:
                slope = np.polyfit(log_eps, log_moments, 1)[0]
                D_q = -slope
            else:
                slope = np.polyfit(log_eps, log_moments, 1)[0]
                D_q = slope / (q - 1.0)
            results[q] = D_q
        else:
            results[q] = 0.0

    return results






def heston_riccati_solution(u, tau, kappa, theta, sigma, rho, r):
    if tau < 0:
        raise ValueError("tau必须非负")
    if abs(sigma) < 1e-12:
        raise ValueError("sigma不能为零")









    raise NotImplementedError("Hole_1: 需要实现Riccati方程解析解")


def heston_characteristic_function(u, S0, v0, T, r, kappa, theta, sigma, rho):
    A, D = heston_riccati_solution(u, T, kappa, theta, sigma, rho, r)
    phi = np.exp(A + D * v0 + 1j * u * log(S0))
    return phi
