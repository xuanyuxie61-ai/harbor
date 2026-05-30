
import numpy as np
from math import sqrt






def golden_section_search(f, a, b, n_max=100, x_tol=1e-12):
    a = float(a)
    b = float(b)
    if a >= b:
        raise ValueError("必须满足 a < b")
    phi = (sqrt(5.0) - 1.0) / 2.0
    nf = 0

    x1 = phi * a + (1.0 - phi) * b
    x2 = (1.0 - phi) * a + phi * b
    f1 = f(x1)
    f2 = f(x2)
    nf += 2

    for it in range(1, n_max + 1):
        if abs(b - a) <= x_tol:
            return a, b, it, nf

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = phi * a + (1.0 - phi) * b
            f1 = f(x1)
            nf += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - phi) * a + phi * b
            f2 = f(x2)
            nf += 1

    return a, b, n_max, nf






def compute_tangent(J, n, p):
    J = np.asarray(J, dtype=np.float64)

    u, s, vh = np.linalg.svd(J)
    t = vh[-1, :].copy()

    norm_t = np.linalg.norm(t)
    if norm_t < 1e-15:
        raise RuntimeError("切向量范数为零，可能遇到分歧点")
    t = t / norm_t
    return t


def continuation_step(f, fp, x0, p0, h, tol=1e-10, it_max=10):
    n = len(x0)
    x0 = np.asarray(x0, dtype=np.float64)


    J0 = fp(n, x0)
    t2 = compute_tangent(J0, n, p0)

    p2 = int(np.argmax(np.abs(t2)))


    x1 = x0 + h * t2
    x1[p0] = x0[p0] + h * t2[p0]


    x = x1.copy()
    alpha = x1[p0]

    for it in range(it_max):
        fx = f(n, x)
        fx = np.append(fx, x[p0] - alpha)
        fx_norm = np.max(np.abs(fx))
        if fx_norm <= tol:
            return 0, x, t2, p2

        J = fp(n, x)
        J_aug = np.zeros((n, n), dtype=np.float64)
        J_aug[:n-1, :] = J
        J_aug[n-1, :] = 0.0
        J_aug[n-1, p0] = 1.0

        try:
            dx = np.linalg.solve(J_aug, -fx)
        except np.linalg.LinAlgError:
            return 1, x, t2, p2
        x = x + dx

    return 1, x, t2, p2


def continuation_trace(f, fp, x_start, p_start, h_init, target_param_index,
                       target_value, max_steps=100, tol=1e-8):
    path = [x_start.copy()]
    x = x_start.copy()
    p = p_start
    h = h_init
    h_min = 1e-8
    h_max = 0.5

    for step in range(max_steps):
        status, x_new, t_new, p_new = continuation_step(f, fp, x, p, h, tol=tol)
        if status != 0:

            h *= 0.5
            if h < h_min:
                break
            continue

        path.append(x_new.copy())
        x = x_new
        p = p_new


        if abs(x[target_param_index] - target_value) < tol:
            break


        if status == 0:
            h = min(h * 1.2, h_max)

    return path






def heston_calibration_objective(market_iv, strikes, maturities, params_to_opt,
                                 fixed_params, param_index):
    from heston_pde_engine import heston_european_call_price

    def objective(param_value):
        params = fixed_params.copy()
        params[params_to_opt[param_index]] = param_value
        error = 0.0
        count = 0
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if i >= market_iv.shape[0] or j >= market_iv.shape[1]:
                    continue
                try:
                    model_price = heston_european_call_price(
                        S0=params['S0'], K=K, T=T, r=params['r'],
                        kappa=params['kappa'], theta=params['theta'],
                        sigma=params['sigma'], rho=params['rho'], v0=params['v0']
                    )

                    market_price = black_scholes_call_price(
                        params['S0'], K, T, params['r'], market_iv[i, j]
                    )
                    diff = model_price - market_price
                    error += diff * diff
                    count += 1
                except Exception:
                    continue
        if count == 0:
            return 1e10
        return error / count

    return objective


def black_scholes_call_price(S0, K, T, r, sigma):
    from math import log, sqrt, exp, erf
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    nd1 = 0.5 * (1.0 + erf(d1 / sqrt(2.0)))
    nd2 = 0.5 * (1.0 + erf(d2 / sqrt(2.0)))
    return S0 * nd1 - K * exp(-r * T) * nd2


def calibrate_rho_golden_section(market_iv, strikes, maturities, fixed_params):
    obj = heston_calibration_objective(market_iv, strikes, maturities,
                                       ['rho'], fixed_params, 0)
    a, b, it, nf = golden_section_search(obj, -0.95, -0.05, n_max=50, x_tol=1e-4)
    best_rho = (a + b) / 2.0
    best_err = obj(best_rho)
    return best_rho, best_err, it, nf
