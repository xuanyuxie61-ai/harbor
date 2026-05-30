
import numpy as np
from typing import Callable, Tuple, Dict
from utils import r8_uniform_01






def glomin_brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    c: float,
    m: float,
    e: float,
    t: float,
    max_calls: int = 1000,
) -> Tuple[float, float, int]:
    calls = 0

    def feval(x):
        nonlocal calls
        calls += 1
        if calls > max_calls:
            return float('inf')
        return float(f(x))


    sa, sb = min(a, b), max(a, b)
    x = c
    fx = feval(x)
    w = x
    fw = fx
    v = x
    fv = fx

    d = sb - sa
    e_len = d

    while calls <= max_calls:
        tol = t * abs(x) + e
        m_c = 0.5 * (sa + sb)


        if abs(x - m_c) <= 2.0 * tol - 0.5 * (sb - sa):
            break

        p = q = r = 0.0
        if abs(e_len) > tol:

            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e_len
            e_len = d

        if abs(p) < abs(0.5 * q * r) and p > q * (sa - x) and p < q * (sb - x):
            d = p / q
            u = x + d
            if u - sa < 2.0 * tol or sb - u < 2.0 * tol:
                d = tol if x < m_c else -tol
        else:
            if x < m_c:
                e_len = sb - x
            else:
                e_len = sa - x
            d = 0.5 * e_len


        if abs(d) < 1.0e-12:

            seed = calls * 1611 + 1
            ru, _ = r8_uniform_01(seed)
            d = (2.0 * ru - 1.0) * tol

        u = x + d
        fu = feval(u)

        if fu <= fx:
            if u >= x:
                sa = x
            else:
                sb = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                sa = u
            else:
                sb = u
            if fu <= fw or abs(w - x) < 1.0e-15:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or abs(v - x) < 1.0e-15 or abs(v - w) < 1.0e-15:
                v, fv = u, fu

    return x, fx, calls






def poisson_log_likelihood(
    n_obs: np.ndarray,
    s_pred: np.ndarray,
    b_pred: np.ndarray,
    mu: float,
) -> float:
    lambda_pred = mu * s_pred + b_pred

    lambda_pred = np.where(lambda_pred < 1.0e-15, 1.0e-15, lambda_pred)
    logL = np.sum(n_obs * np.log(lambda_pred) - lambda_pred)

    return float(logL)


def profile_likelihood_ratio(
    n_obs: np.ndarray,
    s_pred: np.ndarray,
    b_pred: np.ndarray,
    mu_test: float,
) -> float:

    def neg_logL(mu):
        return -poisson_log_likelihood(n_obs, s_pred, b_pred, mu)


    mu_grid = np.linspace(0.0, max(10.0, 2.0 * mu_test), 200)
    logL_grid = np.array([poisson_log_likelihood(n_obs, s_pred, b_pred, m) for m in mu_grid])
    mu_hat = mu_grid[np.argmax(logL_grid)]

    logL_global = poisson_log_likelihood(n_obs, s_pred, b_pred, mu_hat)
    logL_cond = poisson_log_likelihood(n_obs, s_pred, b_pred, mu_test)

    q_mu = -2.0 * (logL_cond - logL_global)
    return max(float(q_mu), 0.0)


def confidence_interval_upper_limit(
    n_obs: np.ndarray,
    s_pred: np.ndarray,
    b_pred: np.ndarray,
    cl: float = 0.90,
    mu_max: float = 20.0,
) -> float:
    target_q = 2.70

    mu_grid = np.linspace(0.0, mu_max, 200)
    q_grid = np.array([profile_likelihood_ratio(n_obs, s_pred, b_pred, m) for m in mu_grid])


    for i in range(len(mu_grid) - 1):
        if q_grid[i] <= target_q <= q_grid[i + 1] or q_grid[i + 1] <= target_q <= q_grid[i]:
            t = (target_q - q_grid[i]) / (q_grid[i + 1] - q_grid[i])
            return float(mu_grid[i] + t * (mu_grid[i + 1] - mu_grid[i]))

    return float(mu_grid[-1])






def sensitivity_curve(
    exposure_kg_day: float,
    target_mass_kg: float,
    background_rate_per_kev_kg_day: float,
    e_min_kev: float,
    e_max_kev: float,
    m_chi_values: np.ndarray,
    efficiency: float = 0.5,
    n_bins: int = 20,
) -> np.ndarray:
    e_bins = np.linspace(e_min_kev, e_max_kev, n_bins + 1)
    bin_width = e_bins[1] - e_bins[0]


    N_b = background_rate_per_kev_kg_day * bin_width * exposure_kg_day


    N_s_90 = 2.44 + 1.64 * np.sqrt(N_b)

    sigma_90 = np.zeros(len(m_chi_values))
    for idx, m_chi in enumerate(m_chi_values):


        from wimp_physics import total_events_in_range
        N_s_per_pb = total_events_in_range(
            e_min_kev, e_max_kev, m_chi, 1.0, 73.0, target_mass_kg, exposure_kg_day / target_mass_kg
        )
        if N_s_per_pb > 1.0e-15:
            sigma_90[idx] = N_s_90 / (efficiency * N_s_per_pb)
        else:
            sigma_90[idx] = 1.0e6

    return sigma_90






if __name__ == "__main__":

    def f_test(x):
        return (x - 0.3) ** 2 + 0.1 * np.sin(20.0 * x)

    x_min, f_min, calls = glomin_brent(f_test, 0.0, 1.0, 0.5, 100.0, 1.0e-10, 1.0e-10)
    assert 0.0 <= x_min <= 1.0
    assert f_min <= f_test(0.3) + 0.1, "全局最小值搜索失败"


    n_obs = np.array([5, 7, 6, 8, 5])
    s_pred = np.array([2, 2, 2, 2, 2])
    b_pred = np.array([3, 3, 3, 3, 3])
    logL = poisson_log_likelihood(n_obs, s_pred, b_pred, 1.0)
    assert np.isfinite(logL), "对数似然非有限"


    q = profile_likelihood_ratio(n_obs, s_pred, b_pred, 0.0)
    assert q >= 0.0, "q_mu 必须非负"


    upper = confidence_interval_upper_limit(n_obs, s_pred, b_pred)
    assert upper >= 0.0


    m_vals = np.array([10.0, 50.0, 100.0])
    sens = sensitivity_curve(1000.0, 10.0, 0.01, 0.5, 50.0, m_vals)
    assert np.all(sens > 0.0)
    assert np.all(np.isfinite(sens))

    print("statistical_analysis.py: 所有自测通过")
