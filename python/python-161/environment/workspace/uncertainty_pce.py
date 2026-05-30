
import numpy as np
from typing import Tuple


def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    H_prev2 = np.ones_like(x)
    H_prev1 = x.copy()
    for k in range(1, n):
        H_curr = x * H_prev1 - k * H_prev2
        H_prev2, H_prev1 = H_prev1, H_curr
    return H_prev1


def he_double_product_integral(i: int, j: int) -> float:
    if i != j:
        return 0.0
    import math
    return np.sqrt(2.0 * np.pi) * math.factorial(i)


def he_triple_product_integral(i: int, j: int, k: int) -> float:
    import math
    if (i + j + k) % 2 == 1:
        return 0.0
    if i > j + k or j > i + k or k > i + j:
        return 0.0



    if max(i, j, k) <= 10:

        xi, wi = np.polynomial.hermite.hermgauss(32)
        Hi = hermite_polynomial(i, xi)
        Hj = hermite_polynomial(j, xi)
        Hk = hermite_polynomial(k, xi)
        return float(np.sum(wi * Hi * Hj * Hk))
    else:

        return 0.0


def pce_time_integrator(
    ti: float,
    tf: float,
    nt: int,
    ui: float,
    np_deg: int,
    alpha_mu: float,
    alpha_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if nt <= 0 or np_deg < 0:
        raise ValueError("nt 必须为正，np_deg 必须非负")
    if alpha_sigma < 0:
        raise ValueError("alpha_sigma 必须非负")

    dt = (tf - ti) / nt
    t = np.zeros(nt + 1)
    u = np.zeros((nt + 1, np_deg + 1))


    u1 = np.zeros(np_deg + 1)
    u1[0] = ui
    t[0] = ti
    u[0, :] = u1

    for it in range(1, nt + 1):
        t2 = ((nt - it) * ti + it * tf) / nt
        u2 = np.zeros(np_deg + 1)

        for k in range(np_deg + 1):
            dp = he_double_product_integral(k, k)
            if dp == 0:
                dp = 1.0

            term = -alpha_mu * u1[k]


            i = 1
            for j in range(np_deg + 1):
                tp = he_triple_product_integral(i, j, k)
                term -= alpha_sigma * u1[j] * tp / dp

            u2[k] = u1[k] + dt * term


        u2 = np.where(np.isfinite(u2), u2, 0.0)

        t[it] = t2
        u1 = u2.copy()
        u[it, :] = u1

    return t, u


def pce_efficiency_uq(
    efficiency_mean: float = 0.20,
    efficiency_std: float = 0.03,
    np_deg: int = 4,
    n_mc: int = 10000,
) -> dict:
    if efficiency_mean <= 0 or efficiency_std < 0:
        raise ValueError("效率均值必须为正，标准差必须非负")



    t_op = 1.0
    alpha_mu = 0.1
    alpha_sigma = 0.05

    t, u_coeff = pce_time_integrator(0.0, t_op, 100, efficiency_mean, np_deg, alpha_mu, alpha_sigma)


    mean_eta = u_coeff[-1, 0]
    var_eta = 0.0
    for k in range(1, np_deg + 1):
        norm = he_double_product_integral(k, k) / np.sqrt(2.0 * np.pi)
        var_eta += u_coeff[-1, k] ** 2 * norm

    std_eta = np.sqrt(max(var_eta, 0.0))


    rng = np.random.default_rng(789)
    xi_samples = rng.standard_normal(n_mc)
    eta_mc = np.zeros(n_mc)
    for idx, xi in enumerate(xi_samples):
        alpha = alpha_mu + alpha_sigma * xi
        eta_mc[idx] = efficiency_mean * np.exp(-alpha * t_op)

    mc_mean = float(np.mean(eta_mc))
    mc_std = float(np.std(eta_mc))


    sensitivity = {}
    total_var = var_eta if var_eta > 0 else 1e-14
    for k in range(1, np_deg + 1):
        norm = he_double_product_integral(k, k) / np.sqrt(2.0 * np.pi)
        contrib = u_coeff[-1, k] ** 2 * norm
        sensitivity[f"order_{k}"] = float(contrib / total_var)

    return {
        "pce_mean_efficiency": float(mean_eta),
        "pce_std_efficiency": float(std_eta),
        "mc_mean_efficiency": mc_mean,
        "mc_std_efficiency": mc_std,
        "variance": float(var_eta),
        "sensitivity_indices": sensitivity,
        "pce_coefficients_final": u_coeff[-1, :].tolist(),
    }


if __name__ == "__main__":
    result = pce_efficiency_uq()
    print("PCE 不确定性量化结果:")
    for k, v in result.items():
        if k != "sensitivity_indices":
            print(f"  {k}: {v}")
    print("  敏感性指标:", result["sensitivity_indices"])
