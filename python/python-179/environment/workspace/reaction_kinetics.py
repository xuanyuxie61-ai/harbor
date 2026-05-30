
import numpy as np
from system_utils import clip_to_range, check_finite






def twoway_reaction_term(u: np.ndarray, k1: float, k2: float) -> np.ndarray:
    k1 = float(k1)
    k2 = float(k2)
    k1 = max(k1, 1e-12)
    k2 = max(k2, 1e-12)
    R = -k1 * u + k2 * (1.0 - u)
    return R


def twoway_exact_solution(t: np.ndarray, u0: float, k1: float, k2: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    k_sum = k1 + k2
    u_star = k2 / k_sum
    return u_star + (u0 - u_star) * np.exp(-k_sum * t)






def vanderpol_reaction_term(u: np.ndarray, mu: float) -> np.ndarray:
    mu = float(mu)
    mu = max(mu, 1e-6)
    u = clip_to_range(np.asarray(u, dtype=float), -1e3, 1e3)
    R = mu * (1.0 - u * u) * u
    return R






def parametric_reaction_source(u: np.ndarray,
                                k1: float, k2: float,
                                mu: float,
                                mix_ratio: float = 0.5) -> np.ndarray:
    mix_ratio = clip_to_range(float(mix_ratio), 0.0, 1.0)
    R_lin = twoway_reaction_term(u, k1, k2)
    R_vdp = vanderpol_reaction_term(u, mu)
    R = mix_ratio * R_lin + (1.0 - mix_ratio) * R_vdp
    check_finite(R, "reaction_source")
    return R






def reaction_jacobian_diagonal(u: np.ndarray,
                                k1: float, k2: float,
                                mu: float,
                                mix_ratio: float = 0.5) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    mix_ratio = clip_to_range(float(mix_ratio), 0.0, 1.0)
    dR_lin = -(k1 + k2)
    dR_vdp = mu * (1.0 - 3.0 * u * u)
    J = mix_ratio * dR_lin + (1.0 - mix_ratio) * dR_vdp
    return J
