
import numpy as np
from typing import List, Tuple


def polynomial_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(p) == 0 or len(q) == 0:
        return np.array([0.0])
    result = np.convolve(p, q)
    return result


def polynomial_power(base: np.ndarray, exponent: int) -> np.ndarray:
    if exponent < 0:
        raise ValueError("指数必须非负")
    if exponent == 0:
        return np.array([1.0])
    result = np.array([1.0])
    current = base.copy()
    exp = exponent
    while exp > 0:
        if exp % 2 == 1:
            result = polynomial_multiply(result, current)
        current = polynomial_multiply(current, current)
        exp //= 2
    return result


class MRMTModel:

    def __init__(self, alphas: np.ndarray, betas: np.ndarray, R_m: float = 1.0):
        self.alphas = np.asarray(alphas, dtype=float)
        self.betas = np.asarray(betas, dtype=float)
        self.R_m = float(R_m)
        if len(self.alphas) != len(self.betas):
            raise ValueError("alphas 与 betas 长度必须一致")
        if np.any(self.alphas <= 0):
            raise ValueError("所有 α_i 必须为正")
        if np.any(self.betas < 0):
            raise ValueError("所有 β_i 必须非负")

    def effective_retardation(self, s: float) -> float:
        if s < 0:
            raise ValueError("s 必须非负")
        return self.R_m + np.sum(self.betas * s / (s + self.alphas))

    def immobile_response_kernel(self, t: float) -> float:
        if t < 0:
            return 0.0
        return float(np.sum(self.betas * self.alphas * np.exp(-self.alphas * t)))

    def compute_immobile_concentration(self, C_mobile_history: np.ndarray,
                                       dt: float) -> np.ndarray:
        n_steps = len(C_mobile_history)
        if n_steps == 0:
            return np.array([])
        S_total = np.zeros(n_steps)

        for alpha_i, beta_i in zip(self.alphas, self.betas):

            k = np.arange(n_steps)
            kernel = beta_i * alpha_i * np.exp(-alpha_i * k * dt)

            conv = np.convolve(C_mobile_history, kernel, mode='full')[:n_steps]
            S_total += conv

        return S_total

    def mobile_zone_equation_rhs(self, C_mobile: np.ndarray,
                                  C_immobile_total: np.ndarray,
                                  dt: float) -> np.ndarray:
        if len(C_mobile) != len(C_immobile_total):
            raise ValueError("浓度数组长度不一致")
        if dt <= 0:
            raise ValueError("dt 必须为正")

        dS_dt = np.zeros_like(C_immobile_total)
        dS_dt[1:] = (C_immobile_total[1:] - C_immobile_total[:-1]) / dt
        return -dS_dt

    def generate_rate_spectrum(self, alpha_min: float, alpha_max: float,
                                n_rates: int, distribution: str = "log_uniform") -> "MRMTModel":
        if alpha_min <= 0 or alpha_max <= alpha_min:
            raise ValueError("速率范围非法")
        if n_rates < 1:
            raise ValueError("速率数必须 ≥ 1")

        log_alphas = np.linspace(np.log(alpha_min), np.log(alpha_max), n_rates)
        alphas_new = np.exp(log_alphas)

        betas_new = 1.0 / np.sqrt(alphas_new)
        betas_new = betas_new / np.sum(betas_new) * np.sum(self.betas) if np.sum(self.betas) > 0 else betas_new
        return MRMTModel(alphas_new, betas_new, self.R_m)

    def breakthrough_curve_moment(self, C_history: np.ndarray, dt: float, moment_order: int) -> float:
        if len(C_history) == 0:
            return 0.0
        t = np.arange(len(C_history)) * dt
        numerator = np.sum((t ** moment_order) * C_history) * dt
        denominator = np.sum(C_history) * dt + 1e-15
        return float(numerator / denominator)


if __name__ == "__main__":

    alphas = np.array([0.01, 0.1, 1.0])
    betas = np.array([0.5, 0.3, 0.2])
    mrmt = MRMTModel(alphas, betas, R_m=1.0)

    C_m = np.exp(-np.linspace(0, 5, 100) * 0.1)
    S = mrmt.compute_immobile_concentration(C_m, dt=0.05)
    assert len(S) == len(C_m)


    p = np.array([1, 2, 3])
    q = np.array([0, 1])
    r = polynomial_multiply(p, q)
    assert np.allclose(r, [0, 1, 2, 3])

    m1 = mrmt.breakthrough_curve_moment(C_m, 0.05, 1)
    assert m1 > 0
    print("mrmt_generator: 自测试通过")
