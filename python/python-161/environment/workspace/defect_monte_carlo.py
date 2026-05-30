
import numpy as np
from typing import Tuple



k_B = 1.380649e-23
m_e = 9.10938356e-31
q = 1.602176634e-19


def uniform_in_triangle(
    n: int, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
) -> np.ndarray:
    v1, v2, v3 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float), np.asarray(v3, dtype=float)
    pts = np.zeros((n, len(v1)))
    rng = np.random.default_rng(123)
    r = rng.random((n, 2))
    for i in range(n):
        a = 1.0 - np.sqrt(r[i, 1])
        b = (1.0 - r[i, 0]) * np.sqrt(r[i, 1])
        c = r[i, 0] * np.sqrt(r[i, 1])
        pts[i] = a * v1 + b * v2 + c * v3
    return pts


def normal_square(n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(456)
    return rng.standard_normal((n, d))


def sample_defect_positions(
    n_defects: int,
    domain: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    v1, v2, v3 = domain
    n_tri = v1.shape[0]
    if n_tri == 0:
        return np.zeros((n_defects, 2))


    areas = np.zeros(n_tri)
    for i in range(n_tri):
        a = v2[i] - v1[i]
        b = v3[i] - v1[i]
        areas[i] = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
    total_area = areas.sum()
    if total_area <= 0:
        counts = np.full(n_tri, n_defects // n_tri)
    else:
        probs = areas / total_area
        counts = np.random.multinomial(n_defects, probs)

    positions = []
    for i in range(n_tri):
        if counts[i] > 0:
            pts = uniform_in_triangle(counts[i], v1[i], v2[i], v3[i])
            positions.append(pts)
    if positions:
        return np.vstack(positions)
    return np.zeros((0, 2))


def defect_density_lognormal(
    n_samples: int, mu_log: float = 24.0, sigma_log: float = 1.0
) -> np.ndarray:
    mu_ln = mu_log * np.log(10.0)
    sigma_ln = sigma_log * np.log(10.0)
    z = normal_square(n_samples, 1).flatten()
    logN = mu_ln + sigma_ln * z
    N_t = np.exp(logN)
    return np.clip(N_t, 1e12, 1e19)


def carrier_lifetime_from_defects(
    N_t: np.ndarray,
    T: float = 300.0,
    sigma_n: float = 1e-16,
    sigma_p: float = 1e-16,
    m_eff: float = 0.15 * 9.10938356e-31,
) -> Tuple[np.ndarray, np.ndarray]:
    if T <= 0 or m_eff <= 0:
        raise ValueError("温度和有效质量必须为正")
    v_th = np.sqrt(3.0 * k_B * T / m_eff)
    v_th_cm = v_th * 100.0
    tau_n = 1.0 / (N_t * sigma_n * v_th_cm)
    tau_p = 1.0 / (N_t * sigma_p * v_th_cm)
    return tau_n, tau_p


class MortalityStyleLifetimeModel:

    def __init__(self, tau_values: np.ndarray):
        self.tau = np.asarray(tau_values)
        self.tau = self.tau[self.tau > 0]
        if len(self.tau) == 0:
            self.tau = np.array([1e-9])

    def pdf(self) -> Tuple[np.ndarray, np.ndarray]:
        log_tau = np.log10(self.tau)
        bins = np.linspace(log_tau.min(), log_tau.max(), max(20, int(np.sqrt(len(self.tau)))))
        counts, edges = np.histogram(log_tau, bins=bins)
        pdf = counts / (counts.sum() * np.diff(edges))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return 10.0 ** centers, pdf

    def cdf(self) -> Tuple[np.ndarray, np.ndarray]:
        log_tau = np.log10(self.tau)
        sorted_log = np.sort(log_tau)
        cdf = np.arange(1, len(sorted_log) + 1) / len(sorted_log)
        return 10.0 ** sorted_log, cdf

    def expected_lifetime(self) -> float:
        return float(np.mean(self.tau))

    def survival_probability(self, t: float) -> float:
        tau_mean = self.expected_lifetime()
        if tau_mean <= 0:
            return 0.0
        return float(np.exp(-t / tau_mean))


def srh_recombination_rate(
    n: float, p: float, n_i: float,
    tau_n: float, tau_p: float,
    E_t: float = 0.0,
    T: float = 300.0,
    N_c: float = 1e19, N_v: float = 1e19,
) -> float:
    kT_eV = 8.617333e-5 * T
    if kT_eV <= 0:
        raise ValueError("温度必须为正")
    n1 = N_c * np.exp(-E_t / kT_eV)
    p1 = N_v * np.exp(E_t / kT_eV)
    denom = tau_p * (n + n1) + tau_n * (p + p1)
    if denom <= 0:
        return 0.0
    return (n * p - n_i * n_i) / denom


if __name__ == "__main__":
    N_t = defect_density_lognormal(1000)
    tau_n, tau_p = carrier_lifetime_from_defects(N_t)
    model = MortalityStyleLifetimeModel(tau_n)
    t_bins, pdf_vals = model.pdf()
    print(f"平均电子寿命: {model.expected_lifetime():.3e} s")
    print(f"100 ns 存活概率: {model.survival_probability(1e-7):.4f}")
    R = srh_recombination_rate(1e15, 1e15, 1e10, tau_n.mean(), tau_p.mean())
    print(f"SRH 复合率: {R:.3e} cm^-3 s^-1")
