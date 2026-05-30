
import numpy as np
from scipy.linalg import expm, solve, det
from typing import Tuple, Optional


class DQMCConfig:

    def __init__(self, nsites: int, beta: float, U: float, t: float, dtau: float):
        if nsites <= 0:
            raise ValueError("nsites > 0  required")
        if beta <= 0 or dtau <= 0:
            raise ValueError("beta, dtau > 0  required")
        if dtau > 0.5:
            raise ValueError("dtau 过大可能导致 Trotter 误差失控，建议 dtau <= 0.5")
        self.nsites = nsites
        self.beta = beta
        self.U = U
        self.t = t
        self.dtau = dtau
        self.L = int(np.ceil(beta / dtau))
        if self.L < 1:
            self.L = 1

        self.lambda_hs = np.arccosh(np.exp(dtau * U / 2.0))
        if np.isnan(self.lambda_hs):
            self.lambda_hs = 0.0


def build_kinetic_matrix(nsites: int, neighbors: list, t: float) -> np.ndarray:
    K = np.zeros((nsites, nsites), dtype=np.float64)
    for i in range(nsites):
        for j in neighbors[i]:
            K[i, j] = -t
    return K


def build_exp_kin(K: np.ndarray, dtau: float) -> np.ndarray:
    return expm(-dtau * K)


def build_b_matrix(exp_kin: np.ndarray, hs_field: np.ndarray, lambda_hs: float, sigma: int) -> np.ndarray:
    diag = np.exp(sigma * lambda_hs * hs_field)
    return exp_kin * diag[np.newaxis, :]


def compute_green_function(Bs: list, stabilize_every: int = 10) -> np.ndarray:
    nsites = Bs[0].shape[0]

    U = np.eye(nsites)
    D = np.ones(nsites)
    Vt = np.eye(nsites)
    for l, B in enumerate(Bs):
        U = B @ U
        if (l + 1) % stabilize_every == 0 or l == len(Bs) - 1:
            U, D, Vt = _svd_stabilize(U, D, Vt)


    M = np.eye(nsites) + (U * D) @ Vt
    G = solve(M, np.eye(nsites), assume_a="gen")
    return G


def _svd_stabilize(U: np.ndarray, D: np.ndarray, Vt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = (U * D) @ Vt
    U_new, s, Vh_new = np.linalg.svd(A, full_matrices=False)

    s = np.where(s > 1e-12, s, 1e-12)
    return U_new, s, Vh_new


def compute_det_ratio(G: np.ndarray, i: int, delta: float) -> float:
    if not (0 <= i < G.shape[0]):
        raise IndexError("i 越界")
    gi = G[i, i]
    ratio = 1.0 + (1.0 - gi) * delta
    return ratio


def truncated_normal_ab_sample(mu: float, sigma: float, a: float, b: float, size: int = 1) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma > 0  required")
    if a >= b:
        raise ValueError("a < b  required")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    from scipy.stats import norm
    alpha_cdf = norm.cdf(alpha)
    beta_cdf = norm.cdf(beta)
    u = np.random.rand(size)
    xi_cdf = alpha_cdf + u * (beta_cdf - alpha_cdf)
    xi = norm.ppf(np.clip(xi_cdf, 1e-10, 1 - 1e-10))
    return mu + sigma * xi


def dqmc_sweep(nsites: int, L: int, hs_field: np.ndarray, B_up: list, B_dn: list,
               lambda_hs: float, exp_kin: np.ndarray, G_up: np.ndarray, G_dn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for l in range(L):
        for i in range(nsites):
            s_old = hs_field[i, l]
            s_new = -s_old

            delta_up = np.exp(2.0 * lambda_hs * s_new) - 1.0
            delta_dn = np.exp(-2.0 * lambda_hs * s_new) - 1.0

            ratio_up = compute_det_ratio(G_up, i, delta_up)
            ratio_dn = compute_det_ratio(G_dn, i, delta_dn)
            ratio = ratio_up * ratio_dn

            if ratio < 0:
                continue
            if np.random.rand() < min(1.0, ratio):

                hs_field[i, l] = s_new
                _update_green(G_up, i, delta_up)
                _update_green(G_dn, i, delta_dn)

                B_up[l] = build_b_matrix(exp_kin, hs_field[:, l], lambda_hs, +1)
                B_dn[l] = build_b_matrix(exp_kin, hs_field[:, l], lambda_hs, -1)
    return hs_field, G_up, G_dn


def _update_green(G: np.ndarray, i: int, delta: float):
    n = G.shape[0]
    u = np.zeros(n)
    u[i] = 1.0
    v = np.zeros(n)
    v[i] = delta

    denom = 1.0 + v @ G @ u
    if abs(denom) < 1e-14:
        return
    G -= np.outer(G @ u, v @ G) / denom


def run_dqmc(config: DQMCConfig, neighbors: list, n_warmup: int = 100, n_measure: int = 200) -> dict:
    nsites = config.nsites
    L = config.L

    hs_field = np.random.choice([-1, 1], size=(nsites, L))

    K = build_kinetic_matrix(nsites, neighbors, config.t)
    exp_kin = build_exp_kin(K, config.dtau)
    B_up = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, +1) for l in range(L)]
    B_dn = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, -1) for l in range(L)]

    G_up = compute_green_function(B_up)
    G_dn = compute_green_function(B_dn)

    for _ in range(n_warmup):
        hs_field, G_up, G_dn = dqmc_sweep(nsites, L, hs_field, B_up, B_dn, config.lambda_hs, exp_kin, G_up, G_dn)

        B_up = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, +1) for l in range(L)]
        B_dn = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, -1) for l in range(L)]
        G_up = compute_green_function(B_up)
        G_dn = compute_green_function(B_dn)

    d_occ_samples = []
    kin_samples = []
    for _ in range(n_measure):
        hs_field, G_up, G_dn = dqmc_sweep(nsites, L, hs_field, B_up, B_dn, config.lambda_hs, exp_kin, G_up, G_dn)
        B_up = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, +1) for l in range(L)]
        B_dn = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, -1) for l in range(L)]
        G_up = compute_green_function(B_up)
        G_dn = compute_green_function(B_dn)

        d_occ = np.mean((1.0 - np.diag(G_up)) * (1.0 - np.diag(G_dn)))
        d_occ_samples.append(d_occ)

        kin = 0.0
        for i in range(nsites):
            for j in neighbors[i]:
                kin += G_up[i, j] + G_dn[i, j]
        kin_samples.append(config.t * kin / nsites)
    return {
        "double_occupancy": np.mean(d_occ_samples),
        "double_occupancy_err": np.std(d_occ_samples) / np.sqrt(len(d_occ_samples)) if len(d_occ_samples) > 1 else 0.0,
        "kinetic_energy": np.mean(kin_samples),
        "kinetic_energy_err": np.std(kin_samples) / np.sqrt(len(kin_samples)) if len(kin_samples) > 1 else 0.0,
    }


if __name__ == "__main__":
    cfg = DQMCConfig(nsites=4, beta=2.0, U=4.0, t=1.0, dtau=0.1)
    neighbors = [[1, 3], [0, 2], [1, 3], [0, 2]]
    res = run_dqmc(cfg, neighbors, n_warmup=20, n_measure=50)
    print(res)
