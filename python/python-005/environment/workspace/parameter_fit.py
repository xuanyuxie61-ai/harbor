# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List
from utils import ensure_positive





def solve_least_squares_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, n = A.shape
    if m < n:
        raise ValueError("QR 最小二乘要求 m ≥ n")
    Q, R = np.linalg.qr(A)
    condR = np.linalg.cond(R)
    if condR > 1e12:

        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        tol = max(A.shape) * np.finfo(float).eps * s[0]
        s_inv = np.array([1.0 / si if si > tol else 0.0 for si in s])
        x = Vt.T @ (s_inv * (U.T @ b))
        return x

    Qtb = Q.T @ b
    x = np.linalg.solve(R[:n, :], Qtb[:n])
    return x





def test_problem_vandermonde(m: int = 20, n: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = np.arange(1, m + 1, dtype=float)
    A = np.vander(i / m, N=n, increasing=True)
    x_exact = np.ones(n)
    b = A @ x_exact
    return A, b, x_exact


def test_problem_rank_deficient() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 9.0],
        [2.0, 1.0, 3.0],
        [5.0, 5.0, 10.0],
        [1.0, 4.0, 5.0],
    ])
    x_exact = np.array([1.0, 1.0, 0.0])
    b = A @ x_exact
    return A, b, x_exact


def run_lls_test_suite() -> dict:
    results = {}

    A, b, xex = test_problem_vandermonde()
    xqr = solve_least_squares_qr(A, b)
    results["vandermonde"] = {
        "cond_A": float(np.linalg.cond(A)),
        "residual_qr": float(np.linalg.norm(A @ xqr - b)),
        "error_vs_exact": float(np.linalg.norm(xqr - xex)),
    }

    A2, b2, xex2 = test_problem_rank_deficient()
    xqr2 = solve_least_squares_qr(A2, b2)
    results["rank_deficient"] = {
        "cond_A": float(np.linalg.cond(A2)),
        "residual_qr": float(np.linalg.norm(A2 @ xqr2 - b2)),
        "error_vs_exact": float(np.linalg.norm(xqr2 - xex2)),
    }
    return results





def theory_Cl_model(params: np.ndarray, l: np.ndarray) -> np.ndarray:
    A, ns, omb, omc, h = params

    A = A * 1e-9
    l_p = 220.0
    l_D = 1500.0

    peak_mod = 1.0 + 0.5 * np.sin(l / l_p * np.pi)
    damping = np.exp(-(l / l_D) ** 2)
    Cl = A * (l / l_p) ** (ns - 1.0) * peak_mod * damping

    return np.maximum(Cl, 1e-30)


class CosmologyFitter:

    def __init__(self, l_data: np.ndarray, Cl_data: np.ndarray,
                 sigma_Cl: np.ndarray, param_names: List[str]):
        self.l = l_data
        self.Cl_data = Cl_data
        self.sigma = sigma_Cl
        self.param_names = param_names
        self.n_params = len(param_names)

    def chi2(self, params: np.ndarray) -> float:
        Cl_th = theory_Cl_model(params, self.l)
        diff = (self.Cl_data - Cl_th) / self.sigma
        return float(np.sum(diff ** 2))

    def jacobian(self, params: np.ndarray, h: float = 1e-5) -> np.ndarray:
        n_l = len(self.l)
        J = np.zeros((n_l, self.n_params))
        for a in range(self.n_params):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[a] += h
            p_minus[a] -= h
            J[:, a] = (theory_Cl_model(p_plus, self.l) -
                       theory_Cl_model(p_minus, self.l)) / (2.0 * h)
        return J

    def fit(self, params0: np.ndarray, max_iter: int = 20,
            tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        params = params0.copy()
        W = np.diag(1.0 / (self.sigma ** 2))
        for it in range(max_iter):
            Cl_th = theory_Cl_model(params, self.l)
            resid = self.Cl_data - Cl_th
            J = self.jacobian(params)

            lhs = J.T @ W @ J
            rhs = J.T @ W @ resid
            try:
                dp = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:

                dp = np.linalg.solve(lhs + 1e-6 * np.eye(self.n_params), rhs)
            params_new = params + dp
            if np.linalg.norm(dp) < tol:
                break

            params_new = np.maximum(params_new, 1e-6)
            params = params_new

        J = self.jacobian(params)
        try:
            cov = np.linalg.inv(J.T @ W @ J)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(J.T @ W @ J)
        return params, cov





def compute_fisher_matrix(l: np.ndarray, params: np.ndarray,
                          sigma_Cl: np.ndarray) -> np.ndarray:
    fitter = CosmologyFitter(l, np.zeros_like(l), sigma_Cl,
                             ["A_s", "n_s", "omb", "omc", "h"])
    J = fitter.jacobian(params)
    W = np.diag(1.0 / (sigma_Cl ** 2))
    return J.T @ W @ J
