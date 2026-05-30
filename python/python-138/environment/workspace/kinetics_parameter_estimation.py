
import numpy as np
from typing import Tuple, Optional


class KineticsParameterEstimator:

    def __init__(self, R_gas: float = 8.314):
        self.R_gas = R_gas

    def _householder_reflection(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        n = len(x)
        if n == 0:
            return np.array([]), 0.0
        sigma = np.dot(x[1:], x[1:])
        v = np.zeros(n)
        v[0] = 1.0
        v[1:] = x[1:]
        if sigma == 0.0:
            return v, 0.0
        mu = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0.0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        beta = 2.0 / (v[0] ** 2 + sigma)
        return v, beta

    def qr_factorize(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = A.shape
        if m < n:
            raise ValueError("QR 分解要求行数不少于列数")
        R = A.copy().astype(float)
        Q = np.eye(m)
        for k in range(min(m - 1, n)):
            x = R[k:, k].copy()
            v, beta = self._householder_reflection(x)
            if beta == 0.0:
                continue

            for j in range(k, n):
                tau = beta * np.dot(v, R[k:, j])
                R[k:, j] -= tau * v

            for j in range(m):
                tau = beta * np.dot(v, Q[k:, j])
                Q[k:, j] -= tau * v
        Q = Q.T
        return Q, R

    def solve_least_squares(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        m, n = A.shape
        if len(b) != m:
            raise ValueError("A 与 b 维度不匹配")
        Q, R = self.qr_factorize(A)

        R_tri = R[:n, :n]
        Qtb = Q[:m, :n].T @ b

        x = self._back_substitution(R_tri, Qtb)
        return x

    def _back_substitution(self, U: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(b)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < 1.0e-14:
                x[i] = 0.0
            else:
                x[i] = (b[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
        return x

    def _predict_rate(
        self, C: np.ndarray, T: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        lnA, Ea, n = params
        A = np.exp(lnA)
        T_safe = np.maximum(T, 200.0)
        C_safe = np.maximum(C, 1.0e-12)
        rate = A * np.exp(-Ea / (self.R_gas * T_safe)) * (C_safe ** max(n, 0.0))
        return np.maximum(rate, 1.0e-20)

    def _gauss_newton_step(
        self,
        C: np.ndarray,
        T: np.ndarray,
        r_exp: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        r_pred = self._predict_rate(C, T, params)
        f = r_pred - r_exp
        lnA, Ea, n = params
        A = np.exp(lnA)
        T_safe = np.maximum(T, 200.0)
        C_safe = np.maximum(C, 1.0e-12)
        exp_term = np.exp(-Ea / (self.R_gas * T_safe))


        dr_dlnA = r_pred
        dr_dEa = -r_pred / (self.R_gas * T_safe)
        dr_dn = r_pred * np.log(C_safe)

        J = np.column_stack([dr_dlnA, dr_dEa, dr_dn])

        JTJ = J.T @ J
        JTf = J.T @ f

        lam_damp = 1.0e-3 * np.trace(JTJ) / 3.0
        H = JTJ + lam_damp * np.eye(3)
        try:
            delta = np.linalg.solve(H, -JTf)
        except np.linalg.LinAlgError:
            delta = -JTf / (np.diag(JTJ) + 1.0e-10)
        return delta

    def estimate_arrhenius_parameters(
        self,
        concentrations: np.ndarray,
        temperatures: np.ndarray,
        rates: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        concentrations = np.asarray(concentrations, dtype=float)
        temperatures = np.asarray(temperatures, dtype=float)
        rates = np.asarray(rates, dtype=float)

        if len(concentrations) < 3:
            raise ValueError("至少需要 3 组数据才能估计 3 个参数")


        valid = (concentrations > 0.0) & (temperatures > 0.0) & (rates > 0.0)
        if np.sum(valid) < 3:
            raise ValueError("有效数据点不足")

        C = concentrations[valid]
        T = temperatures[valid]
        r = rates[valid]


        best_res = float("inf")
        best_params = np.array([np.log(1.0e4), 10000.0, 1.0])
        for n_try in np.linspace(0.0, 2.5, 26):
            y_try = np.log(r) - n_try * np.log(C)
            X_try = np.column_stack([np.ones(len(y_try)), -1.0 / (self.R_gas * T)])
            try:
                beta_try, *_ = np.linalg.lstsq(X_try, y_try, rcond=None)
                lnA_try = beta_try[0]
                Ea_try = max(beta_try[1], 100.0)
                r_pred_try = np.exp(lnA_try) * np.exp(-Ea_try / (self.R_gas * T)) * (C ** n_try)
                res_try = np.linalg.norm(r - r_pred_try)
                if res_try < best_res:
                    best_res = res_try
                    best_params = np.array([lnA_try, Ea_try, n_try])
            except Exception:
                continue

        params = best_params


        for it in range(100):
            delta = self._gauss_newton_step(C, T, r, params)

            alpha = 1.0
            current_res = np.linalg.norm(self._predict_rate(C, T, params) - r)
            for _ in range(10):
                params_new = params + alpha * delta

                params_new[1] = max(params_new[1], 500.0)
                params_new[2] = max(params_new[2], 0.0)
                new_res = np.linalg.norm(self._predict_rate(C, T, params_new) - r)
                if new_res < current_res:
                    break
                alpha *= 0.5
            else:
                break

            if np.linalg.norm(alpha * delta) < 1.0e-8:
                break
            params = params_new

        lnA_est, Ea_est, n_est = params
        A_est = np.exp(lnA_est)
        A_est = max(A_est, 1.0e-6)
        Ea_est = max(Ea_est, 500.0)
        n_est = max(n_est, 0.0)

        r_pred = self._predict_rate(C, T, params)
        residual_norm = np.linalg.norm(r - r_pred)

        return A_est, Ea_est, n_est, residual_norm

    def compute_confidence_intervals(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x_est: np.ndarray,
    ) -> np.ndarray:
        Q, R = self.qr_factorize(A)
        n = x_est.shape[0]
        R1 = R[:n, :n]

        try:
            inv_R = np.linalg.inv(R1)
            cov = inv_R @ inv_R.T
            std_dev = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            std_dev = np.full(n, np.inf)
        return std_dev
