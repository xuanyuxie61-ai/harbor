
import numpy as np
from typing import Callable, Optional


class InverseEstimator:

    def __init__(self, residual_func: Callable[[np.ndarray], np.ndarray],
                 param_names: list[str],
                 param_lower_bounds: Optional[np.ndarray] = None,
                 param_upper_bounds: Optional[np.ndarray] = None):
        self.f = residual_func
        self.param_names = param_names
        self.n_params = len(param_names)
        self.lb = param_lower_bounds
        self.ub = param_upper_bounds

    def _compute_jacobian_fd(self, beta: np.ndarray, h: float = 1e-6) -> np.ndarray:
        beta = np.asarray(beta, dtype=float)
        if len(beta) != self.n_params:
            raise ValueError("参数向量维度不匹配")
        f0 = self.f(beta)
        M = len(f0)
        J = np.zeros((M, self.n_params))
        for j in range(self.n_params):
            beta_plus = beta.copy()
            beta_minus = beta.copy()

            hj = h * max(abs(beta[j]), 1.0)
            beta_plus[j] += hj
            beta_minus[j] -= hj
            f_plus = self.f(beta_plus)
            f_minus = self.f(beta_minus)
            J[:, j] = (f_plus - f_minus) / (2.0 * hj)
        return J

    def _clip_parameters(self, beta: np.ndarray) -> np.ndarray:
        if self.lb is not None:
            beta = np.maximum(beta, self.lb)
        if self.ub is not None:
            beta = np.minimum(beta, self.ub)
        return beta

    def solve_gauss_newton(self, beta0: np.ndarray,
                           max_iter: int = 50,
                           tol: float = 1e-6,
                           damping: float = 1.0) -> dict:
        beta = np.asarray(beta0, dtype=float).copy()
        beta = self._clip_parameters(beta)
        residuals_history = []
        param_history = [beta.copy()]

        for it in range(max_iter):
            f_val = self.f(beta)
            rms = np.sqrt(np.mean(f_val ** 2))
            residuals_history.append(rms)

            J = self._compute_jacobian_fd(beta)
            grad = J.T @ f_val
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tol:
                break


            JTJ = J.T @ J

            JTJ += np.eye(self.n_params) * 1e-8 * np.trace(JTJ)
            try:
                delta = np.linalg.solve(JTJ, -grad)
            except np.linalg.LinAlgError:
                delta = -grad / (grad_norm + 1e-15)


            alpha = damping
            for _ in range(10):
                beta_new = self._clip_parameters(beta + alpha * delta)
                f_new = self.f(beta_new)
                if np.mean(f_new ** 2) < np.mean(f_val ** 2):
                    beta = beta_new
                    break
                alpha *= 0.5
            else:
                beta = self._clip_parameters(beta + alpha * delta)

            param_history.append(beta.copy())

        final_f = self.f(beta)
        return {
            "beta_opt": beta,
            "param_names": self.param_names,
            "final_rms": float(np.sqrt(np.mean(final_f ** 2))),
            "residuals_history": residuals_history,
            "param_history": param_history,
            "n_iter": len(residuals_history),
            "jacobian_condition": float(np.linalg.cond(self._compute_jacobian_fd(beta)))
        }

    def solve_lm(self, beta0: np.ndarray,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 lambda_init: float = 0.01) -> dict:
        beta = np.asarray(beta0, dtype=float).copy()
        beta = self._clip_parameters(beta)
        lam = lambda_init
        residuals_history = []

        for it in range(max_iter):








            raise NotImplementedError("Hole 3: 请实现 LM 迭代核心逻辑")

        final_f = self.f(beta)
        return {
            "beta_opt": beta,
            "param_names": self.param_names,
            "final_rms": float(np.sqrt(np.mean(final_f ** 2))),
            "residuals_history": residuals_history,
            "n_iter": len(residuals_history),
        }






def sherman_morrison_solve(A_inv_u: np.ndarray, v: np.ndarray,
                           A_inv_b: np.ndarray,
                           alpha: float = 1.0) -> np.ndarray:
    denom = 1.0 + alpha * np.dot(v, A_inv_u)
    if abs(denom) < 1e-14:
        raise ValueError("Sherman-Morrison 分母接近零，秩-1 更新奇异")
    return A_inv_b - alpha * A_inv_u * (np.dot(v, A_inv_b) / denom)


def lu_factor_solve(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    return x, lu, piv


if __name__ == "__main__":

    x_data = np.linspace(0, 2, 20)
    y_true = 3.0 * np.exp(-1.5 * x_data)
    noise = np.random.default_rng(0).normal(0, 0.05, size=len(x_data))
    y_obs = y_true + noise

    def residual(beta):
        a, b_param = beta
        return a * np.exp(-b_param * x_data) - y_obs

    est = InverseEstimator(residual, ["a", "b"], lb=np.array([0.1, 0.1]), ub=np.array([10.0, 5.0]))
    result = est.solve_lm(np.array([1.0, 1.0]))
    assert result["final_rms"] < 0.1


    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x0 = np.linalg.solve(A, b)
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    A_inv_u = np.linalg.solve(A, u)
    x_sm = sherman_morrison_solve(A_inv_u, v, x0, alpha=1.0)
    x_direct = np.linalg.solve(A + np.outer(u, v), b)
    assert np.allclose(x_sm, x_direct)
    print("inverse_estimator: 自测试通过")
