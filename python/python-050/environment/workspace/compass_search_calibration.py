
import numpy as np
from typing import Callable, Tuple, Optional


class CompassSearchOptimizer:

    def __init__(self,
                 theta_init: np.ndarray,
                 theta_lower: np.ndarray,
                 theta_upper: np.ndarray,
                 delta_init: float = 1.0,
                 delta_tol: float = 1e-8,
                 k_max: int = 10000,
                 contraction_factor: float = 0.5):
        self.theta = np.asarray(theta_init, dtype=np.float64).copy()
        self.theta_lower = np.asarray(theta_lower, dtype=np.float64)
        self.theta_upper = np.asarray(theta_upper, dtype=np.float64)
        self.delta = float(delta_init)
        self.delta_tol = float(delta_tol)
        self.k_max = int(k_max)
        self.contraction_factor = float(contraction_factor)
        self.history = []

        if len(self.theta) != len(self.theta_lower) or len(self.theta) != len(self.theta_upper):
            raise ValueError("theta_init, theta_lower, theta_upper must have the same length.")

    def _project_to_bounds(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, self.theta_lower, self.theta_upper)

    def _evaluate_objective(self,
                            obj_func: Callable[[np.ndarray], float],
                            theta: np.ndarray) -> float:
        try:
            val = float(obj_func(theta))
            if not np.isfinite(val):
                return 1e20
            return val
        except Exception:
            return 1e20

    def optimize(self, obj_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        m = len(self.theta)
        theta_k = self._project_to_bounds(self.theta.copy())
        J_k = self._evaluate_objective(obj_func, theta_k)

        self.history = [(theta_k.copy(), J_k, self.delta)]

        for k in range(self.k_max):
            if self.delta < self.delta_tol:
                break

            improved = False
            theta_candidate = theta_k.copy()
            J_candidate = J_k


            for j in range(m):
                for sign in [-1.0, 1.0]:
                    theta_trial = theta_k.copy()
                    theta_trial[j] += sign * self.delta
                    theta_trial = self._project_to_bounds(theta_trial)


                    if np.allclose(theta_trial, theta_k, atol=1e-14):
                        continue

                    J_trial = self._evaluate_objective(obj_func, theta_trial)

                    if J_trial < J_candidate:
                        J_candidate = J_trial
                        theta_candidate = theta_trial
                        improved = True

            if improved:
                theta_k = theta_candidate
                J_k = J_candidate
            else:
                self.delta *= self.contraction_factor

            self.history.append((theta_k.copy(), J_k, self.delta))

        self.theta = theta_k
        return theta_k, J_k


def build_calibration_objective(
    observed_velocities: np.ndarray,
    model_velocity_func: Callable[[np.ndarray], np.ndarray],
    weights: Optional[np.ndarray] = None,
    prior_theta: Optional[np.ndarray] = None,
    prior_sigma: Optional[np.ndarray] = None,
    regularization_lambda: float = 0.01
) -> Callable[[np.ndarray], float]:
    obs = np.asarray(observed_velocities, dtype=np.float64)
    w = np.ones_like(obs) if weights is None else np.asarray(weights, dtype=np.float64)

    def objective(theta: np.ndarray) -> float:
        try:
            model_vel = np.asarray(model_velocity_func(theta), dtype=np.float64)
            if model_vel.shape != obs.shape:
                return 1e20
        except Exception:
            return 1e20

        residual = obs - model_vel
        misfit = 0.5 * np.sum(w * (residual ** 2))


        if prior_theta is not None and prior_sigma is not None:
            prior_term = 0.5 * regularization_lambda * np.sum(
                ((theta - prior_theta) / prior_sigma) ** 2
            )
            misfit += prior_term

        return float(misfit)

    return objective


def demo_calibration_problem(theta_true: np.ndarray,
                              noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Callable]:
    theta_true = np.asarray(theta_true, dtype=np.float64)
    a0, q, n = theta_true[:3]


    T_data = np.linspace(240.0, 270.0, 50)

    def model_func(theta: np.ndarray) -> np.ndarray:
        a, qv, nv = theta[:3]
        u = a * (T_data ** nv) * np.exp(-qv / T_data)
        return u

    u_true = model_func(theta_true)
    noise = noise_level * np.mean(np.abs(u_true)) * np.random.randn(len(T_data))
    observed = u_true + noise

    return theta_true, observed, model_func
