
import numpy as np
from typing import List, Tuple, Callable
from linear_algebra import r83p_solve, toeplitz_cholesky_lower


def sample_fisher_information_matrix(states: List[np.ndarray],
                                      actions: List[np.ndarray],
                                      policy_grad_func: Callable,
                                      num_params: int) -> np.ndarray:
    N = len(states)
    if N == 0:
        return np.eye(num_params) * 1.0e-6
    F = np.zeros((num_params, num_params))
    for s, a in zip(states, actions):
        g = policy_grad_func(s, a)
        F += np.outer(g, g)
    F = F / N

    reg = 1.0e-4 * np.trace(F) / num_params
    F = F + reg * np.eye(num_params)
    return F


def conjugate_gradient_solve(A_func: Callable, b: np.ndarray,
                              max_iter: int = 50, tol: float = 1.0e-10,
                              damping: float = 1.0e-3) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs_old = float(r @ r)
    for _ in range(max_iter):
        Ap = A_func(p) + damping * p
        pAp = float(p @ Ap)
        if abs(pAp) < 1.0e-15:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x


def fisher_vector_product(states: List[np.ndarray],
                          actions: List[np.ndarray],
                          policy_grad_func: Callable,
                          v: np.ndarray) -> np.ndarray:

    pass


def natural_gradient_update(theta: np.ndarray,
                            grad: np.ndarray,
                            states: List[np.ndarray],
                            actions: List[np.ndarray],
                            policy_grad_func: Callable,
                            method: str = 'cg',
                            cg_iter: int = 50,
                            cg_damping: float = 1.0e-3) -> np.ndarray:
    if method == 'cg':
        def A_func(v):
            return fisher_vector_product(states, actions, policy_grad_func, v)
        ng = conjugate_gradient_solve(A_func, grad, max_iter=cg_iter,
                                       damping=cg_damping)
    elif method == 'direct':
        num_params = len(theta)
        F = sample_fisher_information_matrix(states, actions, policy_grad_func, num_params)
        try:
            ng = np.linalg.solve(F, grad)
        except np.linalg.LinAlgError:
            ng = np.linalg.lstsq(F, grad, rcond=None)[0]
    else:

        def A_func(v):
            return fisher_vector_product(states, actions, policy_grad_func, v)
        ng = conjugate_gradient_solve(A_func, grad, max_iter=cg_iter,
                                       damping=cg_damping)
    return ng


class NaturalPolicyGradientOptimizer:

    def __init__(self, learning_rate: float = 0.01,
                 cg_iter: int = 50, cg_damping: float = 1.0e-3,
                 max_kl: float = 0.01):
        self.lr = learning_rate
        self.cg_iter = cg_iter
        self.cg_damping = cg_damping
        self.max_kl = max_kl

    def step(self, theta: np.ndarray, grad: np.ndarray,
             states: List[np.ndarray], actions: List[np.ndarray],
             policy_grad_func: Callable) -> np.ndarray:
        ng = natural_gradient_update(
            theta, grad, states, actions, policy_grad_func,
            method='cg', cg_iter=self.cg_iter, cg_damping=self.cg_damping
        )

        step_size = self.lr
        for _ in range(10):
            theta_new = theta + step_size * ng

            Fv = fisher_vector_product(states, actions, policy_grad_func, ng)
            kl_approx = 0.5 * step_size ** 2 * (ng @ Fv)
            if kl_approx <= self.max_kl:
                break
            step_size *= 0.5
        return theta + step_size * ng
