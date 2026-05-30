# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional, Callable


class NewmarkBetaIntegrator:

    def __init__(
        self,
        M: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        gamma: float = 0.5,
        beta: float = 0.25,
        dt: float = 0.01,
        max_iter: int = 10,
        tol: float = 1e-8,
    ):
        self.M = np.asarray(M, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.dt = float(dt)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_dof = self.M.shape[0]



        pass




    def step(
        self,
        u_n: np.ndarray,
        v_n: np.ndarray,
        a_n: np.ndarray,
        a_g: float,
        iso_force_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        solver_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        Gamma: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = self.dt
        beta = self.beta
        gamma = self.gamma


        a_star = (
            (1.0 / (2.0 * beta) - 1.0) * a_n
            + (1.0 / (beta * dt)) * v_n
            + (1.0 / (beta * dt ** 2)) * u_n
        )
        v_star = v_n + dt * (1.0 - gamma) * a_n


        F_ext = -self.M @ Gamma * a_g


        u_new = u_n.copy()


        for _iter in range(self.max_iter):

            a_new = (1.0 / (beta * dt ** 2)) * (u_new - u_n) - (1.0 / (beta * dt)) * v_n - (1.0 / (2.0 * beta) - 1.0) * a_n
            v_new = v_star + gamma * dt * a_new


            F_iso = iso_force_func(u_new, v_new)


            R = F_ext - self.M @ a_new - self.C @ v_new - self.K @ u_new - F_iso




            K_eff = self._K_eff.copy()


            du = solver_func(K_eff, R)


            u_new = u_new + du


            norm_du = float(np.linalg.norm(du))
            norm_u = float(np.linalg.norm(u_new))
            if norm_u > 1e-12:
                rel_err = norm_du / norm_u
            else:
                rel_err = norm_du

            if rel_err < self.tol:
                break
        else:

            pass


        a_new = (
            (1.0 / (beta * dt ** 2)) * (u_new - u_n)
            - (1.0 / (beta * dt)) * v_n
            - (1.0 / (2.0 * beta) - 1.0) * a_n
        )
        v_new = v_star + gamma * dt * a_new

        return u_new, v_new, a_new




    def integrate(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        a0: np.ndarray,
        a_g: np.ndarray,
        iso_force_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        solver_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        Gamma: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_time = len(a_g)
        n_dof = self.n_dof

        U = np.zeros((n_time, n_dof), dtype=float)
        V = np.zeros((n_time, n_dof), dtype=float)
        A = np.zeros((n_time, n_dof), dtype=float)

        U[0, :] = u0
        V[0, :] = v0
        A[0, :] = a0

        u_n = u0.copy()
        v_n = v0.copy()
        a_n = a0.copy()

        for i in range(1, n_time):
            u_n, v_n, a_n = self.step(
                u_n, v_n, a_n, a_g[i], iso_force_func, solver_func, Gamma
            )
            U[i, :] = u_n
            V[i, :] = v_n
            A[i, :] = a_n

        return U, V, A





def backward_euler_step(
    y_n: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    dt: float,
    max_inner_iter: int = 10,
) -> np.ndarray:
    y = y_n.copy()
    for _ in range(max_inner_iter):
        y_new = y_n + dt * f_func(y)
        if np.linalg.norm(y_new - y) < 1e-12:
            break
        y = y_new
    return y
