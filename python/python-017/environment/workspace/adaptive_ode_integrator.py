
import numpy as np
from typing import Callable, Tuple, Optional


class AdaptiveMidpointIntegrator:

    def __init__(self, reltol: float = 1e-6, abstol: float = 1e-8,
                 kappa: float = 0.85, theta: float = 0.5,
                 max_iter: int = 50, newton_tol: float = 1e-10):
        self.reltol = reltol
        self.abstol = abstol
        self.kappa = kappa
        self.theta = theta
        self.max_iter = max_iter
        self.newton_tol = newton_tol

    def _newton_solve(self, y_n: np.ndarray, tau: float,
                      f: Callable[[float, np.ndarray], np.ndarray],
                      t_mid: float) -> np.ndarray:
        m = len(y_n)
        Y = y_n.copy()

        for _ in range(self.max_iter):
            f_val = f(t_mid, Y)

            J = np.zeros((m, m))
            eps = 1e-8
            for j in range(m):
                Y_pert = Y.copy()
                Y_pert[j] += eps
                f_pert = f(t_mid, Y_pert)
                J[:, j] = (f_pert - f_val) / eps


            g = Y - y_n - tau * f_val
            if np.linalg.norm(g) < self.newton_tol:
                break


            A = np.eye(m) - tau * J
            try:
                delta = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:

                A += np.eye(m) * 1e-10
                delta = np.linalg.solve(A, g)
            Y = Y - delta


            if not np.all(np.isfinite(Y)):
                Y = y_n.copy()
                break

        return Y

    def _lte_estimate(self, y_prev2: np.ndarray, y_prev1: np.ndarray,
                      y_new: np.ndarray, tau_n: float, tau_nm1: float) -> float:
        if tau_nm1 <= 0:
            return 0.0

        y_euler = y_prev1 + tau_n * self._last_f_prev1
        scale = self.reltol * np.maximum(np.maximum(np.abs(y_new), np.abs(y_prev1)), self.abstol) + self.abstol

        lte = np.max(np.abs(y_new - y_euler) / scale) * tau_n
        if lte <= 0 or not np.isfinite(lte):
            lte = 1e-20
        return lte

    def integrate(self, f: Callable[[float, np.ndarray], np.ndarray],
                  t0: float, tmax: float, y0: np.ndarray,
                  tau0: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
        m = len(y0)
        t_list = [t0]
        y_list = [y0.copy()]
        tau = tau0
        n_rejected = 0
        n_fsolve_fail = 0

        t = t0
        y = y0.copy()


        y_prev2 = y0.copy()
        y_prev1 = y0.copy()
        self._last_f_prev1 = f(t0, y0)

        count = 0
        while t < tmax and count < 100000:
            count += 1
            tau = min(tau, tmax - t)
            if tau <= 1e-30:
                break

            t_mid = t + 0.5 * tau
            try:
                Y = self._newton_solve(y, tau, f, t_mid)
            except Exception:
                n_fsolve_fail += 1
                tau *= 0.5
                n_rejected += 1
                continue

            y_new = y + tau * f(t_mid, Y)


            if count >= 3:
                lte = self._lte_estimate(y_prev2, y_prev1, y_new, tau,
                                          t_list[-1] - t_list[-2] if len(t_list) > 1 else tau)
                tnmax = max(lte, 1e-20)

                if tnmax > 1.0:

                    n_rejected += 1
                    factor = self.kappa * (1.0 / tnmax) ** (1.0 / 3.0)
                    factor = np.clip(factor, 0.02, 1.5)
                    tau *= factor
                    continue
                else:

                    factor = self.kappa * (1.0 / tnmax) ** (1.0 / 3.0)
                    factor = np.clip(factor, 0.02, 1.5)
                    tau *= factor
            else:

                pass


            y_prev2 = y_prev1.copy()
            y_prev1 = y.copy()
            y = y_new.copy()
            t += tau if count >= 3 else tau
            t_list.append(t)
            y_list.append(y.copy())
            self._last_f_prev1 = f(t, y)

        t_arr = np.array(t_list)
        y_arr = np.array(y_list)
        nstep = len(t_list) - 1
        return t_arr, y_arr, nstep, n_rejected


def tdgl_rhs(t: float, state: np.ndarray,
             gamma_P: float, gamma_M: float,
             dFdP_func: Callable[[np.ndarray], np.ndarray],
             dFdM_func: Callable[[np.ndarray], np.ndarray],
             noise_P: Optional[np.ndarray] = None,
             noise_M: Optional[np.ndarray] = None) -> np.ndarray:
    half = len(state) // 2
    P = state[:half]
    M = state[half:]

    dFdP = dFdP_func(P)
    dFdM = dFdM_func(M)

    dPdt = -gamma_P * dFdP
    dMdt = -gamma_M * dFdM

    if noise_P is not None:
        dPdt += noise_P
    if noise_M is not None:
        dMdt += noise_M

    return np.concatenate([dPdt, dMdt])
