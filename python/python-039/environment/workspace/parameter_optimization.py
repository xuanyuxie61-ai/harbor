
import numpy as np
from typing import Callable, Tuple, Optional, Dict


class QuadraticOptimizer:

    def __init__(self, max_iter: int = 50, x_tol: float = 1e-6,
                 f_tol: float = 1e-8):
        self.max_iter = max_iter
        self.x_tol = x_tol
        self.f_tol = f_tol

    def optimize(self, f: Callable[[float], float],
                 x1: float, x2: float, x3: float) -> Tuple[float, int, float]:

        if abs(x1 - x2) < 1e-15 or abs(x2 - x3) < 1e-15 or abs(x1 - x3) < 1e-15:
            raise ValueError("三个初始点必须互不相同")

        x = [x1, x2, x3]
        fx = [f(xi) for xi in x]

        for it in range(1, self.max_iter + 1):

            idx = np.argsort(fx)
            x = [x[i] for i in idx]
            fx = [fx[i] for i in idx]


            V = np.array([
                [x[0] ** 2, x[0], 1.0],
                [x[1] ** 2, x[1], 1.0],
                [x[2] ** 2, x[2], 1.0]
            ])
            try:
                p = np.linalg.solve(V, fx)
            except np.linalg.LinAlgError:
                break

            a, b, c = p

            if abs(a) < 1e-15:

                x_star = (x[0] + x[2]) / 2.0
            else:
                x_star = -b / (2.0 * a)

            f_star = f(x_star)


            if abs(x_star - x[1]) < self.x_tol and abs(f_star - fx[1]) < self.f_tol:
                return x_star, it, f_star


            if f_star < fx[0]:
                x[2] = x[1]
                fx[2] = fx[1]
                x[1] = x_star
                fx[1] = f_star
            elif f_star < fx[1]:
                x[2] = x[1]
                fx[2] = fx[1]
                x[1] = x_star
                fx[1] = f_star
            elif f_star < fx[2]:
                x[2] = x_star
                fx[2] = f_star
            else:

                x[2] = (x[1] + x[2]) / 2.0
                fx[2] = f(x[2])


        idx_best = np.argmin(fx)
        return x[idx_best], self.max_iter, fx[idx_best]


class QGPParameterFit:

    def __init__(self, optimizer: Optional[QuadraticOptimizer] = None):
        self.optimizer = optimizer if optimizer is not None else QuadraticOptimizer()

    def fit_eta_over_s(self, v2_data: np.ndarray,
                       pt_bins: np.ndarray,
                       centrality: str = '0-5%') -> Tuple[float, float]:

        def theory_v2(eta_s):
            alpha = 5.0
            base_v2 = 0.05 * np.tanh(pt_bins / 2.0)
            suppression = 1.0 / (1.0 + alpha * max(eta_s - 0.08, 0.0))
            return base_v2 * suppression

        def chi2(eta_s):
            theory = theory_v2(eta_s)
            err = v2_data * 0.1 + 0.001
            chi2_val = np.sum(((v2_data - theory) / err) ** 2)
            return chi2_val


        x_opt, iters, chi2_min = self.optimizer.optimize(chi2, 0.0, 0.08, 0.5)
        x_opt = np.clip(x_opt, 0.0, 1.0)
        return float(x_opt), float(chi2_min)

    def fit_cs2(self, mean_pt_data: float,
                T_range: Tuple[float, float] = (0.15, 0.4)) -> Tuple[float, float]:
        T_mid = (T_range[0] + T_range[1]) / 2.0
        pt_thermal = 2.1 * T_mid
        alpha_flow = 1.5

        def model_mean_pt(cs2):
            if cs2 < 0.01:
                return 1e6
            return pt_thermal * (1.0 + alpha_flow * (cs2 - 1.0 / 3.0))

        def residual(cs2):
            return (model_mean_pt(cs2) - mean_pt_data) ** 2

        x_opt, iters, res = self.optimizer.optimize(residual, 0.1, 1.0 / 3.0, 0.5)
        x_opt = np.clip(x_opt, 0.05, 0.5)
        return float(x_opt), float(res)

    def fit_tau0(self, dNch_deta_data: float,
                 epsilon0: float = 15.0,
                 area: float = 150.0) -> Tuple[float, float]:


        def model_dn(tau0):
            return (4.0 / 9.0) * tau0 * s0 * area

        def residual(tau0):
            if tau0 < 0.1:
                return 1e6
            return (model_dn(tau0) - dNch_deta_data) ** 2

        x_opt, iters, res = self.optimizer.optimize(residual, 0.2, 0.6, 2.0)
        x_opt = np.clip(x_opt, 0.1, 5.0)
        return float(x_opt), float(res)

    def fit_all_parameters(self, v2_data: np.ndarray,
                           mean_pt: float,
                           dNch_deta: float) -> Dict[str, float]:
        pt_bins = np.linspace(0.5, 5.0, len(v2_data))

        eta_s, chi2_v2 = self.fit_eta_over_s(v2_data, pt_bins)
        cs2, res_pt = self.fit_cs2(mean_pt)
        tau0, res_dn = self.fit_tau0(dNch_deta)

        return {
            'eta_over_s': eta_s,
            'cs2': cs2,
            'tau0': tau0,
            'chi2_v2': chi2_v2,
            'residual_pt': res_pt,
            'residual_dNch': res_dn
        }
