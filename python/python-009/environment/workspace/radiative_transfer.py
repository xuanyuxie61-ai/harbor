
import numpy as np
from typing import Tuple, Optional, Callable
from sparse_linear_algebra import CRSMatrix, crs_gmres


class RadiativeTransferSolver:

    def __init__(self, wavelength: np.ndarray, planet_radius_m: float):
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.R_p = planet_radius_m
        self.n_wl = len(wavelength)

    def compute_optical_depth(self, pressure: np.ndarray, temperature: np.ndarray,
                              abundance: dict, cross_sections: dict,
                              gravity: np.ndarray,
                              rayleigh_cross_section: Optional[np.ndarray] = None,
                              cloud_optical_depth: Optional[np.ndarray] = None) -> np.ndarray:
        n_layers = len(pressure)








        tau_cumulative = np.zeros((n_layers, self.n_wl), dtype=np.float64)
        return tau_cumulative

    def transit_depth_spectrum(self, pressure: np.ndarray, temperature: np.ndarray,
                               optical_depth: np.ndarray,
                               altitude: np.ndarray) -> np.ndarray:
        n_layers = len(pressure)
        transit_depth = np.zeros(self.n_wl, dtype=np.float64)

        for iwl in range(self.n_wl):
            tau_vert = optical_depth[:, iwl]



            if tau_vert[-1] < 1.0:

                z_eff = altitude[-1]
            elif tau_vert[0] > 1.0:

                z_eff = altitude[0]
            else:

                log_tau = np.log10(np.maximum(tau_vert, 1e-10))
                target_log_tau = 0.0
                z_eff = np.interp(target_log_tau, log_tau, altitude)


            R_eff = self.R_p + z_eff

            R_star = 10.0 * self.R_p
            depth = ((R_eff / R_star)**2 - (self.R_p / R_star)**2) * 1e6
            transit_depth[iwl] = max(depth, 0.0)

        return transit_depth

    def finite_element_rte_solve(self, mu: float, tau_grid: np.ndarray,
                                  source_function: np.ndarray,
                                  boundary_top: float = 0.0,
                                  boundary_bot: Optional[float] = None) -> np.ndarray:
        tau = np.asarray(tau_grid, dtype=np.float64)
        S = np.asarray(source_function, dtype=np.float64)
        n = len(tau)

        if n < 2:
            raise ValueError("光学深度网格至少需 2 个点")
        if len(S) != n:
            raise ValueError("源函数与网格维度不匹配")
        if abs(mu) < 1e-15:
            raise ValueError("μ 不能为零")



        adiag = np.zeros(n, dtype=np.float64)
        aleft = np.zeros(n, dtype=np.float64)
        arite = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)

        for i in range(n - 1):
            h = tau[i + 1] - tau[i]
            if h < 1e-30:
                continue


            gp = h / 2.0 * np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]) + (tau[i] + tau[i + 1]) / 2.0
            w = h / 2.0 * np.array([1.0, 1.0])

            for iq in range(2):
                t = gp[iq]
                ww = w[iq]


                N1 = (tau[i + 1] - t) / h
                N2 = (t - tau[i]) / h


                dN1 = -1.0 / h
                dN2 = 1.0 / h


                S_local = N1 * S[i] + N2 * S[i + 1]



                coeff = 1.0 / mu


                adiag[i] += ww * (N1 * dN1 - coeff * N1 * N1)
                arite[i] += ww * (N1 * dN2 - coeff * N1 * N2)
                rhs[i] += -ww * coeff * N1 * S_local


                aleft[i + 1] += ww * (N2 * dN1 - coeff * N2 * N1)
                adiag[i + 1] += ww * (N2 * dN2 - coeff * N2 * N2)
                rhs[i + 1] += -ww * coeff * N2 * S_local


        I_sol = np.zeros(n, dtype=np.float64)

        if mu > 0:

            adiag[0] = 1.0
            arite[0] = 0.0
            rhs[0] = boundary_top
            aleft[0] = 0.0

            if boundary_bot is not None:
                adiag[-1] = 1.0
                aleft[-1] = 0.0
                rhs[-1] = boundary_bot
        else:

            adiag[-1] = 1.0
            aleft[-1] = 0.0
            rhs[-1] = boundary_bot if boundary_bot is not None else 0.0
            arite[-1] = 0.0

            if boundary_top is not None:
                adiag[0] = 1.0
                arite[0] = 0.0
                rhs[0] = boundary_top


        I_sol = self._thomas_algorithm(aleft, adiag, arite, rhs)


        I_sol = self._diffuse_smooth(I_sol, iterations=1, c=0.1)

        return I_sol

    def _thomas_algorithm(self, a: np.ndarray, b: np.ndarray,
                          c: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = len(b)
        cp = np.zeros(n, dtype=np.float64)
        dp = np.zeros(n, dtype=np.float64)
        x = np.zeros(n, dtype=np.float64)

        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < 1e-30:
                denom = 1e-30
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]

        return x

    def _diffuse_smooth(self, arr: np.ndarray, iterations: int = 1,
                        c: float = 0.1) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64).copy()
        n = len(arr)
        if n < 3:
            return arr

        for _ in range(iterations):
            arr_new = arr.copy()
            arr_new[1:-1] = c * 0.5 * (arr[:-2] + arr[2:]) + (1.0 - c) * arr[1:-1]

            arr_new[0] = c * arr[1] + (1.0 - c) * arr[0]
            arr_new[-1] = c * arr[-2] + (1.0 - c) * arr[-1]
            arr = arr_new

        return arr

    def adaptive_rte_refinement(self, tau_grid: np.ndarray,
                                 source_function: np.ndarray,
                                 mu: float,
                                 tol: float = 1e-4,
                                 max_levels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(tau_grid, dtype=np.float64)
        S = np.asarray(source_function, dtype=np.float64)

        for level in range(max_levels):
            I_coarse = self.finite_element_rte_solve(mu, tau, S)


            tau_fine = np.zeros(2 * len(tau) - 1, dtype=np.float64)
            S_fine = np.zeros(2 * len(tau) - 1, dtype=np.float64)
            for i in range(len(tau) - 1):
                tau_fine[2 * i] = tau[i]
                tau_fine[2 * i + 1] = 0.5 * (tau[i] + tau[i + 1])
                S_fine[2 * i] = S[i]
                S_fine[2 * i + 1] = 0.5 * (S[i] + S[i + 1])
            tau_fine[-1] = tau[-1]
            S_fine[-1] = S[-1]

            I_fine = self.finite_element_rte_solve(mu, tau_fine, S_fine)


            error = np.abs(I_fine[::2] - I_coarse)
            max_err = np.max(error)

            if max_err < tol or level == max_levels - 1:
                return tau_fine, I_fine


            refine_mask = error[:-1] > tol
            if not np.any(refine_mask):
                return tau_fine, I_fine

            new_tau = [tau[0]]
            new_S = [S[0]]
            for i in range(len(tau) - 1):
                if refine_mask[i]:
                    new_tau.append(0.5 * (tau[i] + tau[i + 1]))
                    new_S.append(0.5 * (S[i] + S[i + 1]))
                new_tau.append(tau[i + 1])
                new_S.append(S[i + 1])

            tau = np.array(new_tau)
            S = np.array(new_S)

        return tau, self.finite_element_rte_solve(mu, tau, S)
