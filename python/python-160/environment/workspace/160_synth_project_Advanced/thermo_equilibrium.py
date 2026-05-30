
import math
import numpy as np
from stats_utils import ppchi2, gammad


R_GAS = 8.314462618
P_STD = 101325.0


class ThermoEquilibrium:

    def __init__(self, T=1073.0, P=101325.0):
        self.T = float(T)
        self.P = float(P)

        self.feed = np.array([1.0, 1.4, 0.6], dtype=float)

    def gibbs_free_energy_pure(self, species, T):

        coeffs = {
            'CO':   (-111527.0,  85.23, -0.0132,  0.0, 0.0),
            'CO2':  (-393546.0,  92.11, -0.0151,  0.0, 0.0),
            'H2':   (0.0,       -28.84,  0.0070,  0.0, 0.0),
            'H2O':  (-241826.0,  92.05, -0.0150,  0.0, 0.0),
            'CH4':  (-74863.0,   80.55, -0.0128,  0.0, 0.0),
        }
        if species not in coeffs:
            return 0.0
        a, b, c, d, e = coeffs[species]
        return a + b * T + c * T * math.log(T) + d * T ** 2 + e / T

    def equilibrium_constant(self, reaction, T):
        if T <= 0.0:
            return 1.0
        if reaction == 'WGS':


            dH = -41200.0
            dS = -42.3
        elif reaction == 'BOUDOUARD':


            dH = 172500.0
            dS = 176.5
        elif reaction == 'STEAM':


            dH = 131400.0
            dS = 134.2
        elif reaction == 'METHANATION':


            dH = -74800.0
            dS = -80.8
        else:
            return 1.0
        lnK = -dH / (R_GAS * T) + dS / R_GAS

        lnK = max(min(lnK, 700.0), -700.0)
        return math.exp(lnK)

    def solve_wgs_fixed_point(self, n_CO0, n_CO2_0, n_H2_0, n_H2O_0,
                              T, P, max_iter=100, tol=1.0e-8):
        K = self.equilibrium_constant('WGS', T)
        n_tot0 = n_CO0 + n_CO2_0 + n_H2_0 + n_H2O_0

        def g(xi):
            n_CO = max(n_CO0 - xi, 1.0e-15)
            n_CO2 = max(n_CO2_0 + xi, 1.0e-15)
            n_H2 = max(n_H2_0 + xi, 1.0e-15)
            n_H2O = max(n_H2O_0 - xi, 1.0e-15)
            n_tot = n_CO + n_CO2 + n_H2 + n_H2O
            if n_tot <= 1.0e-15:
                return 0.0

            rhs = (n_CO2 * n_H2) / (n_CO * n_H2O)


            return xi + 0.1 * (rhs - K)

        xi = 0.0
        for it in range(max_iter):
            xi_new = g(xi)
            if abs(xi_new - xi) < tol:
                break

            xi_max = min(n_CO0, n_H2O_0)
            xi_min = -min(n_CO2_0, n_H2_0)
            xi_new = max(xi_min, min(xi_max, xi_new))
            xi = xi_new
        else:

            xi = self._solve_wgs_quadratic(n_CO0, n_CO2_0, n_H2_0, n_H2O_0, K)

        return xi

    def _solve_wgs_quadratic(self, a0, b0, c0, d0, K):
        alpha = 1.0 - K
        beta = b0 + c0 + K * (a0 + d0)
        gamma = b0 * c0 - K * a0 * d0
        if abs(alpha) < 1.0e-12:
            if abs(beta) > 1.0e-12:
                xi = -gamma / beta
            else:
                xi = 0.0
        else:
            disc = beta ** 2 - 4.0 * alpha * gamma
            if disc < 0.0:
                disc = 0.0
            sqrt_disc = math.sqrt(disc)
            xi1 = (-beta + sqrt_disc) / (2.0 * alpha)
            xi2 = (-beta - sqrt_disc) / (2.0 * alpha)

            xi_max = min(a0, d0)
            xi_min = -min(b0, c0)
            candidates = []
            if xi_min <= xi1 <= xi_max:
                candidates.append(xi1)
            if xi_min <= xi2 <= xi_max:
                candidates.append(xi2)
            if candidates:
                xi = min(candidates, key=lambda x: abs(x))
            else:
                xi = 0.0
        return xi

    def newton_solve(self, func, dfunc, x0, tol=1.0e-10, max_iter=50,
                     f_max=1.0e12, df_min=1.0e-14):
        x = float(x0)
        fx = func(x)
        big = 100.0 * abs(fx)
        for it in range(max_iter):
            dfx = dfunc(x)
            if abs(dfx) <= df_min:
                return x, it, 'divergence_small_derivative'
            x_new = x - fx / dfx
            fx_new = func(x_new)
            it += 1
            x = x_new
            fx = fx_new
            if abs(fx) >= f_max:
                return x, it, 'divergence_large_function'
            if abs(fx) <= tol:
                return x, it, 'convergence'
        return x, max_iter, 'max_iterations'

    def solve_composition_newton(self, T, P, feed_C=1.0, feed_H2O=1.0,
                                 feed_O2=0.5):

        b = np.array([feed_C, 2.0 * feed_H2O, feed_H2O + 2.0 * feed_O2], dtype=float)


        x = np.array([0.3, 0.3, 0.4, 0.3, 0.1], dtype=float)

        for it in range(100):
            n_CO, n_CO2, n_H2, n_H2O, n_CH4 = x
            n_tot = max(n_CO + n_CO2 + n_H2 + n_H2O + n_CH4, 1.0e-15)








            K_wgs = self.equilibrium_constant('WGS', T)
            K_steam = self.equilibrium_constant('STEAM', T)

            f = np.zeros(5, dtype=float)
            f[0] = n_CO + n_CO2 + n_CH4 - b[0]
            f[1] = 2.0 * n_H2 + 2.0 * n_H2O + 4.0 * n_CH4 - b[1]
            f[2] = n_CO + 2.0 * n_CO2 + n_H2O - b[2]
            f[3] = (n_CO2 * n_H2) / max(n_CO * n_H2O, 1.0e-30) - K_wgs
            f[4] = (n_CO * n_H2 ** 3) / max(n_CH4 * n_H2O * n_tot ** 2, 1.0e-30) - K_steam


            eps = 1.0e-8
            J = np.zeros((5, 5), dtype=float)
            for j in range(5):
                x_pert = x.copy()
                x_pert[j] += eps
                n_CO_p, n_CO2_p, n_H2_p, n_H2O_p, n_CH4_p = x_pert
                n_tot_p = max(n_CO_p + n_CO2_p + n_H2_p + n_H2O_p + n_CH4_p, 1.0e-15)
                fp = np.zeros(5, dtype=float)
                fp[0] = n_CO_p + n_CO2_p + n_CH4_p - b[0]
                fp[1] = 2.0 * n_H2_p + 2.0 * n_H2O_p + 4.0 * n_CH4_p - b[1]
                fp[2] = n_CO_p + 2.0 * n_CO2_p + n_H2O_p - b[2]
                fp[3] = (n_CO2_p * n_H2_p) / max(n_CO_p * n_H2O_p, 1.0e-30) - K_wgs
                fp[4] = (n_CO_p * n_H2_p ** 3) / max(n_CH4_p * n_H2O_p * n_tot_p ** 2, 1.0e-30) - K_steam
                J[:, j] = (fp - f) / eps


            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -f, rcond=None)[0]


            alpha = 1.0
            for _ in range(10):
                x_new = x + alpha * dx
                x_new = np.maximum(x_new, 1.0e-12)
                f_new = self._residual_composition(x_new, b, T)
                if np.linalg.norm(f_new) < np.linalg.norm(f):
                    break
                alpha *= 0.5

            x = x_new
            if np.linalg.norm(f) < 1.0e-8:
                break

        return x

    def _residual_composition(self, x, b, T):
        n_CO, n_CO2, n_H2, n_H2O, n_CH4 = x
        n_tot = max(n_CO + n_CO2 + n_H2 + n_H2O + n_CH4, 1.0e-15)
        K_wgs = self.equilibrium_constant('WGS', T)
        K_steam = self.equilibrium_constant('STEAM', T)
        f = np.zeros(5, dtype=float)
        f[0] = n_CO + n_CO2 + n_CH4 - b[0]
        f[1] = 2.0 * n_H2 + 2.0 * n_H2O + 4.0 * n_CH4 - b[1]
        f[2] = n_CO + 2.0 * n_CO2 + n_H2O - b[2]
        f[3] = (n_CO2 * n_H2) / max(n_CO * n_H2O, 1.0e-30) - K_wgs
        f[4] = (n_CO * n_H2 ** 3) / max(n_CH4 * n_H2O * n_tot ** 2, 1.0e-30) - K_steam
        return f

    def chi_squared_test(self, observed, expected):
        observed = np.asarray(observed, dtype=float)
        expected = np.asarray(expected, dtype=float)
        mask = expected > 1.0e-15
        chi2 = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
        df = int(np.sum(mask) - 1)
        if df <= 0:
            return chi2, 1.0

        g_val = math.lgamma(df / 2.0)
        p_val, _ = gammad(chi2 / 2.0, df / 2.0)
        return chi2, 1.0 - p_val
