
import numpy as np
from math import log, exp, sqrt, pi, erf
from sparse_matrix_ccs import SparseMatrixCCS
from gmres_iterative import gmres_dense
from special_math_utils import MeshDataManager, ellipse_area_matrix


class HestonPDESolver:

    def __init__(self, S_max, v_max, T, r, kappa, theta, sigma, rho,
                 n_S=80, n_v=40, n_t=100, scheme='CS'):
        self.S_max = float(S_max)
        self.v_max = float(v_max)
        self.T = float(T)
        self.r = float(r)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.n_S = n_S
        self.n_v = n_v
        self.n_t = n_t
        self.scheme = scheme


        self.feller_ratio = 2.0 * kappa * theta / (sigma ** 2)
        if self.feller_ratio < 1.0:

            pass


        self._generate_grid()

        self._build_operators()

    def _generate_grid(self):


        c_s = 5.0
        xi_s = np.linspace(0.0, 1.0, self.n_S + 1)
        self.S_grid = self.S_max * np.sinh(c_s * xi_s) / np.sinh(c_s)
        self.S_grid[0] = 0.0
        self.S_grid[-1] = self.S_max
        self.dS = np.diff(self.S_grid)


        c_v = 8.0
        xi_v = np.linspace(0.0, 1.0, self.n_v + 1)
        self.v_grid = self.v_max * (np.exp(c_v * xi_v) - 1.0) / (np.exp(c_v) - 1.0)
        self.v_grid[0] = 0.0
        self.v_grid[-1] = self.v_max
        self.dv = np.diff(self.v_grid)

        self.dt = self.T / self.n_t

    def _build_operators(self):
        n_total = (self.n_S + 1) * (self.n_v + 1)
        self.n_total = n_total


        entries = {}

        def add_entry(row, col, val):
            key = (row, col)
            entries[key] = entries.get(key, 0.0) + val

        for j in range(1, self.n_v):
            vj = self.v_grid[j]
            dv_j = self.dv[j]
            dv_jm1 = self.dv[j - 1]
            dv_avg = 0.5 * (dv_j + dv_jm1)

            for i in range(1, self.n_S):
                Si = self.S_grid[i]
                dS_i = self.dS[i]
                dS_im1 = self.dS[i - 1]
                dS_avg = 0.5 * (dS_i + dS_im1)
                idx = j * (self.n_S + 1) + i


                coeff_SS = 0.5 * vj * Si ** 2
                alpha_S = 2.0 / (dS_im1 * dS_i * (dS_im1 + dS_i))
                add_entry(idx, idx, -coeff_SS * alpha_S * (dS_im1 + dS_i))
                add_entry(idx, idx + 1, coeff_SS * alpha_S * dS_im1)
                add_entry(idx, idx - 1, coeff_SS * alpha_S * dS_i)


                coeff_vv = 0.5 * self.sigma ** 2 * vj
                alpha_v = 2.0 / (dv_jm1 * dv_j * (dv_jm1 + dv_j))
                add_entry(idx, idx, -coeff_vv * alpha_v * (dv_jm1 + dv_j))
                add_entry(idx, idx + (self.n_S + 1), coeff_vv * alpha_v * dv_jm1)
                add_entry(idx, idx - (self.n_S + 1), coeff_vv * alpha_v * dv_j)



                coeff_Sv = self.rho * self.sigma * vj * Si
                if abs(coeff_Sv) > 1e-12:
                    cross = coeff_Sv / (4.0 * dS_avg * dv_avg)
                    add_entry(idx, idx + 1 + (self.n_S + 1), cross)
                    add_entry(idx, idx + 1 - (self.n_S + 1), -cross)
                    add_entry(idx, idx - 1 + (self.n_S + 1), -cross)
                    add_entry(idx, idx - 1 - (self.n_S + 1), cross)


                coeff_S1 = self.r * Si
                if coeff_S1 >= 0:

                    add_entry(idx, idx, -coeff_S1 / dS_i)
                    add_entry(idx, idx + 1, coeff_S1 / dS_i)
                else:
                    add_entry(idx, idx, coeff_S1 / dS_im1)
                    add_entry(idx, idx - 1, -coeff_S1 / dS_im1)


                coeff_v1 = self.kappa * (self.theta - vj)
                if coeff_v1 >= 0:
                    add_entry(idx, idx, -coeff_v1 / dv_j)
                    add_entry(idx, idx + (self.n_S + 1), coeff_v1 / dv_j)
                else:
                    add_entry(idx, idx, coeff_v1 / dv_jm1)
                    add_entry(idx, idx - (self.n_S + 1), -coeff_v1 / dv_jm1)


                add_entry(idx, idx, -self.r)



        for j in range(self.n_v + 1):
            idx = j * (self.n_S + 1)
            add_entry(idx, idx, 1.0)


        for j in range(self.n_v + 1):
            idx = j * (self.n_S + 1) + self.n_S
            add_entry(idx, idx, 1.0)


        for i in range(1, self.n_S):
            idx = i
            Si = self.S_grid[i]
            dS_i = self.dS[i]
            dS_im1 = self.dS[i - 1]


            coeff_SS = 0.0

            add_entry(idx, idx + 1, self.r * Si / (dS_i + dS_im1))
            add_entry(idx, idx - 1, -self.r * Si / (dS_i + dS_im1))

            dv0 = self.dv[0]
            add_entry(idx, idx, -self.kappa * self.theta / dv0)
            add_entry(idx, idx + (self.n_S + 1), self.kappa * self.theta / dv0)

            add_entry(idx, idx, -self.r)


        for i in range(self.n_S + 1):
            idx = self.n_v * (self.n_S + 1) + i
            add_entry(idx, idx, 1.0)
            if self.n_v >= 1:
                add_entry(idx, idx - (self.n_S + 1), -1.0)


        self.A_dense = np.zeros((n_total, n_total), dtype=np.float64)
        for (row, col), val in entries.items():
            self.A_dense[row, col] = val

    def _apply_boundary_rhs(self, V, t, K):
        rhs = np.zeros(self.n_total, dtype=np.float64)
        tau = self.T - t
        disc = exp(-self.r * tau)


        for j in range(self.n_v + 1):
            idx = j * (self.n_S + 1)
            rhs[idx] = 0.0


        for j in range(self.n_v + 1):
            idx = j * (self.n_S + 1) + self.n_S
            rhs[idx] = self.S_max - K * disc



        return rhs

    def solve_european_call(self, K):
        if K <= 0:
            raise ValueError("行权价K必须为正")


        V = np.zeros(self.n_total, dtype=np.float64)
        for j in range(self.n_v + 1):
            for i in range(self.n_S + 1):
                idx = j * (self.n_S + 1) + i
                Si = self.S_grid[i]
                V[idx] = max(Si - K, 0.0)











        raise NotImplementedError("Hole_2: 需要实现PDE时间推进与边界处理")

    def price_at_spot(self, V_surface, S0, v0):
        if S0 < 0 or S0 > self.S_max or v0 < 0 or v0 > self.v_max:
            raise ValueError("(S0, v0)超出网格范围")


        i = np.searchsorted(self.S_grid, S0) - 1
        i = max(0, min(i, self.n_S - 1))
        j = np.searchsorted(self.v_grid, v0) - 1
        j = max(0, min(j, self.n_v - 1))

        S0_l, S0_r = self.S_grid[i], self.S_grid[i + 1]
        v0_l, v0_r = self.v_grid[j], self.v_grid[j + 1]
        dS_cell = S0_r - S0_l
        dv_cell = v0_r - v0_l
        if dS_cell < 1e-15 or dv_cell < 1e-15:
            return V_surface[j, i]

        w_00 = (S0_r - S0) * (v0_r - v0) / (dS_cell * dv_cell)
        w_10 = (S0 - S0_l) * (v0_r - v0) / (dS_cell * dv_cell)
        w_01 = (S0_r - S0) * (v0 - v0_l) / (dS_cell * dv_cell)
        w_11 = (S0 - S0_l) * (v0 - v0_l) / (dS_cell * dv_cell)

        price = (w_00 * V_surface[j, i] +
                 w_10 * V_surface[j, i + 1] +
                 w_01 * V_surface[j + 1, i] +
                 w_11 * V_surface[j + 1, i + 1])
        return price


def heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0,
                               n_S=60, n_v=30, n_t=80):
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K, T必须为正")
    S_max = max(3.0 * K, 4.0 * S0)
    v_max = max(5.0 * theta, 3.0 * v0, 1.0)
    solver = HestonPDESolver(S_max, v_max, T, r, kappa, theta, sigma, rho,
                             n_S=n_S, n_v=n_v, n_t=n_t)
    V_surface = solver.solve_european_call(K)
    return solver.price_at_spot(V_surface, S0, v0)


def heston_pde_greeks(S0, K, T, r, kappa, theta, sigma, rho, v0,
                      d_param=1e-4):
    V0 = heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0)


    dS = max(S0 * d_param, 1e-3)
    V_up = heston_european_call_price(S0 + dS, K, T, r, kappa, theta, sigma, rho, v0,
                                      n_S=50, n_v=25, n_t=60)
    V_down = heston_european_call_price(S0 - dS, K, T, r, kappa, theta, sigma, rho, v0,
                                        n_S=50, n_v=25, n_t=60)
    delta = (V_up - V_down) / (2.0 * dS)


    dv = max(v0 * d_param, 1e-4)
    V_up = heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0 + dv,
                                      n_S=50, n_v=25, n_t=60)
    V_down = heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, max(v0 - dv, 1e-6),
                                        n_S=50, n_v=25, n_t=60)
    vega = (V_up - V_down) / (2.0 * dv)


    dT = max(T * d_param, 1e-4)
    if T > dT:
        V_T = heston_european_call_price(S0, K, T - dT, r, kappa, theta, sigma, rho, v0,
                                         n_S=50, n_v=25, n_t=60)
        theta_greek = -(V0 - V_T) / dT
    else:
        theta_greek = 0.0


    dr = 1e-4
    V_up = heston_european_call_price(S0, K, T, r + dr, kappa, theta, sigma, rho, v0,
                                      n_S=50, n_v=25, n_t=60)
    V_down = heston_european_call_price(S0, K, T, r - dr, kappa, theta, sigma, rho, v0,
                                        n_S=50, n_v=25, n_t=60)
    rho_greek = (V_up - V_down) / (2.0 * dr)

    return {
        'price': V0,
        'delta': delta,
        'vega': vega,
        'theta': theta_greek,
        'rho': rho_greek
    }
