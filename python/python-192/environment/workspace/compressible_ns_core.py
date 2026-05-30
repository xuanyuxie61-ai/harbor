
import numpy as np
from utils_numerical import safe_divide, check_cfl, limiter_minmod, bisection_root_find
from linear_algebra_engine import lu_decomposition_with_pivot, solve_lu


class CompressibleNSSolver:

    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
                 gamma: float = 1.4, Re: float = 1000.0, Pr: float = 0.71,
                 Ma: float = 0.3, T_wall: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.gamma = gamma
        self.Re = Re
        self.Pr = Pr
        self.Ma = Ma
        self.T_wall = T_wall


        self.rho_inf = 1.0
        self.u_inf = 1.0
        self.T_inf = 1.0
        self.p_inf = self.rho_inf * self.T_inf / (gamma * self.Ma ** 2)


        self.mu = 1.0 / Re
        self.kappa = self.mu * gamma / (Pr * (gamma - 1.0))


        self.x = np.linspace(0.0, Lx, nx)
        self.y = np.linspace(0.0, Ly, ny)


        beta = 3.0
        eta = np.linspace(0.0, 1.0, ny)
        self.y = Ly * (np.exp(beta * eta) - 1.0) / (np.exp(beta) - 1.0)

        self.dx = self.x[1] - self.x[0]
        self.dy = np.diff(self.y)
        self.dy = np.concatenate([self.dy, [self.dy[-1]]])


        self.Q = np.zeros((ny, nx, 4))
        self.initialize_field()


        self.dt = 1e-6
        self.time = 0.0
        self.iter = 0

    def initialize_field(self):
        for j in range(self.ny):
            for i in range(self.nx):
                y_plus = self.y[j] * np.sqrt(self.Re / max(self.x[i], 0.01))

                u_profile = self.u_inf * np.tanh(y_plus * 0.5)

                if j == 0:
                    u_profile = 0.0

                T_profile = self.T_inf + 0.5 * (self.gamma - 1.0) * self.Ma ** 2 * \
                            (1.0 - (u_profile / self.u_inf) ** 2)

                if j == 0:
                    T_profile = self.T_wall

                rho = self.rho_inf / T_profile
                u = u_profile
                v = 0.0
                p = rho * T_profile / (self.gamma * self.Ma ** 2)
                e = p / (rho * (self.gamma - 1.0))
                E = e + 0.5 * (u ** 2 + v ** 2)

                self.Q[j, i, 0] = rho
                self.Q[j, i, 1] = rho * u
                self.Q[j, i, 2] = rho * v
                self.Q[j, i, 3] = rho * E

    def primitive_variables(self, Q: np.ndarray) -> tuple:
        rho = Q[..., 0]
        u = safe_divide(Q[..., 1], rho)
        v = safe_divide(Q[..., 2], rho)
        E = safe_divide(Q[..., 3], rho)
        e = E - 0.5 * (u ** 2 + v ** 2)
        e = np.maximum(e, 1e-14)
        p = (self.gamma - 1.0) * rho * e
        p = np.maximum(p, 1e-14)
        T = p / (rho * (self.gamma - 1.0)) * (self.gamma - 1.0) * self.gamma * self.Ma ** 2

        T = self.gamma * self.Ma ** 2 * p / rho
        return rho, u, v, p, e, T

    def compute_viscous_flux(self, Q: np.ndarray) -> tuple:
        rho, u, v, p, e, T = self.primitive_variables(Q)


        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        dT_dx = np.zeros_like(T)
        dT_dy = np.zeros_like(T)


        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * self.dx)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * self.dy[1:-1, None])
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * self.dx)
        dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * self.dy[1:-1, None])
        dT_dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * self.dx)
        dT_dy[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * self.dy[1:-1, None])


        du_dy[0, :] = (u[1, :] - u[0, :]) / self.dy[0]
        dv_dy[0, :] = (v[1, :] - v[0, :]) / self.dy[0]
        dT_dy[0, :] = (T[1, :] - T[0, :]) / self.dy[0]


        div_u = du_dx + dv_dy
        tau_xx = 2.0 * self.mu * du_dx - (2.0 / 3.0) * self.mu * div_u
        tau_yy = 2.0 * self.mu * dv_dy - (2.0 / 3.0) * self.mu * div_u
        tau_xy = self.mu * (du_dy + dv_dx)


        q_x = -self.kappa * dT_dx
        q_y = -self.kappa * dT_dy


        Fv = np.zeros_like(Q)
        Gv = np.zeros_like(Q)

        Fv[..., 1] = tau_xx
        Fv[..., 2] = tau_xy
        Fv[..., 3] = u * tau_xx + v * tau_xy - q_x

        Gv[..., 1] = tau_xy
        Gv[..., 2] = tau_yy
        Gv[..., 3] = u * tau_xy + v * tau_yy - q_y

        return Fv, Gv

    def roe_average(self, QL: np.ndarray, QR: np.ndarray) -> tuple:
        rhoL, uL, vL, pL, eL, TL = self.primitive_variables(QL)
        rhoR, uR, vR, pR, eR, TR = self.primitive_variables(QR)

        HL = (QL[..., 3] + pL) / rhoL
        HR = (QR[..., 3] + pR) / rhoR

        srL = np.sqrt(rhoL)
        srR = np.sqrt(rhoR)
        denom = srL + srR
        denom = np.where(denom < 1e-14, 1e-14, denom)

        rho_roe = srL * srR
        u_roe = (srL * uL + srR * uR) / denom
        v_roe = (srL * vL + srR * vR) / denom
        H_roe = (srL * HL + srR * HR) / denom

        c2 = (self.gamma - 1.0) * (H_roe - 0.5 * (u_roe ** 2 + v_roe ** 2))
        c2 = np.maximum(c2, 1e-14)
        c_roe = np.sqrt(c2)

        return rho_roe, u_roe, v_roe, H_roe, c_roe

    def roe_flux_x(self, QL: np.ndarray, QR: np.ndarray) -> np.ndarray:
        rhoL, uL, vL, pL, eL, TL = self.primitive_variables(QL)
        rhoR, uR, vR, pR, eR, TR = self.primitive_variables(QR)


        FL = np.zeros_like(QL)
        FR = np.zeros_like(QR)

        EL = eL + 0.5 * (uL ** 2 + vL ** 2)
        ER = eR + 0.5 * (uR ** 2 + vR ** 2)

        FL[..., 0] = rhoL * uL
        FL[..., 1] = rhoL * uL ** 2 + pL
        FL[..., 2] = rhoL * uL * vL
        FL[..., 3] = rhoL * uL * (EL + pL / rhoL)

        FR[..., 0] = rhoR * uR
        FR[..., 1] = rhoR * uR ** 2 + pR
        FR[..., 2] = rhoR * uR * vR
        FR[..., 3] = rhoR * uR * (ER + pR / rhoR)


        rho_roe, u_roe, v_roe, H_roe, c_roe = self.roe_average(QL, QR)


        lam1 = np.abs(u_roe - c_roe)
        lam2 = np.abs(u_roe)
        lam3 = np.abs(u_roe)
        lam4 = np.abs(u_roe + c_roe)


        eps = 0.05 * c_roe
        lam1 = np.where(lam1 < eps, 0.5 * (lam1 ** 2 / eps + eps), lam1)
        lam4 = np.where(lam4 < eps, 0.5 * (lam4 ** 2 / eps + eps), lam4)


        lam_scalar = 0.5 * (lam1 + lam4)
        lam_scalar = lam_scalar[..., None]
        diss = lam_scalar * (QR - QL)

        flux = 0.5 * (FL + FR) - 0.5 * diss
        return flux

    def roe_flux_y(self, QL: np.ndarray, QR: np.ndarray) -> np.ndarray:

        rhoL, uL, vL, pL, eL, TL = self.primitive_variables(QL)
        rhoR, uR, vR, pR, eR, TR = self.primitive_variables(QR)

        FL = np.zeros_like(QL)
        FR = np.zeros_like(QR)

        EL = eL + 0.5 * (uL ** 2 + vL ** 2)
        ER = eR + 0.5 * (uR ** 2 + vR ** 2)

        FL[..., 0] = rhoL * vL
        FL[..., 1] = rhoL * uL * vL
        FL[..., 2] = rhoL * vL ** 2 + pL
        FL[..., 3] = rhoL * vL * (EL + pL / rhoL)

        FR[..., 0] = rhoR * vR
        FR[..., 1] = rhoR * uR * vR
        FR[..., 2] = rhoR * vR ** 2 + pR
        FR[..., 3] = rhoR * vR * (ER + pR / rhoR)

        rho_roe, u_roe, v_roe, H_roe, c_roe = self.roe_average(QL, QR)

        lam1 = np.abs(v_roe - c_roe)
        lam2 = np.abs(v_roe)
        lam3 = np.abs(v_roe)
        lam4 = np.abs(v_roe + c_roe)

        eps = 0.05 * c_roe
        lam1 = np.where(lam1 < eps, 0.5 * (lam1 ** 2 / eps + eps), lam1)
        lam4 = np.where(lam4 < eps, 0.5 * (lam4 ** 2 / eps + eps), lam4)

        lam_scalar = 0.5 * (lam1 + lam4)
        lam_scalar = lam_scalar[..., None]
        diss = lam_scalar * (QR - QL)
        flux = 0.5 * (FL + FR) - 0.5 * diss
        return flux

    def muscl_reconstruct(self, Q: np.ndarray, axis: int = 0) -> tuple:
        if axis == 0:

            QL = Q[:, 1:-2, :].copy()
            QR = Q[:, 2:-1, :].copy()

            dQ_plus = Q[:, 2:-1, :] - Q[:, 1:-2, :]
            dQ_minus = Q[:, 1:-2, :] - Q[:, :-3, :]
            dQ_plus_R = Q[:, 3:, :] - Q[:, 2:-1, :]





            raise NotImplementedError("TODO: complete MUSCL reconstruction for axis=0")
        else:

            QL = Q[1:-2, :, :].copy()
            QR = Q[2:-1, :, :].copy()

            dQ_plus = Q[2:-1, :, :] - Q[1:-2, :, :]
            dQ_minus = Q[1:-2, :, :] - Q[:-3, :, :]
            dQ_plus_R = Q[3:, :, :] - Q[2:-1, :, :]





            raise NotImplementedError("TODO: complete MUSCL reconstruction for axis=1")

        return QL, QR

    def compute_rhs(self, Q: np.ndarray) -> np.ndarray:
        ny, nx = Q.shape[:2]
        rhs = np.zeros_like(Q)


        QL, QR = self.muscl_reconstruct(Q, axis=0)
        flux_x = self.roe_flux_x(QL, QR)
        nfx = flux_x.shape[1]



        for j in range(ny):
            for i in range(2, min(nfx + 2, nx - 1)):
                k = i - 1
                if k > 0 and k < nfx:
                    rhs[j, i, :] -= (flux_x[j, k, :] - flux_x[j, k - 1, :]) / self.dx


        QL, QR = self.muscl_reconstruct(Q, axis=1)
        flux_y = self.roe_flux_y(QL, QR)
        nfy = flux_y.shape[0]

        for j in range(2, min(nfy + 2, ny - 1)):
            for i in range(nx):
                k = j - 1
                if k > 0 and k < nfy:
                    rhs[j, i, :] -= (flux_y[k, i, :] - flux_y[k - 1, i, :]) / self.dy[j]


        Fv, Gv = self.compute_viscous_flux(Q)

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                rhs[j, i, :] += (Fv[j, i + 1, :] - Fv[j, i - 1, :]) / (2.0 * self.dx)
                rhs[j, i, :] += (Gv[j + 1, i, :] - Gv[j - 1, i, :]) / (2.0 * self.dy[j])

        return rhs

    def apply_boundary_conditions(self, Q: np.ndarray) -> np.ndarray:
        Q_bc = Q.copy()


        rho_wall = Q_bc[1, :, 0]
        u_wall = 0.0
        v_wall = 0.0
        T_wall = self.T_wall
        p_wall = rho_wall * T_wall / (self.gamma * self.Ma ** 2)
        e_wall = p_wall / (rho_wall * (self.gamma - 1.0))
        E_wall = e_wall

        Q_bc[0, :, 0] = rho_wall
        Q_bc[0, :, 1] = rho_wall * u_wall
        Q_bc[0, :, 2] = rho_wall * v_wall
        Q_bc[0, :, 3] = rho_wall * E_wall


        Q_bc[-1, :, :] = Q_bc[-2, :, :]


        rho_in = self.rho_inf
        u_in = self.u_inf
        v_in = 0.0
        p_in = self.p_inf
        e_in = p_in / (rho_in * (self.gamma - 1.0))
        E_in = e_in + 0.5 * (u_in ** 2 + v_in ** 2)

        Q_bc[:, 0, 0] = rho_in
        Q_bc[:, 0, 1] = rho_in * u_in
        Q_bc[:, 0, 2] = rho_in * v_in
        Q_bc[:, 0, 3] = rho_in * E_in


        Q_bc[:, -1, :] = Q_bc[:, -2, :]

        return Q_bc

    def update_dt(self):
        rho, u, v, p, e, T = self.primitive_variables(self.Q)
        c = np.sqrt(self.gamma * p / rho)
        nu = self.mu / rho

        u_max = np.max(np.abs(u) + c)
        v_max = np.max(np.abs(v) + c)
        nu_max = np.max(nu)

        dt_conv = 0.5 * self.dx / max(u_max, 1e-14)
        dt_visc = 0.25 * self.dx ** 2 / max(nu_max, 1e-14)
        self.dt = min(dt_conv, dt_visc, 1e-3)

    def step_rk3(self):
        self.update_dt()


        Q1 = self.Q.copy()
        rhs1 = self.compute_rhs(Q1)
        Q1 = Q1 + self.dt * rhs1
        Q1 = self.apply_boundary_conditions(Q1)


        rhs2 = self.compute_rhs(Q1)
        Q2 = 0.75 * self.Q + 0.25 * Q1 + 0.25 * self.dt * rhs2
        Q2 = self.apply_boundary_conditions(Q2)


        rhs3 = self.compute_rhs(Q2)
        self.Q = (1.0 / 3.0) * self.Q + (2.0 / 3.0) * Q2 + (2.0 / 3.0) * self.dt * rhs3
        self.Q = self.apply_boundary_conditions(self.Q)

        self.time += self.dt
        self.iter += 1

    def solve(self, n_steps: int = 100, log_interval: int = 10) -> dict:
        residuals = []
        max_res_history = []

        for step in range(n_steps):
            Q_old = self.Q.copy()
            self.step_rk3()

            res = np.linalg.norm(self.Q - Q_old) / (np.linalg.norm(Q_old) + 1e-14)
            residuals.append(float(res))
            max_res = np.max(np.abs(self.Q - Q_old))
            max_res_history.append(float(max_res))

            if step % log_interval == 0:
                print(f"  Step {step:5d}: t={self.time:.6f}, dt={self.dt:.6e}, res={res:.6e}")


            if step > 50 and res < 1e-8:
                print(f"  Converged at step {step}")
                break

        rho, u, v, p, e, T = self.primitive_variables(self.Q)

        return {
            'Q': self.Q,
            'rho': rho,
            'u': u,
            'v': v,
            'p': p,
            'T': T,
            'time': self.time,
            'iterations': self.iter,
            'residuals': residuals,
            'max_residuals': max_res_history
        }
