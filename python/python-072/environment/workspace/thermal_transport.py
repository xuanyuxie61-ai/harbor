
import numpy as np


class ThermalTransportSolver:

    def __init__(self, nx, ny, dx, dy, dt,
                 thermal_diffusivity=1.0,
                 latent_heat=1.0,
                 specific_heat=1.0,
                 solute_diffusivity_solid=0.01,
                 solute_diffusivity_liquid=1.0,
                 partition_coefficient=0.3,
                 liquidus_slope=-1.0):
        if nx < 3 or ny < 3:
            raise ValueError("网格维度必须至少为 3")
        if dx <= 0 or dy <= 0 or dt <= 0:
            raise ValueError("步长参数必须为正")
        if not (0 < partition_coefficient <= 1.0):
            raise ValueError("分配系数 k_p 必须在 (0, 1] 范围内")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.alpha_T = thermal_diffusivity
        self.latent_heat = latent_heat
        self.specific_heat = specific_heat
        self.D_s = solute_diffusivity_solid
        self.D_l = solute_diffusivity_liquid
        self.k_p = partition_coefficient
        self.m_L = liquidus_slope


        max_diff = max(thermal_diffusivity, solute_diffusivity_liquid)
        dt_diff_limit = 0.25 * min(dx ** 2, dy ** 2) / max_diff
        if dt > dt_diff_limit:

            pass

    def solid_fraction(self, phi):
        return np.clip(0.5 * (1.0 + phi), 0.0, 1.0)

    def effective_diffusivity(self, phi):
        h = self.solid_fraction(phi)
        return self.D_s * h + self.D_l * (1.0 - h)

    def laplacian_with_variable_coeff(self, field, coeff):
        result = np.zeros_like(field)


        coeff_x_plus = np.zeros_like(coeff)
        coeff_x_minus = np.zeros_like(coeff)
        coeff_x_plus[1:-1, :] = 0.5 * (coeff[2:, :] + coeff[1:-1, :])
        coeff_x_minus[1:-1, :] = 0.5 * (coeff[1:-1, :] + coeff[:-2, :])

        result[1:-1, :] += (
            coeff_x_plus[1:-1, :] * (field[2:, :] - field[1:-1, :]) -
            coeff_x_minus[1:-1, :] * (field[1:-1, :] - field[:-2, :])
        ) / (self.dx ** 2)


        coeff_y_plus = np.zeros_like(coeff)
        coeff_y_minus = np.zeros_like(coeff)
        coeff_y_plus[:, 1:-1] = 0.5 * (coeff[:, 2:] + coeff[:, 1:-1])
        coeff_y_minus[:, 1:-1] = 0.5 * (coeff[:, 1:-1] + coeff[:, :-2])

        result[:, 1:-1] += (
            coeff_y_plus[:, 1:-1] * (field[:, 2:] - field[:, 1:-1]) -
            coeff_y_minus[:, 1:-1] * (field[:, 1:-1] - field[:, :-2])
        ) / (self.dy ** 2)


        result[0, :] = result[1, :]
        result[-1, :] = result[-2, :]
        result[:, 0] = result[:, 1]
        result[:, -1] = result[:, -2]

        return result

    def convection_term(self, field, vx, vy):
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)


        mask_pos = vx >= 0
        grad_x[1:-1, :][mask_pos[1:-1, :]] = (
            field[1:-1, :][mask_pos[1:-1, :]] - field[:-2, :][mask_pos[1:-1, :]]
        ) / self.dx
        mask_neg = vx < 0
        grad_x[1:-1, :][mask_neg[1:-1, :]] = (
            field[2:, :][mask_neg[1:-1, :]] - field[1:-1, :][mask_neg[1:-1, :]]
        ) / self.dx


        mask_pos = vy >= 0
        grad_y[:, 1:-1][mask_pos[:, 1:-1]] = (
            field[:, 1:-1][mask_pos[:, 1:-1]] - field[:, :-2][mask_pos[:, 1:-1]]
        ) / self.dy
        mask_neg = vy < 0
        grad_y[:, 1:-1][mask_neg[:, 1:-1]] = (
            field[:, 2:][mask_neg[:, 1:-1]] - field[:, 1:-1][mask_neg[:, 1:-1]]
        ) / self.dy

        return vx * grad_x + vy * grad_y

    def latent_heat_source(self, phi, phi_old):
        h_new = self.solid_fraction(phi)
        h_old = self.solid_fraction(phi_old)
        return (self.latent_heat / self.specific_heat) * (h_new - h_old) / self.dt

    def solute_rejection_source(self, phi, phi_old, C):
        h_new = self.solid_fraction(phi)
        h_old = self.solid_fraction(phi_old)
        dh_dt = (h_new - h_old) / self.dt

        denom = self.k_p + (1.0 - self.k_p) * h_new
        denom = np.maximum(denom, 1e-12)

        return C * (1.0 - self.k_p) * dh_dt / denom

    def temperature_rhs(self, T, phi, phi_old, vx, vy):

        lap_T = np.zeros_like(T)
        lap_T[1:-1, 1:-1] = (
            (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (self.dx ** 2) +
            (T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (self.dy ** 2)
        )
        diffusion = self.alpha_T * lap_T


        convection = self.convection_term(T, vx, vy)


        q_latent = self.latent_heat_source(phi, phi_old)

        return diffusion - convection + q_latent

    def concentration_rhs(self, C, phi, phi_old, vx, vy):














        raise NotImplementedError("HOLE 2: 请实现 concentration_rhs 方法")

    def compute_thermal_undercooling(self, T, T_m):
        return T_m - T

    def compute_solutal_undercooling(self, C, C_e):
        return -self.m_L * (C - C_e)

    def compute_total_undercooling(self, T, C, T_m, C_e, gamma, curvature):
        dT_thermal = self.compute_thermal_undercooling(T, T_m)
        dT_solutal = self.compute_solutal_undercooling(C, C_e)
        return dT_thermal + dT_solutal - gamma * curvature
