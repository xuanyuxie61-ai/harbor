
import numpy as np






class HartmannFlow:

    def __init__(self, G=1.0, Ha=1.0, L=10.0, p0=4.0, Re=10.0, Rm=6.0):
        self.G = G
        self.Ha = Ha
        self.L = L
        self.p0 = p0
        self.Re = Re
        self.Rm = Rm
        self.S = Ha**2 / (Re * Rm)

    def velocity(self, y):
        y = np.asarray(y, dtype=float)
        return (self.G * self.Re / self.Ha / np.tanh(self.Ha)
                * (1.0 - np.cosh(y * self.Ha) / np.cosh(self.Ha)))

    def magnetic_field_b(self, y):
        y = np.asarray(y, dtype=float)
        return self.G / self.S * (np.sinh(y * self.Ha) / np.sinh(self.Ha) - y)

    def pressure(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        b = self.magnetic_field_b(y)
        return -self.G * x - 0.5 * self.S * b**2

    def residual_check(self, y):
        y = np.asarray(y, dtype=float)
        dy = 1e-6

        u_p = self.velocity(y + dy)
        u_m = self.velocity(y - dy)
        u = self.velocity(y)
        uy = (u_p - u_m) / (2 * dy)
        uyy = (u_p - 2 * u + u_m) / dy**2

        b_p = self.magnetic_field_b(y + dy)
        b_m = self.magnetic_field_b(y - dy)
        b = self.magnetic_field_b(y)
        by = (b_p - b_m) / (2 * dy)
        byy = (b_p - 2 * b + b_m) / dy**2

        ur = uyy + self.Re * self.S * by + self.G * self.Re
        br = byy + self.Rm * uy
        return ur, br






def schiller_naumann_cd(re_p):
    re_p = np.asarray(re_p, dtype=float)
    cd = np.zeros_like(re_p)
    mask_low = re_p < 1000.0
    mask_high = ~mask_low
    cd[mask_low] = (24.0 / re_p[mask_low]) * (1.0 + 0.15 * re_p[mask_low]**0.687)
    cd[mask_high] = 0.44
    return cd


def interphase_momentum_exchange(alpha_g, u_g, u_l, rho_l, mu_l, d_b,
                                 C_VM=0.5, C_L=0.25):
    alpha_g = np.asarray(alpha_g, dtype=float)
    u_g = np.asarray(u_g, dtype=float)
    u_l = np.asarray(u_l, dtype=float)


    alpha_g = np.clip(alpha_g, 1e-6, 0.95)

    u_rel = u_g - u_l
    re_p = rho_l * np.abs(u_rel) * d_b / max(mu_l, 1e-12)
    C_D = schiller_naumann_cd(re_p)


    M_D = 0.75 * alpha_g * (rho_l / max(d_b, 1e-9)) * C_D * np.abs(u_rel) * u_rel


    M_VM = C_VM * alpha_g * rho_l * 0.0



    M_L = C_L * alpha_g * rho_l * u_rel * 0.05

    M_gl = M_D + M_VM + M_L
    return M_gl


def effective_viscosity_slurry(mu_l, alpha_s):
    alpha_s = np.asarray(alpha_s, dtype=float)
    alpha_s = np.clip(alpha_s, 0.0, 0.6)
    return mu_l * (1.0 + 2.5 * alpha_s + 7.54 * alpha_s**2)


def two_fluid_momentum_residual(alpha_g, u_g, u_l, p, rho_g, rho_l, mu_eff,
                                g_vec, d_b, dx, dy):
    alpha_g = np.clip(alpha_g, 1e-6, 0.95)
    alpha_l = 1.0 - alpha_g


    M_gl = interphase_momentum_exchange(alpha_g, u_g, u_l, rho_l, mu_eff, d_b)




    dp_dz = 0.0
    res_g = -alpha_g * dp_dz + M_gl + alpha_g * rho_g * g_vec


    res_l = -alpha_l * dp_dz - M_gl + alpha_l * rho_l * g_vec

    return res_g, res_l
