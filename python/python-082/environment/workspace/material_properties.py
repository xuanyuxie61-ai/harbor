
import numpy as np
from utils import validate_positive


class CompositeMaterial:

    def __init__(self, E_f, nu_f, E_m, nu_m, fiber_volume_fraction):
        validate_positive(E_f, "E_f")
        validate_positive(E_m, "E_m")
        if not (0.0 <= fiber_volume_fraction <= 1.0):
            raise ValueError("Fiber volume fraction must be in [0, 1].")
        self.E_f = float(E_f)
        self.nu_f = float(nu_f)
        self.E_m = float(E_m)
        self.nu_m = float(nu_m)
        self.V_f = float(fiber_volume_fraction)
        self.V_m = 1.0 - self.V_f


        self.G_f = self.E_f / (2.0 * (1.0 + self.nu_f))
        self.G_m = self.E_m / (2.0 * (1.0 + self.nu_m))


        self._homogenize()

    def _homogenize(self):
        Vf = self.V_f
        Vm = self.V_m


        self.E1 = self.E_f * Vf + self.E_m * Vm


        xi = 2.0


        eta_E = (self.E_f / self.E_m - 1.0) / (self.E_f / self.E_m + xi)
        self.E2 = self.E_m * (1.0 + xi * eta_E * Vf) / (1.0 - eta_E * Vf)


        eta_G = (self.G_f / self.G_m - 1.0) / (self.G_f / self.G_m + xi)
        self.G12 = self.G_m * (1.0 + xi * eta_G * Vf) / (1.0 - eta_G * Vf)


        self.nu12 = self.nu_f * Vf + self.nu_m * Vm


        self.nu21 = self.nu12 * self.E2 / self.E1


        self.S = np.array([
            [1.0 / self.E1, -self.nu21 / self.E2, 0.0],
            [-self.nu12 / self.E1, 1.0 / self.E2, 0.0],
            [0.0, 0.0, 1.0 / self.G12]
        ])


        self.Q = np.linalg.inv(self.S)


        self.E3 = self.E2
        self.G13 = self.G12
        self.G23 = self.G_m / (1.0 - np.sqrt(Vf) * (1.0 - self.G_m / self.G_f))
        self.nu13 = self.nu12
        self.nu23 = 0.3

    def compute_transformed_stiffness(self, theta_deg):
        theta = np.radians(float(theta_deg))
        c = np.cos(theta)
        s = np.sin(theta)
        c2 = c * c
        s2 = s * s
        s4 = s2 * s2
        c4 = c2 * c2
        s2c2 = s2 * c2

        Q11, Q12 = self.Q[0, 0], self.Q[0, 1]
        Q22, Q66 = self.Q[1, 1], self.Q[2, 2]

        Q_bar = np.zeros((3, 3))
        Q_bar[0, 0] = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * s4
        Q_bar[0, 1] = (Q11 + Q22 - 4.0 * Q66) * s2c2 + Q12 * (s4 + c4)
        Q_bar[1, 0] = Q_bar[0, 1]
        Q_bar[1, 1] = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * c4
        Q_bar[0, 2] = (Q11 - Q12 - 2.0 * Q66) * s * c2 * c + (Q12 - Q22 + 2.0 * Q66) * s2 * s * c
        Q_bar[2, 0] = Q_bar[0, 2]
        Q_bar[1, 2] = (Q11 - Q12 - 2.0 * Q66) * s2 * s * c + (Q12 - Q22 + 2.0 * Q66) * s * c2 * c
        Q_bar[2, 1] = Q_bar[1, 2]
        Q_bar[2, 2] = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2c2 + Q66 * (s4 + c4)
        return Q_bar

    def compute_degraded_stiffness(self, d_f, d_m, d_s):
        d_f = np.clip(d_f, 0.0, 0.99)
        d_m = np.clip(d_m, 0.0, 0.99)
        d_s = np.clip(d_s, 0.0, 0.99)

        E1d = (1.0 - d_f) * self.E1
        E2d = (1.0 - d_m) * self.E2
        G12d = (1.0 - d_s) * self.G12
        nu12d = self.nu12 * (1.0 - d_f)
        nu21d = nu12d * E2d / E1d

        S_d = np.array([
            [1.0 / E1d, -nu21d / E2d, 0.0],
            [-nu12d / E1d, 1.0 / E2d, 0.0],
            [0.0, 0.0, 1.0 / G12d]
        ])


        det_S = np.linalg.det(S_d)
        if abs(det_S) < 1e-20:
            S_d += 1e-12 * np.eye(3)

        Q_d = np.linalg.inv(S_d)
        return Q_d

    def print_properties(self):
        print("-" * 50)
        print("Composite Homogenized Properties")
        print("-" * 50)
        print(f"  Fiber volume fraction V_f = {self.V_f:.4f}")
        print(f"  Longitudinal modulus E_1   = {self.E1:.4f} GPa")
        print(f"  Transverse modulus E_2     = {self.E2:.4f} GPa")
        print(f"  In-plane shear modulus G_12= {self.G12:.4f} GPa")
        print(f"  Major Poisson ratio nu_12  = {self.nu12:.4f}")
        print(f"  Minor Poisson ratio nu_21  = {self.nu21:.4f}")
        print("-" * 50)


def create_carbon_epoxy(V_f=0.6):

    E_f = 230.0
    nu_f = 0.20
    E_m = 3.5
    nu_m = 0.35
    return CompositeMaterial(E_f, nu_f, E_m, nu_m, V_f)
