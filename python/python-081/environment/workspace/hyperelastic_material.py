
import numpy as np


class NeoHookeanMaterial:
    def __init__(self, young_modulus, poisson_ratio, damage=0.0):
        self.E = float(young_modulus)
        self.nu = float(poisson_ratio)
        self.D = float(damage)


        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))


        self.mu_d = self.mu * (1.0 - self.D)
        self.K_d = self.K * (1.0 - self.D)


        if not (0.0 <= self.nu < 0.5):
            raise ValueError(f"Poisson ratio must be in [0, 0.5), got {self.nu}")
        if self.E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {self.E}")
        if not (0.0 <= self.D <= 1.0):
            raise ValueError(f"Damage must be in [0,1], got {self.D}")

    def update_damage(self, D_new):
        self.D = float(np.clip(D_new, 0.0, 1.0))
        self.mu_d = self.mu * (1.0 - self.D)
        self.K_d = self.K * (1.0 - self.D)

    def compute_stress_tangent(self, F):
        F = np.array(F, dtype=float)
        J = np.linalg.det(F)


        J = float(np.clip(J, 0.01, 100.0))
        if J < 1e-8:
            J = 1e-8


        C = F.T @ F


        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            C_inv = np.eye(3)

        I1 = np.trace(C)
        I3 = max(J ** 2, 1e-8)
        I3_inv = 1.0 / I3


        lnJ = np.log(J)



        Jm23 = J ** (-2.0 / 3.0)
        S = self.mu_d * Jm23 * (np.eye(3) - (I1 / 3.0) * C_inv) + self.K_d * lnJ * C_inv


        C_voigt = self._compute_material_tangent(C, C_inv, I1, J, lnJ)

        return S, C_voigt, J

    def _compute_material_tangent(self, C, C_inv, I1, J, lnJ):
        lam = self.K_d - 2.0 * self.mu_d / 3.0
        mu = self.mu_d

        C6 = np.zeros((6, 6))
        for i in range(3):
            C6[i, i] = lam + 2.0 * mu
            for j in range(3):
                if i != j:
                    C6[i, j] = lam
        for i in range(3, 6):
            C6[i, i] = mu

        return C6

    def compute_strain_energy(self, F):
        J = np.linalg.det(F)
        if J < 1e-8:
            J = 1e-8
        C = F.T @ F
        I1 = np.trace(C)
        Jm23 = J ** (-2.0 / 3.0)
        lnJ = np.log(J)
        Psi = 0.5 * self.mu_d * (Jm23 * I1 - 3.0) + 0.5 * self.K_d * (lnJ ** 2)
        return Psi


def compute_deformation_gradient(grad_u):
    grad_u = np.array(grad_u, dtype=float)
    if grad_u.shape != (3, 3):
        raise ValueError("grad_u must be 3x3")
    return np.eye(3) + grad_u


def green_lagrange_strain(F):
    C = F.T @ F
    return 0.5 * (C - np.eye(3))


def voigt_stress_tensor(S):
    return np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[1, 2], S[0, 2]])
