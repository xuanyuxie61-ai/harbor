
import numpy as np


class PoroelasticMaterial:

    def __init__(self, lam, mu, phi, kappa, eta, K_s, K_f, rho_s, rho_f):
        if not (0.0 < phi < 1.0):
            raise ValueError(f"Porosity phi must be in (0,1), got {phi}")
        if lam <= 0.0 or mu <= 0.0:
            raise ValueError("Lamé parameters must be positive.")
        if kappa <= 0.0 or eta <= 0.0:
            raise ValueError("Permeability and viscosity must be positive.")
        if K_s <= 0.0 or K_f <= 0.0:
            raise ValueError("Bulk moduli must be positive.")
        if rho_s <= 0.0 or rho_f <= 0.0:
            raise ValueError("Densities must be positive.")

        self.lam = float(lam)
        self.mu = float(mu)
        self.phi = float(phi)
        self.kappa = float(kappa)
        self.eta = float(eta)
        self.K_s = float(K_s)
        self.K_f = float(K_f)
        self.rho_s = float(rho_s)
        self.rho_f = float(rho_f)


        self._compute_derived()

    def _compute_derived(self):

        self.K_d = self.lam + 2.0 * self.mu / 3.0






        self.alpha = None
        self.M = None


        numerator = 1.0 / self.K_d - 1.0 / self.K_s
        denominator = numerator + self.phi * (1.0 / self.K_f - 1.0 / self.K_s)
        self.B = numerator / denominator if abs(denominator) > 1e-30 else 0.0


        self.K_u = self.K_d + self.alpha ** 2 * self.M


        self.rho_bulk = (1.0 - self.phi) * self.rho_s + self.phi * self.rho_f


        self.D_diff = self.kappa * self.M / self.eta



        self.tortuosity = 1.0 + (1.0 - 1.0 / self.phi) * 0.5
        if self.tortuosity < 1.0:
            self.tortuosity = 1.0


        self.V_p_fast = np.sqrt((self.K_u + 4.0 * self.mu / 3.0) / self.rho_bulk)


        self.V_s = np.sqrt(self.mu / self.rho_bulk)




        denom = self.K_d + 4.0 * self.mu / 3.0
        if abs(denom) > 1e-30:
            self.V_p_slow = np.sqrt(
                (self.kappa / self.eta)
                * self.M
                * (denom + self.alpha ** 2 * self.M)
                / denom
            )
        else:
            self.V_p_slow = 0.0


        c_v = self.kappa * self.M * (self.K_d + 4.0 * self.mu / 3.0) / self.eta
        denom_cv = self.K_d + self.alpha ** 2 * self.M + 4.0 * self.mu / 3.0
        self.c_v = c_v / denom_cv if abs(denom_cv) > 1e-30 else 0.0

    def elastic_matrix(self):
        D = np.array([
            [self.lam + 2.0 * self.mu, self.lam, 0.0],
            [self.lam, self.lam + 2.0 * self.mu, 0.0],
            [0.0, 0.0, self.mu]
        ], dtype=float)
        return D

    def stress_from_strain(self, epsilon):
        D = self.elastic_matrix()
        return D @ epsilon

    def summary(self):
        lines = [
            "=" * 60,
            "Poroelastic Material Summary",
            "=" * 60,
            f"  Lamé parameter λ        = {self.lam:.4e} Pa",
            f"  Shear modulus μ         = {self.mu:.4e} Pa",
            f"  Porosity φ              = {self.phi:.6f}",
            f"  Permeability κ          = {self.kappa:.4e} m²",
            f"  Fluid viscosity η       = {self.eta:.4e} Pa·s",
            f"  Solid bulk modulus K_s  = {self.K_s:.4e} Pa",
            f"  Fluid bulk modulus K_f  = {self.K_f:.4e} Pa",
            f"  Solid density ρ_s       = {self.rho_s:.4e} kg/m³",
            f"  Fluid density ρ_f       = {self.rho_f:.4e} kg/m³",
            "-" * 60,
            "Derived parameters:",
            f"  Drained bulk K_d        = {self.K_d:.4e} Pa",
            f"  Biot-Willis α           = {self.alpha:.6f}",
            f"  Biot modulus M          = {self.M:.4e} Pa",
            f"  Skempton B              = {self.B:.6f}",
            f"  Undrained bulk K_u      = {self.K_u:.4e} Pa",
            f"  Bulk density ρ_bulk     = {self.rho_bulk:.4e} kg/m³",
            f"  Tortuosity a            = {self.tortuosity:.4f}",
            f"  Fast P-wave Vp          = {self.V_p_fast:.4e} m/s",
            f"  Slow P-wave Vp_slow     = {self.V_p_slow:.4e} m/s",
            f"  Shear wave Vs           = {self.V_s:.4e} m/s",
            f"  Hydraulic diffusivity D = {self.D_diff:.4e} m²/s",
            f"  Consolidation c_v       = {self.c_v:.4e} m²/s",
            "=" * 60,
        ]
        return "\n".join(lines)


class BiotConsolidation:

    def __init__(self, material: PoroelasticMaterial):
        self.mat = material

    def solid_stiffness_contribution(self, B_mat, detJ, wq):
        D = self.mat.elastic_matrix()
        return B_mat.T @ D @ B_mat * detJ * wq

    def coupling_contribution(self, B_mat, N_p, alpha, detJ, wq):
        m_vec = np.array([1.0, 1.0, 0.0])
        return alpha * B_mat.T @ np.outer(m_vec, N_p) * detJ * wq

    def compressibility_contribution(self, N_p, M, detJ, wq):
        return (1.0 / M) * np.outer(N_p, N_p) * detJ * wq

    def flow_contribution(self, grad_Np, kappa_over_eta, detJ, wq):
        return kappa_over_eta * grad_Np @ grad_Np.T * detJ * wq

    def mass_matrix_contribution(self, N_u, rho_bulk, detJ, wq):
        return rho_bulk * np.outer(N_u, N_u) * detJ * wq


def compute_characteristic_frequencies(mat: PoroelasticMaterial, length_scale: float):
    omega_c = (mat.eta * mat.phi) / (mat.kappa * mat.rho_f)
    t_consolidation = length_scale ** 2 / mat.c_v if mat.c_v > 1e-30 else np.inf
    t_diffusion = length_scale ** 2 / mat.D_diff if mat.D_diff > 1e-30 else np.inf

    return {
        "omega_c": omega_c,
        "f_c": omega_c / (2.0 * np.pi),
        "t_consolidation": t_consolidation,
        "t_diffusion": t_diffusion,
    }
