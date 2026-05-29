"""
biot_equations.py
=================
Poroelastic medium governing equations based on Biot's theory.

This module implements the fundamental constitutive relations and wave
propagation equations for fluid-saturated poroelastic media.

Core Physics:
-------------
Biot's poroelastic theory describes the coupled mechanical behavior of
a porous solid matrix saturated with a viscous fluid. The theory is
characterized by the following key parameters and equations.

1. Porosity:
       phi = V_pore / V_total

2. Biot-Willis coefficient (effective stress coefficient):
       alpha = 1 - K_d / K_s
   where K_d is the drained bulk modulus of the skeleton and K_s is
the bulk modulus of the solid grains.

3. Skempton pore pressure coefficient:
       B = (1/K_d - 1/K_s) / (1/K_d - 1/K_s + phi*(1/K_f - 1/K_s))
   where K_f is the fluid bulk modulus.

4. Gassmann's equation for undrained bulk modulus:
       K_u = K_d + alpha^2 / M
   where M is the Biot modulus:
       1/M = phi/K_f + (alpha - phi)/K_s

5. Fast P-wave velocity (drained):
       V_p_fast = sqrt((K_u + 4/3*mu) / rho_bulk)

6. Slow P-wave velocity (Biot's second kind):
       V_p_slow = sqrt( (kappa/eta) * (M*K_d + alpha^2*M + 4/3*mu*M) / (K_d + 4/3*mu) )
   In the low-frequency limit, the slow wave is diffusive:
       V_p_slow ≈ sqrt(2 * omega * D_diff)
   where D_diff = kappa * M / eta is the hydraulic diffusivity.

7. Shear wave velocity (unaffected by fluid):
       V_s = sqrt(mu / rho_bulk)

Coupled u-p formulation (2D, dynamic):
--------------------------------------
Let u = (u_x, u_y) be solid displacement and p be fluid pressure.

Momentum balance (solid phase):
   rho_bulk * d^2u/dt^2 = div(sigma) - alpha * grad(p) + f_ext

where the effective stress sigma is:
   sigma = lambda * tr(epsilon) * I + 2*mu * epsilon
with the strain tensor:
   epsilon = 1/2 * (grad(u) + grad(u)^T)

Fluid mass balance (Darcy's law + storage):
   alpha/K_d * dp/dt + alpha * div(du/dt) - div( (kappa/eta) * grad(p) ) = S_f

where S_f is the fluid source term.

For numerical stability, the coupled system can be written in matrix form:
   [ M_uu   0  ] [ ddot(u) ]   [ K_uu   C  ] [ u ]   [ F_u ]
   [ 0      M_p ] [ dot(p)  ] + [ C^T    K_p] [ p ] = [ F_p ]

where:
   M_uu = integral(rho_bulk * phi_u * phi_u^T dOmega)     -- solid mass matrix
   M_p  = integral(1/M * phi_p * phi_p^T dOmega)          -- fluid compressibility
   K_uu = integral( B^T * D * B dOmega )                  -- solid stiffness
   C    = integral( alpha * B^T * m * phi_p dOmega )      -- coupling matrix
   K_p  = integral( (kappa/eta) * grad(phi_p) * grad(phi_p)^T dOmega ) -- flow
   F_u  = external force vector
   F_p  = fluid source vector

Reference:
----------
Biot, M. A. (1956). Theory of propagation of elastic waves in a
fluid-saturated porous solid. I. Low-frequency range. JASA, 28(2), 168-178.
"""

import numpy as np


class PoroelasticMaterial:
    """
    Container for poroelastic material properties with full physical validation.
    """

    def __init__(self, lam, mu, phi, kappa, eta, K_s, K_f, rho_s, rho_f):
        """
        Parameters
        ----------
        lam : float
            First Lamé parameter of drained skeleton (Pa).
        mu : float
            Shear modulus of drained skeleton (Pa).
        phi : float
            Porosity, must be in (0, 1).
        kappa : float
            Permeability (m^2).
        eta : float
            Dynamic fluid viscosity (Pa·s).
        K_s : float
            Bulk modulus of solid grains (Pa).
        K_f : float
            Bulk modulus of pore fluid (Pa).
        rho_s : float
            Density of solid grains (kg/m^3).
        rho_f : float
            Density of fluid (kg/m^3).
        """
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

        # Derived parameters
        self._compute_derived()

    def _compute_derived(self):
        """Compute all derived poroelastic parameters."""
        # Drained bulk modulus
        self.K_d = self.lam + 2.0 * self.mu / 3.0

        # TODO(Hole_1): Compute Biot-Willis coefficient alpha and Biot modulus M.
        # alpha = 1 - K_d / K_s  (effective stress coefficient)
        # 1/M = phi/K_f + (alpha - phi)/K_s  (Biot modulus)
        # These are central to Biot's poroelastic theory and affect coupling
        # matrix C and compressibility matrix M_p in the FEM assembler.
        self.alpha = None  # FIXME
        self.M = None      # FIXME

        # Skempton coefficient
        numerator = 1.0 / self.K_d - 1.0 / self.K_s
        denominator = numerator + self.phi * (1.0 / self.K_f - 1.0 / self.K_s)
        self.B = numerator / denominator if abs(denominator) > 1e-30 else 0.0

        # Undrained bulk modulus (Gassmann)
        self.K_u = self.K_d + self.alpha ** 2 * self.M

        # Bulk density
        self.rho_bulk = (1.0 - self.phi) * self.rho_s + self.phi * self.rho_f

        # Hydraulic diffusivity
        self.D_diff = self.kappa * self.M / self.eta

        # Tortuosity (approximate, Biot's alpha_tort)
        # For simplicity use the Berryman formula approximation
        self.tortuosity = 1.0 + (1.0 - 1.0 / self.phi) * 0.5
        if self.tortuosity < 1.0:
            self.tortuosity = 1.0

        # Fast P-wave velocity (drained frame, low freq)
        self.V_p_fast = np.sqrt((self.K_u + 4.0 * self.mu / 3.0) / self.rho_bulk)

        # Shear wave velocity
        self.V_s = np.sqrt(self.mu / self.rho_bulk)

        # Slow P-wave velocity (high-frequency asymptotic)
        # V_slow^2 = (kappa/eta) * M * (K_d + 4/3*mu + alpha^2*M) / (K_d + 4/3*mu)
        # In practice slow wave is diffusive at low frequencies
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

        # Consolidation coefficient (Terzaghi)
        c_v = self.kappa * self.M * (self.K_d + 4.0 * self.mu / 3.0) / self.eta
        denom_cv = self.K_d + self.alpha ** 2 * self.M + 4.0 * self.mu / 3.0
        self.c_v = c_v / denom_cv if abs(denom_cv) > 1e-30 else 0.0

    def elastic_matrix(self):
        """
        Return the 2D plane-strain elastic matrix D (3x3).

        For plane strain:
            D = [ lam+2*mu   lam         0
                  lam        lam+2*mu    0
                  0          0           mu ]
        """
        D = np.array([
            [self.lam + 2.0 * self.mu, self.lam, 0.0],
            [self.lam, self.lam + 2.0 * self.mu, 0.0],
            [0.0, 0.0, self.mu]
        ], dtype=float)
        return D

    def stress_from_strain(self, epsilon):
        """
        Compute effective stress from strain vector [exx, eyy, exy]^T.
        """
        D = self.elastic_matrix()
        return D @ epsilon

    def summary(self):
        """Return a formatted string summary of material properties."""
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
    """
    Quasi-static Biot consolidation equations (u-p formulation).

    Momentum:  -div(sigma_eff) + alpha * grad(p) = -f_body
    Mass:       div(d/dt(alpha*u + p/M)) - div( (kappa/eta) * grad(p) ) = S_f
    """

    def __init__(self, material: PoroelasticMaterial):
        self.mat = material

    def solid_stiffness_contribution(self, B_mat, detJ, wq):
        """
        Compute element stiffness matrix contribution for solid phase.

        K_uu_e += B^T * D * B * detJ * wq
        """
        D = self.mat.elastic_matrix()
        return B_mat.T @ D @ B_mat * detJ * wq

    def coupling_contribution(self, B_mat, N_p, alpha, detJ, wq):
        """
        Coupling matrix contribution.

        C_e += alpha * B^T * m * N_p * detJ * wq
        where m = [1, 1, 0]^T for 2D.
        """
        m_vec = np.array([1.0, 1.0, 0.0])
        return alpha * B_mat.T @ np.outer(m_vec, N_p) * detJ * wq

    def compressibility_contribution(self, N_p, M, detJ, wq):
        """
        Fluid compressibility matrix.

        M_p_e += (1/M) * N_p * N_p^T * detJ * wq
        """
        return (1.0 / M) * np.outer(N_p, N_p) * detJ * wq

    def flow_contribution(self, grad_Np, kappa_over_eta, detJ, wq):
        """
        Flow (diffusion) matrix contribution.

        K_p_e += (kappa/eta) * grad(N_p) * grad(N_p)^T * detJ * wq
        """
        return kappa_over_eta * grad_Np @ grad_Np.T * detJ * wq

    def mass_matrix_contribution(self, N_u, rho_bulk, detJ, wq):
        """
        Solid mass matrix contribution.

        M_uu_e += rho_bulk * N_u * N_u^T * detJ * wq
        """
        return rho_bulk * np.outer(N_u, N_u) * detJ * wq


def compute_characteristic_frequencies(mat: PoroelasticMaterial, length_scale: float):
    """
    Compute characteristic frequencies for a poroelastic medium.

    The Biot characteristic frequency separates the viscous-drag dominated
    low-frequency regime from the inertial-drag dominated high-frequency regime:

        omega_c = (eta * phi) / (kappa * rho_f)

    The consolidation time for a domain of size L:
        t_c = L^2 / c_v

    The diffusion time for slow P-wave over distance L:
        t_diff = L^2 / D_diff
    """
    omega_c = (mat.eta * mat.phi) / (mat.kappa * mat.rho_f)
    t_consolidation = length_scale ** 2 / mat.c_v if mat.c_v > 1e-30 else np.inf
    t_diffusion = length_scale ** 2 / mat.D_diff if mat.D_diff > 1e-30 else np.inf

    return {
        "omega_c": omega_c,
        "f_c": omega_c / (2.0 * np.pi),
        "t_consolidation": t_consolidation,
        "t_diffusion": t_diffusion,
    }
