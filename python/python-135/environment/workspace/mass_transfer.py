"""
Gas-Liquid Mass Transfer Models for CO2 Absorption
Implements two-film theory with chemical reaction enhancement.
Integrates line_grid for film discretization.

Key equations:
- Two-film theory: 1/K_G = 1/k_G + H/(k_L * E)
- Hatta number: Ha = sqrt(k2 * D_A * c_B) / k_L
- Enhancement factor for second-order reactions:
    E = Ha * sqrt( (E_i - E) / (E_i - 1) )   (van Krevelen-Hoftijzer)
    where E_i = 1 + c_B * D_B / (b * c_Ai * D_A)
"""

import numpy as np
from utils import (
    validate_positive, hatta_number, enhancement_factor_hatta,
    wilke_chang_diffusion, safe_divide, clip_concentration, R_GAS
)
from spectral_methods import SpectralDiffusionSolver


class TwoFilmModel:
    """
    Two-film model with chemical reaction enhancement for CO2 absorption.
    """

    def __init__(self, T, P_total, amine_type="MEA"):
        self.T = T
        self.P_total = P_total
        self.amine_type = amine_type
        # Henry's law constant for CO2 in water [Pa·m^3/mol]
        self.Henry_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T - 1.0 / 298.15))
        # Diffusivity of CO2 in water [m^2/s]
        self.D_CO2 = 1.91e-9 * np.exp(-2194.0 * (1.0 / T - 1.0 / 298.15))
        # Diffusivity of amine in water [m^2/s]
        self.D_amine = 1.0e-9 * np.exp(-1800.0 * (1.0 / T - 1.0 / 298.15))
        # Gas-side mass transfer coefficient [mol/(m^2·s·Pa)]
        self.k_G = 1.0e-5
        # Liquid-side mass transfer coefficient [m/s]
        self.k_L = 2.0e-4
        # Film thickness [m]
        self.delta_L = self.D_CO2 / self.k_L

    def gas_phase_flux(self, P_CO2_bulk, P_CO2_interface):
        """Gas-phase molar flux: N_A = k_G * (P_A,G - P_A,i)"""
        return self.k_G * (P_CO2_bulk - P_CO2_interface)

    def liquid_phase_flux(self, c_CO2_interface, c_CO2_bulk, E):
        """Liquid-phase molar flux: N_A = k_L * E * (c_A,i - c_A,L)"""
        return self.k_L * E * (c_CO2_interface - c_CO2_bulk)

    def interface_concentration(self, P_CO2_bulk, P_CO2_interface, c_CO2_bulk, E):
        """Interface equilibrium: c_A,i = P_A,i / H"""
        c_i = P_CO2_interface / self.Henry_CO2
        return c_i

    def solve_interface(self, P_CO2_bulk, c_CO2_bulk, c_amine_bulk, k2_rate):
        """
        Solve for interface conditions and overall flux.
        Returns: P_i, c_i, E, Ha, N_A
        """
        validate_positive(P_CO2_bulk, "Bulk CO2 partial pressure")
        validate_positive(c_amine_bulk, "Bulk amine concentration")

        # Hatta number
        Ha = hatta_number(k2_rate, self.D_CO2, c_amine_bulk, self.k_L)

        # Instantaneous enhancement factor
        b_stoich = 2.0  # stoichiometric coefficient for MEA
        c_Ai_max = P_CO2_bulk / self.Henry_CO2
        E_infinite = 1.0 + c_amine_bulk * self.D_amine / (b_stoich * c_Ai_max * self.D_CO2)
        E_infinite = np.maximum(E_infinite, 1.0)

        # Enhancement factor (van Krevelen-Hoftijzer approximation)
        if Ha < 0.3:
            E = 1.0 + Ha ** 2 / 3.0
        elif Ha > 10.0 * E_infinite:
            E = E_infinite
        else:
            # Iterative solution of E = Ha * tanh(E) / tanh(Ha * sqrt(...))
            # Simplified: use asymptotic form
            E = np.minimum(Ha, E_infinite)
            for _ in range(20):
                E_new = Ha * np.sqrt((E_infinite - E) / max(E_infinite - 1.0, 1e-6))
                if abs(E_new - E) < 1e-6:
                    break
                E = 0.5 * (E + E_new)

        # Overall mass transfer coefficient
        K_G = 1.0 / (1.0 / self.k_G + self.Henry_CO2 / (self.k_L * E))

        # Overall flux
        N_A = K_G * P_CO2_bulk  # Assume c_bulk ≈ 0 for lean solvent

        P_i = P_CO2_bulk - N_A / self.k_G
        c_i = P_i / self.Henry_CO2

        return {
            "P_interface": P_i,
            "c_interface": c_i,
            "enhancement_factor": E,
            "hatta_number": Ha,
            "E_infinite": E_infinite,
            "flux": N_A,
            "K_G_overall": K_G
        }

    def film_profile(self, P_CO2_bulk, c_CO2_bulk, c_amine_bulk, k2_rate, n_grid=64):
        """
        Compute concentration profile across liquid film using spectral method.
        Based on line_grid and spectral differentiation.
        """
        sol = self.solve_interface(P_CO2_bulk, c_CO2_bulk, c_amine_bulk, k2_rate)
        c_i = sol["c_interface"]
        c_b = c_CO2_bulk
        k = k2_rate * c_amine_bulk

        spectral = SpectralDiffusionSolver(n_cheb=n_grid)
        z, c_profile, flux = spectral.solve_film_diffusion_reaction(
            self.D_CO2, k, self.delta_L, c_i, c_b
        )
        return z, c_profile, flux, sol


def generate_film_grid(n_points, delta, centering=1):
    """
    Generate 1D grid points across liquid film thickness.
    Based on line_grid.m with multiple centering options.
    """
    validate_positive(n_points, "n_points")
    validate_positive(delta, "Film thickness")

    z = np.zeros(n_points)
    a, b = 0.0, delta

    for j in range(n_points):
        idx = j + 1
        if centering == 1:
            # Uniform including endpoints
            if n_points == 1:
                z[j] = 0.5 * (a + b)
            else:
                z[j] = ((n_points - idx) * a + (idx - 1) * b) / (n_points - 1)
        elif centering == 2:
            # Interior points only
            z[j] = ((n_points - idx + 1) * a + idx * b) / (n_points + 1)
        elif centering == 3:
            # Including left endpoint
            z[j] = ((n_points - idx + 1) * a + (idx - 1) * b) / n_points
        elif centering == 4:
            # Including right endpoint
            z[j] = ((n_points - idx) * a + idx * b) / n_points
        elif centering == 5:
            # Midpoint centered
            z[j] = ((2 * n_points - 2 * idx + 1) * a + (2 * idx - 1) * b) / (2 * n_points)
        else:
            raise ValueError(f"Invalid centering: {centering}")

    return z


class PackedColumnModel:
    """
    1D packed column model for CO2 absorption.
    Axial mass and energy balances.
    """

    def __init__(self, column_height, column_diameter, packing_type="random"):
        self.H = column_height
        self.D = column_diameter
        self.packing_type = packing_type
        # Packing-specific parameters
        if packing_type == "random":
            self.a_packing = 200.0  # Interfacial area [m^2/m^3]
            self.eps_void = 0.92    # Void fraction
            self.htu = 0.35         # Height of transfer unit [m]
        elif packing_type == "structured":
            self.a_packing = 250.0
            self.eps_void = 0.95
            self.htu = 0.25
        else:
            self.a_packing = 200.0
            self.eps_void = 0.92
            self.htu = 0.35

        self.cross_section = np.pi * (self.D / 2.0) ** 2

    def axial_profile(self, T, P_total, c_amine, L_flow, G_flow, y_CO2_in, n_z=100):
        """
        Solve axial profile using HTU-NTU method.
        """
        validate_positive(T, "Temperature")
        validate_positive(L_flow, "Liquid flow")
        validate_positive(G_flow, "Gas flow")

        z = generate_film_grid(n_z, self.H, centering=1)

        # Molar flow rates [mol/s]
        n_L = L_flow / 18.015e-3  # assume water density ~1000 kg/m^3
        n_G = G_flow / 22.414e-3  # standard molar volume

        # Convert to superficial velocities [mol/(m^2·s)]
        u_L = n_L / self.cross_section
        u_G = n_G / self.cross_section

        y_CO2 = np.zeros(n_z)
        x_CO2 = np.zeros(n_z)
        T_profile = np.full(n_z, T)

        y_CO2[0] = y_CO2_in

        for i in range(n_z - 1):
            dz = z[i + 1] - z[i]
            P_CO2 = y_CO2[i] * P_total

            # Two-film model at this position
            film = TwoFilmModel(T_profile[i], P_total)
            sol = film.solve_interface(P_CO2, 0.0, c_amine, 5000.0)

            # Mass balance
            N_A = sol["flux"]  # [mol/(m^2·s)]
            dN = N_A * self.a_packing * self.cross_section * dz

            # Update gas composition
            dy = -dN / n_G
            y_CO2[i + 1] = y_CO2[i] + dy
            y_CO2[i + 1] = np.clip(y_CO2[i + 1], 1e-6, 0.5)

            # Temperature change (simplified)
            delta_H_rxn = -85.0e3  # J/mol CO2 absorbed
            dT = -dN * delta_H_rxn / (L_flow * 4184.0)
            T_profile[i + 1] = T_profile[i] + dT

        return z, y_CO2, T_profile
