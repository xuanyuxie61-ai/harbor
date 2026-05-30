
import numpy as np
from utils import (
    validate_positive, hatta_number, enhancement_factor_hatta,
    wilke_chang_diffusion, safe_divide, clip_concentration, R_GAS
)
from spectral_methods import SpectralDiffusionSolver


class TwoFilmModel:

    def __init__(self, T, P_total, amine_type="MEA"):
        self.T = T
        self.P_total = P_total
        self.amine_type = amine_type

        self.Henry_CO2 = 2.82e6 * np.exp(-2044.0 * (1.0 / T - 1.0 / 298.15))

        self.D_CO2 = 1.91e-9 * np.exp(-2194.0 * (1.0 / T - 1.0 / 298.15))

        self.D_amine = 1.0e-9 * np.exp(-1800.0 * (1.0 / T - 1.0 / 298.15))

        self.k_G = 1.0e-5

        self.k_L = 2.0e-4

        self.delta_L = self.D_CO2 / self.k_L

    def gas_phase_flux(self, P_CO2_bulk, P_CO2_interface):
        return self.k_G * (P_CO2_bulk - P_CO2_interface)

    def liquid_phase_flux(self, c_CO2_interface, c_CO2_bulk, E):
        return self.k_L * E * (c_CO2_interface - c_CO2_bulk)

    def interface_concentration(self, P_CO2_bulk, P_CO2_interface, c_CO2_bulk, E):
        c_i = P_CO2_interface / self.Henry_CO2
        return c_i

    def solve_interface(self, P_CO2_bulk, c_CO2_bulk, c_amine_bulk, k2_rate):
        validate_positive(P_CO2_bulk, "Bulk CO2 partial pressure")
        validate_positive(c_amine_bulk, "Bulk amine concentration")


        Ha = hatta_number(k2_rate, self.D_CO2, c_amine_bulk, self.k_L)


        b_stoich = 2.0
        c_Ai_max = P_CO2_bulk / self.Henry_CO2
        E_infinite = 1.0 + c_amine_bulk * self.D_amine / (b_stoich * c_Ai_max * self.D_CO2)
        E_infinite = np.maximum(E_infinite, 1.0)


        if Ha < 0.3:
            E = 1.0 + Ha ** 2 / 3.0
        elif Ha > 10.0 * E_infinite:
            E = E_infinite
        else:


            E = np.minimum(Ha, E_infinite)
            for _ in range(20):
                E_new = Ha * np.sqrt((E_infinite - E) / max(E_infinite - 1.0, 1e-6))
                if abs(E_new - E) < 1e-6:
                    break
                E = 0.5 * (E + E_new)


        K_G = 1.0 / (1.0 / self.k_G + self.Henry_CO2 / (self.k_L * E))


        N_A = K_G * P_CO2_bulk

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
    validate_positive(n_points, "n_points")
    validate_positive(delta, "Film thickness")

    z = np.zeros(n_points)
    a, b = 0.0, delta

    for j in range(n_points):
        idx = j + 1
        if centering == 1:

            if n_points == 1:
                z[j] = 0.5 * (a + b)
            else:
                z[j] = ((n_points - idx) * a + (idx - 1) * b) / (n_points - 1)
        elif centering == 2:

            z[j] = ((n_points - idx + 1) * a + idx * b) / (n_points + 1)
        elif centering == 3:

            z[j] = ((n_points - idx + 1) * a + (idx - 1) * b) / n_points
        elif centering == 4:

            z[j] = ((n_points - idx) * a + idx * b) / n_points
        elif centering == 5:

            z[j] = ((2 * n_points - 2 * idx + 1) * a + (2 * idx - 1) * b) / (2 * n_points)
        else:
            raise ValueError(f"Invalid centering: {centering}")

    return z


class PackedColumnModel:

    def __init__(self, column_height, column_diameter, packing_type="random"):
        self.H = column_height
        self.D = column_diameter
        self.packing_type = packing_type

        if packing_type == "random":
            self.a_packing = 200.0
            self.eps_void = 0.92
            self.htu = 0.35
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
        validate_positive(T, "Temperature")
        validate_positive(L_flow, "Liquid flow")
        validate_positive(G_flow, "Gas flow")

        z = generate_film_grid(n_z, self.H, centering=1)


        n_L = L_flow / 18.015e-3
        n_G = G_flow / 22.414e-3


        u_L = n_L / self.cross_section
        u_G = n_G / self.cross_section

        y_CO2 = np.zeros(n_z)
        x_CO2 = np.zeros(n_z)
        T_profile = np.full(n_z, T)

        y_CO2[0] = y_CO2_in

        for i in range(n_z - 1):
            dz = z[i + 1] - z[i]
            P_CO2 = y_CO2[i] * P_total


            film = TwoFilmModel(T_profile[i], P_total)
            sol = film.solve_interface(P_CO2, 0.0, c_amine, 5000.0)


            N_A = sol["flux"]
            dN = N_A * self.a_packing * self.cross_section * dz


            dy = -dN / n_G
            y_CO2[i + 1] = y_CO2[i] + dy
            y_CO2[i + 1] = np.clip(y_CO2[i + 1], 1e-6, 0.5)


            delta_H_rxn = -85.0e3
            dT = -dN * delta_H_rxn / (L_flow * 4184.0)
            T_profile[i + 1] = T_profile[i] + dT

        return z, y_CO2, T_profile
