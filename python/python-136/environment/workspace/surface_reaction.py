
import numpy as np
from special_functions import arrhenius_rate


class SurfaceReactionError(Exception):
    pass


class LangmuirHinshelwoodKinetics:

    def __init__(self, k0, Ea, KA0, dH_ads_A, KB0, dH_ads_B,
                 reaction_order_A=1.0, reaction_order_B=1.0):
        self.k0 = k0
        self.Ea = Ea
        self.KA0 = KA0
        self.dH_ads_A = dH_ads_A
        self.KB0 = KB0
        self.dH_ads_B = dH_ads_B
        self.reaction_order_A = reaction_order_A
        self.reaction_order_B = reaction_order_B

    def rate(self, CA, CB, temperature):
        if CA < 0 or CB < 0:
            raise SurfaceReactionError("浓度必须非负")
        if temperature <= 0:
            raise SurfaceReactionError("温度必须为正")















        raise NotImplementedError("Hole 3: 请实现 Langmuir-Hinshelwood 反应速率公式")

    def jacobian_entries(self, CA, CB, temperature):
        eps = np.sqrt(np.finfo(float).eps) * max(CA, 1e-12)
        R0 = self.rate(CA, CB, temperature)
        RA = self.rate(CA + eps, CB, temperature)
        RB = self.rate(CA, CB + eps, temperature)
        dRdCA = (RA - R0) / eps
        dRdCB = (RB - R0) / eps
        return dRdCA, dRdCB


class PowerLawKinetics:

    def __init__(self, k0, Ea, nA, nB):
        self.k0 = k0
        self.Ea = Ea
        self.nA = nA
        self.nB = nB

    def rate(self, CA, CB, temperature):
        if CA < 0 or CB < 0:
            raise SurfaceReactionError("浓度必须非负")
        k = arrhenius_rate(self.k0, self.Ea, temperature)
        CA_eff = max(CA, 0.0)
        CB_eff = max(CB, 0.0)
        return k * (CA_eff ** self.nA) * (CB_eff ** self.nB)


class CatalyticParticleModel:

    def __init__(self, kinetics, particle_radius, porosity, tortuosity,
                 lambda_solid, lambda_gas, heat_of_reaction,
                 T_surface, C_surface_A, C_surface_B):
        self.kinetics = kinetics
        self.Rp = particle_radius
        self.porosity = porosity
        self.tortuosity = tortuosity
        self.lambda_eff = lambda_solid * (1.0 - porosity) + lambda_gas * porosity
        self.heat_of_reaction = heat_of_reaction
        self.T_surface = T_surface
        self.C_surface_A = C_surface_A
        self.C_surface_B = C_surface_B

    def effective_diffusivity(self, pore_diameter, temperature, molecular_weight,
                              bulk_diffusivity):
        from special_functions import effective_diffusivity
        return effective_diffusivity(
            pore_diameter, temperature, molecular_weight,
            bulk_diffusivity, self.tortuosity, self.porosity
        )

    def reaction_rate_local(self, CA, CB, T):
        return self.kinetics.rate(CA, CB, T)

    def heat_source(self, CA, CB, T):
        return self.reaction_rate_local(CA, CB, T) * (-self.heat_of_reaction)

    def thiele_modulus(self, D_e, T_surf):
        R_surf = self.reaction_rate_local(self.C_surface_A,
                                          self.C_surface_B, T_surf)
        if self.C_surface_A < np.finfo(float).eps:
            return 0.0
        k_obs = R_surf / self.C_surface_A
        phi = self.Rp * np.sqrt(max(k_obs, 0.0) / max(D_e, 1e-20))
        return phi

    def weisz_prater_criterion(self, eta, D_e, T_surf):
        phi = self.thiele_modulus(D_e, T_surf)
        return eta * (phi ** 2)
