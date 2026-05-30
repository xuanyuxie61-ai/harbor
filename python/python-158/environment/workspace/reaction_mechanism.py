
import numpy as np


R_UNIVERSAL = 8.314462618


SPECIES_NAMES = [
    "N2", "O2", "N", "O", "NO", "NO2", "N2O",
    "CH", "HCN", "NH3", "OH", "H2O", "CO", "CO2", "H2", "CH4"
]
NSPEC = len(SPECIES_NAMES)


MW = np.array([
    28.0134e-3, 31.9988e-3, 14.0067e-3, 15.9994e-3, 30.0061e-3,
    46.0055e-3, 44.0128e-3, 13.0186e-3, 27.0253e-3, 17.0305e-3,
    17.0073e-3, 18.0153e-3, 28.0101e-3, 44.0095e-3, 2.0159e-3, 16.0425e-3
])










REACTIONS = []


def _add_reaction(reacs, prods, nu_reac, nu_prod, A, b, Ea, rtype):
    REACTIONS.append({
        "reacs": list(reacs),
        "prods": list(prods),
        "nu_reac": list(nu_reac),
        "nu_prod": list(nu_prod),
        "A": float(A),
        "b": float(b),
        "Ea": float(Ea),
        "type": str(rtype)
    })






_add_reaction([0, 3], [4, 2], [1, 1], [1, 1],
              A=1.8e11, b=0.0, Ea=319.0e3, rtype="thermal")

_add_reaction([2, 1], [4, 3], [1, 1], [1, 1],
              A=6.4e9, b=1.0, Ea=26.13e3, rtype="thermal")

_add_reaction([2, 10], [4, 11], [1, 1], [1, 1],
              A=3.8e10, b=0.0, Ea=0.0, rtype="thermal")





_add_reaction([8, 3], [4, 7], [1, 1], [1, 1],
              A=1.4e10, b=0.0, Ea=63.6e3, rtype="fuel")

_add_reaction([9, 3], [4, 14], [1, 1], [1, 1],
              A=9.4e9, b=0.0, Ea=26.8e3, rtype="fuel")

_add_reaction([8, 10], [5, 7], [1, 1], [1, 1],
              A=4.0e9, b=0.0, Ea=11.3e3, rtype="fuel")





_add_reaction([7, 0], [8, 2], [1, 1], [1, 1],
              A=2.0e11, b=0.0, Ea=37.7e3, rtype="prompt")







_add_reaction([4, 7], [8, 3], [1, 1], [1, 1],
              A=1.1e11, b=0.0, Ea=0.0, rtype="reburn")

_add_reaction([4, 7], [6, 14], [1, 1], [1, 1],
              A=2.8e10, b=0.0, Ea=0.0, rtype="reburn")





_add_reaction([6, 3], [4, 4], [1, 1], [2, 0],
              A=2.9e10, b=0.0, Ea=116.5e3, rtype="thermal")

_add_reaction([6, 14], [0, 10], [1, 1], [1, 1],
              A=1.6e11, b=0.0, Ea=32.7e3, rtype="thermal")





_add_reaction([4], [0, 1], [2], [1, 1],
              A=2.0e10, b=0.0, Ea=400.0e3, rtype="reburn")

_add_reaction([3, 3], [1], [1, 1], [1],
              A=1.0e13, b=0.0, Ea=0.0, rtype="thermal")

_add_reaction([3, 10], [1, 14], [1, 1], [1, 1],
              A=1.0e12, b=0.0, Ea=0.0, rtype="thermal")

NREAC = len(REACTIONS)


def arrhenius_rate(A: float, b: float, Ea: float, T: float) -> float:
    if T <= 0.0:
        return 0.0

    arg = -Ea / (R_UNIVERSAL * T)
    if arg < -700.0:
        return 0.0
    if arg > 700.0:

        return A * (T ** b) * np.exp(700.0)
    return A * (T ** b) * np.exp(arg)


def compute_production_rates(Y: np.ndarray, T: float, rho: float) -> np.ndarray:
    if rho <= 0.0 or T <= 0.0:
        return np.zeros(NSPEC)
    

    Y = np.maximum(Y, 0.0)
    Y_sum = np.sum(Y)
    if Y_sum > 0.0:
        Y = Y / Y_sum
    else:
        Y = np.zeros(NSPEC)
        Y[0] = 0.79
        Y[1] = 0.21
    

    C = rho * Y / MW
    C = np.maximum(C, 0.0)
    
    omega_molar = np.zeros(NSPEC)
    







    pass
    

    omega_mass = MW * omega_molar
    return omega_mass


def compute_jacobian_fd(Y: np.ndarray, T: float, rho: float, eps: float = 1e-8) -> np.ndarray:
    J = np.zeros((NSPEC, NSPEC))
    omega0 = compute_production_rates(Y, T, rho)
    for j in range(NSPEC):
        Yp = Y.copy()
        dy = max(eps * abs(Y[j]), eps)
        Yp[j] += dy
        omega_p = compute_production_rates(Yp, T, rho)
        J[:, j] = (omega_p - omega0) / dy
    return J


def get_NOx_indices() -> dict:
    return {
        "NO": SPECIES_NAMES.index("NO"),
        "NO2": SPECIES_NAMES.index("NO2"),
        "N2O": SPECIES_NAMES.index("N2O"),
        "HCN": SPECIES_NAMES.index("HCN"),
        "NH3": SPECIES_NAMES.index("NH3"),
    }


def get_pathway_contributions(Y: np.ndarray, T: float, rho: float) -> dict:
    if rho <= 0.0 or T <= 0.0:
        return {"thermal": 0.0, "fuel": 0.0, "prompt": 0.0, "reburn": 0.0}
    
    Y = np.maximum(Y, 0.0)
    Y_sum = np.sum(Y)
    if Y_sum > 0.0:
        Y = Y / Y_sum
    C = rho * Y / MW
    C = np.maximum(C, 0.0)
    
    no_idx = SPECIES_NAMES.index("NO")
    contributions = {"thermal": 0.0, "fuel": 0.0, "prompt": 0.0, "reburn": 0.0}
    
    for r in REACTIONS:
        k_r = arrhenius_rate(r["A"], r["b"], r["Ea"], T)
        q_r = k_r
        for j, spec_idx in enumerate(r["reacs"]):
            nu = r["nu_reac"][j]
            if nu > 0:
                q_r *= (C[spec_idx] ** nu)
        

        no_prod = 0.0
        for j, spec_idx in enumerate(r["prods"]):
            if spec_idx == no_idx:
                no_prod += r["nu_prod"][j] * q_r
        for j, spec_idx in enumerate(r["reacs"]):
            if spec_idx == no_idx:
                no_prod -= r["nu_reac"][j] * q_r
        
        if r["type"] in contributions:
            contributions[r["type"]] += MW[no_idx] * no_prod
    
    return contributions
