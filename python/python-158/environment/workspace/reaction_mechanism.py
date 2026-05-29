"""
reaction_mechanism.py
===================
Detailed chemical reaction mechanism for NOx formation during pulverized coal combustion.

Incorporates reduced NOx chemistry with three principal pathways:
1. Thermal NOx (Zeldovich mechanism)
2. Fuel NOx (volatilization of fuel-nitrogen and subsequent oxidation)
3. Prompt NOx (Fenimore mechanism via CH radicals)

Scientific formulas and rate constants are based on:
- Miller & Bowman, Prog. Energy Combust. Sci., 1989
- Glarborg et al., Combust. Flame, 2018

Mathematical structure:
    dY/dt = S(Y, T)  where Y = mass fractions vector, T = temperature [K]
    S = sum over reactions r of:  nu_{i,r} * k_r(T) * prod_j [X_j]^{alpha_{j,r}}

Arrhenius rate law:
    k_r(T) = A_r * T^{b_r} * exp(-E_r / (R * T))

where R = 8.314 J/(mol·K) is the universal gas constant.
"""

import numpy as np

# Universal gas constant [J/(mol*K)]
R_UNIVERSAL = 8.314462618

# Species indices for the reduced mechanism
SPECIES_NAMES = [
    "N2", "O2", "N", "O", "NO", "NO2", "N2O",
    "CH", "HCN", "NH3", "OH", "H2O", "CO", "CO2", "H2", "CH4"
]
NSPEC = len(SPECIES_NAMES)

# Molecular weights [kg/mol]
MW = np.array([
    28.0134e-3, 31.9988e-3, 14.0067e-3, 15.9994e-3, 30.0061e-3,
    46.0055e-3, 44.0128e-3, 13.0186e-3, 27.0253e-3, 17.0305e-3,
    17.0073e-3, 18.0153e-3, 28.0101e-3, 44.0095e-3, 2.0159e-3, 16.0425e-3
])

# Reaction database: each entry is a dict with
#   reacs: list of reactant species indices
#   prods: list of product species indices
#   nu_reac: stoichiometric coefficients for reactants
#   nu_prod: stoichiometric coefficients for products
#   A: pre-exponential factor [SI units]
#   b: temperature exponent [-]
#   Ea: activation energy [J/mol]
#   type: 'thermal', 'fuel', 'prompt', or 'reburn'
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


# ======================================================================
# 1. THERMAL NOx: Extended Zeldovich mechanism
# ======================================================================
# R1: N2 + O  <=> NO + N
_add_reaction([0, 3], [4, 2], [1, 1], [1, 1],
              A=1.8e11, b=0.0, Ea=319.0e3, rtype="thermal")
# R2: N + O2  <=> NO + O
_add_reaction([2, 1], [4, 3], [1, 1], [1, 1],
              A=6.4e9, b=1.0, Ea=26.13e3, rtype="thermal")
# R3: N + OH  <=> NO + H  (simplified: N + OH -> NO + H2O)
_add_reaction([2, 10], [4, 11], [1, 1], [1, 1],
              A=3.8e10, b=0.0, Ea=0.0, rtype="thermal")

# ======================================================================
# 2. FUEL NOx: Fuel-N release and oxidation
# ======================================================================
# R4: HCN + O  -> NCO + H  (simplified to HCN + O -> NO + CH)
_add_reaction([8, 3], [4, 7], [1, 1], [1, 1],
              A=1.4e10, b=0.0, Ea=63.6e3, rtype="fuel")
# R5: NH3 + O  -> NH2 + OH  (simplified to NH3 + O -> NO + H2)
_add_reaction([9, 3], [4, 14], [1, 1], [1, 1],
              A=9.4e9, b=0.0, Ea=26.8e3, rtype="fuel")
# R6: HCN + OH -> CN + H2O  (simplified to HCN + OH -> NO2 + CH)
_add_reaction([8, 10], [5, 7], [1, 1], [1, 1],
              A=4.0e9, b=0.0, Ea=11.3e3, rtype="fuel")

# ======================================================================
# 3. PROMPT NOx: Fenimore CH + N2 pathway
# ======================================================================
# R7: CH + N2  -> HCN + N
_add_reaction([7, 0], [8, 2], [1, 1], [1, 1],
              A=2.0e11, b=0.0, Ea=37.7e3, rtype="prompt")
# R8: N + O2  -> NO + O  (same as R2, counted separately for pathway analysis)
# already covered by R2

# ======================================================================
# 4. NO REBURN: NO reduction by hydrocarbon radicals
# ======================================================================
# R9: NO + CH  -> HCN + O
_add_reaction([4, 7], [8, 3], [1, 1], [1, 1],
              A=1.1e11, b=0.0, Ea=0.0, rtype="reburn")
# R10: NO + CH2 -> HCN + OH  (simplified as NO + CH -> N2O + H)
_add_reaction([4, 7], [6, 14], [1, 1], [1, 1],
              A=2.8e10, b=0.0, Ea=0.0, rtype="reburn")

# ======================================================================
# 5. N2O PATHWAY: Intermediate from fuel-N
# ======================================================================
# R11: N2O + O  -> 2 NO
_add_reaction([6, 3], [4, 4], [1, 1], [2, 0],
              A=2.9e10, b=0.0, Ea=116.5e3, rtype="thermal")
# R12: N2O + H  -> N2 + OH  (simplified)
_add_reaction([6, 14], [0, 10], [1, 1], [1, 1],
              A=1.6e11, b=0.0, Ea=32.7e3, rtype="thermal")

# ======================================================================
# 6. NO THERMAL DECOMPOSITION (equilibrium limitation)
# ======================================================================
# R13: 2NO -> N2 + O2  (limits unphysical NO accumulation)
_add_reaction([4], [0, 1], [2], [1, 1],
              A=2.0e10, b=0.0, Ea=400.0e3, rtype="reburn")
# R14: O + O -> O2  (O-atom recombination, fast)
_add_reaction([3, 3], [1], [1, 1], [1],
              A=1.0e13, b=0.0, Ea=0.0, rtype="thermal")
# R15: O + OH -> O2 + H  (radical termination, simplified as O + OH -> O2 + H2)
_add_reaction([3, 10], [1, 14], [1, 1], [1, 1],
              A=1.0e12, b=0.0, Ea=0.0, rtype="thermal")

NREAC = len(REACTIONS)


def arrhenius_rate(A: float, b: float, Ea: float, T: float) -> float:
    """
    Compute Arrhenius rate coefficient:
        k(T) = A * T^b * exp(-Ea / (R * T))
    
    Args:
        A: pre-exponential factor
        b: temperature exponent
        Ea: activation energy [J/mol]
        T: temperature [K]
    
    Returns:
        k: rate coefficient [SI units depending on reaction order]
    """
    if T <= 0.0:
        return 0.0
    # Guard against overflow in exp
    arg = -Ea / (R_UNIVERSAL * T)
    if arg < -700.0:
        return 0.0
    if arg > 700.0:
        # Extremely high T, rate dominates
        return A * (T ** b) * np.exp(700.0)
    return A * (T ** b) * np.exp(arg)


def compute_production_rates(Y: np.ndarray, T: float, rho: float) -> np.ndarray:
    """
    Compute species production rates [kg/(m^3·s)] from mass fractions.
    
    The molar concentration of species i is:
        [X_i] = rho * Y_i / MW_i   [mol/m^3]
    
    For each reaction r:
        q_r = k_r(T) * prod_{j in reactants} [X_j]^{nu_{j,r}}
    
    The molar production rate of species i:
        omega_i^{molar} = sum_r (nu_{i,r}^{prod} - nu_{i,r}^{reac}) * q_r
    
    The mass production rate:
        omega_i^{mass} = MW_i * omega_i^{molar}
    
    Args:
        Y: mass fraction array [NSPEC], must sum to 1.0
        T: temperature [K]
        rho: mixture density [kg/m^3]
    
    Returns:
        omega: mass production rates [kg/(m^3·s)]
    """
    if rho <= 0.0 or T <= 0.0:
        return np.zeros(NSPEC)
    
    # Ensure non-negative mass fractions
    Y = np.maximum(Y, 0.0)
    Y_sum = np.sum(Y)
    if Y_sum > 0.0:
        Y = Y / Y_sum
    else:
        Y = np.zeros(NSPEC)
        Y[0] = 0.79
        Y[1] = 0.21
    
    # Molar concentrations [mol/m^3]
    C = rho * Y / MW
    C = np.maximum(C, 0.0)
    
    omega_molar = np.zeros(NSPEC)
    
    # HOLE 1: Compute molar reaction rates and net stoichiometric changes.
    # For each reaction r in REACTIONS:
    #   1. Compute Arrhenius rate coefficient k_r = A * T^b * exp(-Ea/(R*T))
    #   2. Compute reaction rate q_r = k_r * prod_j [C_j]^{nu_j}
    #   3. Update omega_molar for reactants (subtract) and products (add)
    # Finally convert to mass production rates: omega_mass = MW * omega_molar
    # TODO: implement the reaction rate loop and unit conversion
    pass  # HOLE 1
    
    # Convert to mass production rates
    omega_mass = MW * omega_molar
    return omega_mass


def compute_jacobian_fd(Y: np.ndarray, T: float, rho: float, eps: float = 1e-8) -> np.ndarray:
    """
    Compute Jacobian J_{ij} = d(omega_i)/dY_j by finite differences.
    Used for stiffness analysis and implicit time integration.
    """
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
    """Return indices for NOx-related species."""
    return {
        "NO": SPECIES_NAMES.index("NO"),
        "NO2": SPECIES_NAMES.index("NO2"),
        "N2O": SPECIES_NAMES.index("N2O"),
        "HCN": SPECIES_NAMES.index("HCN"),
        "NH3": SPECIES_NAMES.index("NH3"),
    }


def get_pathway_contributions(Y: np.ndarray, T: float, rho: float) -> dict:
    """
    Compute NO formation rate decomposed by pathway:
        thermal, fuel, prompt, reburn
    Returns NO mass production rate [kg/(m^3·s)] for each pathway.
    """
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
        
        # Net production of NO in this reaction
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
