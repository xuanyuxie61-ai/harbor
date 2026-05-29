"""
chemical_kinetics.py
====================
Detailed Chemical Kinetics for H2-Air Combustion with Sparse Jacobian Handling.

Based on seed project 772 (mm_to_st):
- Sparse matrix construction and manipulation for the chemical Jacobian
- Matrix Market inspired sparse coordinate storage

Chemical Mechanism (Reduced H2-O2, 4 species + N2):
--------------------------------------------------
The H2-O2 combustion system is modeled with the following global reactions
and Arrhenius rate laws:

  R1: 2H2 + O2  → 2H2O      (global oxidation)
  R2: H2 + O2   → OH + OH   (chain branching)
  R3: H2 + OH   → H2O + H   (propagation)
  R4: H + O2    → OH + O    (branching)
  R5: OH + H    → H2O       (termination)

Rate law for each reaction i:
  q_i = A_i * T^{n_i} * exp(-E_{a,i} / (R_u T)) * ∏_k [X_k]^{ν'_{k,i}}

where:
  A_i      = pre-exponential factor [units vary]
  n_i      = temperature exponent
  E_{a,i}  = activation energy [J/mol]
  R_u      = universal gas constant = 8.314462618 J/(mol·K)
  [X_k]    = molar concentration of species k [mol/m³]
  ν'_{k,i} = stoichiometric coefficient of reactant k in reaction i

Species production rate:
  ω̇_k = Σ_i (ν''_{k,i} - ν'_{k,i}) * q_i
      = Σ_i ν_{k,i} * q_i

The chemical source term for mass fraction Y_k:
  S_k = W_k * ω̇_k / ρ

where W_k is the molecular weight of species k [kg/mol].

Thermodynamic properties:
  cp_k = a_{0,k} + a_{1,k}T + a_{2,k}T² + a_{3,k}T³ + a_{4,k}T⁴
  h_k  = ∫ cp_k dT = a_{0,k}T + a_{1,k}T²/2 + ... + a_{4,k}T⁵/5

Temperature equation source:
  S_T = - Σ_k h_k * W_k * ω̇_k / (ρ * cp_mix)
      = - Σ_k h_k * S_k / cp_mix
"""

import numpy as np

# Physical constants
R_U = 8.314462618  # J/(mol·K)
P_ATM = 101325.0   # Pa

# Molecular weights [kg/mol]
MW = {
    'H2': 2.01588e-3,
    'O2': 31.9988e-3,
    'H2O': 18.01528e-3,
    'N2': 28.0134e-3,
}

SPECIES_NAMES = ['H2', 'O2', 'H2O', 'N2']


def molar_mass_vector():
    """Return array of molecular weights [kg/mol]."""
    return np.array([MW[s] for s in SPECIES_NAMES])


def mass_fraction_to_mole_fraction(Y):
    """
    Convert mass fractions Y to mole fractions X.
    X_k = (Y_k / W_k) / Σ_j (Y_j / W_j)
    """
    W = molar_mass_vector()
    Y = np.atleast_1d(Y)
    if Y.ndim == 1:
        yw = Y / W
        return yw / yw.sum()
    else:
        # Batch: Y shape (nspecies, npts)
        yw = Y / W[:, None]
        return yw / yw.sum(axis=0)


def mole_fraction_to_mass_fraction(X):
    """
    Convert mole fractions X to mass fractions Y.
    Y_k = X_k * W_k / Σ_j X_j * W_j
    """
    W = molar_mass_vector()
    X = np.atleast_1d(X)
    if X.ndim == 1:
        xw = X * W
        return xw / xw.sum()
    else:
        xw = X * W[:, None]
        return xw / xw.sum(axis=0)


def mixture_molecular_weight(Y):
    """
    Mixture molecular weight [kg/mol].
    W_mix = 1 / Σ_k (Y_k / W_k)
    """
    W = molar_mass_vector()
    Y = np.atleast_1d(Y)
    if Y.ndim == 1:
        yw = Y / W
        yw = np.clip(yw, 1e-30, None)
        return 1.0 / np.sum(yw)
    else:
        yw = Y / W[:, None]
        yw = np.clip(yw, 1e-30, None)
        return 1.0 / np.sum(yw, axis=0)


def density_from_ideal_gas(Y, T, P=P_ATM):
    """
    Ideal gas density: ρ = P * W_mix / (R_u * T)
    """
    W_mix = mixture_molecular_weight(Y)
    return P * W_mix / (R_U * T)


def specific_heat_constant_pressure(T, species_idx=None):
    """
    NASA polynomial for cp/R for H2-O2 species.
    cp_k = R_u * (a0 + a1*T + a2*T² + a3*T³ + a4*T⁴)
    Coefficients valid for T ∈ [200, 6000] K.
    Returns cp [J/(kg·K)] if species_idx is given, else matrix (nspecies, npts).
    """
    # NASA-7 coefficients (low temperature range 200-1000K used here)
    # Format: [a0, a1, a2, a3, a4, a5, a6]
    coeffs = np.array([
        [2.34433112E+00, 7.98052075E-03, -1.94781510E-05, 2.01572094E-08, -7.37611761E-12, 0.0, 0.0],  # H2
        [3.78245636E+00, -2.99673416E-03, 9.84730201E-06, -9.68129509E-09, 3.24372837E-12, 0.0, 0.0],  # O2
        [4.19864056E+00, -2.03643410E-03, 6.52040211E-06, -5.48797062E-09, 1.77197817E-12, 0.0, 0.0],  # H2O
        [3.53100528E+00, -1.23660988E-04, -5.02999433E-07, 2.43530612E-09, -1.40881235E-12, 0.0, 0.0],  # N2
    ])
    T = np.atleast_1d(T)
    # cp/R = a0 + a1*T + a2*T² + a3*T³ + a4*T⁴
    cp_over_R = (coeffs[:, 0:1] + coeffs[:, 1:2] * T + coeffs[:, 2:3] * T**2
                 + coeffs[:, 3:4] * T**3 + coeffs[:, 4:5] * T**4)
    W = molar_mass_vector()[:, None]
    cp = R_U * cp_over_R / W  # J/(kg·K)
    if species_idx is not None:
        return cp[species_idx]
    return cp


def enthalpy(T):
    """
    NASA polynomial for h/(R_u*T) → h [J/kg].
    h_k = R_u * (a0*T + a1*T²/2 + a2*T³/3 + a3*T⁴/4 + a4*T⁵/5)
    """
    coeffs = np.array([
        [2.34433112E+00, 7.98052075E-03, -1.94781510E-05, 2.01572094E-08, -7.37611761E-12],
        [3.78245636E+00, -2.99673416E-03, 9.84730201E-06, -9.68129509E-09, 3.24372837E-12],
        [4.19864056E+00, -2.03643410E-03, 6.52040211E-06, -5.48797062E-09, 1.77197817E-12],
        [3.53100528E+00, -1.23660988E-04, -5.02999433E-07, 2.43530612E-09, -1.40881235E-12],
    ])
    T = np.atleast_1d(T)
    h_over_RT = (coeffs[:, 0:1] + coeffs[:, 1:2] * T / 2.0 + coeffs[:, 2:3] * T**2 / 3.0
                 + coeffs[:, 3:4] * T**3 / 4.0 + coeffs[:, 4:5] * T**4 / 5.0)
    W = molar_mass_vector()[:, None]
    h = R_U * h_over_RT * T / W  # J/kg
    return h


def mixture_cp(Y, T):
    """
    Mixture specific heat: cp_mix = Σ_k Y_k * cp_k(T)
    """
    cp_k = specific_heat_constant_pressure(T)  # (nspecies, npts)
    Y = np.atleast_1d(Y)
    if Y.ndim == 1:
        return np.sum(Y[:, None] * cp_k, axis=0)
    else:
        return np.sum(Y * cp_k, axis=0)


def reaction_rates(Y, T, P=P_ATM):
    """
    Compute species production rates ω̇ [mol/(m³·s)] and temperature source S_T [K/s].

    Returns
    -------
    omega_dot : ndarray, shape (nspecies,) or (nspecies, npts)
        Molar production rates [mol/(m³·s)].
    S_T : float or ndarray
        Temperature source term [K/s].
    q_reactions : ndarray
        Individual reaction rates q_i [mol/(m³·s)].
    """
    Y = np.atleast_1d(Y)
    scalar = (Y.ndim == 1)
    if scalar:
        Y = Y[:, None]

    W = molar_mass_vector()[:, None]
    rho = density_from_ideal_gas(Y, T, P)

    # Molar concentrations [mol/m³]: C_k = ρ * Y_k / W_k
    C = rho * Y / W  # (nspecies, 1) or (nspecies, npts)

    # TODO [Hole 1]: Implement Arrhenius reaction rate computation and species production rates.
    # This is the core chemical kinetics for a reduced H2-O2 mechanism with 3 effective reactions.
    #
    # Required steps:
    #   1. Define pre-exponential factors A [m³/(mol·s) or 1/s], temperature exponents n,
    #      and activation energies Ea [J/mol] for the 3 reactions.
    #   2. Compute rate constants: kf_i = A_i * T^{n_i} * exp(-Ea_i / (R_u * T))
    #   3. Compute reaction rates q_i = kf_i * prod_k(C_k^{ν'_ki}) where C_k = ρ * Y_k / W_k
    #      Reaction 1: 2H2 + O2 → 2H2O      → q1 = kf[0] * C[H2]^2 * C[O2]
    #      Reaction 2: H2 + 0.5O2 → H2O     → q2 = kf[1] * C[H2] * C[O2]
    #      Reaction 3: H2O → H2 + 0.5O2     → q3 = kf[2] * C[H2O]
    #   4. Define stoichiometric matrix ν (nspecies × nreactions) where ν_{k,i} = ν''_{k,i} - ν'_{k,i}
    #   5. Compute species production rates: ω̇_k = Σ_i ν_{k,i} * q_i  [mol/(m³·s)]
    #   6. Compute temperature source: S_T = -Σ_k h_k * W_k * ω̇_k / (ρ * cp_mix)  [K/s]
    #
    # Note: The stoichiometric matrix here must be consistent with the element conservation
    #       verification in stoichiometry_analysis.py.
    raise NotImplementedError("Hole 1: Implement Arrhenius reaction rates and species production")


def chemical_source_terms(Y, T, P=P_ATM):
    """
    Return mass fraction source terms S_k = W_k * ω̇_k / ρ  [1/s]
    and temperature source S_T [K/s].
    """
    omega_dot, S_T, _ = reaction_rates(Y, T, P)
    W = molar_mass_vector()
    rho = density_from_ideal_gas(Y, T, P)
    S_Y = W[:, None] * omega_dot / rho if omega_dot.ndim > 1 else W * omega_dot / rho
    return S_Y, S_T


class SparseChemicalJacobian:
    """
    Sparse chemical Jacobian handling inspired by seed 772 (mm_to_st).
    Stores Jacobian in coordinate format (COO) for memory efficiency.

    The Jacobian J_{ij} = ∂(dY_i/dt) / ∂Y_j for the chemical source terms.
    For nspecies species, J is nspecies × nspecies.
    """

    def __init__(self, nspecies):
        self.nspecies = nspecies
        self.rows = []
        self.cols = []
        self.vals = []

    def add_entry(self, i, j, val):
        """Add a Jacobian entry in coordinate format."""
        self.rows.append(i)
        self.cols.append(j)
        self.vals.append(val)

    def to_dense(self):
        """Convert to dense matrix (for small systems only)."""
        J = np.zeros((self.nspecies, self.nspecies))
        for i, j, v in zip(self.rows, self.cols, self.vals):
            J[i, j] += v
        return J

    def to_coo_array(self):
        """Return as scipy.sparse COO matrix if available, else dict."""
        try:
            from scipy.sparse import coo_array
            return coo_array((self.vals, (self.rows, self.cols)),
                             shape=(self.nspecies, self.nspecies))
        except Exception:
            return {'rows': self.rows, 'cols': self.cols, 'vals': self.vals}


def compute_chemical_jacobian(Y, T, P=P_ATM, eps=1e-8):
    """
    Numerical Jacobian of the chemical source term.
    J_{ij} = (S_i(Y + ε e_j) - S_i(Y - ε e_j)) / (2ε)
    """
    nspecies = len(SPECIES_NAMES)
    S_Y0, _ = chemical_source_terms(Y, T, P)

    jac = SparseChemicalJacobian(nspecies)
    for j in range(nspecies):
        Yp = Y.copy()
        Ym = Y.copy()
        delta = eps * max(Y[j], 1e-10)
        Yp[j] += delta
        Ym[j] -= delta
        # Renormalize to sum=1
        Yp = Yp / Yp.sum()
        Ym = Ym / Ym.sum()
        S_p, _ = chemical_source_terms(Yp, T, P)
        S_m, _ = chemical_source_terms(Ym, T, P)
        dS = (S_p - S_m) / (2.0 * delta)
        for i in range(nspecies):
            if abs(dS[i]) > 1e-20:
                jac.add_entry(i, j, dS[i])
    return jac
