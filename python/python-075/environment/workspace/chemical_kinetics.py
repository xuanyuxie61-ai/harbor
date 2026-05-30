
import numpy as np


R_U = 8.314462618
P_ATM = 101325.0


MW = {
    'H2': 2.01588e-3,
    'O2': 31.9988e-3,
    'H2O': 18.01528e-3,
    'N2': 28.0134e-3,
}

SPECIES_NAMES = ['H2', 'O2', 'H2O', 'N2']


def molar_mass_vector():
    return np.array([MW[s] for s in SPECIES_NAMES])


def mass_fraction_to_mole_fraction(Y):
    W = molar_mass_vector()
    Y = np.atleast_1d(Y)
    if Y.ndim == 1:
        yw = Y / W
        return yw / yw.sum()
    else:

        yw = Y / W[:, None]
        return yw / yw.sum(axis=0)


def mole_fraction_to_mass_fraction(X):
    W = molar_mass_vector()
    X = np.atleast_1d(X)
    if X.ndim == 1:
        xw = X * W
        return xw / xw.sum()
    else:
        xw = X * W[:, None]
        return xw / xw.sum(axis=0)


def mixture_molecular_weight(Y):
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
    W_mix = mixture_molecular_weight(Y)
    return P * W_mix / (R_U * T)


def specific_heat_constant_pressure(T, species_idx=None):


    coeffs = np.array([
        [2.34433112E+00, 7.98052075E-03, -1.94781510E-05, 2.01572094E-08, -7.37611761E-12, 0.0, 0.0],
        [3.78245636E+00, -2.99673416E-03, 9.84730201E-06, -9.68129509E-09, 3.24372837E-12, 0.0, 0.0],
        [4.19864056E+00, -2.03643410E-03, 6.52040211E-06, -5.48797062E-09, 1.77197817E-12, 0.0, 0.0],
        [3.53100528E+00, -1.23660988E-04, -5.02999433E-07, 2.43530612E-09, -1.40881235E-12, 0.0, 0.0],
    ])
    T = np.atleast_1d(T)

    cp_over_R = (coeffs[:, 0:1] + coeffs[:, 1:2] * T + coeffs[:, 2:3] * T**2
                 + coeffs[:, 3:4] * T**3 + coeffs[:, 4:5] * T**4)
    W = molar_mass_vector()[:, None]
    cp = R_U * cp_over_R / W
    if species_idx is not None:
        return cp[species_idx]
    return cp


def enthalpy(T):
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
    h = R_U * h_over_RT * T / W
    return h


def mixture_cp(Y, T):
    cp_k = specific_heat_constant_pressure(T)
    Y = np.atleast_1d(Y)
    if Y.ndim == 1:
        return np.sum(Y[:, None] * cp_k, axis=0)
    else:
        return np.sum(Y * cp_k, axis=0)


def reaction_rates(Y, T, P=P_ATM):
    Y = np.atleast_1d(Y)
    scalar = (Y.ndim == 1)
    if scalar:
        Y = Y[:, None]

    W = molar_mass_vector()[:, None]
    rho = density_from_ideal_gas(Y, T, P)


    C = rho * Y / W


















    raise NotImplementedError("Hole 1: Implement Arrhenius reaction rates and species production")


def chemical_source_terms(Y, T, P=P_ATM):
    omega_dot, S_T, _ = reaction_rates(Y, T, P)
    W = molar_mass_vector()
    rho = density_from_ideal_gas(Y, T, P)
    S_Y = W[:, None] * omega_dot / rho if omega_dot.ndim > 1 else W * omega_dot / rho
    return S_Y, S_T


class SparseChemicalJacobian:

    def __init__(self, nspecies):
        self.nspecies = nspecies
        self.rows = []
        self.cols = []
        self.vals = []

    def add_entry(self, i, j, val):
        self.rows.append(i)
        self.cols.append(j)
        self.vals.append(val)

    def to_dense(self):
        J = np.zeros((self.nspecies, self.nspecies))
        for i, j, v in zip(self.rows, self.cols, self.vals):
            J[i, j] += v
        return J

    def to_coo_array(self):
        try:
            from scipy.sparse import coo_array
            return coo_array((self.vals, (self.rows, self.cols)),
                             shape=(self.nspecies, self.nspecies))
        except Exception:
            return {'rows': self.rows, 'cols': self.cols, 'vals': self.vals}


def compute_chemical_jacobian(Y, T, P=P_ATM, eps=1e-8):
    nspecies = len(SPECIES_NAMES)
    S_Y0, _ = chemical_source_terms(Y, T, P)

    jac = SparseChemicalJacobian(nspecies)
    for j in range(nspecies):
        Yp = Y.copy()
        Ym = Y.copy()
        delta = eps * max(Y[j], 1e-10)
        Yp[j] += delta
        Ym[j] -= delta

        Yp = Yp / Yp.sum()
        Ym = Ym / Ym.sum()
        S_p, _ = chemical_source_terms(Yp, T, P)
        S_m, _ = chemical_source_terms(Ym, T, P)
        dS = (S_p - S_m) / (2.0 * delta)
        for i in range(nspecies):
            if abs(dS[i]) > 1e-20:
                jac.add_entry(i, j, dS[i])
    return jac
