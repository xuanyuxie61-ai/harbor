
import numpy as np
from utils import R_GAS, STANDARD_TEMP, validate_positive, safe_log, van_t_hoff
from spectral_methods import polynomial_fit_2d_vandermonde, evaluate_2d_polynomial


class KentEisenbergModel:

    def __init__(self, amine_type="MEA"):
        self.amine_type = amine_type

        self.K1_0 = 2.5e7
        self.K2_0 = 1.2e3
        self.K3_0 = 2.2e4
        self.Kw_0 = 1.0e-14
        self.Ka_0 = 10.0 ** (-9.5)


        self.dH1 = -52000.0
        self.dH2 = -38000.0
        self.dH3 = -45000.0
        self.dHw = 55000.0
        self.dHa = 30000.0

    def equilibrium_constants(self, T):
        validate_positive(T, "Temperature")
        K1 = van_t_hoff(self.K1_0, self.dH1, T)
        K2 = van_t_hoff(self.K2_0, self.dH2, T)
        K3 = van_t_hoff(self.K3_0, self.dH3, T)
        Kw = van_t_hoff(self.Kw_0, self.dHw, T)
        Ka = van_t_hoff(self.Ka_0, self.dHa, T)
        return K1, K2, K3, Kw, Ka

    def solve_speciation(self, T, alpha, c_amine_total):
        validate_positive(T, "Temperature")
        validate_positive(c_amine_total, "Total amine concentration")
        alpha = np.clip(alpha, 0.0, 0.55)

        K1, K2, K3, Kw, Ka = self.equilibrium_constants(T)


        c_RNH2 = c_amine_total * (1.0 - 2.0 * alpha) / (1.0 - alpha)
        c_RNH2 = np.maximum(c_RNH2, 1e-10)
        c_RNH3 = c_amine_total * alpha / (1.0 - alpha)
        c_RNHCOO = c_amine_total * alpha / (1.0 - alpha)


        c_H = Ka * c_RNH2 / c_RNH3
        c_OH = Kw / c_H
        c_HCO3 = K2 * c_CO2 * c_RNH2 / c_RNH3 if 'c_CO2' in locals() else 0.0

        return {
            "RNH2": c_RNH2,
            "RNH3+": c_RNH3,
            "RNHCOO-": c_RNHCOO,
            "H+": c_H,
            "OH-": c_OH,
            "HCO3-": max(c_HCO3, 1e-10) if 'c_CO2' in locals() else 1e-10
        }

    def CO2_partial_pressure(self, T, alpha, c_amine_total):
        validate_positive(T, "Temperature")
        validate_positive(c_amine_total, "Total amine concentration")
        alpha = np.clip(alpha, 0.01, 0.55)


        A = 35.2
        B = -8300.0
        C = 2.5
        D = -8.0
        E = 12.0

        ln_P = A + B / T + C * np.log(alpha) + D * alpha ** 2 + E * alpha
        P_CO2 = np.exp(ln_P)
        return max(float(P_CO2), 1.0)


class ExtendedUNIQUAC:

    def __init__(self):

        self.q = {"H2O": 1.4, "MEA": 1.5, "CO2": 1.3}
        self.r = {"H2O": 0.92, "MEA": 1.2, "CO2": 1.1}

        self.a_ij = {
            ("H2O", "MEA"): -150.0,
            ("MEA", "H2O"): 200.0,
            ("H2O", "CO2"): 100.0,
            ("CO2", "H2O"): -50.0,
        }

    def activity_coefficient(self, T, x):
        validate_positive(T, "Temperature")
        species = list(x.keys())
        r = np.array([self.r.get(s, 1.0) for s in species])
        q = np.array([self.q.get(s, 1.0) for s in species])
        x_arr = np.array([x[s] for s in species])


        phi = r * x_arr / np.sum(r * x_arr)
        theta = q * x_arr / np.sum(q * x_arr)
        l = 5.0 * (r - q) - (r - 1.0)


        ln_gamma_c = safe_log(phi / x_arr) + 5.0 * q * safe_log(theta / phi) + l - phi / x_arr * np.sum(x_arr * l)


        tau = np.ones((len(species), len(species)))
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                if i != j and (si, sj) in self.a_ij:
                    tau[i, j] = np.exp(-self.a_ij[(si, sj)] / T)

        ln_gamma_r = np.zeros(len(species))
        for i in range(len(species)):
            S = np.sum(theta * tau[:, i])
            ln_gamma_r[i] = q[i] * (1.0 - safe_log(S) - np.sum(theta * tau[i, :] / S))

        ln_gamma = ln_gamma_c + ln_gamma_r
        gamma = np.exp(np.clip(ln_gamma, -50, 50))
        return {s: gamma[i] for i, s in enumerate(species)}


class VLEPolynomialFitter:

    def __init__(self, degree=4):
        self.degree = degree
        self.coeffs = None
        self.cond_num = None

    def fit(self, T_data, alpha_data, P_data):

        self.T_mean = np.mean(T_data)
        self.T_std = np.std(T_data) + 1e-10
        self.alpha_mean = np.mean(alpha_data)
        self.alpha_std = np.std(alpha_data) + 1e-10

        T_norm = (T_data - self.T_mean) / self.T_std
        alpha_norm = (alpha_data - self.alpha_mean) / self.alpha_std
        P_log = np.log10(P_data + 1e-10)

        self.coeffs, self.cond_num = polynomial_fit_2d_vandermonde(
            T_norm, alpha_norm, P_log, degree=self.degree
        )
        return self.coeffs, self.cond_num

    def predict(self, T, alpha):
        if self.coeffs is None:
            raise RuntimeError("Model not fitted yet")
        T_norm = (T - self.T_mean) / self.T_std
        alpha_norm = (alpha - self.alpha_mean) / self.alpha_std
        P_log = evaluate_2d_polynomial(self.coeffs, self.degree, T_norm, alpha_norm)
        return 10.0 ** P_log


def generate_vle_dataset(amine_type="MEA", c_amine=5.0):
    model = KentEisenbergModel(amine_type)
    T_range = np.linspace(298.15, 393.15, 15)
    alpha_range = np.linspace(0.01, 0.5, 15)

    T_data, alpha_data, P_data = [], [], []
    for T in T_range:
        for alpha in alpha_range:
            P = model.CO2_partial_pressure(T, alpha, c_amine)
            T_data.append(T)
            alpha_data.append(alpha)
            P_data.append(P)

    return np.array(T_data), np.array(alpha_data), np.array(P_data)
