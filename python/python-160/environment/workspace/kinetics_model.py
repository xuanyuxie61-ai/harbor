
import math
import numpy as np



R_GAS = 8.314462618


class ArrheniusRate:

    def __init__(self, A, Ea):
        self.A = float(A)
        self.Ea = float(Ea)

    def rate(self, T):
        if T <= 0.0:
            return 0.0
        return self.A * math.exp(-self.Ea / (R_GAS * T))

    def derivative_dk_dT(self, T):
        k = self.rate(T)
        if T <= 0.0:
            return 0.0
        return k * self.Ea / (R_GAS * T ** 2)


class GasificationKinetics:

    def __init__(self):


        self.reactions = {
            'R1': ArrheniusRate(A=1.0e8, Ea=135000.0),
            'R2': ArrheniusRate(A=5.0e7, Ea=125000.0),
            'R3': ArrheniusRate(A=3.0e5, Ea=220000.0),
            'R4': ArrheniusRate(A=2.0e5, Ea=180000.0),
            'R5': ArrheniusRate(A=1.0e4, Ea=120000.0),
            'R6': ArrheniusRate(A=2.5e3, Ea=85000.0),
            'R7': ArrheniusRate(A=3.0e6, Ea=200000.0),
        }

        self.dH = {
            'R1': -393500.0,
            'R2': -221000.0,
            'R3': +172500.0,
            'R4': +131400.0,
            'R5': -74800.0,
            'R6': -41200.0,
            'R7': +206100.0,
        }


        self.species = ['C', 'O2', 'CO2', 'CO', 'H2O', 'H2', 'CH4']
        self.nu = np.array([
            [-1, -2, -1, -1, -1,  0,  0],
            [-1, -1,  0,  0,  0, -1, -1],
            [ 1,  0, -1,  0,  0,  1,  0],
            [ 0,  2,  2,  1,  0, -1,  1],
            [ 0,  0,  0, -1,  0, -1, -1],
            [ 0,  0,  0,  1,  2,  1,  3],
            [ 0,  0,  0,  0,  1,  0, -1],
        ], dtype=float)

    def reaction_rates(self, T, conc):
        conc = np.asarray(conc, dtype=float)
        if len(conc) != len(self.species):
            raise ValueError("Concentration vector length mismatch")

        conc = np.maximum(conc, 0.0)

        c_C, c_O2, c_CO2, c_CO, c_H2O, c_H2, c_CH4 = conc


        r = {}
        r['R1'] = self.reactions['R1'].rate(T) * c_C * c_O2
        r['R2'] = self.reactions['R2'].rate(T) * (c_C ** 2) * c_O2
        r['R3'] = self.reactions['R3'].rate(T) * c_C * c_CO2
        r['R4'] = self.reactions['R4'].rate(T) * c_C * c_H2O
        r['R5'] = self.reactions['R5'].rate(T) * c_C * (c_H2 ** 2)
        r['R6'] = self.reactions['R6'].rate(T) * (c_CO * c_H2O - c_CO2 * c_H2 / self.wgs_equilibrium(T))
        r['R7'] = self.reactions['R7'].rate(T) * c_CH4 * c_H2O

        return r

    def wgs_equilibrium(self, T):
        if T <= 0.0:
            return 1.0
        dH = -41200.0
        dS = -42.3
        lnK = -dH / (R_GAS * T) + dS / R_GAS
        return math.exp(lnK)

    def species_production_rates(self, T, conc):
        r_dict = self.reaction_rates(T, conc)
        r_vec = np.array([r_dict[k] for k in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']], dtype=float)
        omega = self.nu.dot(r_vec)
        return omega

    def heat_of_reaction(self, T, conc):
        r_dict = self.reaction_rates(T, conc)
        q = 0.0
        for key, rate in r_dict.items():
            q += (-self.dH[key]) * rate
        return q


class MarkovReactorState:

    def __init__(self):
        self.states = ['drying', 'pyrolysis', 'combustion', 'reduction', 'exit']
        self.n_states = len(self.states)

        self.P = np.array([
            [0.70, 0.25, 0.00, 0.00, 0.05],
            [0.00, 0.60, 0.30, 0.00, 0.10],
            [0.00, 0.00, 0.50, 0.40, 0.10],
            [0.00, 0.00, 0.00, 0.70, 0.30],
            [0.00, 0.00, 0.00, 0.00, 1.00],
        ], dtype=float)
        self._validate_transition_matrix()

    def _validate_transition_matrix(self):
        row_sums = self.P.sum(axis=1)
        for i in range(self.n_states):
            if abs(row_sums[i] - 1.0) > 1.0e-6:

                if row_sums[i] > 1.0e-15:
                    self.P[i, :] /= row_sums[i]
                else:
                    self.P[i, :] = 0.0
                    self.P[i, -1] = 1.0

    def state_probability(self, initial_state, steps):
        idx = self.states.index(initial_state)
        p0 = np.zeros(self.n_states, dtype=float)
        p0[idx] = 1.0
        Pn = np.linalg.matrix_power(self.P, steps)
        return p0.dot(Pn)

    def expected_residence_steps(self, initial_state):
        idx = self.states.index(initial_state)

        transient = [i for i in range(self.n_states) if self.states[i] != 'exit']
        if idx not in transient:
            return 0.0
        Q = self.P[np.ix_(transient, transient)]
        I = np.eye(len(transient))
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            N = np.linalg.pinv(I - Q)
        local_idx = transient.index(idx)
        return N[local_idx, :].sum()

    def steady_state_distribution(self):
        w, v = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, idx])
        if pi.sum() > 1.0e-15:
            pi = pi / pi.sum()
        else:
            pi = np.zeros(self.n_states)
            pi[-1] = 1.0
        return pi


class BinomialKinetics:

    @staticmethod
    def binomial_coefficient(n, k):
        if k < 0 or k > n or n < 0:
            return 0.0
        if k == 0 or k == n:
            return 1.0
        k = min(k, n - k)
        result = 1.0
        for i in range(1, k + 1):
            result = result * (n - k + i) / i
        return result

    @staticmethod
    def stirling_approximation(n):
        if n <= 0:
            return 1.0
        return math.sqrt(2.0 * math.pi * n) * (n / math.e) ** n * \
               (1.0 + 1.0 / (12.0 * n) + 1.0 / (288.0 * n ** 2))

    @staticmethod
    def probability_mass(n, k, p):
        if p < 0.0 or p > 1.0 or n < 0:
            return 0.0
        if k < 0 or k > n:
            return 0.0

        log_c = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        log_pmf = log_c + k * math.log(p) + (n - k) * math.log(1.0 - p)
        return math.exp(log_pmf)

    @staticmethod
    def expected_value(n, p):
        return n * p

    @staticmethod
    def variance(n, p):
        return n * p * (1.0 - p)
