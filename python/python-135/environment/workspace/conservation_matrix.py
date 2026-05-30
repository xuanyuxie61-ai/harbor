
import numpy as np
from scipy.linalg import null_space
from utils import validate_positive


class StoichiometricAnalysis:

    def __init__(self):

        self.species = ["CO2", "MEA", "H2O", "RNHCOO-", "RNH3+", "HCO3-", "OH-", "H+"]
        self.n_species = len(self.species)








        self.reactions = [
            "CO2 + 2MEA -> RNHCOO- + RNH3+",
            "CO2 + MEA + H2O -> RNH3+ + HCO3-",
            "CO2 + OH- -> HCO3-",
            "H2O -> H+ + OH-",
            "MEA + H+ -> RNH3+",
            "RNHCOO- + H2O -> MEA + HCO3-"
        ]


        self.S = np.zeros((self.n_species, len(self.reactions)))

        self.S[0, 0] = -1.0
        self.S[1, 0] = -2.0
        self.S[3, 0] = 1.0
        self.S[4, 0] = 1.0

        self.S[0, 1] = -1.0
        self.S[1, 1] = -1.0
        self.S[2, 1] = -1.0
        self.S[4, 1] = 1.0
        self.S[5, 1] = 1.0

        self.S[0, 2] = -1.0
        self.S[6, 2] = -1.0
        self.S[5, 2] = 1.0

        self.S[2, 3] = -1.0
        self.S[7, 3] = 1.0
        self.S[6, 3] = 1.0

        self.S[1, 4] = -1.0
        self.S[7, 4] = -1.0
        self.S[4, 4] = 1.0

        self.S[3, 5] = -1.0
        self.S[2, 5] = -1.0
        self.S[1, 5] = 1.0
        self.S[5, 5] = 1.0

    def rank(self):
        return np.linalg.matrix_rank(self.S)

    def null_space_reactions(self):
        return null_space(self.S)

    def conservation_relations(self):

        U, s, Vh = np.linalg.svd(self.S)
        rank = np.sum(s > 1e-10)
        n_conserved = self.n_species - rank
        L = U[:, rank:].T
        return L, n_conserved

    def validate_conservation(self, concentrations):
        if isinstance(concentrations, dict):
            c = np.array([concentrations.get(s, 0.0) for s in self.species])
        else:
            c = np.asarray(concentrations)

        L, n_cons = self.conservation_relations()
        invariants = L @ c

        return {
            "num_conserved": n_cons,
            "invariant_values": invariants,
            "satisfied": np.allclose(invariants, invariants[0] if len(invariants) > 0 else 0, atol=1e-6)
        }

    def compute_elemental_matrix(self):



        E = np.array([
            [1, 2, 0, 3, 2, 1, 0, 0],
            [0, 7, 2, 6, 8, 1, 1, 1],
            [0, 1, 0, 1, 1, 0, 0, 0],
            [2, 1, 1, 3, 1, 3, 1, 0]
        ])
        return E

    def check_elemental_balance(self, reaction_idx):
        E = self.compute_elemental_matrix()
        reaction_vector = self.S[:, reaction_idx]
        balance = E @ reaction_vector
        return balance

    def overall_elemental_balance(self):
        E = self.compute_elemental_matrix()

        return E @ self.S

    def sensitivity_to_rate_constants(self, k_base, delta=0.1):
        sensitivities = {}
        for i, rxn in enumerate(self.reactions):
            k_plus = k_base.copy()
            k_plus[i] *= (1.0 + delta)


            sensitivities[rxn] = {"relative_change": delta}
        return sensitivities


def build_linear_conserved_ode_system():


    S = np.array([[-1.0, 1.0], [1.0, -1.0]])
    k1, k2 = 0.5, 0.3

    def rhs(t, y):
        r = np.array([k1 * y[0], k2 * y[1]])
        return S @ r

    def conserved_quantity(y):
        return y[0] + y[1]

    return rhs, conserved_quantity
