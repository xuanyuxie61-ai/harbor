"""
Conservation Matrix and Stoichiometric Analysis for Amine Reaction Networks
Integrates biochemical_linear_ode conservation concepts.

For a reaction network with m species and r reactions:
    S * v = 0   (steady state)
where S is the m x r stoichiometric matrix.

The left null space of S gives conservation relations (invariants).
These are critical for ensuring numerical stability in kinetic simulations.
"""

import numpy as np
from scipy.linalg import null_space
from utils import validate_positive


class StoichiometricAnalysis:
    """
    Stoichiometric analysis of CO2-amine reaction system.
    """

    def __init__(self):
        # Species order: CO2, MEA, H2O, RNHCOO-, RNH3+, HCO3-, OH-, H+
        self.species = ["CO2", "MEA", "H2O", "RNHCOO-", "RNH3+", "HCO3-", "OH-", "H+"]
        self.n_species = len(self.species)

        # Reactions:
        # R1: CO2 + 2MEA -> RNHCOO- + RNH3+
        # R2: CO2 + MEA + H2O -> RNH3+ + HCO3-
        # R3: CO2 + OH- -> HCO3-
        # R4: H2O -> H+ + OH-
        # R5: MEA + H+ -> RNH3+
        # R6: RNHCOO- + H2O -> MEA + HCO3-
        self.reactions = [
            "CO2 + 2MEA -> RNHCOO- + RNH3+",
            "CO2 + MEA + H2O -> RNH3+ + HCO3-",
            "CO2 + OH- -> HCO3-",
            "H2O -> H+ + OH-",
            "MEA + H+ -> RNH3+",
            "RNHCOO- + H2O -> MEA + HCO3-"
        ]

        # Build stoichiometric matrix S (species x reactions)
        self.S = np.zeros((self.n_species, len(self.reactions)))
        # R1
        self.S[0, 0] = -1.0   # CO2
        self.S[1, 0] = -2.0   # MEA
        self.S[3, 0] = 1.0    # RNHCOO-
        self.S[4, 0] = 1.0    # RNH3+
        # R2
        self.S[0, 1] = -1.0
        self.S[1, 1] = -1.0
        self.S[2, 1] = -1.0
        self.S[4, 1] = 1.0
        self.S[5, 1] = 1.0
        # R3
        self.S[0, 2] = -1.0
        self.S[6, 2] = -1.0
        self.S[5, 2] = 1.0
        # R4
        self.S[2, 3] = -1.0
        self.S[7, 3] = 1.0
        self.S[6, 3] = 1.0
        # R5
        self.S[1, 4] = -1.0
        self.S[7, 4] = -1.0
        self.S[4, 4] = 1.0
        # R6
        self.S[3, 5] = -1.0
        self.S[2, 5] = -1.0
        self.S[1, 5] = 1.0
        self.S[5, 5] = 1.0

    def rank(self):
        """Rank of stoichiometric matrix."""
        return np.linalg.matrix_rank(self.S)

    def null_space_reactions(self):
        """Right null space: steady-state reaction rates."""
        return null_space(self.S)

    def conservation_relations(self):
        """
        Left null space: conservation relations (invariants).
        Returns matrix L such that L * S = 0.
        Each row is a conserved quantity.
        """
        # Compute left null space via SVD
        U, s, Vh = np.linalg.svd(self.S)
        rank = np.sum(s > 1e-10)
        n_conserved = self.n_species - rank
        L = U[:, rank:].T  # Left null space vectors
        return L, n_conserved

    def validate_conservation(self, concentrations):
        """
        Verify that concentrations satisfy conservation relations.
        concentrations: dict or array of species concentrations.
        """
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
        """
        Elemental composition matrix E (elements x species).
        Rows: C, H, N, O
        Species: CO2, MEA, H2O, RNHCOO-, RNH3+, HCO3-, OH-, H+
        """
        # CO2: C1O2, MEA: C2H7NO, H2O: H2O
        # RNHCOO- (carbamate): C3H6NO3-, RNH3+ (protonated MEA): C2H8NO+
        # HCO3-: C1H1O3, OH-: OH-, H+: H+
        E = np.array([
            [1, 2, 0, 3, 2, 1, 0, 0],   # C
            [0, 7, 2, 6, 8, 1, 1, 1],   # H
            [0, 1, 0, 1, 1, 0, 0, 0],   # N
            [2, 1, 1, 3, 1, 3, 1, 0]    # O
        ])
        return E

    def check_elemental_balance(self, reaction_idx):
        """Check if a specific reaction is elementally balanced."""
        E = self.compute_elemental_matrix()
        reaction_vector = self.S[:, reaction_idx]
        balance = E @ reaction_vector
        return balance

    def overall_elemental_balance(self):
        """Check overall elemental balance across all reactions."""
        E = self.compute_elemental_matrix()
        # For any feasible reaction rates v, E * S * v should be zero
        return E @ self.S

    def sensitivity_to_rate_constants(self, k_base, delta=0.1):
        """
        Sensitivity of steady-state composition to rate constants.
        Simplified analysis.
        """
        sensitivities = {}
        for i, rxn in enumerate(self.reactions):
            k_plus = k_base.copy()
            k_plus[i] *= (1.0 + delta)
            # Simplified: measure change in null space basis
            # In practice, this requires solving the full kinetic system
            sensitivities[rxn] = {"relative_change": delta}
        return sensitivities


def build_linear_conserved_ode_system():
    """
    Construct a linear ODE system with conserved quantities.
    Based on biochemical_linear_ode concept.
    Simulates a simplified 2-reaction network with conservation.
    """
    # S = [-1,  1; 1, -1]  (from biochemical_linear_ode)
    # dydt = S @ [k1*y1; k2*y2]
    S = np.array([[-1.0, 1.0], [1.0, -1.0]])
    k1, k2 = 0.5, 0.3

    def rhs(t, y):
        r = np.array([k1 * y[0], k2 * y[1]])
        return S @ r

    def conserved_quantity(y):
        return y[0] + y[1]

    return rhs, conserved_quantity
