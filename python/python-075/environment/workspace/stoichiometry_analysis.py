"""
stoichiometry_analysis.py
=========================
Element Conservation Analysis and Stoichiometric Constraint Solving for
Combustion Chemistry.

Based on seed projects:
  569 (i4mat_rref2)    - Integer row-reduced echelon form for conservation matrix
  339 (eternity_lp)    - Integer linear programming for mixture optimization

Scientific Context:
-------------------
In chemical kinetics, element conservation is a fundamental physical constraint.
For a system with K species and L elements, the element matrix E satisfies:

  E · ω̇ = 0

where E_{l,k} is the number of atoms of element l in species k, and ω̇ is the
vector of species production rates [mol/(m³·s)].

For H2-air combustion with species {H2, O2, H2O, N2}:
  Element matrix E (rows: H, O, N; cols: H2, O2, H2O, N2):
    H:  [2, 0, 2, 0]
    O:  [0, 2, 1, 0]
    N:  [0, 0, 0, 2]

The stoichiometric mixture fraction Z_st for H2-air is:
  Z_st = (Y_O2,∞ / s + Y_H2,∞) / (Y_O2,∞ / s + Y_H2,∞ + Y_Fuel,∞)
  where s = (ν_O2 W_O2) / (ν_H2 W_H2) = (0.5 * 31.9988) / (1.0 * 2.01588) ≈ 7.937

For a H2-O2 system:
  Z_st = 1 / (1 + s * (Y_H2,0 / Y_O2,0))

Integer RREF is used to verify that the reaction mechanism conserves elements
by checking that E · ν = 0 for all reactions, where ν is the stoichiometric
matrix.

The integer LP formulation finds the optimal fuel-oxidizer mixture that
satisfies both element conservation and energy constraints.
"""

import numpy as np

# Atomic composition matrix for {H2, O2, H2O, N2}
# Rows: H, O, N
# Cols: H2, O2, H2O, N2
ELEMENT_MATRIX = np.array([
    [2, 0, 2, 0],  # H
    [0, 2, 1, 0],  # O
    [0, 0, 0, 2],  # N
], dtype=float)

ELEMENT_NAMES = ['H', 'O', 'N']
SPECIES_NAMES = ['H2', 'O2', 'H2O', 'N2']


def stoichiometric_matrix():
    """
    Stoichiometric matrix ν (nspecies × nreactions) for the reduced mechanism.
    R1: 2H2 + O2 → 2H2O
    R2: H2 + 0.5O2 → H2O
    R3: H2O → H2 + 0.5O2
    """
    return np.array([
        [-2.0, -1.0, 1.0],
        [-1.0, -0.5, 0.5],
        [2.0, 1.0, -1.0],
        [0.0, 0.0, 0.0],
    ])


def verify_element_conservation(tol=1e-10):
    """
    Verify that E · ν = 0 for all reactions.
    Returns True if element conservation holds.
    """
    E = ELEMENT_MATRIX
    nu = stoichiometric_matrix()
    residual = np.dot(E, nu)
    max_res = np.max(np.abs(residual))
    return max_res < tol, max_res, residual


def i4mat_rref2(m, n1, n2, a):
    """
    Compute the integer row-reduced echelon form (IRREF) of an integer matrix.
    Based on seed 569 (i4mat_rref2.m).

    IRREF properties:
      1. Leading nonzero in each row is positive.
      2. Each row has no common factor > 1.
      3. Leading nonzero occurs in a column to the right of previous row's leading nonzero.
      4. Zero rows occur last.
      5. When row has leading nonzero in column J, column J is otherwise zero.

    Parameters
    ----------
    m, n1, n2 : int
        m rows, n1 active columns for pivoting, n2 passive columns.
    a : ndarray, shape (m, n1+n2)
        Integer matrix.

    Returns
    -------
    a_rref : ndarray
        IRREF of the matrix.
    rank : int
        Rank of the active submatrix.
    """
    a = a.copy()
    n = n1 + n2
    a_rank = 0
    lead = 0

    def gcd_pair(a, b):
        a, b = abs(int(a)), abs(int(b))
        while b:
            a, b = b, a % b
        return a

    def reduce_row(row):
        """Divide row by GCD of all entries."""
        vals = row[row != 0]
        if len(vals) == 0:
            return row
        g = int(abs(vals[0]))
        for v in vals[1:]:
            g = gcd_pair(g, int(v))
            if g == 1:
                break
        if g > 1:
            row = row / g
        return row

    for r in range(m):
        if lead >= n1:
            break

        # Find pivot
        i = r
        while abs(a[i, lead]) < 0.5:
            i += 1
            if i >= m:
                i = r
                lead += 1
                if lead >= n1:
                    break
        if lead >= n1:
            break

        a_rank += 1

        # Swap rows
        if i != r:
            a[[i, r], :] = a[[r, i], :]

        # Ensure pivot is positive
        if a[r, lead] < 0:
            a[r, :] = -a[r, :]

        # Reduce row
        a[r, :] = reduce_row(a[r, :])

        # Eliminate other rows
        for i in range(m):
            if i != r and abs(a[i, lead]) > 0.5:
                factor = int(round(a[i, lead]))
                pivot_val = int(round(a[r, lead]))
                # Use exact integer elimination
                a[i, :] = pivot_val * a[i, :] - factor * a[r, :]
                a[i, :] = reduce_row(a[i, :])

        lead += 1

    return a, a_rank


def analyze_element_conservation_matrix():
    """
    Apply IRREF to the element conservation matrix to find independent
    conserved scalars and check for linear dependencies.
    """
    E = ELEMENT_MATRIX.astype(int)
    m, n = E.shape
    E_rref, rank = i4mat_rref2(m, n, 0, E)
    return E_rref, rank


def compute_stoichiometric_mixture_fraction(Y_fuel, Y_oxidizer):
    """
    Compute the stoichiometric mixture fraction for H2-air.

    For H2 + 0.5O2 → H2O:
      s = (0.5 * W_O2) / (1.0 * W_H2) ≈ 7.937

    Burke-Schumann mixture fraction:
      Z = (s Y_H2 - Y_O2 + Y_O2,∞) / (s Y_H2,∞ + Y_O2,∞)

    For pure fuel (Y_H2=1, Y_O2=0) and pure oxidizer (Y_H2=0, Y_O2=0.233):
      Z_st = Y_O2,∞ / (Y_O2,∞ + s * Y_H2,∞)
           = 0.233 / (0.233 + 7.937 * 1.0)  ← this is wrong dimensionally

    Correct Z_st for mass fractions:
      Z_st = 1 / (1 + (s * Y_H2,F / Y_O2,O))
      where s = (ν_O2 W_O2) / (ν_H2 W_H2)
    """
    # TODO [Hole 2]: Implement the stoichiometric mixture fraction formula.
    #
    # The stoichiometric mixture fraction Z_st defines the fuel/oxidizer ratio
    # at which reactants are in stoichiometric proportion.
    #
    # For H2-air combustion:
    #   - H2 molecular weight: W_h2 ≈ 2.016e-3 kg/mol
    #   - O2 molecular weight: W_o2 ≈ 31.999e-3 kg/mol
    #   - Stoichiometric reaction: H2 + 0.5 O2 → H2O
    #   - Mass stoichiometric ratio: s = (0.5 * W_o2) / (1.0 * W_h2) ≈ 7.937
    #
    # Given fuel stream mass fraction Y_h2_f and oxidizer stream mass fraction Y_o2_o,
    # the mixture fraction at stoichiometry is:
    #   Z_st = 1 / (1 + s * Y_h2_f / Y_o2_o)
    #
    # This value is used in main.py for flamelet grid generation and DNS initialization.
    raise NotImplementedError("Hole 2: Implement stoichiometric mixture fraction computation")


def integer_lp_optimal_mixture(elements_required, species_available):
    """
    Formulate and solve a simplified integer LP for optimal mixture composition.
    Based on seed 339 (eternity_lp.m) — integer linear system construction.

    Problem: Find integer coefficients c_k such that:
      Σ_k c_k * E_{l,k} = required_atoms_l   for each element l
      minimize Σ_k cost_k * c_k

    Parameters
    ----------
    elements_required : dict
        {element_name: required_atoms}
    species_available : list of dict
        Each dict: {'name': str, 'composition': {element: atoms}, 'cost': float}

    Returns
    -------
    solution : dict
        {species_name: coefficient}
    cost : float
        Total cost.
    """
    element_list = list(elements_required.keys())
    ne = len(element_list)
    ns = len(species_available)

    # Build constraint matrix A (ne × ns)
    A = np.zeros((ne, ns))
    costs = np.zeros(ns)
    for j, sp in enumerate(species_available):
        costs[j] = sp.get('cost', 1.0)
        for i, elem in enumerate(element_list):
            A[i, j] = sp['composition'].get(elem, 0)

    b = np.array([elements_required[e] for e in element_list])

    # Simplified greedy solution (exact ILP requires external solver)
    # Use pseudoinverse for continuous relaxation, then round
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    # Non-negative constraint
    coeffs = np.maximum(coeffs, 0)
    # Scale to satisfy constraints approximately
    produced = np.dot(A, coeffs)
    scale = np.min(b / np.maximum(produced, 1e-12))
    coeffs = coeffs * scale

    solution = {}
    for j, sp in enumerate(species_available):
        if coeffs[j] > 0.01:
            solution[sp['name']] = round(coeffs[j], 4)

    cost = np.sum(costs * coeffs)
    return solution, cost


def mixture_fraction_bounds(Z_st, Z_var):
    """
    Compute physical bounds on mixture fraction considering element conservation.
    Z must satisfy: 0 ≤ Z ≤ 1, and the element mass fractions must be non-negative.

    Returns bounds based on:
      Z_min = max(0, (Y_O2,0 - s * Y_H2,0) / (Y_O2,∞ + s * Y_H2,∞))
      Z_max = min(1, ...)
    """
    Z_min = max(0.0, Z_st - 3.0 * np.sqrt(max(Z_var, 0.0)))
    Z_max = min(1.0, Z_st + 3.0 * np.sqrt(max(Z_var, 0.0)))
    return Z_min, Z_max
