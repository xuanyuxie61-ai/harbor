"""
foreground_model.py
===================
Foreground contamination model for the CMB pipeline,
inspired by input-output (EEIO) economic analysis (1155_EEIO).

In the EEIO framework, the total emission of sector i is
    e_i = sum_j L_{ij} f_j
where L = (I - A)^{-1} is the Leontief inverse and f_j is final demand.

Analogously, foreground contamination in frequency channel nu at pixel p is
    F_nu(p) = sum_c M_{nu,c} S_c(p)
where M is a mixing matrix ( SED x component ) and S_c are the component maps
(synchrotron, free-free, dust, CO, point sources, ...).

This module:
  1. Builds a parametric SED mixing matrix M_nu for 5 foreground components.
  2. Propagates foregrounds through a Leontief-style inverse to get
     the effective "demand" on each CMB multipole.
  3. Runs "carbon-tax"-style recycling scenarios:
     reducing foreground amplitude in one band by a price-like parameter.
"""

from __future__ import annotations
import math
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Foreground components and their SEDs
# ---------------------------------------------------------------------------
COMPONENTS = ["synchrotron", "free_free", "anomalous_dust", "thermal_dust", "point_sources"]

# Reference frequencies and SED parameters
# Synchrotron: power law  T_synch(nu) = A_s * (nu / nu_s)^beta_s
# Free-free:   bremsstrahlung T_ff ~ nu^{-2.1}
# Anomalous dust:  spinning dust, peaked at 20-30 GHz
# Thermal dust:    modified blackbody  T_d ~ nu^{beta_d+2} B_nu(T_d)
# Point sources:   flat spectrum in flux density

NU_REF = 100.0   # GHz
BETA_SYNCH = -3.0
BETA_FF = -2.1
NU_PEAK_AD = 28.0        # GHz
SIGMA_AD = 10.0           # GHz
BETA_DUST = 1.5
T_DUST = 19.6             # K
H_OVER_KB = 4.79924466e-11  # h/k_B in GHz/K  (h/k_B ~ 4.8e-11 K/Hz -> GHz/K)


def planck_bnu(nu_ghz: float, T: float) -> float:
    """Planck function  B_nu(T)  in arbitrary normalisation."""
    x = H_OVER_KB * nu_ghz * 1.0e9 / T
    if x > 500:
        return 0.0
    if x < 1e-8:
        return 2.0 * nu_ghz * nu_ghz * T / (H_OVER_KB * 1e9)
    return nu_ghz ** 3 / (math.exp(x) - 1.0)


def sed_synchrotron(nu_ghz: float) -> float:
    return (nu_ghz / NU_REF) ** BETA_SYNCH


def sed_free_free(nu_ghz: float) -> float:
    return (nu_ghz / NU_REF) ** BETA_FF


def sed_anomalous_dust(nu_ghz: float) -> float:
    """Gaussian peaking at nu_peak with width sigma."""
    return math.exp(-0.5 * ((nu_ghz - NU_PEAK_AD) / SIGMA_AD) ** 2)


def sed_thermal_dust(nu_ghz: float) -> float:
    """Modified blackbody:  nu^{beta+2} B_nu(T_d)."""
    return (nu_ghz / NU_REF) ** (BETA_DUST + 2.0) * planck_bnu(nu_ghz, T_DUST) \
           / max(1e-30, (NU_REF / NU_REF) ** (BETA_DUST + 2.0) * planck_bnu(NU_REF, T_DUST))


def sed_point_sources(nu_ghz: float) -> float:
    return 1.0  # flat in flux density


SED_FUNCTIONS = [sed_synchrotron, sed_free_free, sed_anomalous_dust,
                  sed_thermal_dust, sed_point_sources]


# ---------------------------------------------------------------------------
# Build mixing matrix M_{nu, c}
# ---------------------------------------------------------------------------
def build_mixing_matrix(frequencies_ghz: List[float]) -> List[List[float]]:
    """
    M[nu_idx][c] = SED_c(nu)  for each component c.
    """
    n_nu = len(frequencies_ghz)
    n_c = len(COMPONENTS)
    M = [[0.0] * n_c for _ in range(n_nu)]
    for i, nu in enumerate(frequencies_ghz):
        for c, sed in enumerate(SED_FUNCTIONS):
            M[i][c] = sed(nu)
    return M


# ---------------------------------------------------------------------------
# Leontief-style inverse: propagate foregrounds through the mixing matrix
# ---------------------------------------------------------------------------
def leontief_inverse(A: List[List[float]]) -> List[List[float]]:
    """
    Compute L = (I - A)^{-1} for a small square matrix A.
    Used to invert the foreground mixing.
    """
    n = len(A)
    IminusA = [[(-A[i][j] if i != j else 1.0 - A[i][j]) for j in range(n)] for i in range(n)]
    return invert_matrix(IminusA)


def invert_matrix(M: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan inversion of a square matrix."""
    n = len(M)
    # Augment with identity
    aug = [M[i][:] + [(1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    for k in range(n):
        # Pivot
        max_row = k
        max_val = abs(aug[k][k])
        for i in range(k + 1, n):
            if abs(aug[i][k]) > max_val:
                max_val = abs(aug[i][k])
                max_row = i
        if max_val < 1e-30:
            raise ValueError("Singular matrix in inversion.")
        aug[k], aug[max_row] = aug[max_row], aug[k]
        # Eliminate
        pivot = aug[k][k]
        for j in range(2 * n):
            aug[k][j] /= pivot
        for i in range(n):
            if i != k:
                factor = aug[i][k]
                for j in range(2 * n):
                    aug[i][j] -= factor * aug[k][j]
    return [aug[i][n:] for i in range(n)]


# ---------------------------------------------------------------------------
# Compute foreground contribution to each frequency channel
# ---------------------------------------------------------------------------
def compute_foreground_map(M: List[List[float]],
                             amplitudes: List[float]) -> List[float]:
    """
    F_nu = sum_c M_{nu,c} A_c
    M: (n_nu x n_c), amplitudes: (n_c,)
    Returns: (n_nu,) foreground intensities.
    """
    n_nu = len(M)
    return [sum(M[i][c] * amplitudes[c] for c in range(len(amplitudes))) for i in range(n_nu)]


# ---------------------------------------------------------------------------
# "Carbon tax" recycling scenario
# ---------------------------------------------------------------------------
def recycling_scenario(M: List[List[float]],
                         amplitudes_base: List[float],
                         carbon_price: float,
                         elasticity: float = -0.3,
                         recycling_mechanism: str = "lump_sum") -> Dict[str, float]:
    """
    Analogous to the EEIO carbon-tax recycling scenarios:
      - Reduce foreground amplitude in channel nu by a factor (1 + price * elasticity).
      - Compute the "GDP loss" as reduction in effective CMB signal.
      - Recycle the cost via one of four mechanisms.

    Returns dict with:
      delta_amplitudes : per-component amplitude change
      total_reduction  : total foreground reduction
      gdp_loss         : proxy for CMB information loss
    """
    n_c = len(amplitudes_base)
    delta = [0.0] * n_c
    for c in range(n_c):
        # Component c reduction depends on carbon price and elasticity
        factor = carbon_price * elasticity * (1.0 if c == 3 else 0.5)  # dust most affected
        factor = max(-0.9, min(0.0, factor))   # cap
        delta[c] = amplitudes_base[c] * factor

    new_amplitudes = [amplitudes_base[c] + delta[c] for c in range(n_c)]
    total_reduction = sum(abs(delta[c]) for c in range(n_c))
    gdp_loss = total_reduction * abs(elasticity) * 0.1

    if recycling_mechanism == "lump_sum":
        gdp_loss *= 0.8
    elif recycling_mechanism == "labour_tax":
        gdp_loss *= 0.6
    elif recycling_mechanism == "capital_tax":
        gdp_loss *= 0.7
    elif recycling_mechanism == "green_investment":
        gdp_loss *= 0.5

    return {
        "delta_amplitudes": delta,
        "total_reduction": total_reduction,
        "gdp_loss": gdp_loss,
        "new_amplitudes": new_amplitudes,
    }


# ---------------------------------------------------------------------------
# Full pipeline: foreground model -> mixing -> Leontief propagation
# ---------------------------------------------------------------------------
def foreground_pipeline(frequencies_ghz: List[float],
                          amplitudes_base: List[float],
                          recycling_price: float = 0.0) -> Dict[str, object]:
    """
    Full foreground pipeline:
      1. Build mixing matrix M
      2. Compute baseline foregrounds
      3. Apply recycling scenario
      4. Compute Leontief inverse for back-projection
    """
    M = build_mixing_matrix(frequencies_ghz)
    F_base = compute_foreground_map(M, amplitudes_base)

    # Technical coefficient matrix A ~ outer normalisation
    n_c = len(COMPONENTS)
    n_nu = len(frequencies_ghz)
    # A[i][j] = fraction of component i used by component j
    # Normalise columns of M
    col_sums = [max(1e-30, sum(M[i][c] for i in range(n_nu))) for c in range(n_c)]
    A = [[M[i][j] / col_sums[j] for j in range(n_c)] for i in range(n_c)]

    try:
        L = leontief_inverse(A)
    except ValueError:
        L = [[1.0 if i == j else 0.0 for j in range(n_c)] for i in range(n_c)]

    scenario = recycling_scenario(M, amplitudes_base, recycling_price)
    F_new = compute_foreground_map(M, scenario["new_amplitudes"])

    return {
        "mixing_matrix": M,
        "leontief_inverse": L,
        "foreground_base": F_base,
        "foreground_recycled": F_new,
        "scenario": scenario,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    freqs = [30.0, 70.0, 100.0, 143.0, 217.0, 353.0, 545.0]
    amplitudes = [100.0, 50.0, 30.0, 200.0, 10.0]
    result = foreground_pipeline(freqs, amplitudes, recycling_price=0.1)
    print("Mixing matrix M:")
    for row in result["mixing_matrix"]:
        print("  " + "  ".join(f"{v:8.4f}" for v in row))
    print(f"\nBaseline foregrounds: {result['foreground_base']}")
    print(f"Recycled foregrounds: {result['foreground_recycled']}")
    print(f"Total reduction: {result['scenario']['total_reduction']:.3f}")
