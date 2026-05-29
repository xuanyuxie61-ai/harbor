"""
stiffness_analysis.py
=====================
Stiffness diagnosis of the combustion reaction Jacobian using
Jordan canonical form analysis.

Incorporates jordan_matrix (610): random Jordan matrix generation and
structural analysis of eigenvalue condition.

Scientific motivation: combustion reaction systems are notoriously stiff
because the Jacobian eigenvalues span many orders of magnitude.
The stiffness ratio S = max|Re(lambda)| / min|Re(lambda)| >> 1
indicates the need for implicit time integration.

Key formulas:
    Jordan block for eigenvalue lambda:
        J_k(lambda) = lambda * I + N
    where N is nilpotent with 1s on the superdiagonal.
    
    A general Jordan matrix is block-diagonal:
        J = diag(J_1, J_2, ..., J_k)
    
    The condition number of the eigenvector matrix V in A = V J V^{-1}
    determines the sensitivity of the spectrum to perturbations.

    Stiffness ratio:
        S = |Re(lambda_max)| / |Re(lambda_min)|
    where eigenvalues with Re(lambda) < 0 are considered.
"""

import numpy as np
from reaction_mechanism import compute_jacobian_fd, NSPEC
from reaction_kinetics import ReactorODE
from utils import condition_estimate


# ======================================================================
# 1. Random Jordan matrix generation (from jordan_matrix)
# ======================================================================

def random_composition(n: int, k: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Randomly decompose integer n into k positive integers.
    Used to determine sizes of Jordan blocks.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if k <= 0 or n <= 0:
        return np.array([n])
    if k > n:
        k = n
    # Stars and bars: choose k-1 cut points from n-1 positions
    cuts = rng.choice(n - 1, size=k - 1, replace=False)
    cuts = np.sort(cuts)
    parts = np.zeros(k, dtype=int)
    prev = 0
    for i in range(k - 1):
        parts[i] = cuts[i] - prev
        prev = cuts[i]
    parts[-1] = n - prev
    # Ensure all positive
    parts = np.maximum(parts, 1)
    # Adjust if sum exceeds
    while np.sum(parts) > n:
        idx = np.argmax(parts)
        parts[idx] -= 1
    while np.sum(parts) < n:
        idx = np.argmin(parts)
        parts[idx] += 1
    return parts


def jordan_matrix(n: int, eigenvalues: np.ndarray = None,
                  block_sizes: np.ndarray = None,
                  rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate an n x n Jordan matrix with specified eigenvalues and block sizes.
    
    Args:
        n: matrix dimension
        eigenvalues: 1D array of eigenvalues (one per block)
        block_sizes: 1D array of block sizes
        rng: random number generator
    
    Returns:
        J: n x n Jordan matrix
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if eigenvalues is None:
        # Random eigenvalues with a mix of magnitudes (stiffness-like)
        n_blocks = rng.integers(2, min(n, 8))
        eigenvalues = -10.0 ** rng.uniform(-2, 6, size=n_blocks)
        block_sizes = random_composition(n, n_blocks, rng)
    
    if block_sizes is None:
        block_sizes = np.ones(len(eigenvalues), dtype=int)
    
    J = np.zeros((n, n))
    idx = 0
    for lam, bs in zip(eigenvalues, block_sizes):
        bs = int(bs)
        for i in range(bs):
            J[idx + i, idx + i] = lam
            if i < bs - 1:
                # 80% probability of superdiagonal 1 (from original)
                if rng.random() < 0.8:
                    J[idx + i, idx + i + 1] = 1.0
        idx += bs
    return J


# ======================================================================
# 2. Stiffness analysis of combustion Jacobian
# ======================================================================

def analyze_stiffness(Y: np.ndarray, T: float, rho: float) -> dict:
    """
    Compute stiffness metrics for the combustion reaction system at
    a given composition and temperature.
    """
    # Compute Jacobian
    J = compute_jacobian_fd(Y, T, rho)
    
    # HOLE 3: Stiffness analysis from combustion Jacobian.
    # Steps:
    #   1. Compute eigenvalues of J (handle LinAlgError)
    #   2. Separate real and imaginary parts
    #   3. Filter negative real parts (stable modes)
    #   4. Compute stiffness_ratio = |lambda_fast| / |lambda_slow|
    #   5. Compute chemical time scales tau_i = 1 / |lambda_i|
    #   6. Estimate condition number of J
    #   7. Return dict with all stiffness metrics
    # TODO: implement eigenvalue analysis and stiffness metric extraction
    pass  # HOLE 3
    
    return {
        "eigenvalues": None,
        "real_parts": None,
        "imag_parts": None,
        "stiffness_ratio": 1.0,
        "fastest_time_scale": np.inf,
        "slowest_time_scale": 0.0,
        "condition_number": 1.0,
        "n_negative_eigenvalues": 0,
        "spectral_abscissa": 0.0,
    }


def generate_test_jordan_spectrum(n: int = 16) -> dict:
    """
    Generate a synthetic Jordan matrix with spectrum mimicking a stiff
    combustion system and analyze its structural properties.
    """
    rng = np.random.default_rng(123)
    # Eigenvalues spanning 8 orders of magnitude (combustion-like)
    n_blocks = 6
    eigenvalues = -np.array([1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4])
    block_sizes = random_composition(n, n_blocks, rng)
    
    J = jordan_matrix(n, eigenvalues=eigenvalues, block_sizes=block_sizes, rng=rng)
    
    # Perturbation analysis: how sensitive are eigenvalues?
    eigvals = np.linalg.eigvals(J)
    
    # Add small random perturbation
    delta = 1e-10
    J_pert = J + delta * rng.standard_normal((n, n))
    eigvals_pert = np.linalg.eigvals(J_pert)
    
    sensitivity = np.abs(eigvals_pert - eigvals) / delta
    
    return {
        "J": J,
        "eigenvalues": eigvals,
        "block_sizes": block_sizes,
        "max_sensitivity": np.max(sensitivity),
        "mean_sensitivity": np.mean(sensitivity),
    }


# ======================================================================
# 3. Stability region analysis for integration methods
# ======================================================================

def stability_function_backward_euler(z: complex) -> complex:
    """
    Stability function of backward Euler:
        R(z) = 1 / (1 - z)
    """
    denom = 1.0 - z
    if abs(denom) < 1e-300:
        return 0.0
    return 1.0 / denom


def is_a_stable(z: complex) -> bool:
    """
    Check if |R(z)| <= 1 for backward Euler.
    Backward Euler is A-stable: stable for all Re(z) < 0.
    """
    R = stability_function_backward_euler(z)
    return abs(R) <= 1.0 + 1e-12


def recommend_timestep(stiffness_info: dict, safety: float = 0.1) -> float:
    """
    Recommend maximum explicit timestep based on fastest mode:
        dt_max_explicit = safety / |lambda_fast|
    If stiffness ratio > 1e4, recommend implicit method.
    """
    fastest = stiffness_info.get("fastest_time_scale", 1e-6)
    ratio = stiffness_info.get("stiffness_ratio", 1.0)
    
    dt_explicit = safety * fastest
    if ratio > 1e4:
        method = "implicit (backward Euler or BDF2)"
    elif ratio > 1e2:
        method = "semi-implicit or Rosenbrock"
    else:
        method = "explicit RK"
    
    return {
        "dt_max_explicit": dt_explicit,
        "recommended_method": method,
        "stiffness_ratio": ratio,
    }
