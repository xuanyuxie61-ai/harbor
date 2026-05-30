
import numpy as np
from reaction_mechanism import compute_jacobian_fd, NSPEC
from reaction_kinetics import ReactorODE
from utils import condition_estimate






def random_composition(n: int, k: int, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    if k <= 0 or n <= 0:
        return np.array([n])
    if k > n:
        k = n

    cuts = rng.choice(n - 1, size=k - 1, replace=False)
    cuts = np.sort(cuts)
    parts = np.zeros(k, dtype=int)
    prev = 0
    for i in range(k - 1):
        parts[i] = cuts[i] - prev
        prev = cuts[i]
    parts[-1] = n - prev

    parts = np.maximum(parts, 1)

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
    if rng is None:
        rng = np.random.default_rng(42)
    
    if eigenvalues is None:

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

                if rng.random() < 0.8:
                    J[idx + i, idx + i + 1] = 1.0
        idx += bs
    return J






def analyze_stiffness(Y: np.ndarray, T: float, rho: float) -> dict:

    J = compute_jacobian_fd(Y, T, rho)
    

    try:
        eigvals = np.linalg.eigvals(J)
    except np.linalg.LinAlgError:
        eigvals = np.zeros(NSPEC)
    
    re_eig = np.real(eigvals)
    im_eig = np.imag(eigvals)
    

    neg_re = re_eig[re_eig < 0]
    
    stiffness_ratio = 1.0
    if len(neg_re) > 1:
        lambda_fast = np.max(np.abs(neg_re))
        lambda_slow = np.min(np.abs(neg_re))
        if lambda_slow > 1e-30:
            stiffness_ratio = lambda_fast / lambda_slow
    

    time_scales = np.zeros_like(re_eig)
    for i, lam in enumerate(re_eig):
        if abs(lam) > 1e-30:
            time_scales[i] = 1.0 / abs(lam)
    

    cond = condition_estimate(J)
    
    return {
        "eigenvalues": eigvals,
        "real_parts": re_eig,
        "imag_parts": im_eig,
        "stiffness_ratio": stiffness_ratio,
        "fastest_time_scale": np.min(time_scales[time_scales > 0]) if np.any(time_scales > 0) else np.inf,
        "slowest_time_scale": np.max(time_scales[time_scales > 0]) if np.any(time_scales > 0) else 0.0,
        "condition_number": cond,
        "n_negative_eigenvalues": np.sum(re_eig < 0),
        "spectral_abscissa": np.max(re_eig),
    }


def generate_test_jordan_spectrum(n: int = 16) -> dict:
    rng = np.random.default_rng(123)

    n_blocks = 6
    eigenvalues = -np.array([1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4])
    block_sizes = random_composition(n, n_blocks, rng)
    
    J = jordan_matrix(n, eigenvalues=eigenvalues, block_sizes=block_sizes, rng=rng)
    

    eigvals = np.linalg.eigvals(J)
    

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






def stability_function_backward_euler(z: complex) -> complex:
    denom = 1.0 - z
    if abs(denom) < 1e-300:
        return 0.0
    return 1.0 / denom


def is_a_stable(z: complex) -> bool:
    R = stability_function_backward_euler(z)
    return abs(R) <= 1.0 + 1e-12


def recommend_timestep(stiffness_info: dict, safety: float = 0.1) -> float:
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
