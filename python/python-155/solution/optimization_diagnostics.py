"""
Optimization and convergence diagnostics for quantum walk parameters.
Incorporates: nonlin_newton (Newton-Raphson optimization),
              levels (random contour sampling),
              full_deck_simulation (coupon collector / cover time),
              steinerberger (pathological integration tests).
"""
import numpy as np
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Newton-Raphson method (from nonlin_newton)
# ---------------------------------------------------------------------------
def newton_raphson(f: Callable[[float], float],
                   df: Callable[[float], float],
                   x0: float,
                   tol: float = 1e-10,
                   max_iter: int = 50,
                   divergence_factor: float = 100.0) -> Tuple[float, bool, int]:
    """Find root of f(x) = 0 using Newton's method.
    Returns (root, converged, iterations).
    Includes robust boundary handling:
      - Stops if |f'(x)| < 1e-8
      - Stops if |f(x)| exceeds divergence_factor * |f(x0)|
    """
    x = x0
    f0 = f(x0)
    for it in range(max_iter):
        fx = f(x)
        if abs(fx) <= tol:
            return x, True, it
        dfx = df(x)
        if abs(dfx) < 1e-8:
            return x, False, it
        if abs(fx) > divergence_factor * max(abs(f0), 1.0):
            return x, False, it
        x_new = x - fx / dfx
        x = x_new
    return x, False, max_iter


def find_optimal_coin_angle(success_prob_func: Callable[[float], float],
                            angle0: float = 0.5,
                            bracket: Tuple[float, float] = (0.0, np.pi / 2.0)) -> dict:
    """Find optimal coin angle by maximizing success probability.
    Uses golden-section search combined with local Newton refinement.
    """
    a, b = bracket
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    c = a + resphi * (b - a)
    d = b - resphi * (b - a)

    for _ in range(30):
        if success_prob_func(c) < success_prob_func(d):
            b = d
            d = c
            c = a + resphi * (b - a)
        else:
            a = c
            c = d
            d = b - resphi * (b - a)
        if abs(b - a) < 1e-8:
            break

    opt_angle = (a + b) / 2.0
    prob = success_prob_func(opt_angle)

    return {
        "optimal_angle": float(opt_angle),
        "success_probability": float(prob),
        "search_interval": bracket
    }


def find_critical_gamma(gap_func: Callable[[float], float],
                        gamma0: float = 1.0) -> dict:
    """Find critical gamma that maximizes spectral gap.
    Uses Newton's method on the derivative of the gap function.
    """
    def obj(g):
        return -gap_func(g)

    def obj_deriv(g, h=1e-6):
        return (-gap_func(g + h) + gap_func(g - h)) / (2.0 * h)

    # Golden section first
    a, b = 0.01, 5.0
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    for _ in range(25):
        if obj(c) < obj(d):
            b = d
        else:
            a = c
        c = a + resphi * (b - a)
        d = b - resphi * (b - a)

    gamma_opt = (a + b) / 2.0
    return {
        "critical_gamma": float(gamma_opt),
        "spectral_gap": float(gap_func(gamma_opt))
    }


# ---------------------------------------------------------------------------
# Random level sampling for landscape analysis (from levels)
# ---------------------------------------------------------------------------
def random_level_sample(values: np.ndarray, num_levels: int,
                        seed: Optional[int] = None) -> np.ndarray:
    """Sample random contour levels from a probability landscape."""
    if seed is not None:
        np.random.seed(seed)
    if num_levels <= 0 or values.size == 0:
        return np.array([])
    indices = np.random.choice(values.size,
                               size=min(num_levels, values.size),
                               replace=False)
    return np.sort(values.flat[indices])


def analyze_search_landscape(prob_evolution: np.ndarray,
                             num_levels: int = 20) -> dict:
    """Analyze the probability landscape using random level sampling."""
    levels = random_level_sample(prob_evolution, num_levels)
    stats = {
        "levels": levels.tolist(),
        "min": float(np.min(prob_evolution)),
        "max": float(np.max(prob_evolution)),
        "mean": float(np.mean(prob_evolution)),
        "std": float(np.std(prob_evolution)),
        "median": float(np.median(prob_evolution)),
        "level_spacing_mean": float(np.mean(np.diff(levels))) if len(levels) > 1 else 0.0
    }
    return stats


def detect_local_maxima(evolution: np.ndarray) -> List[Tuple[int, float]]:
    """Detect local maxima in an evolution array."""
    maxima = []
    for i in range(1, len(evolution) - 1):
        if evolution[i] > evolution[i - 1] and evolution[i] > evolution[i + 1]:
            maxima.append((i, float(evolution[i])))
    return maxima


# ---------------------------------------------------------------------------
# Coupon collector / cover time analysis (from full_deck_simulation)
# ---------------------------------------------------------------------------
def coupon_collector_simulation(n_items: int, num_trials: int = 1000,
                                seed: Optional[int] = None) -> dict:
    """Simulate the coupon collector problem (cover time).
    Compares classical random walk cover time with quantum walk expectations.
    """
    if seed is not None:
        np.random.seed(seed)
    draws = np.zeros(num_trials, dtype=int)
    for t in range(num_trials):
        collected = np.zeros(n_items, dtype=bool)
        count = 0
        while not np.all(collected):
            idx = np.random.randint(0, n_items)
            collected[idx] = True
            count += 1
        draws[t] = count

    # Theoretical exact values
    H_n = np.sum(1.0 / np.arange(1, n_items + 1))
    expected_exact = n_items * H_n
    # Variance exact
    var_exact = 0.0
    for i in range(2, n_items + 1):
        var_exact += (i - 1.0) / ((n_items - i + 1.0) ** 2)
    var_exact *= n_items

    return {
        "n_items": n_items,
        "num_trials": num_trials,
        "empirical_mean": float(np.mean(draws)),
        "empirical_std": float(np.std(draws)),
        "empirical_min": int(np.min(draws)),
        "empirical_max": int(np.max(draws)),
        "theoretical_expected": float(expected_exact),
        "theoretical_variance": float(var_exact),
        "theoretical_std": float(np.sqrt(var_exact))
    }


def compare_classical_quantum_cover_time(n: int, graph_degree: float = 4.0) -> dict:
    """Theoretical comparison of classical vs quantum cover times.
    Classical cover time on regular graph: O(n log n)
    Quantum search time: O(sqrt(n / degree))
    """
    classical = n * np.log(max(n, 2))
    quantum = np.sqrt(n / graph_degree)
    return {
        "n_vertices": n,
        "classical_cover_time": float(classical),
        "quantum_search_time": float(quantum),
        "speedup_factor": float(classical / quantum) if quantum > 0 else 1.0
    }


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------
def convergence_rate_analysis(errors: np.ndarray) -> dict:
    """Analyze convergence rate from error sequence.
    Fits log(error) vs iteration to estimate convergence order.
    """
    if len(errors) < 3:
        return {"order": None, "rate": 0.0}
    # Exclude zero errors
    valid = errors > 1e-15
    if np.sum(valid) < 3:
        return {"order": None, "rate": 0.0}
    iters = np.arange(len(errors))[valid]
    log_err = np.log(errors[valid])
    # Linear fit: log(e_k) = log(C) + p * log(k)
    A = np.vstack([np.log(iters + 1), np.ones(len(iters))]).T
    p, logC = np.linalg.lstsq(A, log_err, rcond=None)[0]
    return {
        "estimated_order": float(-p),
        "prefactor": float(np.exp(logC)),
        "final_error": float(errors[-1])
    }


def residual_norm_history(A_mult: Callable, b: np.ndarray,
                          x_history: List[np.ndarray]) -> np.ndarray:
    """Compute residual norms for a sequence of iterates."""
    residuals = np.array([np.linalg.norm(b - A_mult(x)) for x in x_history])
    return residuals
