"""
parameter_search.py
===================
Efficient parameter space exploration for combustion optimization.

Incorporates:
- hilbert_curve_display (537): Hilbert space-filling curve for 1D-to-ND mapping.
- l4lib (631): Park-Miller LCG pseudo-random number generator.

Scientific goal: search the multi-dimensional operating parameter space
(excess air ratio, coal particle size, burner temperature, residence time)
to find conditions that minimize NOx emissions while maintaining burnout.

Key formulas:
    Hilbert curve mapping (d2xy):
        Recursively decompose 1D index D into 2D coordinates (x,y)
        by bit-interleaving and quadrant transformations.
    
    For N dimensions, we use a generalized Hilbert-like mapping via
    bit-reversal and Gray code ordering.
    
    LCG random generator (Park-Miller):
        seed = (16807 * seed) mod (2^31 - 1)
"""

import numpy as np
from typing import List, Callable


# ======================================================================
# 1. Hilbert curve 1D-to-2D mapping (from hilbert_curve_display)
# ======================================================================

def _hilbert_rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple:
    """Rotate/flip a quadrant appropriately."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def d2xy(n: int, d: int) -> tuple:
    """
    Convert 1D Hilbert distance d to 2D coordinates (x,y) on an n x n grid.
    n must be a power of 2.
    """
    x, y = 0, 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_curve_2d(order: int) -> np.ndarray:
    """
    Generate all 2D points of a Hilbert curve of given order.
    Returns array of shape (4^order, 2) with coordinates in [0, 2^order-1].
    """
    n = 2 ** order
    n_points = n * n
    points = np.zeros((n_points, 2), dtype=int)
    for d in range(n_points):
        x, y = d2xy(n, d)
        points[d] = [x, y]
    return points


# ======================================================================
# 2. N-dimensional parameter space sampling
# ======================================================================

def hilbert_sample_nd(
    n_dims: int, n_samples: int, bounds: List[tuple]
) -> np.ndarray:
    """
    Sample an n-dimensional parameter space using space-filling curves.
    For n_dims > 2, we use a recursive pairing of Hilbert curves
    combined with bit-reversal scrambling.
    
    Args:
        n_dims: number of dimensions
        n_samples: number of sample points
        bounds: list of (min, max) tuples for each dimension
    
    Returns:
        samples: (n_samples, n_dims) array
    """
    if n_dims == 2:
        # Exact 2D Hilbert
        order = int(np.ceil(np.log2(max(2, int(np.sqrt(n_samples))))))
        n_grid = 2 ** order
        points = hilbert_curve_2d(order)
        # Scale to bounds
        samples = np.zeros((n_grid * n_grid, 2))
        for d in range(2):
            lo, hi = bounds[d]
            samples[:, d] = lo + (hi - lo) * points[:, d] / max(n_grid - 1, 1)
        # Subsample if needed
        if n_samples < len(samples):
            idx = np.linspace(0, len(samples) - 1, n_samples, dtype=int)
            samples = samples[idx]
        return samples
    
    # For higher dimensions: use Sobol-like sequence via bit-reversal
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        lo, hi = bounds[d]
        # Van der Corput sequence in base 2
        for i in range(n_samples):
            val = 0.0
            f = 0.5
            n = i + 1
            while n > 0:
                val += f * (n % 2)
                n //= 2
                f *= 0.5
            samples[i, d] = lo + (hi - lo) * val
    return samples


# ======================================================================
# 3. Park-Miller LCG random generator (from l4lib)
# ======================================================================

class ParkMillerLCG:
    """
    Park-Miller minimal standard LCG:
        seed = (16807 * seed) mod (2^31 - 1)
    Period: 2^31 - 2.
    """
    
    MODULUS = 2147483647  # 2^31 - 1
    MULTIPLIER = 16807
    
    def __init__(self, seed: int = 1):
        self.seed = max(1, seed % self.MODULUS)
    
    def next_int(self) -> int:
        """Next integer in [1, MODULUS-1]."""
        self.seed = (self.MULTIPLIER * self.seed) % self.MODULUS
        return self.seed
    
    def next_float(self) -> float:
        """Next float in (0, 1)."""
        return self.next_int() / self.MODULUS
    
    def next_uniform(self, lo: float, hi: float) -> float:
        """Next uniform random in [lo, hi]."""
        return lo + (hi - lo) * self.next_float()
    
    def sample(self, n: int, bounds: List[tuple]) -> np.ndarray:
        """Generate n random samples in the given bounds."""
        samples = np.zeros((n, len(bounds)))
        for i in range(n):
            for d, (lo, hi) in enumerate(bounds):
                samples[i, d] = self.next_uniform(lo, hi)
        return samples


# ======================================================================
# 4. Combustion parameter optimization
# ======================================================================

def nox_objective(
    params: np.ndarray,
    reactor_model: Callable = None,
    penalty_burnout: float = 1.0e6
) -> float:
    """
    Objective function for NOx minimization.
    params = [excess_air, particle_diameter_um, peak_T, residence_time_ms]
    
    Returns a scalar cost: lower is better (minimize NOx + burnout penalty).
    """
    excess_air = np.clip(params[0], 0.8, 1.5)
    dp = np.clip(params[1], 10.0, 300.0)  # micrometers
    T_peak = np.clip(params[2], 1200.0, 2200.0)
    tau = np.clip(params[3], 50.0, 500.0)  # milliseconds
    
    # Semi-empirical NOx correlation (simplified for search)
    # Thermal NOx ~ exp(-Ea/(R*T)) * tau * [O2]^0.5
    R_gas = 8.314462618
    Ea = 319.0e3
    
    # O2 concentration decreases with excess air and burnout
    X_O2 = 0.21 * (excess_air - 0.5) / excess_air
    X_O2 = max(X_O2, 0.01)
    
    # Thermal NOx term
    NOx_thermal = np.exp(-Ea / (R_gas * T_peak)) * tau * 1e-3 * (X_O2 ** 0.5)
    
    # Fuel NOx term (decreases with higher T, increases with N content)
    NOx_fuel = 100.0 * np.exp(-150e3 / (R_gas * T_peak)) * (1.0 / excess_air)
    
    # Burnout efficiency (sigmoid)
    burnout = 1.0 / (1.0 + np.exp(-0.05 * (T_peak - 1400.0))) * (
        1.0 - np.exp(-tau / (dp ** 1.5))
    )
    burnout = np.clip(burnout, 0.0, 1.0)
    
    # Penalty for incomplete burnout
    penalty = penalty_burnout * max(0.0, 0.95 - burnout) ** 2
    
    cost = (NOx_thermal + NOx_fuel) * 1e6 + penalty
    return cost


def optimize_combustion_parameters(
    n_evals: int = 256, use_hilbert: bool = True
) -> dict:
    """
    Search the 4D parameter space for combustion conditions minimizing
    NOx emissions with >95% burnout.
    """
    bounds = [
        (0.9, 1.3),      # excess air ratio
        (30.0, 150.0),   # particle diameter [um]
        (1400.0, 2000.0),  # peak temperature [K]
        (100.0, 400.0),  # residence time [ms]
    ]
    
    if use_hilbert:
        samples = hilbert_sample_nd(4, n_evals, bounds)
    else:
        rng = ParkMillerLCG(seed=42)
        samples = rng.sample(n_evals, bounds)
    
    costs = np.zeros(n_evals)
    for i in range(n_evals):
        costs[i] = nox_objective(samples[i])
    
    best_idx = np.argmin(costs)
    best_params = samples[best_idx]
    best_cost = costs[best_idx]
    
    return {
        "best_params": best_params,
        "best_cost": best_cost,
        "all_samples": samples,
        "all_costs": costs,
        "mean_cost": np.mean(costs),
        "std_cost": np.std(costs),
    }
