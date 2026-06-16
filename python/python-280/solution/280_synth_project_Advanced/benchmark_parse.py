"""
benchmark_parse.py — Benchmark result parsing, aggregation, and convergence analysis.

Seed reference: 1092_omnibenchmark (result parsing and metric aggregation
from hierarchical directory structures).

Core idea
=========
For verification and validation, we run the damage simulation on a series
of meshes and compare:
  1. Convergence of global quantities (energy, peak load, crack path)
  2. Mesh-independence of the nonlocal damage regularization
  3. Order of accuracy of the FD scheme
  4. Computational cost scaling

Following the omnibenchmark pattern (seed 1092), we organize results
in a hierarchical structure:
    results/
      case_{mesh_size}/
        mesh_info.json
        energy_history.json
        force_displacement.json
        damage_field.npy
        percolation_diagnostics.json

And provide parsers to aggregate metrics across cases:
    - L2 error norm vs reference solution
    - L∞ error norm
    - Energy conservation error
    - Peak load error
    - Crack path deviation

The convergence rate is estimated via linear regression on log-log scale:
    ||e_h|| = C * h^p
    log ||e_h|| = log C + p * log h
    → slope p = convergence order

For the nonlocal damage model, we expect:
    - Mesh-independent results when h < ℓ_c / 3
    - p ≈ fd_order for smooth solutions
    - p ≈ 1-2 for solutions with damage localization
"""

import math
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from config import SimulationConfig


# ===================================================================
# Result path parsing  (seed 1092)
# ===================================================================

def parse_result_path(path: str) -> Dict[str, str]:
    """Parse a benchmark result directory path to extract metadata.

    Expected format:  results/case_{mesh_size}/
    or:  results/{backend}_{repeat}/{generator}_{dataset}/{method}/

    Returns a dict with parsed fields.
    """
    parts = os.path.normpath(path).split(os.sep)
    result = {"raw_path": path, "parts": parts}

    basename = parts[-1] if parts else ""

    # Try to extract mesh size
    if basename.startswith("case_"):
        try:
            result["mesh_size"] = int(basename[5:])
        except ValueError:
            result["mesh_size"] = -1
    else:
        # Try regex-like parsing for numeric content
        digits = "".join(c for c in basename if c.isdigit())
        result["mesh_size"] = int(digits) if digits else -1

    result["case_name"] = basename
    return result


def parse_metric_file(filepath: str) -> Dict[str, Any]:
    """Parse a JSON metric file and return the contents.

    Handles both flat and nested metric structures.
    """
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError):
        return {}


# ===================================================================
# Error norm computation
# ===================================================================

def compute_error_norms(computed: np.ndarray,
                        reference: np.ndarray,
                        dx: float, dy: float) -> Dict[str, float]:
    """Compute error norms between computed and reference solutions.

    L2 norm:   ||e||_2 = sqrt(∫ (u_h - u_ref)² dA)
    L∞ norm:   ||e||_∞ = max |u_h - u_ref|
    L1 norm:   ||e||_1 = ∫ |u_h - u_ref| dA
    Relative:  ||e|| / ||u_ref||

    Also computes the convergence order if multiple mesh sizes are available.
    """
    error = computed - reference
    dA = dx * dy

    l2_error = math.sqrt(float(np.sum(error ** 2) * dA))
    linf_error = float(np.max(np.abs(error)))
    l1_error = float(np.sum(np.abs(error)) * dA)

    # Reference norms
    ref_l2 = math.sqrt(float(np.sum(reference ** 2) * dA))
    ref_linf = float(np.max(np.abs(reference)))

    rel_l2 = l2_error / max(ref_l2, 1.0e-30)
    rel_linf = linf_error / max(ref_linf, 1.0e-30)

    return {
        "L2_error": l2_error,
        "Linf_error": linf_error,
        "L1_error": l1_error,
        "relative_L2": rel_l2,
        "relative_Linf": rel_linf,
        "ref_L2_norm": ref_l2,
        "ref_Linf_norm": ref_linf,
    }


# ===================================================================
# Convergence rate estimation
# ===================================================================

def estimate_convergence_order(mesh_sizes: List[int],
                               errors: List[float]) -> Dict[str, float]:
    """Estimate the convergence order from mesh refinement study.

    Uses least-squares fit on log-log scale:
        log(error) = log(C) + p * log(h)

    where h = 1/N (N = number of elements per side).

    The slope p is the convergence order.
    """
    if len(mesh_sizes) < 2 or len(errors) < 2:
        return {"order": 0.0, "constant": 0.0, "r_squared": 0.0}

    h_values = [1.0 / max(N, 1) for N in mesh_sizes]
    log_h = np.log(h_values)
    log_e = np.log(np.maximum(errors, 1.0e-30))

    # Least squares: log_e = a + p * log_h
    n = len(log_h)
    sum_x = np.sum(log_h)
    sum_y = np.sum(log_e)
    sum_xx = np.sum(log_h ** 2)
    sum_xy = np.sum(log_h * log_e)

    denom = n * sum_xx - sum_x ** 2
    if abs(denom) < 1.0e-30:
        return {"order": 0.0, "constant": 0.0, "r_squared": 0.0}

    p = (n * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - p * sum_x) / n
    C = math.exp(a)

    # R²
    y_mean = np.mean(log_e)
    ss_tot = np.sum((log_e - y_mean) ** 2)
    ss_res = np.sum((log_e - (a + p * log_h)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1.0e-30)

    return {
        "order": float(p),
        "constant": float(C),
        "r_squared": float(r_squared),
        "n_data_points": n,
    }


# ===================================================================
# Benchmark case manager
# ===================================================================

class BenchmarkManager:
    """Manages benchmark simulations across multiple mesh sizes."""

    def __init__(self, base_cfg: SimulationConfig, output_dir: str = "results_280"):
        self.base_cfg = base_cfg
        self.output_dir = output_dir
        self.results: List[Dict] = []

    def create_case_config(self, mesh_size: int) -> SimulationConfig:
        """Create a configuration for a specific mesh size."""
        import copy
        cfg = copy.deepcopy(self.base_cfg)
        cfg.numerical.nx = mesh_size
        cfg.numerical.ny = mesh_size
        cfg.output_dir = os.path.join(self.output_dir, f"case_{mesh_size}")
        return cfg

    def record_result(self, mesh_size: int,
                      errors: Dict[str, float],
                      diagnostics: Dict) -> None:
        """Record results from a single benchmark case."""
        result = {
            "mesh_size": mesh_size,
            "dx": self.base_cfg.numerical.domain_x[1] / max(mesh_size - 1, 1),
            "errors": errors,
            "diagnostics": diagnostics,
        }
        self.results.append(result)

    def compute_convergence(self,
                            error_key: str = "relative_L2") -> Dict:
        """Compute convergence rates across all recorded cases."""
        if len(self.results) < 2:
            return {"order": 0.0, "message": "Insufficient data"}

        # Sort by mesh size
        sorted_results = sorted(self.results, key=lambda r: r["mesh_size"])

        mesh_sizes = [r["mesh_size"] for r in sorted_results]
        errors = [r["errors"].get(error_key, 1.0) for r in sorted_results]

        convergence = estimate_convergence_order(mesh_sizes, errors)
        convergence["error_key"] = error_key
        convergence["mesh_sizes"] = mesh_sizes
        convergence["errors"] = errors

        return convergence

    def generate_report(self) -> str:
        """Generate a text report of benchmark results."""
        lines = []
        lines.append("=" * 70)
        lines.append("BENCHMARK RESULTS — Multi-scale Damage Evolution")
        lines.append("=" * 70)

        if not self.results:
            lines.append("No benchmark results recorded.")
            return "\n".join(lines)

        sorted_results = sorted(self.results, key=lambda r: r["mesh_size"])

        lines.append(f"\n{'Mesh':>6s} {'dx [m]':>12s} {'L2 error':>12s} "
                     f"{'L∞ error':>12s} {'Rel L2':>10s}")
        lines.append("-" * 60)

        for r in sorted_results:
            lines.append(
                f"{r['mesh_size']:6d} {r['dx']:12.6e} "
                f"{r['errors'].get('L2_error', 0):12.6e} "
                f"{r['errors'].get('Linf_error', 0):12.6e} "
                f"{r['errors'].get('relative_L2', 0):10.4e}"
            )

        # Convergence
        conv = self.compute_convergence("relative_L2")
        lines.append(f"\nConvergence order (L2): {conv['order']:.3f}")
        lines.append(f"R² = {conv.get('r_squared', 0):.6f}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ===================================================================
# Analytical reference solution (K-field)
# ===================================================================

def k_field_reference_solution(x: np.ndarray, y: np.ndarray,
                               K_I: float,
                               crack_tip_x: float, crack_tip_y: float,
                               E: float, nu: float) -> Dict[str, np.ndarray]:
    """Compute the Williams asymptotic K-field near a crack tip.

    For mode-I loading, the near-tip displacement field is:
        u_x = K_I/(2μ) * sqrt(r/(2π)) * cos(θ/2) * [κ - 1 + 2sin²(θ/2)]
        u_y = K_I/(2μ) * sqrt(r/(2π)) * sin(θ/2) * [κ + 1 - 2cos²(θ/2)]

    where κ = 3 - 4ν (plane strain), μ = E/(2(1+ν)).

    This serves as a reference solution for verifying the FD scheme
    in the near-tip region.
    """
    mu = E / (2.0 * (1.0 + nu))
    kappa = 3.0 - 4.0 * nu  # plane strain

    r = np.sqrt((x - crack_tip_x) ** 2 + (y - crack_tip_y) ** 2)
    r = np.maximum(r, 1.0e-15)
    theta = np.arctan2(y - crack_tip_y, x - crack_tip_x)

    coeff = K_I / (2.0 * mu) * np.sqrt(r / (2.0 * math.pi))

    u_x_ref = coeff * np.cos(theta / 2.0) * (kappa - 1.0 + 2.0 * np.sin(theta / 2.0) ** 2)
    u_y_ref = coeff * np.sin(theta / 2.0) * (kappa + 1.0 - 2.0 * np.cos(theta / 2.0) ** 2)

    # Stress field
    sqrt_2pi_r = np.sqrt(2.0 * math.pi * r)
    sqrt_2pi_r = np.maximum(sqrt_2pi_r, 1.0e-15)

    sxx_ref = K_I / sqrt_2pi_r * np.cos(theta / 2.0) * (1.0 - np.sin(theta / 2.0) * np.sin(3.0 * theta / 2.0))
    syy_ref = K_I / sqrt_2pi_r * np.cos(theta / 2.0) * (1.0 + np.sin(theta / 2.0) * np.sin(3.0 * theta / 2.0))
    sxy_ref = K_I / sqrt_2pi_r * np.sin(theta / 2.0) * np.cos(theta / 2.0) * np.cos(3.0 * theta / 2.0)

    return {
        "u_x": u_x_ref,
        "u_y": u_y_ref,
        "sxx": sxx_ref,
        "syy": syy_ref,
        "sxy": sxy_ref,
        "r": r,
        "theta": theta,
    }
