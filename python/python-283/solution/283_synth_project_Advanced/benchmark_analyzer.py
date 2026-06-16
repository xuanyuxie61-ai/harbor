"""
benchmark_analyzer.py
=====================
Benchmark result parser and analyzer for the defect-state computation
pipeline. Ported from 1092_omnibenchmark_omnibenchmark_paper_code.

The omnibenchmark project defines a hierarchical result directory layout:
    out-{backend}-{rep}/data/{dataset}/clustering/{method}
and provides utilities to parse TSV performance files and aggregate
results across multiple backends and repetitions.

We adapt this to the perovskite defect-state computation context:
    out-{solver}-{rep}/defect/{material}_{dopant}/{method}
where:
    solver  : the Poisson/drift-diffusion solver used
              (e.g. 'HO-FD6', 'DG-P3', 'Gummel-SRH')
    rep     : repetition index (for statistical robustness)
    material: 'MAPbI3', 'FAPbI3', 'CsPbBr3'
    dopant  : 'V_I', 'Pb_i', 'I_i', 'V_Pb', etc.
    method  : the numerical method used (HO-FD, DG, FEM, ML)

Performance metrics tracked:
    - MAE of predicted vs reference defect transition levels [eV]
    - wall-clock time [s]
    - residual convergence rate
    - memory footprint [MB]
"""

from __future__ import annotations
import csv
import gzip
import json
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Path parser (from 1092 parse_results.py)
# ============================================================================
def parse_result_path(path: Path) -> Dict[str, str]:
    """Parse a result path and extract components.
    Pattern: out-{solver}-{rep}/defect/{material}_{dopant}/{method}
    """
    parts = path.parts
    result: dict = {}
    # Parse out-{solver}-{rep}
    if parts:
        out_match = re.match(r'out-([^-]+)-(\d+)', parts[0])
        if out_match:
            result['solver'] = out_match.group(1)
            result['rep'] = out_match.group(2)
    # Find defect_{material}_{dopant} part
    for part in parts:
        if part.startswith('defect-'):
            # Pattern: defect-{material}_{dopant}
            defect_match = re.match(r'defect-([^_]+)_(.+)', part)
            if defect_match:
                result['material'] = defect_match.group(1)
                result['dopant'] = defect_match.group(2)
            break
    # Method is the last component
    if parts:
        result['method'] = parts[-1]
    result['path'] = str(path)
    return result


def parse_performance_file(perf_file: Path) -> Optional[Dict]:
    """Parse a performance file (TSV format) and return a dict of metrics."""
    if not perf_file.exists():
        return None
    try:
        with open(perf_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                result = {}
                for key, value in row.items():
                    if value:
                        value = value.strip()
                        if key == 'h:m:s':
                            result[key] = value
                        else:
                            try:
                                result[key] = float(value)
                            except ValueError:
                                result[key] = value
                return result
    except Exception:
        return None
    return None


# ============================================================================
# Synthetic benchmark data for perovskite defect solvers
# ============================================================================
def generate_benchmark_data(seed: int = 283) -> List[dict]:
    """Generate a synthetic benchmark dataset for comparing defect-state
    solvers across materials and defect species."""
    rng = np.random.default_rng(seed)
    materials = ['MAPbI3', 'FAPbI3', 'CsPbBr3']
    dopants = ['V_I', 'Pb_i', 'I_i', 'V_Pb']
    solvers = ['HO-FD6', 'DG-P3', 'Gummel-SRH', 'FEM-P2']
    methods = ['direct', 'iterative', 'ML-precond']
    entries = []
    for mat in materials:
        for dop in dopants:
            for solver in solvers:
                for method in methods:
                    for rep in range(3):
                        # Synthetic MAE (eV) with solver/method dependence
                        base_mae = {
                            'HO-FD6': 0.05,
                            'DG-P3': 0.08,
                            'Gummel-SRH': 0.12,
                            'FEM-P2': 0.10,
                        }[solver]
                        method_factor = {
                            'direct': 1.0,
                            'iterative': 1.1,
                            'ML-precond': 0.9,
                        }[method]
                        mat_factor = {
                            'MAPbI3': 1.0,
                            'FAPbI3': 1.2,
                            'CsPbBr3': 0.8,
                        }[mat]
                        mae = base_mae * method_factor * mat_factor
                        mae += rng.normal(0.0, 0.01)
                        mae = max(mae, 0.01)
                        wall_time = (0.5 + rng.exponential(1.0)) \
                            * {'HO-FD6': 1.0, 'DG-P3': 2.0,
                               'Gummel-SRH': 1.5, 'FEM-P2': 1.2}[solver]
                        entries.append({
                            'material': mat,
                            'dopant': dop,
                            'solver': solver,
                            'method': method,
                            'rep': rep,
                            'mae_eV': round(mae, 4),
                            'wall_time_s': round(wall_time, 3),
                            'residual_final': round(1e-6 * (1 + rng.random()),
                                                    8),
                        })
    return entries


# ============================================================================
# Aggregation
# ============================================================================
def aggregate_benchmarks(entries: List[dict],
                         groupby: str = 'solver') -> dict:
    """Aggregate benchmark entries by a grouping key.
    Returns dict mapping group -> {mean, std, min, max, count} of MAE."""
    groups: dict = {}
    for e in entries:
        g = e.get(groupby, 'unknown')
        groups.setdefault(g, []).append(e['mae_eV'])
    agg = {}
    for g, vals in groups.items():
        arr = np.array(vals)
        agg[g] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'count': len(vals),
        }
    return agg


def rank_solvers(agg: dict) -> List[Tuple[str, float, float]]:
    """Rank solvers by mean MAE (ascending). Returns list of (solver, mean, std)."""
    ranking = [(g, v['mean'], v['std']) for g, v in agg.items()]
    ranking.sort(key=lambda x: x[1])
    return ranking


def best_solver_per_material(entries: List[dict]) -> dict:
    """Find the best solver (by mean MAE) for each material."""
    mat_solver: dict = {}
    for e in entries:
        key = (e['material'], e['solver'])
        mat_solver.setdefault(key, []).append(e['mae_eV'])
    best: dict = {}
    for (mat, solver), vals in mat_solver.items():
        mean = float(np.mean(vals))
        if mat not in best or mean < best[mat][1]:
            best[mat] = (solver, mean)
    return best


# ============================================================================
# Statistical tests
# ============================================================================
def paired_t_test(vals_a: NDArray, vals_b: NDArray) -> dict:
    """Two-sided paired t-test for H0: mean(vals_a - vals_b) = 0.
    Returns t-statistic and approximate p-value."""
    n = len(vals_a)
    if n < 2:
        return {'t_stat': 0.0, 'p_value': 1.0}
    d = vals_a - vals_b
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)
    if d_std < 1e-30:
        return {'t_stat': 0.0, 'p_value': 1.0}
    t_stat = d_mean / (d_std / math.sqrt(n))
    # Approximate p-value using Student's t with n-1 df
    # Use a normal approximation for large n
    df = n - 1
    x = df / (df + t_stat ** 2)
    # Incomplete beta approximation (simplified):
    # For reporting purposes we give a rough p-value.
    from math import erf
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / math.sqrt(2.0))))
    return {'t_stat': float(t_stat), 'p_value': float(p_value)}


# ============================================================================
# Driver
# ============================================================================
def run_benchmark_analysis() -> dict:
    """Run the full benchmark analysis pipeline."""
    entries = generate_benchmark_data()
    agg_solver = aggregate_benchmarks(entries, groupby='solver')
    agg_material = aggregate_benchmarks(entries, groupby='material')
    agg_method = aggregate_benchmarks(entries, groupby='method')
    ranking = rank_solvers(agg_solver)
    best = best_solver_per_material(entries)
    # Pairwise t-test between top two solvers
    if len(ranking) >= 2:
        s1, s2 = ranking[0][0], ranking[1][0]
        v1 = np.array([e['mae_eV'] for e in entries if e['solver'] == s1])
        v2 = np.array([e['mae_eV'] for e in entries if e['solver'] == s2])
        # Pad to equal length
        n_min = min(len(v1), len(v2))
        ttest = paired_t_test(v1[:n_min], v2[:n_min])
    else:
        ttest = {'t_stat': 0.0, 'p_value': 1.0}
    return {
        "n_entries": len(entries),
        "agg_by_solver": agg_solver,
        "agg_by_material": agg_material,
        "agg_by_method": agg_method,
        "solver_ranking": ranking,
        "best_per_material": best,
        "ttest_top_two": ttest,
    }
