"""
percolation_failure.py — Percolation-based failure detection for damage fields.

Seed reference: 865_percolation_simulation (2D site percolation,
connected-component labeling via flood fill).

Core idea
=========
Material failure is detected when damage sites percolate — i.e., when a
connected cluster of highly damaged elements spans the specimen from one
loaded boundary to the other. This is a topological phase transition
analogous to the percolation threshold p_c in statistical mechanics.

For a 2D lattice with occupation probability p (site percolation):
  - Square lattice, von Neumann (4-connectivity):  p_c ≈ 0.5927
  - Square lattice, Moore (8-connectivity):        p_c ≈ 0.4073

As damage evolves, the "occupied" sites are those with D > D_threshold.
When the largest cluster spans the domain, macroscopic failure occurs.

We compute:
  - Connected components via stack-based flood fill
  - Cluster size distribution
  - Spanning probability
  - Percolation order parameter P∞ (fraction in largest cluster)
  - Correlation length ξ (characteristic cluster size)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SimulationConfig


# ===================================================================
# Connected component labeling  (seed 865_percolation_simulation)
# ===================================================================

def label_connected_components(occupancy: np.ndarray,
                               connectivity: str = "von_neumann"
                               ) -> Tuple[np.ndarray, int]:
    """Label connected components using stack-based flood fill.

    occupancy: boolean 2D array (True = damaged site).
    connectivity: "von_neumann" (4-conn) or "moore" (8-conn).

    Returns:
      labels: integer array, 0 = unoccupied, 1..n_clusters = cluster IDs
      n_clusters: number of distinct clusters

    Non-recursive (explicit stack) to avoid Python recursion limits.
    """
    ny, nx = occupancy.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    cluster_id = 0

    # Neighbor offsets
    if connectivity == "moore":
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:  # von Neumann
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for j in range(ny):
        for i in range(nx):
            if occupancy[j, i] and labels[j, i] == 0:
                # New cluster: flood fill
                cluster_id += 1
                labels[j, i] = cluster_id
                stack = [(j, i)]

                while stack:
                    cj, ci = stack.pop()
                    for dj, di in offsets:
                        nj, ni = cj + dj, ci + di
                        if (0 <= nj < ny and 0 <= ni < nx
                                and occupancy[nj, ni]
                                and labels[nj, ni] == 0):
                            labels[nj, ni] = cluster_id
                            stack.append((nj, ni))

    return labels, cluster_id


# ===================================================================
# Cluster analysis
# ===================================================================

def analyze_cluster_sizes(labels: np.ndarray,
                          n_clusters: int,
                          occupancy: np.ndarray
                          ) -> Dict:
    """Compute cluster size statistics.

    Returns:
      cluster_sizes: array of sizes for each cluster
      max_cluster_size: size of the largest cluster
      mean_cluster_size: average cluster size
      total_occupied: total number of occupied sites
      percolation_strength: P∞ = max_cluster_size / total_occupied
    """
    if n_clusters == 0:
        return {
            "cluster_sizes": np.array([]),
            "max_cluster_size": 0,
            "mean_cluster_size": 0.0,
            "total_occupied": 0,
            "percolation_strength": 0.0,
            "n_clusters": 0,
            "size_distribution": {},
            "susceptibility": 0.0,
        }

    sizes = []
    for cid in range(1, n_clusters + 1):
        s = int(np.sum(labels == cid))
        sizes.append(s)

    sizes_arr = np.array(sizes)
    total = int(np.sum(occupancy))
    max_size = int(sizes_arr.max())
    mean_size = float(sizes_arr.mean())

    # Percolation strength (order parameter)
    P_inf = max_size / max(total, 1)

    # Size distribution (histogram)
    unique_sizes, counts = np.unique(sizes_arr, return_counts=True)
    size_dist = {int(s): int(c) for s, c in zip(unique_sizes, counts)}

    # Second moment (susceptibility): χ = Σ s² n(s) / Σ s n(s)
    if total > 0 and mean_size > 0:
        chi = float(np.sum(sizes_arr ** 2)) / max(total, 1)
    else:
        chi = 0.0

    return {
        "cluster_sizes": sizes_arr,
        "max_cluster_size": max_size,
        "mean_cluster_size": mean_size,
        "total_occupied": total,
        "percolation_strength": float(P_inf),
        "n_clusters": n_clusters,
        "size_distribution": size_dist,
        "susceptibility": chi,
    }


# ===================================================================
# Spanning cluster detection
# ===================================================================

def check_spanning_cluster(labels: np.ndarray,
                           n_clusters: int,
                           ny: int, nx: int) -> Dict[str, bool]:
    """Check if any cluster spans the domain.

    A cluster spans:
      - horizontally: touches both left (i=0) and right (i=nx-1) boundaries
      - vertically: touches both bottom (j=0) and top (j=ny-1) boundaries
      - diagonally: touches both diagonal pairs

    Returns dict of spanning flags.
    """
    spans_horizontal = False
    spans_vertical = False

    for cid in range(1, n_clusters + 1):
        mask = (labels == cid)
        touches_left = np.any(mask[:, 0])
        touches_right = np.any(mask[:, nx - 1])
        touches_bottom = np.any(mask[0, :])
        touches_top = np.any(mask[ny - 1, :])

        if touches_left and touches_right:
            spans_horizontal = True
        if touches_bottom and touches_top:
            spans_vertical = True

    return {
        "spans_horizontal": spans_horizontal,
        "spans_vertical": spans_vertical,
        "spans_any": spans_horizontal or spans_vertical,
    }


# ===================================================================
# Correlation length estimation
# ===================================================================

def estimate_correlation_length(occupancy: np.ndarray,
                                dx: float, dy: float) -> float:
    """Estimate the correlation length ξ from the pair correlation function.

    The pair correlation function g(r) measures the probability of finding
    two occupied sites separated by distance r, relative to random:
        g(r) = P(both sites occupied at distance r) / p²

    For percolation near the critical point:
        g(r) ~ r^{-(d-2+η)} exp(-r/ξ)

    The correlation length ξ diverges as:
        ξ ~ |p - p_c|^{-ν}

    with ν = 4/3 for 2D percolation.

    We estimate ξ from the second moment of the cluster size distribution:
        ξ² = Σ_s s * R²(s) / Σ_s s

    where R²(s) is the mean squared radius of gyration of s-clusters.

    Simplified: ξ ≈ sqrt(mean_cluster_size) * h
    """
    if not np.any(occupancy):
        return 0.0

    h = min(dx, dy)
    labels, n_clusters = label_connected_components(occupancy)
    cluster_info = analyze_cluster_sizes(labels, n_clusters, occupancy)

    if cluster_info["total_occupied"] == 0:
        return 0.0

    # Estimate from mean cluster radius
    total_r2 = 0.0
    total_mass = 0

    for cid in range(1, n_clusters + 1):
        mask = (labels == cid)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx = np.mean(xs)
        cy = np.mean(ys)
        r2_mean = np.mean((xs - cx) ** 2 * dx ** 2 + (ys - cy) ** 2 * dy ** 2)
        s = int(mask.sum())
        total_r2 += s * r2_mean
        total_mass += s

    if total_mass > 0:
        xi = math.sqrt(total_r2 / total_mass)
    else:
        xi = 0.0

    return float(xi)


# ===================================================================
# Percolation analysis of damage field
# ===================================================================

def percolation_analysis(damage: np.ndarray,
                         threshold: float,
                         cfg: SimulationConfig) -> Dict:
    """Complete percolation analysis of the damage field.

    1. Create occupancy map: D > threshold → occupied
    2. Label connected components
    3. Compute cluster statistics
    4. Check for spanning clusters (failure criterion)
    5. Estimate correlation length

    Returns comprehensive diagnostics.
    """
    dx = cfg.dx()
    dy = cfg.dy()
    ny, nx = damage.shape
    connectivity = cfg.numerical.connectivity_mode

    # Occupancy
    occupancy = damage > threshold
    p_occupied = float(occupancy.sum()) / occupancy.size

    # Connected components
    labels, n_clusters = label_connected_components(occupancy, connectivity)

    # Cluster statistics
    cluster_info = analyze_cluster_sizes(labels, n_clusters, occupancy)

    # Spanning check
    spanning = check_spanning_cluster(labels, n_clusters, ny, nx)

    # Correlation length
    xi = estimate_correlation_length(occupancy, dx, dy)

    # Failure criterion: spanning cluster OR percolation strength > 0.5
    failure = spanning["spans_any"] or cluster_info["percolation_strength"] > 0.5

    # Distance from theoretical percolation threshold
    # (for infinite lattice; finite-size shifted)
    if connectivity == "von_neumann":
        p_c_infinite = 0.5927
    else:
        p_c_infinite = 0.4073

    # Finite-size correction: p_c(L) ≈ p_c(∞) + a * L^{-1/ν}
    L = max(nx, ny)
    nu_2d = 4.0 / 3.0  # 2D percolation exponent
    L_char = L * min(dx, dy) / max(cfg.numerical.domain_x[1] - cfg.numerical.domain_x[0], 1.0e-15)
    p_c_finite = p_c_infinite + 0.5 * L_char ** (-1.0 / nu_2d)

    return {
        "occupancy_fraction": p_occupied,
        "n_clusters": n_clusters,
        "max_cluster_size": cluster_info["max_cluster_size"],
        "mean_cluster_size": cluster_info["mean_cluster_size"],
        "percolation_strength": cluster_info["percolation_strength"],
        "susceptibility": cluster_info["susceptibility"],
        "spans_horizontal": spanning["spans_horizontal"],
        "spans_vertical": spanning["spans_vertical"],
        "spans_any": spanning["spans_any"],
        "correlation_length": xi,
        "p_c_infinite": p_c_infinite,
        "p_c_finite_estimate": p_c_finite,
        "distance_to_threshold": p_occupied - p_c_finite,
        "failure_detected": failure,
    }
