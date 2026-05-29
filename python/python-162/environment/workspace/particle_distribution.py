"""
particle_distribution.py
================================================================================
Particle size distribution (PSD) modeling, clustering, and tree-based
enumeration for lithium electrode microstructure.

Injects core algorithms from:
  - 039_asa113         (transfer/swap clustering for particle classification)
  - 1291_treepack      (Prüfer codes, tree enumeration, Catalan numbers)
  - 1267_toms179       (incomplete beta for PSD confidence bounds)

Scientific role:
  Real electrodes contain polydisperse active material particles.
  This module:
    1. Generates representative PSDs (log-normal + beta mixtures).
    2. Clusters particles into size classes using transfer/swap optimization.
    3. Builds tree-structured hierarchical representations for fast lookup.
    4. Computes statistical confidence intervals on effective diffusivity
       using incomplete beta functions.
================================================================================
"""

import numpy as np
from typing import List, Tuple
from quadrature_special import incomplete_beta_ratio, log_gamma


# ==============================================================================
# Particle size distribution generation
# ==============================================================================

def lognormal_psd(n_particles: int, mu_ln: float, sigma_ln: float,
                  r_min: float = 1e-7, r_max: float = 20e-6) -> np.ndarray:
    """Generate particle radii from a log-normal distribution."""
    radii = np.random.lognormal(mu_ln, sigma_ln, size=n_particles)
    radii = np.clip(radii, r_min, r_max)
    return radii


def beta_mixture_psd(n_particles: int, alpha1: float, beta1: float,
                     alpha2: float, beta2: float, mix: float = 0.5,
                     r_min: float = 1e-7, r_max: float = 20e-6) -> np.ndarray:
    """
    Generate particle radii from a bimodal beta mixture mapped to [r_min, r_max].
    Uses incomplete beta for CDF inversion.
    """
    radii = np.zeros(n_particles, dtype=float)
    for i in range(n_particles):
        u = np.random.rand()
        # Binary search for inverse CDF of beta mixture
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) * 0.5
            val = mix * incomplete_beta_ratio(mid, alpha1, beta1) + \
                  (1.0 - mix) * incomplete_beta_ratio(mid, alpha2, beta2)
            if val is np.nan:
                break
            if val < u:
                lo = mid
            else:
                hi = mid
        radii[i] = r_min + lo * (r_max - r_min)
    return radii


# ==============================================================================
# Transfer/Swap clustering (from 039_asa113)
# ==============================================================================

def _criterion_variance(data: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Within-cluster sum of squared deviations (WCSS)."""
    total = 0.0
    for k in range(n_classes):
        mask = labels == k
        if np.sum(mask) == 0:
            continue
        cluster = data[mask]
        mean = np.mean(cluster)
        total += np.sum((cluster - mean) ** 2)
    return total


def transfer_clustering(data: np.ndarray, init_labels: np.ndarray,
                        n_classes: int, max_iter: int = 100) -> np.ndarray:
    """
    Transfer algorithm: iteratively move each particle to the class
    that most reduces the within-cluster variance.
    Maps from 039_asa113/trnsfr.m.
    """
    labels = init_labels.copy()
    n = len(data)
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            current = labels[i]
            best_class = current
            best_crit = _criterion_variance(data, labels, n_classes)
            for k in range(n_classes):
                if k == current:
                    continue
                labels[i] = k
                crit = _criterion_variance(data, labels, n_classes)
                if crit < best_crit:
                    best_crit = crit
                    best_class = k
                    improved = True
            labels[i] = best_class
        if not improved:
            break
    return labels


def swap_clustering(data: np.ndarray, labels: np.ndarray,
                    n_classes: int, max_iter: int = 50) -> np.ndarray:
    """
    Swap algorithm: iteratively swap pairs of particles between classes
    to reduce variance. Maps from 039_asa113/swap.m.
    """
    n = len(data)
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                crit_before = _criterion_variance(data, labels, n_classes)
                labels[i], labels[j] = labels[j], labels[i]
                crit_after = _criterion_variance(data, labels, n_classes)
                if crit_after < crit_before:
                    improved = True
                else:
                    labels[i], labels[j] = labels[j], labels[i]
        if not improved:
            break
    return labels


def cluster_particles(radii: np.ndarray, n_classes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster particles into n_classes size groups using transfer+swap.
    Returns (labels, class_centers).
    """
    # Initialize with sorted quantile bins
    sorted_idx = np.argsort(radii)
    labels = np.zeros(len(radii), dtype=int)
    for k in range(n_classes):
        lo = int(k * len(radii) / n_classes)
        hi = int((k + 1) * len(radii) / n_classes)
        labels[sorted_idx[lo:hi]] = k
    labels = transfer_clustering(radii, labels, n_classes)
    labels = swap_clustering(radii, labels, n_classes)
    centers = np.array([np.mean(radii[labels == k]) for k in range(n_classes)])
    return labels, centers


# ==============================================================================
# Tree-based hierarchical enumeration (from 1291_treepack)
# ==============================================================================

def catalan_number(n: int) -> int:
    """Catalan number C_n = (2n)! / ((n+1)! * n!). Maps from 1291_treepack/catalan.m."""
    if n < 0:
        return 0
    if n == 0:
        return 1
    val = 1.0
    for k in range(2, n + 1):
        val = val * (n + k) / k
    return int(val / (n + 1) + 0.5)


def tree_arc_to_pruefer(inode: np.ndarray, jnode: np.ndarray) -> np.ndarray:
    """
    Convert tree edge list to Prüfer code. Maps from 1291_treepack/tree_arc_to_pruefer.m.
    """
    n = len(inode) + 1
    degree = np.zeros(n, dtype=int)
    for i in range(n - 1):
        degree[inode[i]] += 1
        degree[jnode[i]] += 1
    code = np.zeros(n - 2, dtype=int)
    for i in range(n - 2):
        # Find leaf with smallest label
        leaf = np.argmin(np.where(degree == 1, np.arange(n), n))
        # Find its neighbor
        for e in range(n - 1):
            if inode[e] == leaf:
                neighbor = jnode[e]
                break
            elif jnode[e] == leaf:
                neighbor = inode[e]
                break
        code[i] = neighbor
        degree[leaf] = 0
        degree[neighbor] -= 1
    return code


def pruefer_to_tree(code: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Prüfer code to tree edge list. Maps from 1291_treepack/pruefer_to_tree_2.m.
    """
    n = len(code) + 2
    degree = np.ones(n, dtype=int)
    for c in code:
        degree[c] += 1
    inode = np.zeros(n - 1, dtype=int)
    jnode = np.zeros(n - 1, dtype=int)
    ptr = 0
    for c in code:
        leaf = np.argmin(np.where(degree == 1, np.arange(n), n))
        inode[ptr] = leaf
        jnode[ptr] = c
        ptr += 1
        degree[leaf] = 0
        degree[c] -= 1
    # Last edge
    leaves = np.where(degree == 1)[0]
    inode[ptr] = leaves[0]
    jnode[ptr] = leaves[1]
    return inode, jnode


class ParticleHierarchy:
    """
    Hierarchical tree representation of particle clusters for fast lookup
    and combinatorial enumeration of microstructural configurations.
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        # Use a star tree for simplicity
        self.inode = np.zeros(n_classes - 1, dtype=int)
        self.jnode = np.zeros(n_classes - 1, dtype=int)
        for i in range(1, n_classes):
            self.inode[i - 1] = 0
            self.jnode[i - 1] = i

    def to_pruefer(self) -> np.ndarray:
        return tree_arc_to_pruefer(self.inode, self.jnode)

    @classmethod
    def from_pruefer(cls, code: np.ndarray):
        n = len(code) + 2
        obj = cls(n)
        obj.inode, obj.jnode = pruefer_to_tree(code)
        return obj

    def diameter(self) -> int:
        """Tree diameter via BFS."""
        n = self.n_classes
        adj = [[] for _ in range(n)]
        for i in range(len(self.inode)):
            a, b = self.inode[i], self.jnode[i]
            adj[a].append(b)
            adj[b].append(a)

        def bfs(start: int):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            far = start
            while queue:
                u = queue.pop(0)
                far = u
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            return far, max(dist)

        u, _ = bfs(0)
        v, d = bfs(u)
        return d


# ==============================================================================
# Effective diffusivity statistics
# ==============================================================================

def effective_diffusivity_bruggeman(epsilon: float, tau: float = 1.5) -> float:
    """
    Bruggeman relation for effective diffusivity:
    D_eff = epsilon / tau * D_bulk, often approximated as epsilon^1.5.
    """
    return epsilon ** tau


def psd_confidence_interval(radii: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval on particle size distribution using
    incomplete beta ratio for order statistics.
    """
    n = len(radii)
    sorted_r = np.sort(radii)
    alpha_ci = (1.0 - confidence) * 0.5
    # Lower bound index via incomplete beta
    lo_idx = max(0, min(n - 1, int(incomplete_beta_ratio(alpha_ci, 1.0, n) * n) - 1))
    hi_idx = max(0, min(n - 1, int(incomplete_beta_ratio(1.0 - alpha_ci, 1.0, n) * n)))
    return float(sorted_r[lo_idx]), float(sorted_r[hi_idx])
