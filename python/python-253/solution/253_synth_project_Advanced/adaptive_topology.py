"""
adaptive_topology.py — Topology reconfiguration for adaptive mesh refinement.

When solving the Regge-Wheeler equation on a 1+1 grid, the required spatial
resolution is non-uniform: near the black-hole horizon and near the peak
of the potential barrier the solution has the largest gradients, while
in the asymptotic regions a coarser grid suffices.

This module provides three *topology reconfiguration strategies* inspired
by distributed multi-agent consensus:

  1. SIMPLIFY  : coarsen the mesh by removing interior nodes whose
                 interpolation error is below a tolerance.
  2. RESILIENT : refine the mesh by inserting new nodes where the
                 second-difference indicator exceeds a threshold.
  3. DECOMPOSE : split the grid into independent sub-topologies that
                 can be evolved in parallel with halo exchange.

Each strategy operates on a "mesh graph" whose adjacency matrix encodes
which grid points are neighbours; reconfiguration amounts to modifying
the adjacency matrix while preserving the boundary nodes.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
#  Mesh graph representation
# ---------------------------------------------------------------------------
class MeshGraph:
    """1D mesh stored as an adjacency structure.

    Attributes
    ----------
    x     : (N,) array of node coordinates (tortoise r_*).
    adj   : (N, N) adjacency matrix (tridiagonal for a 1D chain).
    u     : (N,) field values (optional, used for error estimation).
    """

    def __init__(self, x: np.ndarray, u: np.ndarray = None):
        N = len(x)
        self.x = np.asarray(x, dtype=float)
        self.u = np.asarray(u, dtype=float) if u is not None else np.zeros(N)
        self.adj = np.zeros((N, N), dtype=float)
        for i in range(N - 1):
            self.adj[i, i + 1] = 1.0
            self.adj[i + 1, i] = 1.0

    @property
    def n_nodes(self) -> int:
        return len(self.x)

    def neighbour_list(self, i: int) -> List[int]:
        return list(np.where(self.adj[i] > 0)[0])

    def indegree(self) -> np.ndarray:
        return np.sum(self.adj, axis=0)

    def outdegree(self) -> np.ndarray:
        return np.sum(self.adj, axis=1)

    def copy(self) -> "MeshGraph":
        g = MeshGraph(self.x.copy(), self.u.copy())
        g.adj = self.adj.copy()
        return g


# ---------------------------------------------------------------------------
#  Error indicator based on second differences
# ---------------------------------------------------------------------------
def second_difference_indicator(g: MeshGraph) -> np.ndarray:
    """Compute  eta_i = |u_{i+1} - 2 u_i + u_{i-1}|  as a refinement indicator."""
    eta = np.zeros(g.n_nodes)
    for i in range(1, g.n_nodes - 1):
        eta[i] = abs(g.u[i + 1] - 2.0 * g.u[i] + g.u[i - 1])
    return eta


# ---------------------------------------------------------------------------
#  Strategy 1: SIMPLIFY  (coarsen)
# ---------------------------------------------------------------------------
def simplify_topology(g: MeshGraph, tol: float) -> MeshGraph:
    """Remove interior nodes whose interpolation error is below tol.

    Boundary nodes (i = 0 and i = N-1) are never removed.
    """
    keep = [0]
    for i in range(1, g.n_nodes - 1):
        # linear interpolation error between neighbours
        xL, xR = g.x[i - 1], g.x[i + 1]
        uL, uR = g.u[i - 1], g.u[i + 1]
        u_interp = uL + (uR - uL) * (g.x[i] - xL) / max(xR - xL, 1.0e-300)
        if abs(g.u[i] - u_interp) > tol:
            keep.append(i)
    keep.append(g.n_nodes - 1)
    keep = sorted(set(keep))
    g2 = MeshGraph(g.x[keep], g.u[keep])
    return g2


# ---------------------------------------------------------------------------
#  Strategy 2: RESILIENT  (refine)
# ---------------------------------------------------------------------------
def resilient_topology(g: MeshGraph, threshold: float,
                       max_nodes: int = 10000) -> MeshGraph:
    """Insert midpoints where the second-difference indicator exceeds threshold."""
    eta = second_difference_indicator(g)
    new_x, new_u = [g.x[0]], [g.u[0]]
    for i in range(g.n_nodes - 1):
        if eta[i + 1] > threshold and len(new_x) < max_nodes - 2:
            x_mid = 0.5 * (g.x[i] + g.x[i + 1])
            u_mid = 0.5 * (g.u[i] + g.u[i + 1])
            new_x.append(x_mid)
            new_u.append(u_mid)
        new_x.append(g.x[i + 1])
        new_u.append(g.u[i + 1])
    return MeshGraph(np.array(new_x), np.array(new_u))


# ---------------------------------------------------------------------------
#  Strategy 3: DECOMPOSE  (split into parallel sub-topologies)
# ---------------------------------------------------------------------------
def decompose_topology(g: MeshGraph, n_parts: int,
                       halo: int = 2) -> List[MeshGraph]:
    """Split the mesh into n_parts overlapping sub-meshes for parallel evolution."""
    N = g.n_nodes
    base = N // n_parts
    rem = N % n_parts
    parts = []
    start = 0
    for k in range(n_parts):
        size = base + (1 if k < rem else 0)
        end = start + size
        lo = max(0, start - halo)
        hi = min(N, end + halo)
        sub = MeshGraph(g.x[lo:hi], g.u[lo:hi])
        parts.append(sub)
        start = end
    return parts


# ---------------------------------------------------------------------------
#  Consensus-style state update across sub-topologies (resilience)
# ---------------------------------------------------------------------------
def consensus_step(parts: List[MeshGraph], dt: float,
                   coupling: float = 0.1) -> List[MeshGraph]:
    """Perform one diffusive-consensus step across the sub-topologies.

    Each node updates:  u_i <- u_i + dt * coupling * sum_j (u_j - u_i)
    where the sum runs over neighbours in the same sub-topology and
    halo nodes are averaged between adjacent subdomains.
    """
    for g in parts:
        u_new = g.u.copy()
        for i in range(1, g.n_nodes - 1):
            s = 0.0
            for j in g.neighbour_list(i):
                s += g.u[j] - g.u[i]
            u_new[i] = g.u[i] + dt * coupling * s
        g.u = u_new
    return parts


# ---------------------------------------------------------------------------
#  Topology quality metrics
# ---------------------------------------------------------------------------
def topology_quality(g: MeshGraph) -> Dict[str, float]:
    """Return quality metrics for the current mesh topology."""
    dx = np.diff(g.x)
    return {
        "n_nodes": float(g.n_nodes),
        "min_dx": float(np.min(dx)) if len(dx) else 0.0,
        "max_dx": float(np.max(dx)) if len(dx) else 0.0,
        "mean_dx": float(np.mean(dx)) if len(dx) else 0.0,
        "aspect_ratio": float(np.max(dx) / max(np.min(dx), 1.0e-300)),
    }
