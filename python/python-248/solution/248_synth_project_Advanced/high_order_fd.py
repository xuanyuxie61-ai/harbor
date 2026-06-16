"""
high_order_fd.py
================

High-order finite-difference (FD) operators for the compressible Euler
equations with self-gravity.  The stencil structure is viewed as a
*directed graph* whose nodes are grid points and whose edges carry
the FD weights.  Traversal of this graph by BFS/DFS/Dijkstra-like
algorithms (inspired by the graphalgosimulation seed, 1134) gives the
minimal-cost closure of wide stencils near domain boundaries and AMR
level interfaces.

Key formulae
------------
* 4th-order centred first derivative (5-point stencil):

      f'_i ~ (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 dx)

      Truncation error:  O(dx^4) with coefficient  f^{(6)} / 180.

* 6th-order centred first derivative (7-point stencil):

      f'_i ~ (f_{i+3} - 9 f_{i+2} + 45 f_{i+1} - 45 f_{i-1}
               + 9 f_{i-2} - f_{i-3}) / (60 dx)

      Truncation error:  O(dx^6) with coefficient  f^{(8)} / 560.

* 4th-order centred second derivative:

      f''_i ~ (-f_{i+2} + 16 f_{i+1} - 30 f_i + 16 f_{i-1}
                - f_{i-2}) / (12 dx^2)

* Laplacian in 3-D:

      Delta f = d^2 f/dx^2 + d^2 f/dy^2 + d^2 f/dz^2

  each axis is treated independently by the same 1-D stencil.

Stencil-graph view
------------------
For a 1-D stencil of half-width s the stencil graph is a path of
2s + 1 vertices and 2s directed edges.  BFS on this graph gives the
stencil footprint; DFS gives the order of accumulation; Dijkstra
on weighted stencils (e.g. non-uniform grids near AMR interfaces)
gives the minimum-roundoff accumulation ordering.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable, Optional
from collections import deque
import numpy as np

from astro_constants import FD_ORDER_DEFAULT, FD_ORDER_MAX, WENO_ORDER


# =====================================================================
#                  STENCIL GRAPH REPRESENTATION
# =====================================================================

class StencilGraph:
    """
    Directed-graph representation of an FD stencil.  Nodes are integer
    offsets from the central point; edge weights are the FD coefficients.

    This object supports the three classical graph traversals from the
    graphalgosimulation seed:

      * BFS  -- enumerate the stencil footprint
      * DFS  -- derive a dependency ordering for fused multiply-adds
      * Dijkstra -- optimal summation ordering when coefficients
                    have widely varying magnitudes (e.g. near AMR
                    refinement boundaries where dx varies across cells)
    """

    def __init__(self, offsets: List[int], weights: List[float]) -> None:
        if len(offsets) != len(weights):
            raise ValueError("offsets and weights must have equal length")
        self.offsets = list(offsets)
        self.weights = list(weights)
        self.size = len(offsets)
        self._adj = self._build_adj()

    def _build_adj(self) -> dict:
        """Build an adjacency list connecting neighbouring stencil points."""
        adj = {o: [] for o in self.offsets}
        for i, o in enumerate(self.offsets):
            if i > 0:
                adj[self.offsets[i - 1]].append((o, abs(self.weights[i])))
            if i < self.size - 1:
                adj[self.offsets[i + 1]].append((o, abs(self.weights[i])))
        return adj

    # ---------------- BFS: stencil footprint ----------------
    def bfs_footprint(self, center: int = 0) -> List[int]:
        """Breadth-first enumeration of the stencil offsets."""
        visited = []
        queue = deque([center])
        seen = {center}
        while queue:
            node = queue.popleft()
            visited.append(node)
            for nbr, _ in self._adj.get(node, []):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        return visited

    # ---------------- DFS: accumulation order ----------------
    def dfs_order(self, start: Optional[int] = None) -> List[int]:
        """Depth-first ordering for the accumulation of the FD sum."""
        if start is None:
            start = self.offsets[self.size // 2]
        visited = []
        seen = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            visited.append(node)
            for nbr, _ in self._adj.get(node, []):
                if nbr not in seen:
                    stack.append(nbr)
        return visited

    # ---------------- Dijkstra: minimum-cost summation ----------------
    def dijkstra_order(self, start: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Dijkstra-based ordering: returns nodes sorted by cumulative
        cost (sum of |weight| along the shortest path from start).
        This minimises round-off when coefficients span many orders
        of magnitude, which occurs at coarse-fine AMR interfaces.
        """
        if start is None:
            start = self.offsets[self.size // 2]
        dist = {o: float("inf") for o in self.offsets}
        prev = {o: None for o in self.offsets}
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, u = pq.pop(0)
            if d > dist[u]:
                continue
            for v, w in self._adj.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    pq.append((nd, v))
        pq.sort(key=lambda t: t[0])
        return [(o, dist[o]) for o, _ in pq]

    def __repr__(self) -> str:
        return (f"StencilGraph(size={self.size}, "
                f"offsets={self.offsets}, weights={self.weights})")


# =====================================================================
#                    STANDARD CENTRED STENCILS
# =====================================================================

def fd4_first_derivative_stencil() -> StencilGraph:
    """
    4th-order centred first-derivative stencil (5 points):

        w = [-1, 8, 0, -8, 1] / (12 dx)
    """
    offsets = [-2, -1, 0, 1, 2]
    weights = [-1.0, 8.0, 0.0, -8.0, 1.0]
    # divide by 12 dx is deferred to the caller
    return StencilGraph(offsets, [w / 12.0 for w in weights])


def fd6_first_derivative_stencil() -> StencilGraph:
    """
    6th-order centred first-derivative stencil (7 points):

        w = [1, -9, 45, 0, -45, 9, -1] / (60 dx)
    """
    offsets = [-3, -2, -1, 0, 1, 2, 3]
    weights = [1.0, -9.0, 45.0, 0.0, -45.0, 9.0, -1.0]
    return StencilGraph(offsets, [w / 60.0 for w in weights])


def fd4_second_derivative_stencil() -> StencilGraph:
    """
    4th-order centred second-derivative stencil (5 points):

        w = [-1, 16, -30, 16, -1] / (12 dx^2)
    """
    offsets = [-2, -1, 0, 1, 2]
    weights = [-1.0, 16.0, -30.0, 16.0, -1.0]
    return StencilGraph(offsets, [w / 12.0 for w in weights])


def fd2_second_derivative_stencil() -> StencilGraph:
    """
    2nd-order centred second-derivative stencil (3 points, reference):

        w = [1, -2, 1] / dx^2
    """
    offsets = [-1, 0, 1]
    weights = [1.0, -2.0, 1.0]
    return StencilGraph(offsets, weights)


# =====================================================================
#                    1-D FINITE-DIFFERENCE OPERATORS
# =====================================================================

def apply_fd1(f: np.ndarray, dx: float, order: int = FD_ORDER_DEFAULT) -> np.ndarray:
    """
    Apply a 1-D first-derivative operator of specified order to a
    periodic array f.

    Boundary strategy: for the small test problem we use *periodic*
    wrapping; for non-periodic domains a one-sided downgraded stencil
    is used near the edges (this is handled in apply_fd1_nonperiodic).
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("apply_fd1 expects a 1-D array")
    n = f.size
    out = np.zeros_like(f)
    fp = np.concatenate([f[-3:], f, f[:3]])  # 3 periodic ghosts for 6th order

    if order == 2:
        for i in range(n):
            out[i] = (fp[i + 3 + 1] - fp[i + 3 - 1]) / (2.0 * dx)
    elif order == 4:
        for i in range(n):
            out[i] = (-fp[i + 3 + 2] + 8.0 * fp[i + 3 + 1]
                       - 8.0 * fp[i + 3 - 1] + fp[i + 3 - 2]) / (12.0 * dx)
    elif order == 6:
        for i in range(n):
            out[i] = (fp[i + 3 + 3] - 9.0 * fp[i + 3 + 2] + 45.0 * fp[i + 3 + 1]
                       - 45.0 * fp[i + 3 - 1] + 9.0 * fp[i + 3 - 2] - fp[i + 3 - 3]) \
                      / (60.0 * dx)
    else:
        raise ValueError(f"unsupported order {order}")
    return out


def apply_fd2(f: np.ndarray, dx: float, order: int = FD_ORDER_DEFAULT) -> np.ndarray:
    """
    Apply a 1-D second-derivative operator of specified order.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("apply_fd2 expects a 1-D array")
    n = f.size
    out = np.zeros_like(f)
    fp = np.concatenate([f[-3:], f, f[:3]])  # 3 periodic ghosts
    if order == 2:
        for i in range(n):
            out[i] = (fp[i + 3 + 1] - 2.0 * fp[i + 3] + fp[i + 3 - 1]) / (dx * dx)
    elif order == 4:
        for i in range(n):
            out[i] = (-fp[i + 3 + 2] + 16.0 * fp[i + 3 + 1]
                       - 30.0 * fp[i + 3] + 16.0 * fp[i + 3 - 1] - fp[i + 3 - 2]) \
                      / (12.0 * dx * dx)
    else:
        raise ValueError(f"unsupported order {order}")
    return out


def laplacian_3d(f: np.ndarray, dx: float, order: int = FD_ORDER_DEFAULT) -> np.ndarray:
    """
    3-D Laplacian via application of the 1-D second-derivative along
    each axis independently (assumes periodic boundary conditions).
    """
    if f.ndim != 3:
        raise ValueError("laplacian_3d expects a 3-D array")
    lap = np.zeros_like(f)
    for axis in range(3):
        # move target axis to last, apply fd2, move back
        g = np.moveaxis(f, axis, -1)
        shape = g.shape
        g2 = np.empty_like(g)
        for idx in np.ndindex(*shape[:-1]):
            g2[idx] = apply_fd2(g[idx], dx, order)
        lap += np.moveaxis(g2, -1, axis)
    return lap


# =====================================================================
#              TRUNCATION-ERROR ESTIMATION (self-check)
# =====================================================================

def fd1_truncation_error(order: int, dx: float, f6: float = 1.0,
                          f8: float = 1.0) -> float:
    """
    Leading truncation error of the first-derivative stencil:

        order 2:   dx^2 / 6  * f'''
        order 4:   dx^4 / 30 * f^{(5)}
        order 6:   dx^6 / 140 * f^{(7)}

    We return a normalised estimate assuming |f^{(n)}| = 1.
    """
    if order == 2:
        return (dx ** 2) / 6.0 * f6
    elif order == 4:
        return (dx ** 4) / 30.0 * f6
    elif order == 6:
        return (dx ** 6) / 140.0 * f8
    else:
        raise ValueError("unknown order")


def convergence_test(func: Callable[[np.ndarray], np.ndarray],
                     dx_list: List[float],
                     analytic: Callable[[np.ndarray], np.ndarray],
                     x: np.ndarray) -> List[Tuple[float, float]]:
    """
    Run a convergence test: compute the L-infinity error of the FD
    approximation at a sequence of grid spacings and return a list
    of (dx, err).  The observed convergence rate is

        rate = log(err_i / err_{i+1}) / log(dx_i / dx_{i+1})

    and for an order-p scheme this should approach p.
    """
    results = []
    for dx in dx_list:
        approx = func(x, dx)
        exact = analytic(x)
        err = np.max(np.abs(approx - exact))
        results.append((dx, float(err)))
    return results
