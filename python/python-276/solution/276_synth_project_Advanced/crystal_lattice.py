"""
crystal_lattice.py — 2D crystal lattice, Miller reduction & migration graph
============================================================================

This module builds the host crystal lattice, identifies defect sites, and
exposes the graph of nearest-neighbour migration pathways that a point defect
can traverse. Three independent ideas from the seed projects are combined:

1. **Euclidean GCD** (from `342_euclid/gcd{1,2}.m`)
   Reduces a Miller-index direction [u v] to its primitive form by dividing
   out g = gcd(u, v). Also used to decide whether two lattice vectors
   R₁ = n₁ a₁ + m₁ a₂ and R₂ = n₂ a₁ + m₂ a₂ are commensurate:
       R₁ ∥ R₂   ⇔   n₁ m₂ − n₂ m₁ = 0   (cross product)
   after dividing each (n, m) pair by its GCD.

2. **Directed-graph forward-star representation** (from `286_digraph_arc/`)
   The set of atom-to-atom migration edges (e.g. a vacancy hopping to a
   nearest-neighbour site) is stored as a forward-star adjacency:
       arcfir[1..N+1], fwdarc[1..E]
   This is the classical LINPACK-era compressed format, repurposed here to
   drive kinetic-Monte-Carlo-style enumeration of defect configurations.

3. **Boundary-edge extraction from a triangulation** (from `1331_triangulation_boundary`)
   Given a Delaunay triangulation of the atomic positions, the boundary of
   the defect cluster is the set of directed edges that appear exactly once.
   The function `extract_boundary` reproduces the Burkardt algorithm verbatim
   in Python, returning ordered boundary-node paths.

Together these three pieces give us (i) a rigorous definition of the
supercell, (ii) a discrete state space for the defect, and (iii) a way to
identify the cluster of atoms that relax around the defect.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Dict


# -------------------------------------------------------------------------
# (1) Euclidean GCD — adapted from 342_euclid/gcd1.m, gcd2.m
# -------------------------------------------------------------------------
def gcd_euclid(a: int, b: int) -> int:
    """Greatest common divisor by repeated subtraction (Burkardt gcd1.m).

    The classical Euclidean algorithm:
        gcd(a, b) = gcd(a - b, b)   if a > b
        gcd(a, b) = gcd(a, b - a)   if b > a
        gcd(a, a) = a

    We use the subtraction form (rather than modulo) to mirror the seed
    implementation exactly. The result is always non-negative.
    """
    a, b = abs(int(a)), abs(int(b))
    if a == 0 and b == 0:
        return 0
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    return a


def gcd_euclid_fast(a: int, b: int) -> int:
    """Modulo-based GCD (Burkardt gcd2.m) — used where speed matters."""
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a


def reduce_miller(u: int, v: int) -> Tuple[int, int]:
    """Reduce Miller direction [u v] to primitive form by dividing gcd(u, v)."""
    g = gcd_euclid_fast(u, v)
    if g == 0:
        return (0, 0)
    return (u // g, v // g)


# -------------------------------------------------------------------------
# (2) Lattice construction
# -------------------------------------------------------------------------
class CrystalLattice:
    """A 2D crystal lattice (hexagonal or square) with N×N primitive cells.

    For hexagonal, the primitive vectors are
        a1 = a * (1, 0)
        a2 = a * (1/2, sqrt(3)/2)
    with a two-atom basis at (0, 0) and a * (1/2, 1/(2 sqrt(3))).

    The returned `positions` array has shape (N_sites, 2) in Bohr.
    """
    def __init__(self, a: float, n_cells: int, basis: str = "hexagonal"):
        self.a = float(a)
        self.n_cells = int(n_cells)
        self.basis = basis
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        N = self.n_cells
        a = self.a
        sites: List[np.ndarray] = []

        if self.basis == "hexagonal":
            a1 = np.array([a, 0.0])
            a2 = np.array([0.5 * a, 0.5 * math.sqrt(3.0) * a])
            basis_vecs = [
                np.array([0.0, 0.0]),
                np.array([0.5 * a, a / (2.0 * math.sqrt(3.0))]),
            ]
            for n1 in range(N):
                for n2 in range(N):
                    R = n1 * a1 + n2 * a2
                    for b in basis_vecs:
                        sites.append(R + b)
            self.a1, self.a2 = a1, a2
            self.basis_vecs = basis_vecs
        else:  # square
            a1 = np.array([a, 0.0])
            a2 = np.array([0.0, a])
            basis_vecs = [np.array([0.0, 0.0])]
            for n1 in range(N):
                for n2 in range(N):
                    R = n1 * a1 + n2 * a2
                    for b in basis_vecs:
                        sites.append(R + b)
            self.a1, self.a2 = a1, a2
            self.basis_vecs = basis_vecs

        self.positions = np.asarray(sites, dtype=np.float64)  # (N_sites, 2)
        self.n_sites = self.positions.shape[0]

    # ------------------------------------------------------------------
    def supercell_area(self) -> float:
        """Area of the supercell in Bohr²."""
        cross = abs(self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0])
        return cross * (self.n_cells ** 2)

    # ------------------------------------------------------------------
    def nearest_neighbours(self, cutoff_factor: float = 1.15
                           ) -> List[Tuple[int, int, float]]:
        """Return list of (i, j, d) pairs with d < cutoff_factor * d_min.

        For hexagonal, d_min = a / sqrt(3). We use a 1.15 factor to catch
        the 3 nearest neighbours of every site without false positives.
        """
        N = self.n_sites
        pos = self.positions
        d_min = self.a / math.sqrt(3.0) if self.basis == "hexagonal" else self.a
        cutoff = cutoff_factor * d_min
        edges: List[Tuple[int, int, float]] = []
        for i in range(N):
            for j in range(i + 1, N):
                rij = pos[j] - pos[i]
                # Minimum-image convention for periodic supercell
                L = self.n_cells * self.a
                if self.basis == "hexagonal":
                    # project onto a1, a2 and wrap
                    inv = np.linalg.inv(np.stack([self.a1, self.a2], axis=1))
                    frac = inv @ rij
                    frac -= np.round(frac)
                    rij = frac[0] * self.a1 + frac[1] * self.a2
                else:
                    rij -= L * np.round(rij / L)
                d = float(np.linalg.norm(rij))
                if d < cutoff and d > 1e-10:
                    edges.append((i, j, d))
        return edges


# -------------------------------------------------------------------------
# (3) Forward-star representation of the migration graph
#     (from 286_digraph_arc/digraph_adj_to_arc.m and digraph_arc_to_star.m)
# -------------------------------------------------------------------------
class MigrationGraph:
    """Directed graph of defect-migration edges in forward-star form.

    Given edges (u, v), we construct:
        arcfir[i] = index in fwdarc of the first edge leaving node i
        fwdarc[k] = destination node of the k-th edge
    The convention matches Burkardt's `digraph_arc_to_star.m`:
        arcfir has length (node_num + 1)
        arcfir[node_num + 1] = edge_num + 1   (sentinel)
    """
    def __init__(self, edges: List[Tuple[int, int, float]], n_nodes: int):
        self.n_nodes = int(n_nodes)
        # make edges bi-directed (migration is reversible)
        pairs: List[Tuple[int, int]] = []
        self.weight: Dict[Tuple[int, int], float] = {}
        for (u, v, d) in edges:
            pairs.append((int(u), int(v)))
            pairs.append((int(v), int(u)))
            self.weight[(int(u), int(v))] = d
            self.weight[(int(v), int(u))] = d

        # ---- Burkardt's algorithm: sort by (source, destination) ----
        pairs_sorted = sorted(pairs, key=lambda e: (e[0], e[1]))
        self.edge_num = len(pairs_sorted)

        # forward-star representation (0-based):
        #   arcfir[i] = index in fwdarc of first edge leaving node i
        #   arcfir[n_nodes] = sentinel = edge_num
        #   fwdarc[k] = destination node of the k-th edge
        self.arcfir = np.zeros(self.n_nodes + 1, dtype=np.int64)
        self.fwdarc = np.zeros(self.edge_num, dtype=np.int64)

        # count edges per source node
        src_count = np.zeros(self.n_nodes, dtype=np.int64)
        for (u, v) in pairs_sorted:
            src_count[u] += 1

        # prefix sum → arcfir[i] = start of edges leaving node i
        running = 0
        for i in range(self.n_nodes):
            self.arcfir[i] = running
            running += src_count[i]
        self.arcfir[self.n_nodes] = running  # sentinel

        # fill destinations
        cursor = np.zeros(self.n_nodes, dtype=np.int64)
        for (u, v) in pairs_sorted:
            pos = self.arcfir[u] + cursor[u]
            self.fwdarc[pos] = v
            cursor[u] += 1

    # ------------------------------------------------------------------
    def degree_in(self, node: int) -> int:
        """In-degree of a node (number of edges arriving)."""
        # An edge (u, v) arrives at v if fwdarc[k] == v for some k.
        # Use numpy to count.
        return int(np.sum(self.fwdarc == node))

    def degree_out(self, node: int) -> int:
        """Out-degree of a node."""
        return int(self.arcfir[node + 1] - self.arcfir[node])

    # ------------------------------------------------------------------
    def is_eulerian(self) -> bool:
        """Check the Eulerian property: indeg(v) == outdeg(v) for all v.

        Port of `digraph_arc_is_eulerian.m`. For the defect-migration graph
        this is always true (every hop has a reverse hop), but we keep the
        check as a sanity test and for completeness.
        """
        for v in range(self.n_nodes):
            if self.degree_in(v) != self.degree_out(v):
                return False
        return True


# -------------------------------------------------------------------------
# (4) Boundary extraction from triangulation
#     (from 1331_triangulation_boundary/boundary_edge_to_path.m and
#      triangulation_node_to_boundary_edge.m)
# -------------------------------------------------------------------------
def extract_boundary(triangle_node: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the boundary edges of a triangulation.

    Algorithm (Burkardt `triangulation_node_to_boundary_edge.m`):
      1. For every triangle, emit its three directed edges with the smaller
         node index first.
      2. Sort all 3*T edges lexicographically.
      3. Scan the sorted list: an edge that appears exactly once is a
         boundary edge; twice means interior.
      4. Chain boundary edges into ordered paths.

    Parameters
    ----------
    triangle_node : (T, 3) int array
        Each row lists the three node indices (0-based) of a triangle.

    Returns
    -------
    boundary_edge : (B, 2) int array
        Each row is an oriented boundary edge.
    boundary_path : 1-D int array
        Ordered sequence of boundary nodes (one connected loop assumed).
    """
    T = triangle_node.shape[0]
    edge = np.zeros((3 * T, 3), dtype=np.int64)
    k = 0
    for t in range(T):
        i, j, m = triangle_node[t]
        for (a, b) in [(i, j), (j, m), (m, i)]:
            if a <= b:
                edge[k] = [a, b, 0]
            else:
                edge[k] = [b, a, 1]
            k += 1

    # sort by (first node, second node, flag)
    order = np.lexsort((edge[:, 2], edge[:, 1], edge[:, 0]))
    edge = edge[order]

    # detect singletons
    boundary_edges: List[Tuple[int, int]] = []
    k = 0
    n_edges = edge.shape[0]
    while k < n_edges:
        j = k
        while j < n_edges - 1 and np.array_equal(edge[j, :2], edge[j + 1, :2]):
            j += 1
        if j == k:
            # appeared exactly once
            a, b = int(edge[k, 0]), int(edge[k, 1])
            # restore orientation from original flag
            if edge[k, 2] == 1:
                boundary_edges.append((b, a))
            else:
                boundary_edges.append((a, b))
        k = j + 1

    if len(boundary_edges) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)

    boundary_edge = np.asarray(boundary_edges, dtype=np.int64)

    # chain into path (Burkardt boundary_edge_to_path.m)
    path = [boundary_edge[0, 0], boundary_edge[0, 1]]
    remaining = list(range(1, len(boundary_edges)))
    while remaining:
        last = path[-1]
        found = False
        for idx, r in enumerate(remaining):
            if boundary_edge[r, 0] == last:
                path.append(boundary_edge[r, 1])
                remaining.pop(idx)
                found = True
                break
            elif boundary_edge[r, 1] == last:
                path.append(boundary_edge[r, 0])
                remaining.pop(idx)
                found = True
                break
        if not found:
            break
    boundary_path = np.asarray(path, dtype=np.int64)
    return boundary_edge, boundary_path


# -------------------------------------------------------------------------
# Convenience: build a lattice + migration graph for the default config
# -------------------------------------------------------------------------
def build_default_lattice(a: float, n_cells: int
                          ) -> Tuple[CrystalLattice, MigrationGraph]:
    """Build the host crystal and its defect-migration graph."""
    lat = CrystalLattice(a=a, n_cells=n_cells, basis="hexagonal")
    edges = lat.nearest_neighbours(cutoff_factor=1.15)
    graph = MigrationGraph(edges=edges, n_nodes=lat.n_sites)
    return lat, graph
