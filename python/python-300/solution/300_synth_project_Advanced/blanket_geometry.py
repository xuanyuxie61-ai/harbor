"""
blanket_geometry.py
===================
Heterogeneous pebble-bed breeding-blanket geometry built from a
**1-D Voronoi tessellation** of pebble centres, with a **vertex-to-element**
map that assigns the correct material cross section to every spatial cell.

The HCPB (Helium-Cooled Pebble Bed) blanket concept packs millimetre-sized
Li2O or Li4SiO4 ceramic pebbles inside Eurofer cooling channels.  The
*local* packing fraction  f_p(x)  varies with position because of wall
effects and random packing; we model it as a 1-D Voronoi tessellation
along the radial coordinate.

The algorithm:
  1. Draw N_pebble random centres on [0, L] with a hard-core constraint
     (minimum spacing  d_min).
  2. Construct the Voronoi cells around each centre.
  3. For every FD cell i, compute the fraction of the cell that lies
     inside a pebble; this defines the *local material composition*.
  4. Build a vertex-to-element (V-to-E) sparse map that takes the list of
     cell-averaged cross sections at the Voronoi *vertices* and produces
     the cross-section value at the centre of every FD cell.

Adapted from seed projects:
    * 1395_voronoi_display -> 1-D Voronoi cell construction
    * 756_mesh_vtoe        -> vertex-to-element sparse mapping
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple

import physics_constants as pc


# ---------------------------------------------------------------------------
# 1-D Voronoi tessellation
# ---------------------------------------------------------------------------
def poisson_disk_1d(xmin: float, xmax: float, r_min: float,
                     max_attempts: int = 200, seed: int = 42) -> List[float]:
    """Generate a 1-D Poisson-disk point set with minimum spacing r_min.

    This is the 1-D analogue of Bridson's fast algorithm: a candidate is
    accepted if it is at distance >= r_min from every existing point.
    """
    if xmax <= xmin:
        raise ValueError("xmax must exceed xmin")
    points: List[float] = []
    state = seed & 0xFFFFFFFF
    attempts = 0
    while attempts < max_attempts * (xmax - xmin) / max(r_min, 1.0e-12):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        cand = xmin + (xmax - xmin) * (state / 0xFFFFFFFF)
        ok = True
        for p in points:
            if abs(cand - p) < r_min:
                ok = False
                break
        if ok:
            points.append(cand)
        attempts += 1
        if len(points) > 5000:
            break
    return sorted(points)


def voronoi_edges_1d(centres: Sequence[float],
                      xmin: float, xmax: float) -> List[float]:
    """Return the sorted Voronoi edge positions including the outer walls.

    For centres sorted x_1 < x_2 < ... < x_N the interior edges are the
    midpoints  e_k = (x_k + x_{k+1}) / 2,  and we append the boundaries
    xmin and xmax.
    """
    if not centres:
        return [xmin, xmax]
    edges: List[float] = [xmin]
    for k in range(len(centres) - 1):
        edges.append(0.5 * (centres[k] + centres[k + 1]))
    edges.append(xmax)
    return edges


def voronoi_cell_lengths(edges: Sequence[float]) -> List[float]:
    """Return the length of each Voronoi cell."""
    return [edges[k + 1] - edges[k] for k in range(len(edges) - 1)]


# ---------------------------------------------------------------------------
# Vertex-to-element sparse mapping (from mesh_vtoe)
# ---------------------------------------------------------------------------
class VertexToElementMap:
    """Sparse V-to-E interpolation operator.

    Given a list of Voronoi edges (vertices in the FEM sense) and a list of
    FD cell centres, the map V2E takes a vector v_vtx of per-vertex values
    and returns a vector v_cell of per-cell values by piecewise-linear
    interpolation inside each Voronoi cell.

    The operator is represented in COO-style sparse form (i, j, w).
    """

    def __init__(
        self,
        edges: Sequence[float],
        cell_centres: Sequence[float],
    ) -> None:
        self.edges = list(edges)
        self.cell_centres = list(cell_centres)
        self.n_cells = len(cell_centres)
        self.n_vertices = len(edges)
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []
        self._build()

    def _build(self) -> None:
        edges = self.edges
        for i, x in enumerate(self.cell_centres):
            # locate bracketing edge index k: edges[k] <= x < edges[k+1]
            k = 0
            for j in range(len(edges) - 1):
                if edges[j] <= x <= edges[j + 1]:
                    k = j
                    break
            else:
                # outside: clamp
                if x < edges[0]:
                    k = 0
                else:
                    k = len(edges) - 2
            x0, x1 = edges[k], edges[k + 1]
            dx = x1 - x0 if x1 != x0 else 1.0
            u = (x - x0) / dx
            u = max(0.0, min(1.0, u))
            self.rows.append(i); self.cols.append(k); self.vals.append(1.0 - u)
            self.rows.append(i); self.cols.append(k + 1); self.vals.append(u)

    def apply(self, v_vtx: Sequence[float]) -> List[float]:
        """Apply V2E to a per-vertex vector -> per-cell vector."""
        if len(v_vtx) != self.n_vertices:
            raise ValueError("dimension mismatch")
        out = [0.0] * self.n_cells
        for r, c, w in zip(self.rows, self.cols, self.vals):
            out[r] += w * v_vtx[c]
        return out


# ---------------------------------------------------------------------------
# Blanket region builder
# ---------------------------------------------------------------------------
class BlanketRegion:
    """Composite radial blanket: first wall | pebble bed | back wall.

    Parameters
    ----------
    L_total_cm : total radial thickness (cm).
    fw_cm : first-wall (Eurofer) thickness.
    bw_cm : back-wall (Eurofer) thickness.
    n_cells : number of FD cells across the full radial domain.
    pebble_diameter_cm : mean pebble diameter.
    packing_fraction : volume fraction of pebbles inside the bed region.
    seed : PRNG seed for pebble-centre placement.
    """

    def __init__(
        self,
        L_total_cm: float = 80.0,
        fw_cm: float = 2.0,
        bw_cm: float = 1.5,
        n_cells: int = 128,
        pebble_diameter_cm: float = 0.06,   # 600 um
        packing_fraction: float = 0.62,
        seed: int = 12345,
    ) -> None:
        self.L_total = L_total_cm
        self.fw_cm = fw_cm
        self.bw_cm = bw_cm
        self.n_cells = n_cells
        self.pebble_diam = pebble_diameter_cm
        self.pack_frac = packing_fraction
        self.seed = seed

        # Bed region bounds
        self.x_fw = fw_cm
        self.x_bed_end = L_total_cm - bw_cm
        self.L_bed = max(self.x_bed_end - self.x_fw, 1.0e-6)

        # Generate pebble centres in the bed region with a hard-core
        # distance equal to the pebble diameter.
        self.pebble_centres = poisson_disk_1d(
            self.x_fw + 0.5 * pebble_diameter_cm,
            self.x_bed_end - 0.5 * pebble_diameter_cm,
            r_min=pebble_diameter_cm,
            seed=seed,
        )
        self.n_pebbles = len(self.pebble_centres)

        # Voronoi edges inside the bed region
        self.bed_edges = voronoi_edges_1d(
            self.pebble_centres, self.x_fw, self.x_bed_end
        )

        # FD grid (uniform)
        self.dx = L_total_cm / n_cells
        self.x_edges = [i * self.dx for i in range(n_cells + 1)]
        self.x_centres = [0.5 * (self.x_edges[i] + self.x_edges[i + 1])
                          for i in range(n_cells)]

        # Vertex-to-element map for mapping pebble-centre data onto the FD grid
        # We use a slightly richer vertex set: walls + pebble centres.
        vertex_set = [0.0, fw_cm]
        vertex_set.extend(self.pebble_centres)
        vertex_set.append(self.x_bed_end)
        vertex_set.append(L_total_cm)
        vertex_set = sorted(set(vertex_set))
        self.vertices = vertex_set
        self.v2e = VertexToElementMap(vertex_set, self.x_centres)

        # Pre-compute per-cell pebble-fraction f_p(x)
        self.pebble_fraction = self._compute_pebble_fraction()

    # ------------------------------------------------------------------
    def _compute_pebble_fraction(self) -> List[float]:
        """Return f_p[i], the fraction of FD cell i that is inside a pebble.

        A point x is "inside a pebble" if it lies within radius R of any
        pebble centre.  We integrate this indicator exactly over each FD
        cell by exploiting the sorted pebble list.
        """
        R = 0.5 * self.pebble_diam
        centres = self.pebble_centres
        f: List[float] = []
        for i in range(self.n_cells):
            a = self.x_edges[i]
            b = self.x_edges[i + 1]
            length = b - a
            if length <= 0.0:
                f.append(0.0)
                continue
            # union of intervals [c - R, c + R] intersected with [a, b]
            covered = 0.0
            for c in centres:
                lo = max(a, c - R)
                hi = min(b, c + R)
                if hi > lo:
                    covered += hi - lo
            # cap at length (overlaps may double-count; we accept the
            # first-come approximation as this is a small-scale model)
            covered = min(covered, length)
            f.append(covered / length)
        return f

    # ------------------------------------------------------------------
    def material_tag(self, cell_index: int) -> str:
        """Return the dominant material tag for FD cell `cell_index`.

        Returns one of 'fw', 'bed', 'bw'.  Within the bed region the local
        pebble fraction determines whether the cell is 'li2o' or 'eurofer'
        matrix.
        """
        x = self.x_centres[cell_index]
        if x < self.x_fw:
            return 'fw'
        if x >= self.x_bed_end:
            return 'bw'
        return 'li2o_pebble' if self.pebble_fraction[cell_index] > 0.5 else 'eurofer_matrix'

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        """Printable summary of the constructed geometry."""
        mean_f = sum(self.pebble_fraction) / max(self.n_cells, 1)
        return {
            "L_total_cm": self.L_total,
            "fw_cm": self.fw_cm,
            "bw_cm": self.bw_cm,
            "L_bed_cm": self.L_bed,
            "n_cells": self.n_cells,
            "n_pebbles": self.n_pebbles,
            "mean_pebble_fraction": mean_f,
            "dx_cm": self.dx,
        }
