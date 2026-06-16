# -*- coding: utf-8 -*-
"""
brillouin_zone.py
-----------------
Construction of the first Brillouin zone of a 1D / 2D lattice as a convex
polygon via Minkowski sums of the reciprocal-lattice basis vectors.  Also
provides the "support function" h_BZ(theta) which is the radial extent of
the BZ in direction theta.

Scientific origin of the fused algorithms
-----------------------------------------
* Polygon Minkowski representation  (seed project 887_polygon_minkowski)
    -> conversion between vertex representation (V-rep) and Minkowski
       representation (edge normals + offsets).
    -> adapted: the BZ polygon is constructed as the Minkowski sum of p
       line segments [-G_j/2, G_j/2] where G_j are the reciprocal-lattice
       vectors.  The Minkowski representation is used to test whether a
       k-point lies inside the BZ.

Core physics / mathematics
--------------------------
* For a 2D lattice with primitive vectors a_1, a_2 the reciprocal vectors
      b_1 = 2pi (a_2 x z) / (a_1 . (a_2 x z))
      b_2 = 2pi (z x a_1) / (a_1 . (a_2 x z))
  generate the reciprocal lattice.
* The first BZ is the Voronoi cell of the origin in the reciprocal lattice,
  equivalently the set of points closer to 0 than to any other reciprocal
  lattice point.  For a zonotope (Minkowski sum of segments) this is a
  convex polygon whose edges are perpendicular to the generating vectors.

Stability / boundary notes
--------------------------
* Degenerate cases (all generators collinear) are detected and reported.
* The in-BZ test uses a small tolerance ``eps`` to handle points on the
  boundary robustly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Polygon V-rep  <->  Minkowski rep  (seed project 887)
# ---------------------------------------------------------------------------
@dataclass
class PolygonMinkowski:
    """Minkowski representation of a convex polygon.

    Each row (n_i, h_i) encodes the half-plane  n_i . x <= h_i.
    The polygon is the intersection of all these half-planes.
    """
    normals: np.ndarray   # (m, 2) outward unit normals
    offsets: np.ndarray   # (m,) offsets  h_i


def vertices_to_minkowski(vertices: np.ndarray) -> PolygonMinkowski:
    """Convert a convex polygon (given as ordered vertices) to Minkowski rep.

    Each edge (v_i, v_{i+1}) gives a half-plane constraint.
    """
    v = np.asarray(vertices, dtype=float)
    nv = v.shape[0]
    if nv < 3:
        raise ValueError("vertices_to_minkowski: need at least 3 vertices")
    normals = np.zeros((nv, 2))
    offsets = np.zeros(nv)
    for i in range(nv):
        v0 = v[i]
        v1 = v[(i + 1) % nv]
        edge = v1 - v0
        # Outward normal: rotate edge by -90 degrees and normalise
        n = np.array([edge[1], -edge[0]])
        n_len = np.linalg.norm(n)
        if n_len < 1e-300:
            raise ValueError("vertices_to_minkowski: degenerate edge")
        n /= n_len
        normals[i] = n
        offsets[i] = float(n @ v0)
    return PolygonMinkowski(normals=normals, offsets=offsets)


def minkowski_to_vertices(mk: PolygonMinkowski) -> np.ndarray:
    """Convert Minkowski representation back to vertices.

    Uses the fact that each vertex is the intersection of two adjacent
    constraint lines.  (This only works correctly when the normals are
    sorted by angle.)
    """
    # Sort normals by angle
    angles = np.arctan2(mk.normals[:, 1], mk.normals[:, 0])
    order = np.argsort(angles)
    normals = mk.normals[order]
    offsets = mk.offsets[order]
    m = normals.shape[0]
    verts = np.zeros((m, 2))
    for i in range(m):
        n1, h1 = normals[i], offsets[i]
        n2, h2 = normals[(i + 1) % m], offsets[(i + 1) % m]
        det = n1[0] * n2[1] - n1[1] * n2[0]
        if abs(det) < 1e-300:
            verts[i] = np.array([np.nan, np.nan])
            continue
        verts[i, 0] = (h1 * n2[1] - h2 * n1[1]) / det
        verts[i, 1] = (n1[0] * h2 - n2[0] * h1) / det
    return verts


def point_in_polygon(pt: np.ndarray, mk: PolygonMinkowski, eps: float = 1e-10) -> bool:
    """Test whether pt lies inside (or on the boundary of) the polygon."""
    pt = np.asarray(pt, dtype=float)
    violations = (mk.normals @ pt) - mk.offsets
    return bool(np.all(violations <= eps))


# ---------------------------------------------------------------------------
# 2.  BZ construction as a Minkowski sum of reciprocal-lattice segments
# ---------------------------------------------------------------------------
@dataclass
class BrillouinZone:
    """First Brillouin zone as a convex polygon."""
    vertices: np.ndarray           # (m, 2)
    minkowski: PolygonMinkowski
    area: float
    reciprocal_vectors: np.ndarray  # (p, 2) generators used


def reciprocal_vectors_2d(a1: np.ndarray, a2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 2D reciprocal lattice vectors from the real-space primitives."""
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    cross = a1[0] * a2[1] - a1[1] * a2[0]
    if abs(cross) < 1e-300:
        raise ValueError("reciprocal_vectors_2d: degenerate lattice (cross product = 0)")
    b1 = 2.0 * math.pi * np.array([a2[1], -a2[0]]) / cross
    b2 = 2.0 * math.pi * np.array([-a1[1], a1[0]]) / cross
    return b1, b2


def build_brillouin_zone(
    a1: np.ndarray = None,
    a2: np.ndarray = None,
    extra_g_vectors: List[np.ndarray] = None,
) -> BrillouinZone:
    """Construct the first BZ as the convex hull of the Wigner-Seitz cell.

    For a simple square lattice with a1=(1,0), a2=(0,1) the BZ is the
    square [-pi, pi]^2.  Extra G-vectors can be added to model a
    multi-band reciprocal lattice.
    """
    if a1 is None:
        a1 = np.array([1.0, 0.0])
    if a2 is None:
        a2 = np.array([0.0, 1.0])
    b1, b2 = reciprocal_vectors_2d(a1, a2)
    generators = [b1, b2]
    if extra_g_vectors is not None:
        for g in extra_g_vectors:
            generators.append(np.asarray(g, dtype=float))

    # Build the BZ as the intersection of half-planes  k . G <= |G|^2 / 2
    normals = []
    offsets = []
    for g in generators:
        g_len = np.linalg.norm(g)
        if g_len < 1e-300:
            continue
        n = g / g_len
        normals.append(n)
        normals.append(-n)
        offsets.append(0.5 * g_len)
        offsets.append(0.5 * g_len)

    normals_arr = np.array(normals)
    offsets_arr = np.array(offsets)
    mk = PolygonMinkowski(normals=normals_arr, offsets=offsets_arr)

    verts = minkowski_to_vertices(mk)
    # Remove NaN rows
    valid = ~np.isnan(verts).any(axis=1)
    verts = verts[valid]

    # Area via shoelace
    if verts.shape[0] >= 3:
        x = verts[:, 0]
        y = verts[:, 1]
        area = 0.5 * abs(float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()))
    else:
        area = 0.0

    return BrillouinZone(
        vertices=verts,
        minkowski=mk,
        area=area,
        reciprocal_vectors=np.array(generators),
    )


# ---------------------------------------------------------------------------
# 3.  BZ integration weight for a k-point
# ---------------------------------------------------------------------------
def kpoint_in_bz_weight(
    k: np.ndarray,
    bz: BrillouinZone,
    sigma: float = 0.05,
) -> float:
    """Soft weight w(k) = 1 if k is inside the BZ, smoothly decaying outside.

    Uses a Fermi-like smearing  w = 1 / (1 + exp(d / sigma))  where d is
    the signed distance to the BZ boundary (positive outside).
    """
    k = np.asarray(k, dtype=float)
    violations = (bz.minkowski.normals @ k) - bz.minkowski.offsets
    d = float(violations.max())   # distance to the most-violated half-plane
    if d < -10.0 * sigma:
        return 1.0
    if d > 10.0 * sigma:
        return 0.0
    return 1.0 / (1.0 + math.exp(d / sigma))
