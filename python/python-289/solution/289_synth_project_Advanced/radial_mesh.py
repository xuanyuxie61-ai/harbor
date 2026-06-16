# -*- coding: utf-8 -*-
"""
radial_mesh.py
==============

1-D finite-element mesh along the radial (flux-surface) coordinate ``x``
together with the 3-D phase-space I/O and the poloidal triangulation
utilities needed by the gyrokinetic solver.

The design mirrors the classical ``fem1d`` suite (Burkardt) but is
rewritten for the gyrokinetic context:

    - ``RadialMesh1D``            port of ``fem1d_heat_steady`` and
                                  ``fem1d_function_10_display`` --
                                  piecewise-linear (P1) hat basis on a
                                  possibly non-uniform 1-D mesh.
    - ``HatBasis``                port of ``basic_hat`` / ``rd_lin_spline``
                                  -- the scalar hat and its gradient, the
                                  building blocks of the FEM assembly.
    - ``TriangularPoloidalMesh``  port of ``triangulation_triangle_neighbors``
                                  and ``xyzf_display`` -- an unstructured
                                  triangulation of the (R, Z) poloidal
                                  cross-section used to represent the
                                  equilibrium magnetic geometry in the
                                  local limit.
    - ``PhaseSpaceIO``            port of ``xyzf_display`` -- binary
                                  read/write of the 5-D gyrokinetic
                                  distribution function (x, y, z, v_par, mu).

The weak formulation of the radial FEM solve is

    integral_0^Lx   kappa(x) dphi/dx dv/dx  dx   =   integral_0^Lx  f(x) v(x) dx

for all test functions v in H^1 with v(0)=v(Lx)=0 (Dirichlet).  For the
Rosenbluth-Hinton zonal-flow residual (see ``zonal_flow.py``) the stiffness
is the (k_perp rho_s)^2 weighted by the adiabatic response and the
right-hand side is the initial vorticity.

Governing equation in strong form (Poisson / quasineutrality):

    - d/dx ( kappa(x) dphi/dx ) + kperp^2 (1 - Gamma0) phi = rho(x)

with  Gamma0 = I0(b) exp(-b),  b = k_perp^2 rho_s^2 / 2,  I0 the modified
Bessel function,  and  kappa(x) a (possibly tensorial) radial diffusion
coefficient that encodes the neoclassical polarisation.

References:
    [1] Burkardt, ``fem1d_heat_steady``, ``fem1d_function_10_display``.
    [2] Burkardt, ``triangulation_triangle_neighbors``, ``xyzf_display``.
    [3] Borggaard et al., ``rd_lin_spline`` -- FEM reaction-diffusion.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from physics_constants import PI, EPS_SQRT


# ============================================================================
# 1-D piecewise-linear (P1) hat basis -- port of fem_neumann/basic_hat.m
# ============================================================================
def basic_hat(x: np.ndarray) -> np.ndarray:
    """Evaluate the reference hat function  phi(x) = max(0, 1 - |x|)."""
    x = np.asarray(x, dtype=np.float64)
    f = np.zeros_like(x)
    m = (x + 1.0)
    f += m * ((-1.0 <= x) & (x < 0.0))
    m = (1.0 - x)
    f += m * ((0.0 <= x) & (x <= 1.0))
    return f


def basic_hat_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of the reference hat  (zero a.e. outside [-1, 1])."""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    g += np.where((-1.0 <= x) & (x < 0.0), 1.0, 0.0)
    g += np.where((0.0 <= x) & (x <= 1.0), -1.0, 0.0)
    return g


# ============================================================================
# 1-D radial mesh with P1 FEM -- port of fem1d_heat_steady / fem1d_function_10_display
# ============================================================================
@dataclass
class RadialMesh1D:
    """Piecewise-linear FEM mesh on [a, b].

    Attributes
    ----------
    nodes : (N,) array, sorted
    elements : (N-1, 2) int array of node indices per element
    values : optional (N,) array, the finite-element coefficients
    """

    nodes: np.ndarray
    elements: np.ndarray
    values: Optional[np.ndarray] = None

    # ---------- constructors ----------
    @classmethod
    def uniform(cls, a: float, b: float, n_nodes: int) -> "RadialMesh1D":
        if n_nodes < 2:
            raise ValueError("n_nodes must be >= 2")
        nodes = np.linspace(a, b, n_nodes)
        elements = np.column_stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)])
        return cls(nodes=nodes, elements=elements)

    @classmethod
    def from_file(cls, prefix: str) -> "RadialMesh1D":
        """Read node/element/value files (port of fem1d_function_10_display)."""
        nodes = np.loadtxt(f"{prefix}_nodes.txt")
        elements = np.loadtxt(f"{prefix}_elements.txt", dtype=int) - 1   # 1-indexed
        try:
            values = np.loadtxt(f"{prefix}_values.txt")
        except OSError:
            values = None
        return cls(nodes=nodes, elements=elements, values=values)

    # ---------- evaluation ----------
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the FE function sum_j u_j phi_j(x) at points x.

        If ``values`` is not set, returns zero.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.values is None:
            return np.zeros_like(x)
        out = np.zeros_like(x)
        # vectorised: find containing element
        idx = np.searchsorted(self.nodes, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.nodes) - 2)
        h = self.nodes[idx + 1] - self.nodes[idx]
        h = np.where(h > 0.0, h, 1.0)
        t = (x - self.nodes[idx]) / h
        out = self.values[idx] * (1.0 - t) + self.values[idx + 1] * t
        return out

    def save(self, prefix: str) -> None:
        """Port of fem1d_function_10_display output format."""
        np.savetxt(f"{prefix}_nodes.txt", self.nodes)
        np.savetxt(f"{prefix}_elements.txt", self.elements + 1, fmt="%d")
        if self.values is not None:
            np.savetxt(f"{prefix}_values.txt", self.values)

    # ---------- assembly ----------
    def assemble_stiffness(
        self, kappa: Callable[[np.ndarray], np.ndarray] | float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble stiffness K and mass M matrices for the P1 basis.

            K_ij = int kappa(x) phi_i'(x) phi_j'(x) dx
            M_ij = int phi_i(x) phi_j(x) dx

        For piecewise-constant kappa per element the element stiffness is
            K^e = kappa_e / h_e [[1, -1], [-1, 1]]
        and the element mass is
            M^e = h_e / 6  [[2, 1], [1, 2]].

        Returns
        -------
        K, M : (N, N) dense arrays
        """
        N = self.nodes.size
        K = np.zeros((N, N), dtype=np.float64)
        M = np.zeros((N, N), dtype=np.float64)
        xmid = 0.5 * (self.nodes[self.elements[:, 0]] + self.nodes[self.elements[:, 1]])
        h = self.nodes[self.elements[:, 1]] - self.nodes[self.elements[:, 0]]
        h = np.where(h > 0.0, h, 1.0)
        if callable(kappa):
            k = np.asarray(kappa(xmid), dtype=np.float64)
        else:
            k = np.full(xmid.size, float(kappa))
        k = np.where(np.isfinite(k) & (k > 0.0), k, EPS_SQRT)

        Ke = np.empty((self.elements.shape[0], 2, 2))
        Me = np.empty_like(Ke)
        Ke[:, 0, 0] = k / h
        Ke[:, 0, 1] = -k / h
        Ke[:, 1, 0] = -k / h
        Ke[:, 1, 1] = k / h
        Me[:, 0, 0] = h / 3.0
        Me[:, 0, 1] = h / 6.0
        Me[:, 1, 0] = h / 6.0
        Me[:, 1, 1] = h / 3.0

        for e in range(self.elements.shape[0]):
            idx = self.elements[e]
            K[np.ix_(idx, idx)] += Ke[e]
            M[np.ix_(idx, idx)] += Me[e]
        return K, M

    def solve_dirichlet(
        self,
        rhs: Callable[[np.ndarray], np.ndarray],
        kappa: Callable[[np.ndarray], np.ndarray] | float = 1.0,
        ua: float = 0.0,
        ub: float = 0.0,
    ) -> np.ndarray:
        """Solve the steady 1-D problem  -(kappa u')' = f, u(a)=ua, u(b)=ub.

        This is a direct port of ``fem1d_heat_steady.m`` (Burkardt).
        """
        K, _ = self.assemble_stiffness(kappa)
        N = self.nodes.size
        f = np.asarray(rhs(self.nodes), dtype=np.float64)
        # assemble RHS vector
        F = np.zeros(N)
        for e in range(self.elements.shape[0]):
            i, j = self.elements[e]
            h = self.nodes[j] - self.nodes[i]
            F[i] += 0.5 * h * (f[i] + f[j]) * 0.5
            F[j] += 0.5 * h * (f[i] + f[j]) * 0.5
        # apply Dirichlet BC
        F[0] = ua
        F[-1] = ub
        K[0, :] = 0.0
        K[-1, :] = 0.0
        K[0, 0] = 1.0
        K[-1, -1] = 1.0
        # solve
        u = np.linalg.solve(K, F)
        self.values = u
        return u


# ============================================================================
# Triangular poloidal mesh -- port of triangulation_triangle_neighbors / xyzf_display
# ============================================================================
@dataclass
class TriangularPoloidalMesh:
    """Unstructured triangulation of the (R, Z) poloidal cross-section.

    The local limit of toroidal geometry describes a flux surface by its
    major radius R0 and minor radius a; the mesh below tiles the surface
    (R, Z) in (r, theta) polar coordinates.  Neighbour information is
    required for conservative finite-volume advection of the gyrocentre
    density along the equilibrium drift.

    The neighbour-finding algorithm is a direct port of
    ``triangulation_triangle_neighbors.m`` (Burkardt): two triangles are
    neighbours if they share exactly two nodes.
    """

    nodes: np.ndarray              # (Nn, 3)   x, y, z  (we use x=R, y=Z, z unused)
    triangles: np.ndarray          # (Nt, 3)   0-indexed node ids
    neighbours: np.ndarray         # (Nt, 3)   opposite-triangle id or -1

    # ---------- constructors ----------
    @classmethod
    def annular(cls, R0: float, a: float, nr: int = 6, nt: int = 24) -> "TriangularPoloidalMesh":
        """Generate an annulus triangulation around (R0, 0) with minor radius a."""
        rs = np.linspace(max(0.1 * a, R0 - a), R0 + a, nr + 1)
        thetas = np.linspace(0.0, 2.0 * PI, nt + 1)[:-1]
        Rs, Ts = np.meshgrid(rs, thetas, indexing="ij")
        pts = np.column_stack([
            Rs.ravel() + 0.0 * Ts.ravel(),
            Ts.ravel() * 0.0 + 0.0,     # Z = 0 in this toy version (midplane)
            np.zeros(Rs.size),
        ])
        tris: List[List[int]] = []
        for i in range(nr):
            for j in range(nt):
                a_ = i * nt + j
                b_ = i * nt + (j + 1) % nt
                c_ = (i + 1) * nt + j
                d_ = (i + 1) * nt + (j + 1) % nt
                tris.append([a_, b_, d_])
                tris.append([a_, d_, c_])
        triangles = np.asarray(tris, dtype=int)
        neighbours = compute_triangle_neighbours(triangles)
        return cls(nodes=pts, triangles=triangles, neighbours=neighbours)

    @classmethod
    def from_xyzf(cls, prefix: str) -> "TriangularPoloidalMesh":
        """Port of xyzf_display: read ``prefix.xyz`` and ``prefix.xyzf``."""
        xyz_path = prefix + ".xyz"
        xyzf_path = prefix + ".xyzf"
        with open(xyz_path, "r") as fh:
            nodes = np.loadtxt(fh)
        with open(xyzf_path, "r") as fh:
            triangles = np.loadtxt(fh, dtype=int) - 1     # 1-indexed
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, -1)
        if triangles.ndim == 1:
            triangles = triangles.reshape(1, -1)
        # trim to 3 columns if needed
        if nodes.shape[1] > 3:
            nodes = nodes[:, :3]
        elif nodes.shape[1] < 3:
            pad = np.zeros((nodes.shape[0], 3 - nodes.shape[1]))
            nodes = np.hstack([nodes, pad])
        if triangles.shape[1] > 3:
            triangles = triangles[:, :3]
        neighbours = compute_triangle_neighbours(triangles)
        return cls(nodes=nodes, triangles=triangles, neighbours=neighbours)

    def save_xyzf(self, prefix: str) -> None:
        np.savetxt(prefix + ".xyz", self.nodes)
        np.savetxt(prefix + ".xyzf", self.triangles + 1, fmt="%d")


def compute_triangle_neighbours(triangles: np.ndarray) -> np.ndarray:
    """Port of ``triangulation_triangle_neighbors.m``.

    For each triangle t with nodes (a, b, c), the k-th neighbour (k = 0, 1, 2)
    is the triangle sharing the edge opposite node k.  Returns (Nt, 3)
    array of neighbour indices; -1 denotes a boundary edge.
    """
    Nt = triangles.shape[0]
    neigh = -np.ones((Nt, 3), dtype=np.int64)
    # edge -> triangle map
    edge_map = {}
    for t in range(Nt):
        a, b, c = triangles[t]
        for k, (p, q) in enumerate([(b, c), (a, c), (a, b)]):
            e = (min(p, q), max(p, q))
            if e in edge_map:
                t2, k2 = edge_map[e]
                neigh[t, k] = t2
                neigh[t2, k2] = t
                del edge_map[e]
            else:
                edge_map[e] = (t, k)
    return neigh


# ============================================================================
# 5-D phase-space I/O -- port of xyzf_display (here for structured grid)
# ============================================================================
class PhaseSpaceIO:
    """Binary dump / load of the gyrokinetic distribution function.

    Storage order:  (x, y, z, v_par, mu)  where (x, y, z) are the
    3 spatial coordinates (here reduced to 1 radial + 2 trivial),
    v_par is the parallel velocity, mu the magnetic moment.

    File layout (little-endian IEEE-754 float64):
        header (json utf-8) + length-prefix
        raw (N_x N_y N_z N_vp N_mu) * float64
    """

    HEADER_KEY = "gyrokinetic_phase_space_v1"

    @staticmethod
    def save(path: str, data: np.ndarray, meta: Optional[dict] = None) -> None:
        if data.ndim != 5:
            raise ValueError("PhaseSpaceIO.save: data must be 5-D")
        meta = dict(meta or {})
        meta["header"] = PhaseSpaceIO.HEADER_KEY
        meta["shape"] = list(data.shape)
        hdr = json.dumps(meta).encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(len(hdr).to_bytes(4, "little"))
            fh.write(hdr)
            data.astype("<f8", copy=False).tofile(fh)

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, dict]:
        with open(path, "rb") as fh:
            L = int.from_bytes(fh.read(4), "little")
            meta = json.loads(fh.read(L).decode("utf-8"))
            shape = tuple(meta["shape"])
            data = np.fromfile(fh, dtype="<f8").reshape(shape)
        return data, meta


# ============================================================================
# Sanity self-check
# ============================================================================
if __name__ == "__main__":
    mesh = RadialMesh1D.uniform(0.0, 1.0, 11)
    u = mesh.solve_dirichlet(lambda x: np.ones_like(x), kappa=1.0, ua=0.0, ub=0.0)
    exact = 0.5 * mesh.nodes * (1.0 - mesh.nodes)
    print("FEM steady-heat max error:", np.max(np.abs(u - exact)))
    # triangulation
    tri = TriangularPoloidalMesh.annular(R0=3.0, a=1.0, nr=3, nt=8)
    print("triangles:", tri.triangles.shape[0], "neighbours:",
          (tri.neighbours >= 0).sum())
    # hat
    xs = np.linspace(-1.5, 1.5, 7)
    print("hat:", basic_hat(xs))
    print("hat_grad:", basic_hat_grad(xs))
