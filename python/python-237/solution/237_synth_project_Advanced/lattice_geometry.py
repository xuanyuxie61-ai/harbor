"""
Lattice geometry and mesh generation.

Adapted from:
  - 378_fem_to_gmsh (FEM mesh construction)
  - 1348_triangulation_quality (simplex/plaquette quality metrics)

This module builds the 4D hypercubic lattice structure, enumerates
plaquettes, and computes plaquette quality metrics.

Key formulas:
    Plaquette: U_{mu nu}(x) = U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
    Plaquette action density:
        S_p(x, mu, nu) = 1 - (1/2) Re Tr U_{mu nu}(x)
"""

import numpy as np
from typing import List, Tuple, NamedTuple
from constants import LatticeParams, su2_identity


class SiteIndex(NamedTuple):
    t: int
    x: int
    y: int
    z: int


class PlaquetteIndex(NamedTuple):
    site: SiteIndex
    mu: int
    nu: int


class LatticeGeometry:
    """4D hypercubic lattice with periodic boundary conditions."""

    def __init__(self, params: LatticeParams):
        self.p = params
        self.Nt, self.Ns = params.Nt, params.Ns
        self.shape = params.shape
        self._flat_size = params.V

    def site_to_flat(self, s: SiteIndex) -> int:
        return ((s.t * self.Ns + s.x) * self.Ns + s.y) * self.Ns + s.z

    def flat_to_site(self, i: int) -> SiteIndex:
        z = i % self.Ns
        y = (i // self.Ns) % self.Ns
        x = (i // (self.Ns ** 2)) % self.Ns
        t = i // (self.Ns ** 3)
        return SiteIndex(t, x, y, z)

    def neighbor(self, s: SiteIndex, mu: int, forward: bool = True) -> SiteIndex:
        coords = list(s)
        L = self.Nt if mu == 0 else self.Ns
        if forward:
            coords[mu] = (coords[mu] + 1) % L
        else:
            coords[mu] = (coords[mu] - 1) % L
        return SiteIndex(*coords)

    def all_sites(self) -> List[SiteIndex]:
        sites = []
        for t in range(self.Nt):
            for x in range(self.Ns):
                for y in range(self.Ns):
                    for z in range(self.Ns):
                        sites.append(SiteIndex(t, x, y, z))
        return sites

    def all_plaquettes(self) -> List[PlaquetteIndex]:
        plaq = []
        for s in self.all_sites():
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    plaq.append(PlaquetteIndex(s, mu, nu))
        return plaq

    def n_plaquettes(self) -> int:
        return self._flat_size * 6


def compute_site_quality_metric(gauge_field, geo: LatticeGeometry, s) -> np.ndarray:
    """Per-site gauge field quality metric (sum of action densities)."""
    q = np.zeros(6)
    idx = 0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            U_p = _plaquette_at(gauge_field, geo, s, mu, nu)
            tr = 0.5 * (U_p[0, 0].real + U_p[1, 1].real)
            q[idx] = 1.0 - tr
            idx += 1
    return q


def _plaquette_at(gauge_field, geo: LatticeGeometry, s, mu: int, nu: int):
    s_mu = geo.neighbor(s, mu, forward=True)
    s_nu = geo.neighbor(s, nu, forward=True)
    U_mu_s = gauge_field.link(s, mu)
    U_nu_smu = gauge_field.link(s_mu, nu)
    U_mu_snu_dag = gauge_field.link(s_nu, mu).conj().T
    U_nu_s_dag = gauge_field.link(s, nu).conj().T
    return U_mu_s @ U_nu_smu @ U_mu_snu_dag @ U_nu_s_dag


def global_plaquette_quality(gauge_field, geo: LatticeGeometry) -> dict:
    sites = geo.all_sites()
    values = []
    for s in sites:
        for mu in range(4):
            for nu in range(mu + 1, 4):
                U_p = _plaquette_at(gauge_field, geo, s, mu, nu)
                tr = 0.5 * (U_p[0, 0].real + U_p[1, 1].real)
                values.append(tr)
    arr = np.array(values)
    return {
        'mean': float(arr.mean()),
        'var': float(arr.var()),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }
