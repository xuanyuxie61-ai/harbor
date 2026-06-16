"""
multizone_accretion.py
======================
Four-zone accretion model inspired by simulated moving-bed (SMB)
chromatography (CADET / Klatt et al., 2002; He et al., 2018).

Physical analogy
----------------
In SMB chromatography, four functional zones (adsorption, regeneration,
desorption, purification) are arranged in a loop and the feed/collect
ports are periodically switched.  The analogue for dark matter halo
accretion is a four-zone "accretion ring" around the proto-halo:

    Zone 1 - Cold-flow filaments  (high inflow, low metallicity)
    Zone 2 - Shock-heated halo gas (quasi-virial, high entropy)
    Zone 3 - Recycled winds       (outflowing, metal-enriched)
    Zone 4 - Intergalactic void    (low density, dark matter only)

Matter cycles through these zones, with the effective "switching"
given by the halo dynamical time t_dyn.  In each zone the transport
is governed by an equilibrium-dispersive equation

    dc/dt + u dc/dz = D d^2c/dz^2 - (1 - eps)/eps * dq/dt,

where c is the species concentration (here: baryon fraction, metal
mass fraction, specific angular momentum), u is the zone inflow
velocity, D is an effective dispersion, and q is the "adsorbed"
mass fraction bound to subhaloes.

We discretise each of the 8 columns (2 per zone) with a uniform
grid and integrate the equilibrium-dispersive model forward in time
with a simple explicit upwind + central-difference scheme.
"""

from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np


# ---------- Physical parameters ----------------------------------------------

class AccretionZone:
    """Parameters for one of the four zones."""
    __slots__ = ("name", "velocity", "dispersion", "porosity",
                 "k_ads", "n_columns")

    def __init__(self, name: str, velocity: float, dispersion: float,
                 porosity: float, k_ads: float, n_columns: int = 2):
        self.name = name
        self.velocity = velocity
        self.dispersion = dispersion
        self.porosity = porosity
        self.k_ads = k_ads
        self.n_columns = n_columns


def default_four_zones() -> List[AccretionZone]:
    return [
        AccretionZone("cold_filament", velocity=1.2, dispersion=0.05,
                      porosity=0.4, k_ads=0.8, n_columns=2),
        AccretionZone("virial_gas",   velocity=0.3, dispersion=0.10,
                      porosity=0.7, k_ads=0.3, n_columns=2),
        AccretionZone("recycled_wind",velocity=-0.6,dispersion=0.08,
                      porosity=0.6, k_ads=0.5, n_columns=2),
        AccretionZone("void_inflow",  velocity=0.2, dispersion=0.03,
                      porosity=0.3, k_ads=0.9, n_columns=2),
    ]


# ---------- Equilibrium-dispersive column solver -----------------------------

class Column:
    """1D equilibrium-dispersive column with linear isotherm."""

    def __init__(self, n_cells: int = 32, length: float = 1.0):
        self.n_cells = n_cells
        self.length = length
        self.dz = length / max(n_cells, 1)
        self.c = np.zeros(n_cells)         # mobile-phase concentration
        self.q = np.zeros(n_cells)         # adsorbed concentration

    def step_explicit(self, zone: AccretionZone, dt: float,
                      c_in: float) -> float:
        """Advance the column by one explicit Euler step.
        Returns the outlet concentration."""
        eps = zone.porosity
        u = zone.velocity
        D = zone.dispersion
        k = zone.k_ads
        dz = self.dz
        c, q = self.c, self.q
        n = self.n_cells

        # CFL-safe clipping
        cfl_u = abs(u) * dt / dz
        cfl_d = 2.0 * D * dt / dz ** 2
        if cfl_u + cfl_d > 0.9:
            dt_loc = 0.8 * dz / (abs(u) + 2.0 * D / dz + 1e-9)
        else:
            dt_loc = dt

        c_new = c.copy()
        q_new = q.copy()
        for j in range(n):
            c_left = c[j - 1] if j > 0 else c_in
            c_right = c[j + 1] if j < n - 1 else c[j]
            grad_c = (c_right - c_left) / (2.0 * dz)
            lap_c = (c_right - 2.0 * c[j] + c_left) / (dz ** 2)
            ads_rate = k * (c[j] - q[j])
            c_new[j] = c[j] + dt_loc * (
                -u * grad_c + D * lap_c - (1.0 - eps) / eps * ads_rate
            )
            q_new[j] = q[j] + dt_loc * ads_rate
        # clip to maintain physical positivity
        self.c = np.clip(c_new, 0.0, None)
        self.q = np.clip(q_new, 0.0, None)
        return float(self.c[-1])


# ---------- Four-zone loop ---------------------------------------------------

class FourZoneAccretionRing:
    """Loop of 8 columns (2 per zone) with periodic port switching."""

    def __init__(self, n_cells_per_column: int = 24):
        self.zones = default_four_zones()
        self.columns: List[List[Column]] = []
        for z in self.zones:
            self.columns.append([Column(n_cells=n_cells_per_column)
                                 for _ in range(z.n_columns)])
        self.switch_counter = 0

    def feed(self, species: str = "baryon") -> np.ndarray:
        """Initialise the inlet concentration pattern for the given
        species.  Supported: baryon, metal, angular_momentum."""
        if species == "baryon":
            return np.array([0.16, 0.10, 0.02, 0.04])
        if species == "metal":
            return np.array([0.001, 0.010, 0.020, 0.0005])
        if species == "angular_momentum":
            return np.array([1.0, 0.7, 0.3, 0.5])
        raise ValueError(f"unknown species {species}")

    def advance(self, dt: float, n_steps: int,
                species: str = "baryon") -> dict:
        """Advance the ring for n_steps and return outlet histories."""
        inlet = self.feed(species)
        history = {z.name: [] for z in self.zones}
        switch_every = max(1, 20)
        for step in range(n_steps):
            for iz, zone in enumerate(self.zones):
                c_in = inlet[iz]
                for col in self.columns[iz]:
                    c_out = col.step_explicit(zone, dt, c_in)
                    c_in = c_out
                history[zone.name].append(c_out)
            self.switch_counter += 1
            if self.switch_counter % switch_every == 0:
                self._switch_ports()
        return {k: np.array(v) for k, v in history.items()}

    def _switch_ports(self) -> None:
        """Rotate the zone assignments by one position."""
        self.zones = self.zones[1:] + self.zones[:1]
        self.columns = self.columns[1:] + self.columns[:1]


# ---------- Self-check --------------------------------------------------------

def self_check() -> dict:
    ring = FourZoneAccretionRing(n_cells_per_column=16)
    hist = ring.advance(dt=0.01, n_steps=40, species="baryon")
    return {name: {"final": float(arr[-1]),
                   "mean": float(arr.mean())}
            for name, arr in hist.items()}
