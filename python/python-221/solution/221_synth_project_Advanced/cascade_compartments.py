"""
cascade_compartments.py
=======================

Compartmental modeling of the hadronic cascade (from 1101_LCNP-KIST):
- Compartment-based cascade dynamics with saturation
- Viability scoring for event acceptance

Scientific context:
-------------------
The hadronic cascade in a high-energy collision is modeled as a system
of coupled compartments, each representing a rapidity slice or a parton
species. The evolution follows:

    dN_i/dt = sum_j T_{ij} N_j - sum_j T_{ji} N_i - gamma_i N_i + S_i(t)

where:
- N_i is the multiplicity in compartment i
- T_{ij} is the transition rate from compartment j to i
- gamma_i is the hadronization rate (loss to observable hadrons)
- S_i(t) is the source term from perturbative parton shower

This is analogous to the compartmental model in neuroscience (from 1101)
where instead of neural compartments we have rapidity compartments, and
instead of membrane potential we have parton multiplicity.

The viability score determines whether an event passes the analysis cuts:
    V = prod_c s_c
where s_c in [0, 1] is the score for cut c.
"""

import math
import random
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# Section 1: Compartment model (from 1101_LCNP-KIST)
# ===========================================================================

class CascadeCompartment:
    """
    A single compartment in the hadronic cascade model.

    Analogous to a neural compartment (from the neuron model), but
    representing a rapidity slice with parton multiplicity N_i.

    Attributes
    ----------
    length : float
        Rapidity extent of the compartment.
    diameter : float
        Effective transverse size (related to pT bin).
    n_partons : float
        Current parton multiplicity.
    saturation : float
        Saturation parameter (0 = empty, 1 = fully saturated).
    """

    def __init__(self, length: float = 1.0, diameter: float = 1.0,
                 n_partons: float = 0.0, nseg: int = 5):
        self.length = length
        self.diameter = diameter
        self.n_partons = n_partons
        self.nseg = nseg
        self.saturation = 0.0
        self._update_saturation()

    def _update_saturation(self):
        """Michaelis-Menten saturation: S = N / (K + N) with K = 1."""
        k_sat = 1.0
        n = max(0.0, self.n_partons)
        self.saturation = n / (k_sat + n + 1e-12)

    def add_partons(self, delta_n: float):
        """Add delta_n partons to this compartment."""
        self.n_partons = max(0.0, self.n_partons + delta_n)
        self._update_saturation()

    def remove_partons(self, delta_n: float):
        """Remove delta_n partons (hadronization loss)."""
        self.n_partons = max(0.0, self.n_partons - delta_n)
        self._update_saturation()


class CascadeSystem:
    """
    System of coupled compartments representing the hadronic cascade.

    The compartments are arranged in rapidity: y_min to y_max.
    Adjacent compartments exchange partons via diffusion (DGLAP-like)
    and each compartment loses partons to hadronization.
    """

    def __init__(self, n_compartments: int, y_min: float = -3.0,
                 y_max: float = 3.0):
        if n_compartments < 1:
            raise ValueError(f"CascadeSystem: n_compartments={n_compartments} < 1")
        self.n = n_compartments
        self.y_min = y_min
        self.y_max = y_max
        self.dy = (y_max - y_min) / n_compartments
        self.compartments = [
            CascadeCompartment(length=self.dy, diameter=1.0)
            for _ in range(n_compartments)
        ]
        # Transition rates (DGLAP-like diffusion in rapidity)
        self.diffusion_rate = 0.1
        self.hadronization_rate = 0.05

    def initialize_from_parton_shower(self, parton_rapidities: List[float],
                                      parton_energies: List[float]):
        """Initialize compartment occupancies from a list of partons."""
        for y, e in zip(parton_rapidities, parton_energies):
            # Find compartment index
            idx = int((y - self.y_min) / self.dy)
            idx = max(0, min(self.n - 1, idx))
            self.compartments[idx].add_partons(e)

    def step(self, dt: float, source: Optional[List[float]] = None):
        """
        Advance the cascade by dt using Euler method:

        dN_i/dt = D*(N_{i-1} - 2*N_i + N_{i+1})/dy^2 - gamma*N_i + S_i

        This is a discrete diffusion equation with source and sink.
        """
        n = self.n
        new_n = [0.0] * n
        inv_dy2 = 1.0 / (self.dy * self.dy) if self.dy > 0 else 0.0
        for i in range(n):
            ni = self.compartments[i].n_partons
            # Diffusion from neighbors
            n_left = self.compartments[i - 1].n_partons if i > 0 else ni
            n_right = self.compartments[i + 1].n_partons if i < n - 1 else ni
            laplacian = (n_left - 2.0 * ni + n_right) * inv_dy2
            # Source
            si = source[i] if source is not None else 0.0
            # Update
            dndt = self.diffusion_rate * laplacian - self.hadronization_rate * ni + si
            new_n[i] = max(0.0, ni + dt * dndt)
        # Apply
        for i in range(n):
            self.compartments[i].n_partons = new_n[i]
            self.compartments[i]._update_saturation()

    def total_multiplicity(self) -> float:
        """Total parton multiplicity across all compartments."""
        return sum(c.n_partons for c in self.compartments)

    def entropy(self) -> float:
        """
        Information entropy of the multiplicity distribution:
            S = -sum_i p_i * ln(p_i)
        where p_i = N_i / sum N_j.
        """
        total = self.total_multiplicity()
        if total < 1e-10:
            return 0.0
        s = 0.0
        for c in self.compartments:
            p = c.n_partons / total
            if p > 1e-15:
                s -= p * math.log(p)
        return s

    def get_multiplicities(self) -> List[float]:
        """Return list of multiplicities."""
        return [c.n_partons for c in self.compartments]



# ===========================================================================
# Section 2: Viability scoring (from 1279_AlbertoAlaldu_ppa_app_systemic_reduction)
# ===========================================================================

class EventViability:
    """
    Compute the viability score for an event based on analysis cuts.

    V = prod_c s_c

    where s_c in [0, 1] is the score for cut c. An event is accepted
    if V > threshold (typically 0.5).
    """

    def __init__(self):
        self.cuts: Dict[str, Tuple[float, float, float, float]] = {}
        # Each cut: name -> (min_val, safe_min, safe_max, max_val)

    def add_cut(self, name: str, min_val: float, safe_min: float,
                safe_max: float, max_val: float):
        """
        Add a cut with a sigmoid-like transition:
            s = 1 inside [safe_min, safe_max]
            s -> 0 as value -> min_val or max_val
        """
        self.cuts[name] = (min_val, safe_min, safe_max, max_val)

    def score_cut(self, name: str, value: float) -> float:
        """Compute the score for a single cut."""
        if name not in self.cuts:
            return 1.0
        mn, smin, smax, mx = self.cuts[name]
        # Lower bound score
        if value < mn:
            return 0.0
        if value < smin:
            s_low = (value - mn) / (smin - mn + 1e-12)
        else:
            s_low = 1.0
        # Upper bound score
        if value > mx:
            return 0.0
        if value > smax:
            s_high = (mx - value) / (mx - smax + 1e-12)
        else:
            s_high = 1.0
        return s_low * s_high

    def compute_viability(self, values: Dict[str, float]) -> float:
        """Compute total viability V = prod s_c."""
        v = 1.0
        for name in self.cuts:
            val = values.get(name, 0.0)
            v *= self.score_cut(name, val)
        return v


def build_default_event_cuts(pt_min: float = 20.0, eta_max: float = 5.0,
                             mjj_min: float = 100.0) -> EventViability:
    """
    Build a default set of analysis cuts for a dijet + X search:
        - pT of each jet > pt_min (GeV)
        - |eta| of each jet < eta_max
        - Invariant mass of dijet system > mjj_min (GeV)
    """
    ev = EventViability()
    ev.add_cut('pt_jet1', 0.0, pt_min, 1000.0, 5000.0)
    ev.add_cut('pt_jet2', 0.0, pt_min, 1000.0, 5000.0)
    ev.add_cut('eta_jet1', -10.0, -eta_max, eta_max, 10.0)
    ev.add_cut('eta_jet2', -10.0, -eta_max, eta_max, 10.0)
    ev.add_cut('mjj', 0.0, mjj_min, 5000.0, 10000.0)
    return ev


# ===========================================================================
# Section 3: Levels / contour extraction (from 667_levels)
# ===========================================================================

def levels_extract(func: callable, level_num: int,
                   x_range: Tuple[float, float],
                   y_range: Tuple[float, float],
                   data_num: int = 51, seed: int = 42
                   ) -> List[float]:
    """
    Extract contour levels from a 2D function by random sampling.

    Evaluates f(x, y) on a regular grid and selects `level_num` random
    function values as contour levels.

    In the physics context: f could be the cross-section as a function
    of two scales (mu_R, mu_F), and the contour levels represent
    iso-cross-section curves used for scale uncertainty estimation.
    """
    rng = random.Random(seed)
    x_vals = [x_range[0] + i * (x_range[1] - x_range[0]) / (data_num - 1)
              for i in range(data_num)]
    y_vals = [y_range[0] + i * (y_range[1] - y_range[0]) / (data_num - 1)
              for i in range(data_num)]
    # Evaluate on grid
    f_vals = []
    for xi in x_vals:
        for yi in y_vals:
            try:
                f_vals.append(func(xi, yi))
            except (ValueError, ZeroDivisionError):
                f_vals.append(0.0)
    # Select random levels
    levels = []
    for _ in range(level_num):
        idx = rng.randint(0, len(f_vals) - 1)
        levels.append(f_vals[idx])
    return sorted(set(levels))
