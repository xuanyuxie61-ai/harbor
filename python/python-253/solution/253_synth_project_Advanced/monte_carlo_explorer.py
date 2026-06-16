"""
monte_carlo_explorer.py — Monte Carlo sampling for parameter-space exploration.

Gravitational-wave template banks must cover the astrophysical parameter
space (component masses, spins, sky location) with a minimal mismatch.
This module implements:

  * A "card-dealing" randomisation inspired by the clock-solitaire seed
    project: the parameter space is divided into 13 "piles" (one per
    mass-ratio bin), and a "game" consists of drawing parameters from
    each pile in an order determined by a permutation.
  * Random sampling of the binary parameter space.
  * Mismatch evaluation between pairs of waveforms.
  * Template-bank coverage statistics (fraction of parameter space
    with mismatch < threshold).
  * Seeded reproducibility for small-scale experiments.
"""

from __future__ import annotations
import math
import random
import numpy as np
from typing import List, Tuple, Dict, Callable

from physics_constants import BinaryParameters


# ---------------------------------------------------------------------------
#  Card-dealing permutation sampler (from clock_solitaire)
# ---------------------------------------------------------------------------
class PileDeck:
    """A deck of N cards dealt into 13 piles of 4 cards each.

    Mirrors the clock-solitaire card-dealing structure: a permutation
    of 1..N is reshaped into a matrix (n_piles, cards_per_pile),
    negated to indicate "face down", and the simulation draws cards
    by following the pile index given by each card's rank.
    """

    def __init__(self, n_piles: int = 13, cards_per_pile: int = 4,
                 seed: int = 42):
        self.n_piles = n_piles
        self.cards_per_pile = cards_per_pile
        self.rng = random.Random(seed)

    def deal(self) -> np.ndarray:
        """Return a freshly shuffled pile matrix of shape (n_piles, cards_per_pile)."""
        N = self.n_piles * self.cards_per_pile
        perm = self.rng.sample(range(1, N + 1), N)
        piles = np.array(perm).reshape(self.n_piles, self.cards_per_pile)
        return -piles  # face down


# ---------------------------------------------------------------------------
#  Parameter-space bins
# ---------------------------------------------------------------------------
def mass_ratio_bins(n_bins: int = 13) -> List[Tuple[float, float]]:
    """Return n_bins contiguous mass-ratio intervals in (0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return [(float(edges[k]), float(edges[k + 1])) for k in range(n_bins)]


def chirp_mass_bins(n_bins: int = 8, Mc_min: float = 5.0,
                    Mc_max: float = 100.0) -> List[Tuple[float, float]]:
    """Return log-spaced chirp-mass bins."""
    edges = np.geomspace(Mc_min, Mc_max, n_bins + 1)
    return [(float(edges[k]), float(edges[k + 1])) for k in range(n_bins)]


# ---------------------------------------------------------------------------
#  Random binary parameter generator
# ---------------------------------------------------------------------------
def sample_binary(seed: int = 42,
                  m_range: Tuple[float, float] = (5.0, 80.0),
                  chi_range: Tuple[float, float] = (-0.9, 0.9)) -> BinaryParameters:
    """Draw a random binary-black-hole parameter set with uniform priors."""
    rng = random.Random(seed)
    m1 = rng.uniform(*m_range)
    m2 = rng.uniform(*m_range)
    if m2 > m1:
        m1, m2 = m2, m1
    chi1 = rng.uniform(*chi_range)
    chi2 = rng.uniform(*chi_range)
    return BinaryParameters(m1_si=m1, m2_si=m2, chi1=chi1, chi2=chi2,
                            f_low_hz=20.0, distance_mpc=100.0, ell=2)


# ---------------------------------------------------------------------------
#  Mismatch function
# ---------------------------------------------------------------------------
def mismatch(h1: np.ndarray, h2: np.ndarray,
             dt: float = 1.0 / 4096.0) -> float:
    """Compute the noise-agnostic mismatch between two waveforms.

        mismatch = 1 - max_{t_c, phi_c} <h1, h2> / sqrt(<h1,h1> <h2,h2>).
    The maximisation over time and phase shifts is done by FFT cross-correlation.
    """
    N = min(len(h1), len(h2))
    a = h1[:N]
    b = h2[:N]
    na = np.sqrt(np.sum(a * a))
    nb = np.sqrt(np.sum(b * b))
    if na < 1.0e-300 or nb < 1.0e-300:
        return 1.0
    A = np.fft.rfft(a)
    B = np.fft.rfft(b)
    cc = np.fft.irfft(A * np.conj(B), n=N)
    max_cc = float(np.max(np.abs(cc)))
    overlap = max_cc / (na * nb)
    return max(0.0, 1.0 - overlap)


# ---------------------------------------------------------------------------
#  Template-bank coverage test
# ---------------------------------------------------------------------------
def coverage_fraction(n_templates: int, n_signals: int,
                      mismatch_threshold: float = 0.03,
                      seed: int = 42) -> Dict[str, float]:
    """Monte Carlo estimate of the template-bank coverage fraction.

    Draws n_templates random template waveforms and n_signals random
    "injection" signals; reports the fraction of injections whose minimum
    mismatch to any template is below mismatch_threshold.
    """
    rng = random.Random(seed)
    # Generate template parameters
    templates = [sample_binary(seed=seed + k) for k in range(n_templates)]
    # Generate injections
    injections = [sample_binary(seed=seed + n_templates + k)
                  for k in range(n_signals)]
    # Mismatch matrix (small-scale)
    n_covered = 0
    for j, inj in enumerate(injections):
        # use chirp mass difference as a cheap mismatch proxy
        mc_inj = inj.chirp_mass
        min_dm = min(abs(mc_inj - t.chirp_mass) / max(t.chirp_mass, 1.0)
                     for t in templates)
        # map dm to mismatch via a simple quadratic model
        approx_mm = min(1.0, 10.0 * min_dm ** 2)
        if approx_mm < mismatch_threshold:
            n_covered += 1
    frac = n_covered / max(n_signals, 1)
    return {
        "n_templates": n_templates,
        "n_signals": n_signals,
        "threshold": mismatch_threshold,
        "covered": n_covered,
        "coverage_fraction": frac,
    }


# ---------------------------------------------------------------------------
#  Clock-solitaire-style parameter sweep
# ---------------------------------------------------------------------------
def clock_sweep(n_rounds: int = 52, seed: int = 42) -> Dict[str, object]:
    """Perform a clock-solitaire-style random parameter sweep.

    We deal a 13x4 deck, then draw cards following the solitaire rule:
    the rank of each card determines the next pile.  Each card is mapped
    to a random binary parameter set, and we record the sequence of
    chirp masses drawn.
    """
    deck = PileDeck(n_piles=13, cards_per_pile=4, seed=seed)
    pile = deck.deal()
    pile = pile.copy()  # make it mutable
    sequence = []
    k = 0
    rng = random.Random(seed)
    # pick starting pile
    i = rng.randint(0, 12)
    max_steps = min(n_rounds, 52)
    while k < max_steps:
        j = pile[i, 0]
        if j > 0:
            break
        pile[i, 0] = -j
        # rotate
        pile[i] = np.roll(pile[i], -1)
        # map card to a chirp mass
        mc = 5.0 + 95.0 * ((abs(j) - 1) / 52.0)
        sequence.append(mc)
        k += 1
        # next pile = card rank (1-based) mod 13
        i = (abs(j) - 1) % 13
    return {
        "steps_taken": k,
        "chirp_masses": sequence,
        "mean_mc": float(np.mean(sequence)) if sequence else 0.0,
        "std_mc": float(np.std(sequence)) if sequence else 0.0,
    }
