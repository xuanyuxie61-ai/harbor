"""
shock_quantization.py
=====================

K-means quantization of the Mach number field for automatic shock
detection in the compressible-Euler simulation.

The seed project 583_image_quantization uses MATLAB's k-means to reduce
the number of gray levels in an image.  We lift this technique to the
3-D Mach number field M(x) = |v(x)| / c_s(x): by clustering the cells
into K discrete "Mach states" we obtain a piecewise-constant
approximation that cleanly separates shocks (high-Mach cells) from
smooth flow (low-Mach cells).

Key mappings
------------
* Image pixel intensities     -> cell Mach numbers
* K grayscale shades          -> K Mach states (subsonic, transonic,
                                  supersonic, hypersonic, ...)
* K-means clustering          -> optimal partition of the Mach field
* Representative centroids    -> characteristic Mach number of each
                                  shock-intensity class

Algorithm
---------
Given a Mach field {M_i}_{i=1}^{N}, find centroids {c_k}_{k=1}^K and
assignments {a_i in {1..K}} minimising

    J = sum_{i=1}^{N} (M_i - c_{a_i})^2.

We use Lloyd's algorithm with k-means++ initialisation:

    1. Initialise c_1 = M_{uniform random}, c_k = M_j with
       P(j) proportional to min_{l < k} |M_j - c_l|^2.
    2. Assign: a_i = argmin_k |M_i - c_k|^2.
    3. Update: c_k = mean_{a_i = k}(M_i).
    4. Repeat 2-3 until assignments do not change.

Physical interpretation
-----------------------
For galaxy-formation hydrodynamics the Mach states map to physical
regimes:

    M < 0.5       : subsonic / incompressible-like
    0.5 < M < 1.0 : transonic
    1.0 < M < 3.0 : mildly supersonic (typical ISM turbulence)
    3.0 < M < 10  : supersonic (SNR-driven shells, accretion shocks)
    M > 10        : hypersonic (relativistic jets, merger shocks)

Quantization preserves the mass-weighted Mach number distribution,
which enters the turbulent dissipation rate

    epsilon ~ rho sigma^3 / L   (for Kolmogorov)

through the centroid values.

References
----------
- Lloyd, S. P. 1982, IEEE Trans. Inform. Theory 28, 129
- Arthur, D., & Vassilvitskii, S. 2007, SODA '07, 1027 (k-means++)
- Ryu, D., et al. 2003, ApJ 593, 599 (cosmological shock detection)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Optional, Dict

from astro_constants import DOMAIN_SIZE_KPC, KILOPARSEC_CGS


# =====================================================================
#                       K-MEANS UTILITIES
# =====================================================================

def kmeans_pp_init(data: np.ndarray, k: int,
                    rng: Optional[np.random.Generator] = None
                    ) -> np.ndarray:
    """
    K-means++ initialisation (Arthur & Vassilvitskii 2007).

    Pick the first centroid uniformly at random, then each subsequent
    centroid c_k from the remaining data with probability

        P(x) proportional to min_{l < k} |x - c_l|^2.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = data.size
    if k > n:
        k = n
    centroids = np.empty(k)
    idx0 = rng.integers(n)
    centroids[0] = data[idx0]
    for m in range(1, k):
        d2 = np.array([min((x - centroids[l]) ** 2 for l in range(m))
                       for x in data])
        d2_sum = d2.sum()
        if d2_sum < 1.0e-60:
            centroids[m] = data[rng.integers(n)]
            continue
        probs = d2 / d2_sum
        idx = rng.choice(n, p=probs)
        centroids[m] = data[idx]
    return centroids


def kmeans_1d(data: np.ndarray, k: int, max_iter: int = 200,
               tol: float = 1.0e-6, seed: int = 42
               ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    1-D Lloyd's k-means with k-means++ initialisation.

    Returns (centroids_sorted, assignments, n_iter).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = data.size
    k = min(k, n)
    if k <= 0:
        raise ValueError("k must be positive")
    centroids = kmeans_pp_init(data, k, rng)
    assign = np.zeros(n, dtype=int)
    n_iter = 0
    for _ in range(max_iter):
        # assignment step
        new_assign = np.argmin(
            np.abs(data[:, None] - centroids[None, :]), axis=1
        )
        n_iter += 1
        if np.array_equal(new_assign, assign) and n_iter > 1:
            break
        assign = new_assign
        # update step
        for m in range(k):
            members = data[assign == m]
            if members.size > 0:
                centroids[m] = members.mean()
    # sort centroids
    order = np.argsort(centroids)
    centroids_sorted = centroids[order]
    remap = {old: new for new, old in enumerate(order)}
    assign_sorted = np.array([remap[a] for a in assign])
    return centroids_sorted, assign_sorted, n_iter


# =====================================================================
#                     MACH-NUMBER QUANTIZATION
# =====================================================================

class ShockQuantizer:
    """
    Quantize a 3-D Mach number field into K discrete shock-intensity
    classes and report the mass fraction in each class.

    The K classes are ordered by centroid Mach number, so class 0 is
    the most subsonic and class K-1 is the most supersonic.
    """

    def __init__(self, k: int = 5, seed: int = 42) -> None:
        self.k = k
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.assignments: Optional[np.ndarray] = None
        self.n_iter = 0
        self.mass_fractions: Optional[np.ndarray] = None

    def fit(self, mach_3d: np.ndarray,
            rho_3d: Optional[np.ndarray] = None) -> "ShockQuantizer":
        """
        Compute the k-means partition of the Mach field.

        Parameters
        ----------
        mach_3d : ndarray
            3-D array of cell Mach numbers  M = |v| / c_s.
        rho_3d : ndarray, optional
            3-D density array for mass-weighted class fractions.
        """
        flat = mach_3d.flatten()
        # clip outliers for robustness (Mach numbers above 100 are rare)
        flat_clipped = np.clip(flat, 0.0, 100.0)
        centroids, assign, n_iter = kmeans_1d(
            flat_clipped, self.k, seed=self.seed
        )
        self.centroids = centroids
        self.assignments = assign.reshape(mach_3d.shape)
        self.n_iter = n_iter
        if rho_3d is not None:
            rho_flat = rho_3d.flatten()
            total_mass = np.sum(rho_flat)
            if total_mass > 0:
                self.mass_fractions = np.array([
                    np.sum(rho_flat[assign == m]) / total_mass
                    for m in range(self.k)
                ])
            else:
                self.mass_fractions = np.zeros(self.k)
        else:
            self.mass_fractions = np.array([
                np.sum(assign == m) / assign.size for m in range(self.k)
            ])
        return self

    def classify(self, mach_3d: np.ndarray) -> np.ndarray:
        """Assign new Mach values to the existing centroid classes."""
        if self.centroids is None:
            raise RuntimeError("must call fit() first")
        flat = mach_3d.flatten()
        assign = np.argmin(
            np.abs(flat[:, None] - self.centroids[None, :]), axis=1
        )
        return assign.reshape(mach_3d.shape)

    # -----------------------------------------------------------------
    #  physical labelling
    # -----------------------------------------------------------------
    def physical_labels(self) -> List[str]:
        """Assign human-readable labels to the centroids."""
        labels = []
        for c in self.centroids:
            if c < 0.5:
                labels.append("subsonic")
            elif c < 1.0:
                labels.append("transonic")
            elif c < 3.0:
                labels.append("mildly_supersonic")
            elif c < 10.0:
                labels.append("supersonic")
            else:
                labels.append("hypersonic")
        return labels

    def shock_mask(self, mach_threshold: float = 1.5) -> np.ndarray:
        """Return a boolean mask of cells classified as shocked (M > threshold)."""
        if self.centroids is None:
            raise RuntimeError("must call fit() first")
        shocked_classes = [m for m, c in enumerate(self.centroids)
                           if c >= mach_threshold]
        mask = np.isin(self.assignments, shocked_classes)
        return mask

    # -----------------------------------------------------------------
    #  reporting
    # -----------------------------------------------------------------
    def summary(self) -> str:
        if self.centroids is None:
            return "ShockQuantizer not fitted"
        labels = self.physical_labels()
        lines = [
            f"ShockQuantizer summary (K={self.k}, n_iter={self.n_iter}):",
            "  class    centroid     mass_fraction   regime",
            "  -----    --------     -------------   ------",
        ]
        for m, c in enumerate(self.centroids):
            mf = (self.mass_fractions[m] if self.mass_fractions is not None
                  else float("nan"))
            lines.append(f"  {m:5d}    {c:8.4f}     {mf:13.4f}    {labels[m]}")
        return "\n".join(lines)


# =====================================================================
#                  MACH FIELD COMPUTATION
# =====================================================================

def mach_field(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                rho: np.ndarray, p: np.ndarray,
                gamma: float = 5.0 / 3.0) -> np.ndarray:
    """
    Compute the Mach number field:

        M(x) = |v(x)| / c_s(x),     c_s = sqrt(gamma P / rho).
    """
    cs = np.sqrt(gamma * p / np.maximum(rho, 1.0e-60))
    v_abs = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return v_abs / np.maximum(cs, 1.0e-60)
