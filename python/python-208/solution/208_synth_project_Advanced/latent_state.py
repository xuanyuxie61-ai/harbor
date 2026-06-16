"""
latent_state.py
===============

Four-dimensional latent-state representation of the multi-fidelity UQ
system state.  Adapted from `xyzl_display` of project 1426, which
handles point and line data in 3D + scalar tag.

Purpose
-------
We embed each training sample (xi, y, level, cost) into a 4-D latent
space

    z = (z_1, z_2, z_3, z_4) in R^4

where
  z_1, z_2, z_3 = first 3 principal components of xi (after whitening)
  z_4           = normalized log-cost of the fidelity level

This gives a unified geometric representation of the UQ system state
that can be used for:
  - nearest-neighbor queries in the adaptive sampler,
  - diversity penalties in batch selection,
  - diagnostic summaries of the sample distribution.

The `XYZL` data structure (from project 1426) stores:
  - N points in 4-D,
  - M "lines" (ordered lists of point indices) that form connected
    trajectories through the state-space trajectory of the adaptive
    sampling loop.

We provide no visualization (per project requirements); the data
structure is kept for algorithmic use only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Latent point in 4-D.
# ----------------------------------------------------------------------
@dataclass
class LatentPoint:
    """A point in the 4-D latent state space."""
    z1: float
    z2: float
    z3: float
    z4: float      # fidelity-cost coordinate
    label: int = 0   # fidelity level
    tag: str = ""    # sample name


# ----------------------------------------------------------------------
# 4-D latent cloud with lines (XYZL format).
# ----------------------------------------------------------------------
@dataclass
class LatentCloud:
    """A point + line cloud in 4-D latent space (XYZL-style)."""
    points: List[LatentPoint] = field(default_factory=list)
    lines: List[List[int]] = field(default_factory=list)

    def add_point(self, p: LatentPoint) -> int:
        idx = len(self.points)
        self.points.append(p)
        return idx

    def add_line(self, indices: List[int]) -> None:
        for i in indices:
            if i < 0 or i >= len(self.points):
                raise ValueError("add_line: index out of range.")
        self.lines.append(list(indices))

    def n_points(self) -> int:
        return len(self.points)

    def n_lines(self) -> int:
        return len(self.lines)


# ----------------------------------------------------------------------
# Projection from (xi, y, level, cost) to latent z in R^4.
# ----------------------------------------------------------------------
def project_to_latent(
    xi: List[float],
    y: float,
    level: int,
    cost: float,
    W,                  # WhiteningTransform from coordinate_transform
    n_levels: int = 4,
) -> LatentPoint:
    """Project a sample (xi, y, level, cost) into 4-D latent space.

    z_1, z_2, z_3 = first 3 components of whitened xi.
    z_4           = log10(cost + 1) / log10(max_cost)  [normalized 0..1].
    """
    from coordinate_transform import encode
    eta = encode(xi, W)
    z1 = eta[0] if len(eta) > 0 else 0.0
    z2 = eta[1] if len(eta) > 1 else 0.0
    z3 = eta[2] if len(eta) > 2 else 0.0
    # Normalize cost to [0, 1] using a log scale.
    max_cost_exp = 3.0  # assume max cost ~ 10^3
    z4 = math.log10(max(cost, 1.0) + 1.0) / max_cost_exp
    z4 = min(max(z4, 0.0), 1.0)
    return LatentPoint(z1=z1, z2=z2, z3=z3, z4=z4, label=level)


# ----------------------------------------------------------------------
# Distance in latent space.
# ----------------------------------------------------------------------
def latent_distance(a: LatentPoint, b: LatentPoint) -> float:
    """Euclidean distance in 4-D latent space."""
    return math.sqrt(
        (a.z1 - b.z1) ** 2 +
        (a.z2 - b.z2) ** 2 +
        (a.z3 - b.z3) ** 2 +
        (a.z4 - b.z4) ** 2
    )


def nearest_neighbor(query: LatentPoint, cloud: LatentCloud) -> Tuple[int, float]:
    """Find the nearest neighbor to `query` in `cloud`."""
    if not cloud.points:
        raise ValueError("nearest_neighbor: empty cloud.")
    best_i = 0
    best_d = latent_distance(query, cloud.points[0])
    for i, p in enumerate(cloud.points):
        d = latent_distance(query, p)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


# ----------------------------------------------------------------------
# Bounding box of the latent cloud.
# ----------------------------------------------------------------------
def latent_bounding_box(
    cloud: LatentCloud,
) -> Tuple[List[float], List[float]]:
    """Compute the axis-aligned bounding box of the latent cloud."""
    if not cloud.points:
        return [0.0] * 4, [0.0] * 4
    lo = [cloud.points[0].z1, cloud.points[0].z2,
          cloud.points[0].z3, cloud.points[0].z4]
    hi = list(lo)
    for p in cloud.points[1:]:
        coords = [p.z1, p.z2, p.z3, p.z4]
        for k in range(4):
            if coords[k] < lo[k]:
                lo[k] = coords[k]
            if coords[k] > hi[k]:
                hi[k] = coords[k]
    return lo, hi


# ----------------------------------------------------------------------
# Summary of the latent cloud.
# ----------------------------------------------------------------------
def latent_summary(cloud: LatentCloud) -> Dict:
    """Diagnostic summary of the latent cloud."""
    if not cloud.points:
        return {"n_points": 0, "n_lines": 0}
    lo, hi = latent_bounding_box(cloud)
    return {
        "n_points": cloud.n_points(),
        "n_lines": cloud.n_lines(),
        "bbox_lo": lo,
        "bbox_hi": hi,
        "bbox_diag": math.sqrt(sum((hi[k] - lo[k]) ** 2 for k in range(4))),
    }
