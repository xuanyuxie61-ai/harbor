"""
parameter_box.py
----------------
Parameter-space bounding, discretization, and region-filling utilities.

Adapted from seed project 1377_usa_box_plot (box filling).

Scientific Background
=====================
When mapping exceptional-point manifolds in a multi-dimensional
parameter space, it is essential to define a bounding box (hyper-rectangle)
that contains the region of interest. The box is then subdivided into
smaller cells for systematic or adaptive searches.

For a d-dimensional parameter space with bounds [a_i, b_i], the volume is

    V = Π_{i=1}^{d} (b_i - a_i).

A uniform grid with N_i points along dimension i has cell volume

    V_cell = V / (Π_i N_i).

Adaptive refinement concentrates cells near regions where the discriminant
|Δ| is small, using octree-like subdivision.
"""

import numpy as np


class ParameterBox:
    """
    Represents a d-dimensional axis-aligned parameter box.
    """
    def __init__(self, bounds):
        """
        Parameters
        ----------
        bounds : list of tuple
            [(min_1, max_1), (min_2, max_2), ...]
        """
        self.bounds = [tuple(b) for b in bounds]
        self.dim = len(bounds)
        for lo, hi in self.bounds:
            if lo >= hi:
                raise ValueError("Each bound must satisfy min < max.")

    def volume(self):
        """
        Hyper-volume of the box.
        """
        vol = 1.0
        for lo, hi in self.bounds:
            vol *= (hi - lo)
        return vol

    def center(self):
        """
        Center point of the box.
        """
        return np.array([(lo + hi) / 2.0 for lo, hi in self.bounds])

    def contains(self, point):
        """
        Check whether a point lies inside the box (inclusive).
        """
        point = np.asarray(point)
        if point.shape[0] != self.dim:
            return False
        for i, (lo, hi) in enumerate(self.bounds):
            if not (lo <= point[i] <= hi):
                return False
        return True

    def random_point(self, seed=None):
        """
        Sample a uniform random point inside the box.
        """
        rng = np.random.default_rng(seed)
        return np.array([lo + rng.random() * (hi - lo) for lo, hi in self.bounds])

    def grid(self, n_points_per_dim):
        """
        Generate a uniform Cartesian grid inside the box.

        Parameters
        ----------
        n_points_per_dim : int or list of int

        Returns
        -------
        grid : ndarray, shape (N_total, dim)
        """
        if isinstance(n_points_per_dim, int):
            n_points_per_dim = [n_points_per_dim] * self.dim
        axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(self.bounds, n_points_per_dim)]
        mesh = np.meshgrid(*axes, indexing='ij')
        grid = np.stack([m.ravel() for m in mesh], axis=1)
        return grid

    def subdivide(self):
        """
        Split the box into 2^d sub-boxes by bisecting each dimension.

        Returns
        -------
        sub_boxes : list of ParameterBox
        """
        sub_boxes = []
        # Generate all combinations of lower/upper halves
        for bits in range(2 ** self.dim):
            new_bounds = []
            for i in range(self.dim):
                lo, hi = self.bounds[i]
                mid = (lo + hi) / 2.0
                if (bits >> i) & 1:
                    new_bounds.append((mid, hi))
                else:
                    new_bounds.append((lo, mid))
            sub_boxes.append(ParameterBox(new_bounds))
        return sub_boxes


def adaptive_box_refinement(discriminant_func, initial_box, max_level=6, threshold=1e-3):
    """
    Recursively subdivide parameter boxes, keeping only those that
    contain or are near exceptional points (|Δ| < threshold).

    Parameters
    ----------
    discriminant_func : callable
        f(point_array) -> array of discriminant magnitudes.
    initial_box : ParameterBox
    max_level : int
        Maximum recursion depth.
    threshold : float

    Returns
    -------
    leaf_boxes : list of ParameterBox
        Boxes at the finest level that are near EPs.
    """
    leaf_boxes = []

    def recurse(box, level):
        center = box.center()
        delta_center = abs(discriminant_func(center))
        # Also sample corners for robustness
        corners = []
        for bits in range(2 ** box.dim):
            corner = []
            for i in range(box.dim):
                lo, hi = box.bounds[i]
                corner.append(hi if (bits >> i) & 1 else lo)
            corners.append(corner)
        corners = np.array(corners)
        deltas = np.abs(discriminant_func(corners))
        min_delta = min(delta_center, deltas.min())

        if min_delta < threshold or level >= max_level:
            if min_delta < threshold:
                leaf_boxes.append(box)
            return

        for sub in box.subdivide():
            recurse(sub, level + 1)

    recurse(initial_box, 0)
    return leaf_boxes
