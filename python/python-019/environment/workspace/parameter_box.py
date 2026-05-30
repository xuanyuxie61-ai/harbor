
import numpy as np


class ParameterBox:
    def __init__(self, bounds):
        self.bounds = [tuple(b) for b in bounds]
        self.dim = len(bounds)
        for lo, hi in self.bounds:
            if lo >= hi:
                raise ValueError("Each bound must satisfy min < max.")

    def volume(self):
        vol = 1.0
        for lo, hi in self.bounds:
            vol *= (hi - lo)
        return vol

    def center(self):
        return np.array([(lo + hi) / 2.0 for lo, hi in self.bounds])

    def contains(self, point):
        point = np.asarray(point)
        if point.shape[0] != self.dim:
            return False
        for i, (lo, hi) in enumerate(self.bounds):
            if not (lo <= point[i] <= hi):
                return False
        return True

    def random_point(self, seed=None):
        rng = np.random.default_rng(seed)
        return np.array([lo + rng.random() * (hi - lo) for lo, hi in self.bounds])

    def grid(self, n_points_per_dim):
        if isinstance(n_points_per_dim, int):
            n_points_per_dim = [n_points_per_dim] * self.dim
        axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(self.bounds, n_points_per_dim)]
        mesh = np.meshgrid(*axes, indexing='ij')
        grid = np.stack([m.ravel() for m in mesh], axis=1)
        return grid

    def subdivide(self):
        sub_boxes = []

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
    leaf_boxes = []

    def recurse(box, level):
        center = box.center()
        delta_center = abs(discriminant_func(center))

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
