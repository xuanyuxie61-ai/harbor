
import numpy as np
from math import factorial
from scipy.spatial import Delaunay
from typing import List, Tuple


class StateSpaceTriangulation:

    def __init__(self, states: np.ndarray):
        self.states = np.asarray(states, dtype=float)
        if self.states.ndim == 1:
            self.states = self.states.reshape(-1, 1)
        self.N, self.d = self.states.shape
        if self.N < self.d + 1:
            raise ValueError("StateSpaceTriangulation: not enough points")
        self.tri = Delaunay(self.states)
        self.simplices = self.tri.simplices

    def find_simplex(self, point: np.ndarray) -> int:
        return int(self.tri.find_simplex(point))

    def barycentric_coordinates(self, point: np.ndarray, simplex_idx: int) -> np.ndarray:
        if simplex_idx < 0 or simplex_idx >= len(self.simplices):
            return None
        vertices = self.states[self.simplices[simplex_idx]]

        M = np.vstack([vertices.T, np.ones(self.d + 1)])
        rhs = np.append(point, 1.0)
        try:
            lam = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return lam

    def interpolate(self, point: np.ndarray, values: np.ndarray) -> float:
        idx = self.find_simplex(point)
        if idx < 0:

            dists = np.linalg.norm(self.states - point, axis=1)
            return float(values[np.argmin(dists)])
        lam = self.barycentric_coordinates(point, idx)
        if lam is None:
            dists = np.linalg.norm(self.states - point, axis=1)
            return float(values[np.argmin(dists)])
        verts = self.simplices[idx]
        return float(np.dot(lam, values[verts]))

    def simplex_volumes(self) -> np.ndarray:
        vols = []
        for simp in self.simplices:
            verts = self.states[simp]

            M = np.zeros((self.d, self.d))
            for i in range(self.d):
                M[:, i] = verts[i + 1] - verts[0]
            vol = abs(np.linalg.det(M)) / factorial(self.d)
            vols.append(vol)
        return np.array(vols)


def adaptive_mesh_refinement(states: np.ndarray, values: np.ndarray,
                              threshold: float = 0.1, max_points: int = 500) -> np.ndarray:
    points = np.asarray(states, dtype=float).copy()
    vals = np.asarray(values, dtype=float).copy()
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    for _ in range(20):
        if len(points) >= max_points:
            break
        tri = StateSpaceTriangulation(points)
        new_points = []
        for simp in tri.simplices:
            verts = points[simp]
            vvals = vals[simp]

            max_diff = np.max(vvals) - np.min(vvals)
            diam = np.max([np.linalg.norm(verts[i] - verts[j])
                           for i in range(len(verts)) for j in range(i + 1, len(verts))])
            if diam < 1.0e-10:
                continue
            grad_est = max_diff / diam
            if grad_est > threshold:
                centroid = np.mean(verts, axis=0)
                new_points.append(centroid)
        if len(new_points) == 0:
            break

        new_points = np.array(new_points)

        kept = []
        for p in new_points:
            if len(points) == 0 or np.min(np.linalg.norm(points - p, axis=1)) > 1.0e-6:
                kept.append(p)
                if len(kept) + len(points) >= max_points:
                    break
        if len(kept) == 0:
            break
        kept = np.array(kept)
        points = np.vstack([points, kept])

        for i in range(len(kept)):
            dists = np.linalg.norm(points[:len(points) - len(kept) + i] - kept[i], axis=1)
            vals = np.append(vals, vals[np.argmin(dists)])
    return points
