
import numpy as np
from spatial_mesh import TetMesh


class EnvironmentField:

    def __init__(self, mesh: TetMesh, nodal_values: np.ndarray):
        self.mesh = mesh
        self.nodal_values = np.asarray(nodal_values, dtype=float)
        if self.nodal_values.shape[0] != mesh.nodes.shape[0]:
            raise ValueError("Nodal values length must match number of mesh nodes.")

    def evaluate(self, p: np.ndarray):
        return self.mesh.interpolate_nodal_field(p, self.nodal_values)

    def gradient(self, p: np.ndarray, eps: float = 1e-5):
        p = np.asarray(p, dtype=float)
        grad = np.zeros(3, dtype=float)
        f0 = self.evaluate(p)
        if np.isnan(f0):
            return grad
        for i in range(3):
            p_plus = p.copy()
            p_plus[i] += eps
            f_plus = self.evaluate(p_plus)
            p_minus = p.copy()
            p_minus[i] -= eps
            f_minus = self.evaluate(p_minus)
            if not (np.isnan(f_plus) or np.isnan(f_minus)):
                grad[i] = (f_plus - f_minus) / (2.0 * eps)
        return grad


def generate_gradient_field(mesh: TetMesh, direction: np.ndarray, magnitude: float = 1.0):
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-14)
    vals = magnitude * mesh.nodes.dot(direction)
    return EnvironmentField(mesh, vals)


def generate_gaussian_bump_field(mesh: TetMesh, center: np.ndarray, sigma: float = 0.2, amplitude: float = 1.0):
    center = np.asarray(center, dtype=float)
    diffs = mesh.nodes - center[np.newaxis, :]
    dists2 = np.sum(diffs ** 2, axis=1)
    vals = amplitude * np.exp(-dists2 / (2.0 * sigma ** 2))
    return EnvironmentField(mesh, vals)


def sample_field_at_positions(field: EnvironmentField, positions: np.ndarray):
    positions = np.asarray(positions, dtype=float)
    values = np.array([field.evaluate(p) for p in positions], dtype=float)
    return values
