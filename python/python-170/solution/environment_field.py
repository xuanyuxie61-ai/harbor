"""
environment_field.py
====================
Finite-element interpolation of environmental scalar fields.

Incorporates:
  - fem_to_tec / fem_read (from 380_fem_to_tec)

Scientific role:
  The swarm operates in an environment characterized by scalar fields
  such as temperature T(x), chemical concentration c(x), or illumination
  I(x). These fields are defined on the tetrahedral mesh and interpolated
  to arbitrary robot positions using finite-element shape functions (linear
  Lagrange basis on tetrahedra). The gradient of the field drives the
  chemotaxis / thermotaxis behavior of individual robots.

  The FEM framework provides:
    u_h(x) = sum_j u_j * phi_j(x)
    grad(u_h)|_{T} = sum_j u_j * grad(phi_j)|_{T}
  where phi_j are the piecewise-linear hat functions.
"""

import numpy as np
from spatial_mesh import TetMesh


class EnvironmentField:
    """
    Scalar environmental field defined on a tetrahedral mesh.
    """

    def __init__(self, mesh: TetMesh, nodal_values: np.ndarray):
        """
        Parameters
        ----------
        mesh : TetMesh
        nodal_values : ndarray, shape (N_n,)
            Scalar value at each node.
        """
        self.mesh = mesh
        self.nodal_values = np.asarray(nodal_values, dtype=float)
        if self.nodal_values.shape[0] != mesh.nodes.shape[0]:
            raise ValueError("Nodal values length must match number of mesh nodes.")

    def evaluate(self, p: np.ndarray):
        """
        Evaluate field at point p via barycentric interpolation.

        Parameters
        ----------
        p : ndarray, shape (3,)

        Returns
        -------
        value : float
            NaN if outside mesh.
        """
        return self.mesh.interpolate_nodal_field(p, self.nodal_values)

    def gradient(self, p: np.ndarray, eps: float = 1e-5):
        """
        Approximate gradient at point p by finite differences.

        Parameters
        ----------
        p : ndarray, shape (3,)
        eps : float
            Perturbation size.

        Returns
        -------
        grad : ndarray, shape (3,)
        """
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
    """
    Generate a linear scalar field phi(x) = magnitude * <direction, x>
    on the mesh nodes.

    Parameters
    ----------
    mesh : TetMesh
    direction : ndarray, shape (3,)
    magnitude : float

    Returns
    -------
    field : EnvironmentField
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-14)
    vals = magnitude * mesh.nodes.dot(direction)
    return EnvironmentField(mesh, vals)


def generate_gaussian_bump_field(mesh: TetMesh, center: np.ndarray, sigma: float = 0.2, amplitude: float = 1.0):
    """
    Generate a Gaussian bump field on the mesh.

        phi(x) = amplitude * exp(-||x - center||^2 / (2*sigma^2))

    Parameters
    ----------
    mesh : TetMesh
    center : ndarray, shape (3,)
    sigma : float
    amplitude : float

    Returns
    -------
    field : EnvironmentField
    """
    center = np.asarray(center, dtype=float)
    diffs = mesh.nodes - center[np.newaxis, :]
    dists2 = np.sum(diffs ** 2, axis=1)
    vals = amplitude * np.exp(-dists2 / (2.0 * sigma ** 2))
    return EnvironmentField(mesh, vals)


def sample_field_at_positions(field: EnvironmentField, positions: np.ndarray):
    """
    Sample a field at multiple robot positions.

    Parameters
    ----------
    field : EnvironmentField
    positions : ndarray, shape (N, 3)

    Returns
    -------
    values : ndarray, shape (N,)
    """
    positions = np.asarray(positions, dtype=float)
    values = np.array([field.evaluate(p) for p in positions], dtype=float)
    return values
