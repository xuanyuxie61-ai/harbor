"""
grid_refinement.py
Grid refinement and resampling for molecular potential maps.

Derived from: 578_image_double

In molecular dynamics, electrostatic potential grids often require
adaptive refinement near charged residues and DNA damage sites.
This module implements doubling (dyadic refinement) of 2D and 3D
scalar fields, preserving integral conservation.

Given a coarse grid field Phi_{ij}, the refined field Phi2_{IJ} satisfies:
    Phi2[2*i-1, 2*j-1] = Phi[i,j]   (pixel replication)
    sum_{I,J in child(i,j)} Phi2[I,J] * dV2 = Phi[i,j] * dV1

For bilinear refinement (higher order):
    Phi2[2*i, 2*j] = Phi[i,j]
    Phi2[2*i+1, 2*j] = (Phi[i,j] + Phi[i+1,j])/2
    Phi2[2*i, 2*j+1] = (Phi[i,j] + Phi[i,j+1])/2
    Phi2[2*i+1, 2*j+1] = (Phi[i,j] + Phi[i+1,j] + Phi[i,j+1] + Phi[i+1,j+1])/4
"""

import numpy as np


def double_grid_nearest(field):
    """
    Double the resolution of a 2D or 3D field by nearest-neighbor replication.

    Parameters
    ----------
    field : ndarray, shape (m, n) or (m, n, p)

    Returns
    -------
    field2 : ndarray
        Shape (2*m, 2*n) or (2*m, 2*n, 2*p).
    """
    if field.ndim == 2:
        m, n = field.shape
        field2 = np.zeros((2 * m, 2 * n), dtype=field.dtype)
        for i in range(m):
            for j in range(n):
                field2[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = field[i, j]
        return field2
    elif field.ndim == 3:
        m, n, p = field.shape
        field2 = np.zeros((2 * m, 2 * n, 2 * p), dtype=field.dtype)
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    field2[
                        2 * i : 2 * i + 2,
                        2 * j : 2 * j + 2,
                        2 * k : 2 * k + 2,
                    ] = field[i, j, k]
        return field2
    else:
        raise ValueError("Only 2D and 3D fields supported.")


def double_grid_bilinear(field):
    """
    Double the resolution of a 2D field using bilinear interpolation.

    Parameters
    ----------
    field : ndarray, shape (m, n)

    Returns
    -------
    field2 : ndarray, shape (2*m-1, 2*n-1)
    """
    if field.ndim != 2:
        raise ValueError("Bilinear doubling only supports 2D fields.")
    m, n = field.shape
    field2 = np.zeros((2 * m - 1, 2 * n - 1), dtype=float)

    for i in range(m):
        for j in range(n):
            field2[2 * i, 2 * j] = field[i, j]

    # Horizontal interpolation
    for i in range(m):
        for j in range(n - 1):
            field2[2 * i, 2 * j + 1] = 0.5 * (field[i, j] + field[i, j + 1])

    # Vertical interpolation
    for i in range(m - 1):
        for j in range(n):
            field2[2 * i + 1, 2 * j] = 0.5 * (field[i, j] + field[i + 1, j])

    # Diagonal interpolation
    for i in range(m - 1):
        for j in range(n - 1):
            field2[2 * i + 1, 2 * j + 1] = 0.25 * (
                field[i, j] + field[i + 1, j] + field[i, j + 1] + field[i + 1, j + 1]
            )

    return field2


def adaptive_refinement_2d(
    field, threshold, max_level=3, refinement_func="bilinear"
):
    """
    Adaptively refine regions of a 2D field where the gradient magnitude
    exceeds a threshold. This targets high-gradient regions such as
    protein-DNA interfaces and damage sites.

    Parameters
    ----------
    field : ndarray, shape (m, n)
    threshold : float
        Gradient magnitude threshold for refinement.
    max_level : int
        Maximum refinement levels.
    refinement_func : str
        'nearest' or 'bilinear'.

    Returns
    -------
    refined_fields : list of ndarray
        Fields at each refinement level.
    """
    refined_fields = [field.copy()]
    current = field.copy()

    for _ in range(max_level):
        m, n = current.shape
        if m < 3 or n < 3:
            break

        # Compute gradient magnitude
        gx = np.zeros_like(current)
        gy = np.zeros_like(current)
        gx[:, :-1] = current[:, 1:] - current[:, :-1]
        gy[:-1, :] = current[1:, :] - current[:-1, :]
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Check if any region needs refinement
        if np.max(grad_mag) < threshold:
            break

        if refinement_func == "bilinear":
            current = double_grid_bilinear(current)
        else:
            current = double_grid_nearest(current)
        refined_fields.append(current)

    return refined_fields


def integrate_field_2d(field, dx, dy):
    """
    Integrate a 2D scalar field using the trapezoidal rule.

    Parameters
    ----------
    field : ndarray, shape (m, n)
    dx, dy : float

    Returns
    -------
    integral : float
    """
    return np.trapz(np.trapz(field, dx=dx, axis=0), dx=dy)


def integrate_field_3d(field, dx, dy, dz):
    """
    Integrate a 3D scalar field.

    Parameters
    ----------
    field : ndarray, shape (m, n, p)
    dx, dy, dz : float

    Returns
    -------
    integral : float
    """
    temp = np.trapz(field, dx=dx, axis=0)
    temp = np.trapz(temp, dx=dy, axis=0)
    return np.trapz(temp, dx=dz)
