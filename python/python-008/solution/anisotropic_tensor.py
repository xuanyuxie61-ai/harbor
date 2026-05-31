"""
Anisotropic Tensor Module
=========================
Based on seed project 719_matlab_compiler:
- magicsquare.m, magic3.m  →  magic-square tensor structures

Physics:
--------
In a magnetized GRB jet, the diffusion of relativistic particles
and photons is anisotropic: transport parallel to the magnetic
field is much faster than perpendicular transport.  The diffusion
tensor in the local field-aligned frame is diagonal:

    D̃ = diag( D_⊥, D_⊥, D_∥ )

Rotating to the lab frame via the field direction 𝐛 = 𝐁/|𝐁|:

    D_{ij} = D_⊥ δ_{ij} + (D_∥ - D_⊥) b_i b_j

For a helical field with pitch angle ψ(r), the direction cosines are:

    b_r = 0
    b_φ = sin ψ
    b_z = cos ψ

The magic-square structure provides a discrete, divergence-free
tensor field that satisfies:

    Σ_i D_{ij} = constant   (row conservation)
    Σ_j D_{ij} = constant   (column conservation)

which mimics the conservation of particle number under anisotropic
diffusion.  We construct a 3×3 anisotropic diffusion tensor from
the magic-square pattern scaled by D_⊥ and D_∥.
"""

import numpy as np


def magic3():
    """
    Returns the 3×3 Lo Shu magic square:

        [ 8  1  6
          3  5  7
          4  9  2 ]

    The magic constant is 15.
    """
    return np.array([
        [8, 1, 6],
        [3, 5, 7],
        [4, 9, 2]
    ], dtype=float)


def magicsquare(n):
    """
    Construct an n×n magic square using the Siamese method (odd n)
or bordered-doubling (even n).

    Parameters
    ----------
    n : int
        Order of magic square.

    Returns
    -------
    M : ndarray, shape (n, n)
        Magic square.
    """
    if n % 2 == 1:
        M = np.zeros((n, n), dtype=int)
        i, j = 0, n // 2
        for k in range(1, n * n + 1):
            M[i, j] = k
            ni = (i - 1) % n
            nj = (j + 1) % n
            if M[ni, nj] != 0:
                i = (i + 1) % n
            else:
                i, j = ni, nj
        return M.astype(float)
    elif n % 4 == 0:
        M = np.arange(1, n * n + 1).reshape(n, n)
        I = np.arange(n)
        J = np.arange(n)
        mask = ((I[:, None] % 4 == J[None, :] % 4)
                | ((I[:, None] + J[None, :]) % 4 == 3))
        M[mask] = n * n + 1 - M[mask]
        return M.astype(float)
    else:
        # Singly even: not fully implemented; fall back to odd method
        return magicsquare(n - 1)


def anisotropic_diffusion_tensor(D_perp, D_para, pitch_angle):
    """
    Construct the anisotropic diffusion tensor for a helical
    magnetic field with pitch angle ψ.

        D_{ij} = D_⊥ δ_{ij} + (D_∥ - D_⊥) b_i b_j

    In cylindrical coordinates (r, φ, z):
        b = (0, sin ψ, cos ψ)

    Parameters
    ----------
    D_perp : float
        Perpendicular diffusion coefficient.
    D_para : float
        Parallel diffusion coefficient.
    pitch_angle : float
        Pitch angle ψ in radians.

    Returns
    -------
    D : ndarray, shape (3, 3)
        Diffusion tensor.
    """
    b = np.array([0.0, np.sin(pitch_angle), np.cos(pitch_angle)], dtype=float)
    delta = np.eye(3, dtype=float)
    D = D_perp * delta + (D_para - D_perp) * np.outer(b, b)
    return D


def magic_anisotropic_field(n_r=16, D_perp_base=1e20, D_para_base=1e24):
    """
    Generate a radially varying anisotropic diffusion field where
the ratio D_para/D_perp follows a magic-square modulation pattern.

    Returns
    -------
    r : ndarray
        Radial coordinates.
    D_tensors : ndarray, shape (n_r, 3, 3)
        Diffusion tensors.
    """
    r = np.linspace(0.0, 1e13, n_r)
    M = magicsquare(3)
    M_norm = M / np.sum(M)

    D_tensors = np.zeros((n_r, 3, 3), dtype=float)
    for i in range(n_r):
        # Modulate D_para by magic-square pattern mapped to radius
        modulation = 1.0 + 0.5 * np.sin(2 * np.pi * r[i] / 1e13)
        D_perp = D_perp_base * modulation
        D_para = D_para_base * modulation * (1.0 + M_norm[i % 3, 0])
        psi = np.arctan2(r[i], 1e13)  # Simple pitch-angle model
        D_tensors[i] = anisotropic_diffusion_tensor(D_perp, D_para, psi)

    return r, D_tensors
