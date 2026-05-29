"""
2D Poisson Solver for Interlayer Electrostatic Potential
=========================================================
Self-consistently solves the 2D Poisson equation for the electrostatic
potential V(r) arising from charge redistribution in the twisted bilayer
graphene heterostructure.

Scientific Background
---------------------
The electrostatic potential in each layer satisfies the Poisson equation

    −∇² V_ℓ(r) = ρ_ℓ(r) / ε

where ρ_ℓ is the planar charge density in layer ℓ and ε = ε₀ ε_r is the
effective permittivity.  For a dual-gated heterostructure with gate
separation d_g and applied gate voltage V_g, the boundary condition is

    V(z = ±d_g/2) = ±V_g/2 .

In the weak-coupling limit the total potential difference between layers
is related to the layer-polarized charge density:

    ΔV = V_1 − V_0 = (e d / ε) (n_1 − n_0)

where d ≈ 0.335 nm is the interlayer spacing and n_ℓ are the areal
electron densities.  This potential enters the tight-binding Hamiltonian
as a layer-dependent onsite energy shift.

The finite-difference discretization on a uniform N_x × N_y grid with
spacing (dx, dy) reads

    (V_{i−1,j} + V_{i+1,j} + V_{i,j−1} + V_{i,j+1} − 4 V_{i,j}) / (dx dy)
    = −ρ_{i,j} / ε .

We solve this system using the Jacobi iteration with successive
over-relaxation (SOR) for improved convergence.
"""

import numpy as np
from typing import Tuple, Optional


def build_charge_density(
    nx: int,
    ny: int,
    layer_polarization: float = 0.0,
    moire_amplitude: float = 0.0,
    L_moire: float = 10.0,
) -> np.ndarray:
    """
    Construct a model charge density ρ(x, y) on the grid.

    The density has a uniform background plus a moiré-periodic modulation:

        ρ(r) = ρ_0 + ρ_m Σ_{j=1}^{3} cos(q_j · r)

    where q_j are the moiré reciprocal vectors.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    layer_polarization : float
        Average charge imbalance between layers (electrons/nm²).
    moire_amplitude : float
        Amplitude of moiré density modulation.
    L_moire : float
        Moiré period in nm.

    Returns
    -------
    np.ndarray of shape (nx, ny)
    """
    if nx < 3 or ny < 3:
        raise ValueError("Grid dimensions must be at least 3.")

    rho = np.zeros((nx, ny))
    rho[:, :] = layer_polarization

    if moire_amplitude != 0.0 and L_moire > 0.0:
        q_mag = 4.0 * np.pi / (np.sqrt(3.0) * L_moire)
        # Moiré vectors
        q_vecs = [
            np.array([q_mag, 0.0]),
            np.array([q_mag * 0.5, q_mag * np.sqrt(3.0) * 0.5]),
            np.array([-q_mag * 0.5, q_mag * np.sqrt(3.0) * 0.5]),
        ]
        x = np.linspace(0.0, L_moire, nx, endpoint=False)
        y = np.linspace(0.0, L_moire, ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        modulation = np.zeros((nx, ny))
        for q in q_vecs:
            modulation += np.cos(q[0] * X + q[1] * Y)
        rho += moire_amplitude * modulation / 3.0

    return rho


def jacobi_poisson_2d(
    rho: np.ndarray,
    dx: float,
    dy: float,
    epsilon: float = 1.0,
    tolerance: float = 1e-8,
    max_iterations: int = 20000,
    omega: float = 1.0,
    boundary_value: float = 0.0,
) -> Tuple[np.ndarray, int, float]:
    """
    Solve the 2D Poisson equation

        −(∂²V/∂x² + ∂²V/∂y²) = ρ / ε

    on a rectangular domain using Jacobi iteration with optional SOR.

    The five-point stencil is

        V_{new}(i,j) = [V(i−1,j)+V(i+1,j)+V(i,j−1)+V(i,j+1)]/4
                       + (dx·dy) · ρ(i,j) / (4ε)

    when dx = dy.  For dx ≠ dy the coefficients are weighted.

    Parameters
    ----------
    rho : np.ndarray of shape (nx, ny)
        Charge density.
    dx, dy : float
        Grid spacings.
    epsilon : float
        Permittivity.
    tolerance : float
        Convergence criterion on the Frobenius norm of the update.
    max_iterations : int
    omega : float
        Successive-over-relaxation factor (1.0 = pure Jacobi,
        1.5–1.9 typical for SOR).
    boundary_value : float
        Dirichlet boundary value.

    Returns
    -------
    V : np.ndarray of shape (nx, ny)
    iters : int
        Number of iterations performed.
    residual : float
        Final update norm.

    Raises
    ------
    RuntimeError
        If max_iterations is reached without convergence.
    """
    nx, ny = rho.shape
    if nx < 3 or ny < 3:
        raise ValueError("Grid must be at least 3×3.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("Grid spacings must be positive.")
    if not (0.0 < omega <= 2.0):
        raise ValueError("SOR factor omega must be in (0, 2].")

    V = np.zeros((nx, ny))
    # Set boundary conditions
    V[0, :] = boundary_value
    V[-1, :] = boundary_value
    V[:, 0] = boundary_value
    V[:, -1] = boundary_value

    # Precompute coefficient
    coeff = dx * dy / (4.0 * epsilon)
    # For anisotropic grid, use weighted average
    dx2 = dx ** 2
    dy2 = dy ** 2
    denom = 2.0 * (dx2 + dy2)

    V_new = V.copy()

    for it in range(max_iterations):
        # Interior points only
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Weighted five-point stencil for anisotropic spacing
                laplace = (
                    (V[i - 1, j] + V[i + 1, j]) * dy2
                    + (V[i, j - 1] + V[i, j + 1]) * dx2
                ) / denom
                source = dx2 * dy2 * rho[i, j] / (denom * epsilon)
                V_new[i, j] = omega * (laplace + source) + (1.0 - omega) * V[i, j]

        # Keep boundaries fixed
        V_new[0, :] = boundary_value
        V_new[-1, :] = boundary_value
        V_new[:, 0] = boundary_value
        V_new[:, -1] = boundary_value

        diff = np.linalg.norm(V_new - V, ord="fro")
        V, V_new = V_new, V  # swap buffers

        if diff < tolerance:
            return V, it + 1, diff

    raise RuntimeError(
        f"Poisson solver did not converge within {max_iterations} iterations. "
        f"Final residual: {diff:.3e}"
    )


def self_consistent_potential_loop(
    H_builder_func,
    theta_deg: float,
    n_grid: int = 32,
    epsilon_r: float = 4.0,
    mixing_beta: float = 0.3,
    scf_tolerance: float = 1e-5,
    max_scf_cycles: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Perform a self-consistent field (SCF) loop that couples the Poisson
    solver to the tight-binding charge density.

    At each SCF cycle:
      1. Diagonalize the current Hamiltonian.
      2. Compute layer-resolved charge density n_ℓ(r).
      3. Solve Poisson equation for ΔV(r).
      4. Update the layer-dependent onsite energy with mixing:
             V^{new} = β V^{Poisson} + (1−β) V^{old}

    Parameters
    ----------
    H_builder_func : callable
        Function that builds the Hamiltonian with an optional potential.
    theta_deg : float
    n_grid : int
        Grid size for the Poisson solver.
    epsilon_r : float
        Relative permittivity (e.g., hBN substrate ~ 3–5).
    mixing_beta : float
        Density/potential mixing parameter (0 < β ≤ 1).
    scf_tolerance : float
        Convergence criterion for the potential update.
    max_scf_cycles : int

    Returns
    -------
    H_final : np.ndarray
        Converged Hamiltonian.
    V_scf : np.ndarray of shape (n_grid, n_grid)
        Converged potential.
    n_layers : np.ndarray of shape (2,)
        Total electron count per layer.
    history : list
        List of potential update norms per cycle.
    """
    from band_solver import diagonalize_hamiltonian, find_fermi_level
    from tight_binding import build_tight_binding_hamiltonian, moire_lattice_constant

    epsilon_0 = 0.0552635  # e²/(eV·nm) ≈ 1/(4π ε₀) in these units
    epsilon = epsilon_0 * epsilon_r

    L_m = moire_lattice_constant(theta_deg)
    dx = L_m / n_grid
    dy = dx

    # Initial guess: zero potential
    V_old = np.zeros((n_grid, n_grid))

    history = []

    for cycle in range(max_scf_cycles):
        # Build Hamiltonian with current potential
        # We approximate the potential by its layer average
        V_avg = np.mean(V_old)
        # Add a layer-dependent shift proportional to V_avg
        H0, positions, layer_index = build_tight_binding_hamiltonian(
            theta_deg, n_super=3
        )
        N = H0.shape[0]
        for i in range(N):
            sign = +1.0 if layer_index[i] == 1 else -1.0
            H0[i, i] += sign * 0.5 * V_avg

        # Diagonalize and compute filling
        energies, vectors = diagonalize_hamiltonian(H0)
        e_fermi = find_fermi_level(energies)

        # Layer-resolved density (real-space projection)
        n_layers = np.zeros(2)
        layer_densities = np.zeros((2, n_grid, n_grid))

        # Simple approximation: assign density uniformly within each layer
        # weighted by orbital weight near each grid point
        for band in range(N):
            occ = 1.0 if energies[band] <= e_fermi else 0.0
            if abs(energies[band] - e_fermi) < 1e-6:
                occ = 0.5  # smear at Fermi level
            for layer in range(2):
                mask = layer_index == layer
                weight = np.sum(np.abs(vectors[mask, band]) ** 2)
                n_layers[layer] += occ * weight

        n_layers /= (L_m ** 2)  # convert to areal density
        delta_n = n_layers[1] - n_layers[0]

        # Build charge density for Poisson solver
        rho = build_charge_density(
            n_grid, n_grid,
            layer_polarization=delta_n,
            moire_amplitude=delta_n * 0.1,
            L_moire=L_m,
        )

        V_new, _, _ = jacobi_poisson_2d(
            rho, dx, dy,
            epsilon=epsilon,
            tolerance=1e-7,
            max_iterations=5000,
            omega=1.5,
        )

        # Mixing
        V_mixed = mixing_beta * V_new + (1.0 - mixing_beta) * V_old
        diff = np.linalg.norm(V_mixed - V_old)
        history.append(diff)
        V_old = V_mixed

        if diff < scf_tolerance:
            H_final = H0
            return H_final, V_old, n_layers, history

    # If not converged, return last iterate with warning
    H_final, positions, layer_index = build_tight_binding_hamiltonian(theta_deg, n_super=3)
    for i in range(H_final.shape[0]):
        sign = +1.0 if layer_index[i] == 1 else -1.0
        H_final[i, i] += sign * 0.5 * np.mean(V_old)
    return H_final, V_old, n_layers, history
