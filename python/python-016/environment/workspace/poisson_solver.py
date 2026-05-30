
import numpy as np
from typing import Tuple, Optional


def build_charge_density(
    nx: int,
    ny: int,
    layer_polarization: float = 0.0,
    moire_amplitude: float = 0.0,
    L_moire: float = 10.0,
) -> np.ndarray:
    if nx < 3 or ny < 3:
        raise ValueError("Grid dimensions must be at least 3.")

    rho = np.zeros((nx, ny))
    rho[:, :] = layer_polarization

    if moire_amplitude != 0.0 and L_moire > 0.0:
        q_mag = 4.0 * np.pi / (np.sqrt(3.0) * L_moire)

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
    nx, ny = rho.shape
    if nx < 3 or ny < 3:
        raise ValueError("Grid must be at least 3×3.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("Grid spacings must be positive.")
    if not (0.0 < omega <= 2.0):
        raise ValueError("SOR factor omega must be in (0, 2].")

    V = np.zeros((nx, ny))

    V[0, :] = boundary_value
    V[-1, :] = boundary_value
    V[:, 0] = boundary_value
    V[:, -1] = boundary_value


    coeff = dx * dy / (4.0 * epsilon)

    dx2 = dx ** 2
    dy2 = dy ** 2
    denom = 2.0 * (dx2 + dy2)

    V_new = V.copy()

    for it in range(max_iterations):

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):

                laplace = (
                    (V[i - 1, j] + V[i + 1, j]) * dy2
                    + (V[i, j - 1] + V[i, j + 1]) * dx2
                ) / denom
                source = dx2 * dy2 * rho[i, j] / (denom * epsilon)
                V_new[i, j] = omega * (laplace + source) + (1.0 - omega) * V[i, j]


        V_new[0, :] = boundary_value
        V_new[-1, :] = boundary_value
        V_new[:, 0] = boundary_value
        V_new[:, -1] = boundary_value

        diff = np.linalg.norm(V_new - V, ord="fro")
        V, V_new = V_new, V

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
    from band_solver import diagonalize_hamiltonian, find_fermi_level
    from tight_binding import build_tight_binding_hamiltonian, moire_lattice_constant

    epsilon_0 = 0.0552635
    epsilon = epsilon_0 * epsilon_r

    L_m = moire_lattice_constant(theta_deg)
    dx = L_m / n_grid
    dy = dx


    V_old = np.zeros((n_grid, n_grid))

    history = []

    for cycle in range(max_scf_cycles):


        V_avg = np.mean(V_old)

        H0, positions, layer_index = build_tight_binding_hamiltonian(
            theta_deg, n_super=3
        )
        N = H0.shape[0]
        for i in range(N):
            sign = +1.0 if layer_index[i] == 1 else -1.0
            H0[i, i] += sign * 0.5 * V_avg


        energies, vectors = diagonalize_hamiltonian(H0)
        e_fermi = find_fermi_level(energies)


        n_layers = np.zeros(2)
        layer_densities = np.zeros((2, n_grid, n_grid))



        for band in range(N):
            occ = 1.0 if energies[band] <= e_fermi else 0.0
            if abs(energies[band] - e_fermi) < 1e-6:
                occ = 0.5
            for layer in range(2):
                mask = layer_index == layer
                weight = np.sum(np.abs(vectors[mask, band]) ** 2)
                n_layers[layer] += occ * weight

        n_layers /= (L_m ** 2)
        delta_n = n_layers[1] - n_layers[0]


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


        V_mixed = mixing_beta * V_new + (1.0 - mixing_beta) * V_old
        diff = np.linalg.norm(V_mixed - V_old)
        history.append(diff)
        V_old = V_mixed

        if diff < scf_tolerance:
            H_final = H0
            return H_final, V_old, n_layers, history


    H_final, positions, layer_index = build_tight_binding_hamiltonian(theta_deg, n_super=3)
    for i in range(H_final.shape[0]):
        sign = +1.0 if layer_index[i] == 1 else -1.0
        H_final[i, i] += sign * 0.5 * np.mean(V_old)
    return H_final, V_old, n_layers, history
