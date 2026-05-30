# -*- coding: utf-8 -*-
import numpy as np
from utils import condition_number, fermi_dirac, H_BAR, landau_level_energy





def cg_ne_solve(A, b, x0=None, max_iter=None, tol=1e-10):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    z = A.T @ r
    d = z.copy()

    for i in range(1, max_iter + 1):
        Ad = A @ d
        denom = np.dot(Ad, Ad)
        if denom < 1e-30:

            return x, False, np.linalg.norm(r)

        alpha = np.dot(z, z) / denom
        x = x + alpha * d
        r = b - A @ x

        residual_norm = np.linalg.norm(r)
        if residual_norm < tol:
            return x, True, residual_norm

        z_old = z
        z = A.T @ r

        denom_beta = np.dot(z_old, z_old)
        if denom_beta < 1e-30:
            return x, False, residual_norm

        beta = np.dot(z, z) / denom_beta
        d = z + beta * d

    return x, False, residual_norm






def newton_solve(F, J, x0, tol=1e-8, max_iter=50, damping=0.8):
    x = np.asarray(x0, dtype=float).copy()
    for k in range(max_iter):
        fx = np.asarray(F(x), dtype=float)
        jac = np.asarray(J(x), dtype=float)


        cond = condition_number(jac)
        if cond > 1e14:

            jac = jac + 1e-10 * np.eye(jac.shape[0])

        try:
            delta = np.linalg.solve(jac, fx)
        except np.linalg.LinAlgError:

            delta, _, _ = cg_ne_solve(jac, fx, tol=tol * 0.1)

        x_new = x - damping * delta

        if np.linalg.norm(x_new - x) < tol:
            return x_new, True, k + 1

        x = x_new

    return x, False, max_iter






def coulomb_interaction_2d(r, epsilon_r=12.0):
    r = np.asarray(r, dtype=float)
    a_cutoff = 0.01
    r_safe = np.sqrt(r ** 2 + a_cutoff ** 2)

    return 1.0 / (epsilon_r * r_safe)


def build_hartree_fock_matrix(
    basis_orbitals, grid_x, grid_y, occupied_indices, dx, dy, epsilon_r=12.0, mixing=0.5
):
    N_basis = basis_orbitals.shape[0]
    Nx, Ny = grid_x.shape


    density = np.zeros((Nx, Ny), dtype=float)
    for idx in occupied_indices:
        if idx < 0 or idx >= N_basis:
            raise ValueError(f"占据态指标 {idx} 超出范围 [0, {N_basis})")
        phi = basis_orbitals[idx]
        density += np.abs(phi) ** 2



    V_H = np.zeros((Nx, Ny), dtype=float)
    for ix in range(Nx):
        for iy in range(Ny):
            dr_x = grid_x - grid_x[ix, iy]
            dr_y = grid_y - grid_y[ix, iy]
            dr = np.sqrt(dr_x ** 2 + dr_y ** 2)
            V_c = coulomb_interaction_2d(dr, epsilon_r)
            V_H[ix, iy] = np.sum(V_c * density) * dx * dy



    H_HF = np.zeros((N_basis, N_basis), dtype=complex)


    for alpha in range(N_basis):
        H_HF[alpha, alpha] += landau_level_energy(alpha // 2, 10.0, 1.0)


    for alpha in range(N_basis):
        phi_a = basis_orbitals[alpha]
        H_HF[alpha, alpha] += np.sum(np.conj(phi_a) * V_H * phi_a) * dx * dy


    for alpha in range(N_basis):
        for beta in range(alpha + 1, N_basis):

            overlap = np.sum(np.conj(basis_orbitals[alpha]) * basis_orbitals[beta]) * dx * dy
            H_HF[alpha, beta] = 0.01 * overlap
            H_HF[beta, alpha] = np.conj(H_HF[alpha, beta])

    return H_HF, density


def self_consistent_hf(
    N_electrons, N_basis, B, lB, grid_x, grid_y,
    max_iter=30, tol=1e-6, epsilon_r=12.0, mixing=0.5
):
    from landau_levels import landau_orbital_wavefunction

    Nx, Ny = grid_x.shape
    dx = grid_x[1, 0] - grid_x[0, 0] if Nx > 1 else 1.0
    dy = grid_y[0, 1] - grid_y[0, 0] if Ny > 1 else 1.0
















    raise NotImplementedError("基函数初始构造待实现")



    occupied = list(range(min(N_electrons, N_basis)))

    density_old = np.zeros((Nx, Ny), dtype=float)

    for iteration in range(max_iter):
        H_HF, density_new = build_hartree_fock_matrix(
            basis_orbitals, grid_x, grid_y, occupied, dx, dy, epsilon_r, mixing
        )


        if iteration > 0:
            density = mixing * density_new + (1.0 - mixing) * density_old
        else:
            density = density_new


        H_HF = 0.5 * (H_HF + np.conj(H_HF.T))
        energies, C = np.linalg.eigh(H_HF)


        sort_idx = np.argsort(energies.real)
        energies = energies[sort_idx]
        C = C[:, sort_idx]


        new_orbitals = np.zeros_like(basis_orbitals)
        for alpha in range(N_basis):
            for beta in range(N_basis):
                new_orbitals[alpha] += C[beta, alpha] * basis_orbitals[beta]

            norm = np.sqrt(np.sum(np.abs(new_orbitals[alpha]) ** 2) * dx * dy)
            if norm > 1e-14:
                new_orbitals[alpha] /= norm

        basis_orbitals = new_orbitals
        occupied = list(range(min(N_electrons, N_basis)))


        density_diff = np.max(np.abs(density - density_old))
        if density_diff < tol:
            return energies, basis_orbitals, density, True

        density_old = density.copy()

    return energies, basis_orbitals, density_old, False





def test_hartree_fock():
    print("=" * 60)
    print("[hartree_fock_solver.py] Hartree-Fock求解器测试")
    print("=" * 60)


    print("\n1. 共轭梯度法(CGNE)测试:")
    A = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, 1.0]], dtype=float)
    b = np.array([4.0, 5.0, 3.0], dtype=float)
    x, converged, res = cg_ne_solve(A, b, tol=1e-12)
    print(f"   解 x = {x}")
    print(f"   收敛: {converged}, 残差: {res:.2e}")
    print(f"   ||Ax-b|| = {np.linalg.norm(A @ x - b):.2e}")


    print("\n2. Newton迭代测试 (求解 x² - 2 = 0):")
    def F(x):
        return np.array([x[0] ** 2 - 2.0])
    def J(x):
        return np.array([[2.0 * x[0]]])
    x_sol, conv, nit = newton_solve(F, J, np.array([1.5]), tol=1e-12)
    print(f"   解 x = {x_sol[0]:.10f}")
    print(f"   收敛: {conv}, 迭代次数: {nit}")


    print("\n3. 自洽Hartree-Fock测试 (简化2×2格点):")
    B = 10.0
    lB = np.sqrt(1.0 / B)
    x = np.linspace(-2.0 * lB, 2.0 * lB, 15)
    y = np.linspace(-2.0 * lB, 2.0 * lB, 15)
    grid_x, grid_y = np.meshgrid(x, y)

    energies, orbitals, density, conv = self_consistent_hf(
        N_electrons=2, N_basis=4, B=B, lB=lB,
        grid_x=grid_x, grid_y=grid_y,
        max_iter=10, tol=1e-4
    )
    print(f"   收敛: {conv}")
    print(f"   能级: {energies[:4].real}")
    print(f"   最大密度: {np.max(density):.6f}")

    print("\n[hartree_fock_solver.py] 测试完成。\n")


if __name__ == "__main__":
    test_hartree_fock()
