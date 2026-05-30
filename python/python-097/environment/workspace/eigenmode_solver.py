
import numpy as np
from physics_constants import EPSILON_0, MU_0


def power_method_eigenmode(A_func, x0, it_max=1000, tol=1e-10):
    x = x0.copy()
    n = x.size
    x = x.reshape(n)
    x = x / np.linalg.norm(x)

    lambda_old = 0.0
    convergence_history = []

    for it_num in range(1, it_max + 1):
        y = A_func(x)
        lambda_val = np.dot(x, y)
        y_norm = np.linalg.norm(y)
        if y_norm < 1e-30:
            break
        x_new = y / y_norm
        if lambda_val < 0:
            x_new = -x_new
            lambda_val = -y_norm

        delta_lambda = abs(lambda_val - lambda_old)
        cos_xy = np.dot(x, x_new)
        sin_xy = np.sqrt(max(0.0, 1.0 - cos_xy ** 2))

        convergence_history.append((it_num, lambda_val, delta_lambda, sin_xy))

        if delta_lambda <= tol and sin_xy <= tol:
            x = x_new
            break

        x = x_new
        lambda_old = lambda_val

    return lambda_val, x, it_num, convergence_history


def inverse_power_method(A_func, x0, sigma_shift, it_max=500, tol=1e-10):
    x = x0.copy().reshape(-1)
    x = x / np.linalg.norm(x)



    lambda_old = 0.0

    for it_num in range(1, it_max + 1):


        Ax = A_func(x)
        y = x / (sigma_shift + 1e-10)
        for _ in range(5):
            residual = x - (Ax - sigma_shift * x)
            y = y + 0.1 * residual
            Ay = A_func(y)
            residual = x - (Ay - sigma_shift * y)
            if np.linalg.norm(residual) < tol:
                break


        Ay = A_func(y)
        lambda_val = np.dot(y, Ay) / np.dot(y, y)

        y_norm = np.linalg.norm(y)
        if y_norm < 1e-30:
            break
        x_new = y / y_norm

        delta = abs(lambda_val - lambda_old)
        if delta < tol:
            x = x_new
            break

        x = x_new
        lambda_old = lambda_val

    return lambda_val, x, it_num


def build_fd_helmholtz_operator_2d(nx, ny, dx, dy, epsilon, mu):
    def A_func(psi_vec):
        psi = psi_vec.reshape((nx, ny))
        laplacian = np.zeros_like(psi)


        laplacian[1:-1, 1:-1] = (
            (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx ** 2 +
            (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dy ** 2
        )


        laplacian[0, :] = 0.0
        laplacian[-1, :] = 0.0
        laplacian[:, 0] = 0.0
        laplacian[:, -1] = 0.0


        result = -laplacian
        return result.reshape(-1)

    return A_func


def compute_cavity_modes_2d(nx, ny, dx, dy, epsilon, mu, n_modes=3, max_iter=500):
    A_func = build_fd_helmholtz_operator_2d(nx, ny, dx, dy, epsilon, mu)

    modes = []
    psi0 = np.random.randn(nx, ny)
    psi0[0, :] = 0.0
    psi0[-1, :] = 0.0
    psi0[:, 0] = 0.0
    psi0[:, -1] = 0.0

    for mode_idx in range(n_modes):

        sigma_guess = (mode_idx + 1) * np.pi ** 2 * (1.0 / dx ** 2 + 1.0 / dy ** 2) / 10.0
        lambda_val, eigenvec, it_num = inverse_power_method(
            A_func, psi0.flatten(), sigma_guess, it_max=max_iter
        )


        field = eigenvec.reshape((nx, ny))
        for prev_mode in modes:
            overlap = np.sum(field * prev_mode['field'])
            field = field - overlap * prev_mode['field']

        field = field / (np.linalg.norm(field) + 1e-30)



        eps_avg = np.mean(epsilon)
        mu_avg = np.mean(mu)
        omega = np.sqrt(max(lambda_val, 0.0) / (eps_avg * mu_avg))
        frequency = omega / (2.0 * np.pi)
        wavenumber = np.sqrt(max(lambda_val, 0.0))

        modes.append({
            'frequency': frequency,
            'wavenumber': wavenumber,
            'field': field,
            'eigenvalue': lambda_val,
            'iterations': it_num,
        })


        psi0 = np.random.randn(nx, ny)
        psi0[0, :] = 0.0
        psi0[-1, :] = 0.0
        psi0[:, 0] = 0.0
        psi0[:, -1] = 0.0

    return modes


def power_flow_pagerank_analysis(E, H, dx, dy, dz, damping=0.15, n_iter=100):
    Ex, Ey, Ez = E
    Hx, Hy, Hz = H
    nx, ny, nz = Ex.shape


    Sx = Ey * Hz - Ez * Hy
    Sy = Ez * Hx - Ex * Hz
    Sz = Ex * Hy - Ey * Hx
    S_mag = np.sqrt(Sx ** 2 + Sy ** 2 + Sz ** 2)


    N = nx * ny * nz
    s_flat = S_mag.flatten()



    from physics_constants import electromagnetic_energy_density
    eps = np.ones_like(Ex) * EPSILON_0
    mu = np.ones_like(Ex) * MU_0
    w = electromagnetic_energy_density(E, H, eps, mu)
    w_flat = w.flatten()


    rank = np.ones(N) / N


    for _ in range(n_iter):


        if np.sum(w_flat) > 1e-30:
            energy_weights = w_flat / np.sum(w_flat)
        else:
            energy_weights = np.ones(N) / N

        rank = (1.0 - damping) * energy_weights + damping * rank
        rank = rank / (np.sum(rank) + 1e-30)

    rank_field = rank.reshape((nx, ny, nz))
    return rank_field
