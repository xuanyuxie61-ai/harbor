
import numpy as np
from math import sqrt, pi



RHO0 = 0.16

DIFFUSION_COEF = 2.5


def reaction_source(rho, rho0=RHO0, alpha=0.5, beta=0.1):
    x = rho / rho0
    logistic = rho * (1.0 - x)
    correction = 1.0 + alpha * (x - 0.5) ** 2
    cubic_damping = beta * rho ** 3
    return logistic * correction - cubic_damping


def reaction_source_derivative(rho, rho0=RHO0, alpha=0.5, beta=0.1):
    x = rho / rho0
    term1 = (1.0 - 2.0 * x) * (1.0 + alpha * (x - 0.5) ** 2)
    term2 = rho * (1.0 - x) * 2.0 * alpha * (x - 0.5) / rho0
    term3 = 3.0 * beta * rho ** 2
    return term1 + term2 - term3


def ftcs_density_evolution_1d(r_grid, rho_initial, D, t_max, nt,
                               rho0=RHO0, alpha=0.5, beta=0.1,
                               left_bc='neumann', right_bc='neumann'):
    N = len(r_grid)
    dr = r_grid[1] - r_grid[0]
    dt = t_max / nt


    s = D * dt / (dr ** 2)
    if s > 0.5:

        nt = int(np.ceil(t_max * D / (0.45 * dr ** 2)))
        dt = t_max / nt
        s = D * dt / (dr ** 2)

    rho = rho_initial.copy()
    save_interval = max(1, nt // 20)
    history = [rho.copy()]

    for step in range(nt):
        rho_new = rho.copy()

        for i in range(1, N - 1):
            r = r_grid[i]

            d2rho = (rho[i + 1] - 2.0 * rho[i] + rho[i - 1]) / (dr ** 2)


            if r > 1e-6:
                drho = (rho[i + 1] - rho[i - 1]) / (2.0 * dr)
                laplacian = d2rho + (2.0 / r) * drho
            else:

                laplacian = 3.0 * d2rho

            source = reaction_source(rho[i], rho0, alpha, beta)
            rho_new[i] = rho[i] + dt * (D * laplacian + source)


            if rho_new[i] < 0:
                rho_new[i] = 0.0

            if rho_new[i] > 3.0 * rho0:
                rho_new[i] = 3.0 * rho0


        if left_bc == 'neumann':
            rho_new[0] = rho_new[1]
        elif left_bc == 'dirichlet':
            rho_new[0] = rho0

        if right_bc == 'neumann':
            rho_new[N - 1] = rho_new[N - 2]
        elif right_bc == 'dirichlet':
            rho_new[N - 1] = 0.0

        rho = rho_new
        if (step + 1) % save_interval == 0:
            history.append(rho.copy())

    return rho, np.array(history), s


def total_nucleon_number(r_grid, rho):
    integrand = 4.0 * pi * rho * r_grid ** 2
    return np.trapezoid(integrand, r_grid)


def rms_radius(r_grid, rho):
    num = np.trapezoid(rho * r_grid ** 4, r_grid)
    den = np.trapezoid(rho * r_grid ** 2, r_grid)
    if den < 1e-15:
        return 0.0
    return sqrt(num / den)


def surface_thickness(r_grid, rho, rho0=RHO0):

    r90 = None
    r10 = None
    for i in range(len(r_grid) - 1, 0, -1):
        if r90 is None and rho[i] < 0.9 * rho0:
            r90 = r_grid[i]
        if r10 is None and rho[i] < 0.1 * rho0:
            r10 = r_grid[i]
            break
    if r90 is not None and r10 is not None:
        return r10 - r90
    return 0.0


def density_moment(r_grid, rho, n):
    integrand = 4.0 * pi * rho * r_grid ** (n + 2)
    return np.trapezoid(integrand, r_grid)
