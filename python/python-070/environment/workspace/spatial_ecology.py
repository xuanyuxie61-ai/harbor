
import numpy as np
from utils import NumericalConfig, safe_divide






def streamfunction_phi(Z, C_param):
    return (1.0 - np.cos(C_param * np.pi * Z)) * ((1.0 - Z) ** 2)


def streamfunction_dphi(Z, C_param):
    term1 = C_param * np.pi * np.sin(C_param * np.pi * Z) * ((1.0 - Z) ** 2)
    term2 = -2.0 * (1.0 - np.cos(C_param * np.pi * Z)) * (1.0 - Z)
    return term1 + term2


def divergence_free_velocity(n, X, Y, C_param):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = np.clip(X, 0.0, 1.0)
    Y = np.clip(Y, 0.0, 1.0)

    phi_X = streamfunction_phi(X, C_param)
    phi_Y = streamfunction_phi(Y, C_param)
    dphi_X = streamfunction_dphi(X, C_param)
    dphi_Y = streamfunction_dphi(Y, C_param)

    U = 10.0 * phi_X * dphi_Y
    V = -10.0 * phi_Y * dphi_X
    return U, V


def compute_divergence(X, Y, U, V, dx, dy):
    dUdx = np.gradient(U, dx, axis=0)
    dVdy = np.gradient(V, dy, axis=1)
    return dUdx + dVdy






def sphere_stereograph(points_sphere):
    points_sphere = np.asarray(points_sphere, dtype=float)
    if points_sphere.ndim == 1:
        points_sphere = points_sphere.reshape(1, -1)

    p1 = points_sphere[:, 0]
    p2 = points_sphere[:, 1]
    p3 = points_sphere[:, 2]

    denom = 1.0 + p3
    denom = np.where(np.abs(denom) < NumericalConfig.EPS, NumericalConfig.EPS, denom)

    q1 = 2.0 * p1 / denom
    q2 = 2.0 * p2 / denom
    q3 = np.ones_like(q1)

    return np.column_stack([q1, q2, q3])


def sphere_stereograph_inverse(points_plane):
    points_plane = np.asarray(points_plane, dtype=float)
    if points_plane.ndim == 1:
        points_plane = points_plane.reshape(1, -1)

    e1 = points_plane[:, 0]
    e2 = points_plane[:, 1]

    norm_sq = e1 ** 2 + e2 ** 2
    denom = 4.0 + norm_sq

    p1 = 4.0 * e1 / denom
    p2 = 4.0 * e2 / denom
    p3 = (4.0 - norm_sq) / denom

    return np.column_stack([p1, p2, p3])


def icosahedron_vertices():
    phi = 0.5 * (1.0 + np.sqrt(5.0))

    vertices = np.array([
        [0.0, 1.0, phi],
        [0.0, 1.0, -phi],
        [0.0, -1.0, phi],
        [0.0, -1.0, -phi],
        [1.0, phi, 0.0],
        [1.0, -phi, 0.0],
        [-1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0],
        [phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
        [-phi, 0.0, -1.0]
    ], dtype=float)


    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms






def advection_diffusion_2d_step(C, U, V, D, dx, dy, dt, lambda_mortality=0.0):
    nx, ny = C.shape
    C_new = C.copy()


    u_max = np.max(np.abs(U))
    v_max = np.max(np.abs(V))
    cfl_limit = min(dx / (u_max + NumericalConfig.EPS),
                    dy / (v_max + NumericalConfig.EPS))
    diff_limit = 0.5 / (D * (1.0 / dx ** 2 + 1.0 / dy ** 2) + NumericalConfig.EPS)
    dt_safe = min(cfl_limit, diff_limit, dt)
    if dt_safe < dt:
        dt = dt_safe

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            if U[i, j] >= 0:
                adv_x = U[i, j] * (C[i, j] - C[i - 1, j]) / dx
            else:
                adv_x = U[i, j] * (C[i + 1, j] - C[i, j]) / dx

            if V[i, j] >= 0:
                adv_y = V[i, j] * (C[i, j] - C[i, j - 1]) / dy
            else:
                adv_y = V[i, j] * (C[i, j + 1] - C[i, j]) / dy


            diff_x = (C[i + 1, j] - 2.0 * C[i, j] + C[i - 1, j]) / (dx ** 2)
            diff_y = (C[i, j + 1] - 2.0 * C[i, j] + C[i, j - 1]) / (dy ** 2)

            C_new[i, j] = C[i, j] - dt * (adv_x + adv_y) \
                          + dt * D * (diff_x + diff_y) \
                          - dt * lambda_mortality * C[i, j]


    C_new[0, :] = C_new[1, :]
    C_new[-1, :] = C_new[-2, :]
    C_new[:, 0] = C_new[:, 1]
    C_new[:, -1] = C_new[:, -2]

    return C_new


def simulate_larval_dispersal(nx, ny, Lx, Ly, C0_center, C0_sigma,
                               U, V, D, T_total, dt, lambda_mortality=0.0):
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)


    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    cx, cy = C0_center
    C = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * C0_sigma ** 2))

    n_steps = int(T_total / dt)
    C_history = []
    times = []

    for step in range(n_steps):
        C = advection_diffusion_2d_step(C, U, V, D, dx, dy, dt, lambda_mortality)
        if step % max(1, n_steps // 10) == 0:
            C_history.append(C.copy())
            times.append(step * dt)

    return C, times, C_history
