
import numpy as np


def laplace_radial_2d_exact(x, y, a, b):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x ** 2 + y ** 2

    r2 = np.where(r2 < 1e-14, 1e-14, r2)
    r = np.sqrt(r2)

    u = a * np.log(r) + b
    ux = a * x / r2
    uy = a * y / r2
    uxx = a * (-2.0 * x ** 2 / r2 + 1.0) / r2
    uxy = -2.0 * a * x * y / r2 ** 2
    uyy = a * (-2.0 * y ** 2 / r2 + 1.0) / r2
    return u, ux, uy, uxx, uxy, uyy


def laplace_radial_3d_exact(x, y, z, a, b):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    r2 = x ** 2 + y ** 2 + z ** 2
    r2 = np.where(r2 < 1e-14, 1e-14, r2)
    r = np.sqrt(r2)

    u = a / r + b
    ux = -a * x / r ** 3
    uy = -a * y / r ** 3
    uz = -a * z / r ** 3
    return u, ux, uy, uz


def burgers_flux(u):
    return 0.5 * u ** 2


def burgers_flux_derivative(u):
    return u


def godunov_nf(u_left, u_right):
    u_left = np.asarray(u_left, dtype=float)
    u_right = np.asarray(u_right, dtype=float)
    ustar = np.empty_like(u_left)

    mask1 = u_right <= u_left
    mask2 = ~mask1


    cond_a = mask1 & ((u_left + u_right) / 2.0 > 0)
    cond_b = mask1 & ((u_left + u_right) / 2.0 <= 0)
    ustar[cond_a] = u_left[cond_a]
    ustar[cond_b] = u_right[cond_b]


    cond_c = mask2 & (u_left > 0)
    cond_d = mask2 & (u_right < 0)
    cond_e = mask2 & ~(cond_c | cond_d)
    ustar[cond_c] = u_left[cond_c]
    ustar[cond_d] = u_right[cond_d]
    ustar[cond_e] = 0.0

    return burgers_flux(ustar)


def burgers_time_inviscid_godunov(u0, nx, nt, t_max, bc_type='periodic'):
    a = -1.0
    b = 1.0
    dx = (b - a) / nx
    x = np.linspace(a, b, nx)
    dt = t_max / nt

    U = np.zeros((nt + 1, nx))
    u = u0(x).astype(float)
    U[0, :] = u

    for i in range(nt):
        unew = np.empty_like(u)

        if bc_type == 'periodic':
            unew[0] = u[0] - dt / dx * (godunov_nf(u[0], u[1]) - godunov_nf(u[-1], u[0]))
            unew[1:-1] = u[1:-1] - dt / dx * (
                godunov_nf(u[1:-1], u[2:]) - godunov_nf(u[:-2], u[1:-1])
            )
            unew[-1] = u[-1] - dt / dx * (godunov_nf(u[-1], u[0]) - godunov_nf(u[-2], u[-1]))
        else:
            unew[0] = u[0]
            unew[1:-1] = u[1:-1] - dt / dx * (
                godunov_nf(u[1:-1], u[2:]) - godunov_nf(u[:-2], u[1:-1])
            )
            unew[-1] = u[-1]

        u = unew
        U[i + 1, :] = u

    return U, x


def windkessel_pressure_outflow(Q_in, R, C, dt, n_steps):
    P = np.zeros(n_steps)
    P[0] = Q_in[0] * R
    alpha = dt / (R * C)
    for n in range(n_steps - 1):
        P[n + 1] = (P[n] + R * Q_in[n] * alpha) / (1.0 + alpha)
    return P


def poiseuille_flow_rate(radius, delta_p, length, mu=3.5e-3):
    if radius <= 0 or length <= 0:
        return 0.0
    return np.pi * radius ** 4 * delta_p / (8.0 * mu * length)


def compute_vascular_pressure_field(nodes, edges, radius, inflow_node, outflow_nodes, P_in, P_out_base):





    raise NotImplementedError("HOLE_2: 血管网络压力场求解待实现")
