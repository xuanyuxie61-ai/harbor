
import numpy as np
from constants import EARTH_RADIUS_KM, get_prem_density


def assemble_fem_1d(x_nodes, k_fun, source_fun, time=0.0):
    n = len(x_nodes)
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        h = x_nodes[i + 1] - x_nodes[i]
        if h <= 0:
            raise ValueError(f"Element {i} has non-positive length: {h}")

        x_mid = 0.5 * (x_nodes[i] + x_nodes[i + 1])
        k_val = k_fun(x_mid, time)
        f_val = source_fun(x_mid, time)




        ke = (k_val / h) * np.array([[1.0, -1.0],
                                     [-1.0, 1.0]])



        be = (f_val * h / 2.0) * np.array([1.0, 1.0])


        A[i:i + 2, i:i + 2] += ke
        b[i:i + 2] += be

    return A, b


def apply_boundary_conditions_1d(A, b, x_nodes, bc_type_left='dirichlet',
                                  bc_val_left=0.0, bc_type_right='neumann',
                                  bc_val_right=0.0):
    n = len(x_nodes)
    A = A.copy()
    b = b.copy()


    if bc_type_left == 'dirichlet':
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = bc_val_left
    elif bc_type_left == 'neumann':


        b[0] += bc_val_left


    if bc_type_right == 'dirichlet':
        A[n - 1, :] = 0.0
        A[n - 1, n - 1] = 1.0
        b[n - 1] = bc_val_right
    elif bc_type_right == 'neumann':
        b[n - 1] += bc_val_right

    return A, b


def solve_steady_state_density_1d(r_nodes_km, k_diffusion=None,
                                   rho_core=13.0, rho_surface=2.7):
    r_nodes = np.asarray(r_nodes_km, dtype=np.float64)
    n = len(r_nodes)
    if n < 2:
        raise ValueError("At least 2 nodes required")
    if r_nodes[0] < -1e-10 or abs(r_nodes[-1] - EARTH_RADIUS_KM) > 1.0:

        pass

    if k_diffusion is None:
        k_fun = lambda x, t: 1.0
    else:
        k_fun = lambda x, t: float(k_diffusion)


    def source_fun(x, t):
        r_ratio = x / EARTH_RADIUS_KM
        r_ratio = max(0.0, min(1.0, r_ratio))
        rho_prem = get_prem_density(r_ratio)



        return rho_prem * 0.1

    A, b = assemble_fem_1d(r_nodes, k_fun, source_fun)


    A, b = apply_boundary_conditions_1d(
        A, b, r_nodes,
        bc_type_left='dirichlet', bc_val_left=rho_core,
        bc_type_right='dirichlet', bc_val_right=rho_surface
    )


    try:
        rho = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:

        rho = np.linalg.lstsq(A, b, rcond=None)[0]

    return rho, r_nodes


def backward_euler_step_1d(A, M, u_old, dt, f_vec):
    n = len(u_old)
    lhs = M + dt * A
    rhs = M @ u_old + dt * f_vec

    try:
        u_new = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        u_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return u_new


def assemble_mass_matrix_1d(x_nodes):
    n = len(x_nodes)
    M = np.zeros((n, n), dtype=np.float64)

    for i in range(n - 1):
        h = x_nodes[i + 1] - x_nodes[i]
        if h <= 0:
            continue
        me = (h / 6.0) * np.array([[2.0, 1.0],
                                   [1.0, 2.0]])
        M[i:i + 2, i:i + 2] += me

    return M


def solve_time_dependent_density_1d(r_nodes_km, t_init=0.0, t_final=1.0,
                                     n_steps=100, k_diffusion=1.0):
    r_nodes = np.asarray(r_nodes_km, dtype=np.float64)
    n = len(r_nodes)
    dt = (t_final - t_init) / n_steps

    A, _ = assemble_fem_1d(r_nodes, lambda x, t: k_diffusion,
                           lambda x, t: 0.0)
    M = assemble_mass_matrix_1d(r_nodes)


    rho = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r_ratio = r_nodes[i] / EARTH_RADIUS_KM
        rho[i] = get_prem_density(max(0.0, min(1.0, r_ratio)))

    rho_history = [rho.copy()]
    t_history = [t_init]

    for step in range(n_steps):
        t = t_init + (step + 1) * dt
        _, b = assemble_fem_1d(r_nodes, lambda x, tt: k_diffusion,
                               lambda x, tt: 0.1 * get_prem_density(
                                   max(0.0, min(1.0, x / EARTH_RADIUS_KM))), t)

        A_bc, b_bc = apply_boundary_conditions_1d(
            A.copy(), b, r_nodes,
            bc_type_left='dirichlet', bc_val_left=13.0,
            bc_type_right='dirichlet', bc_val_right=2.7
        )

        rho = backward_euler_step_1d(A_bc, M, rho, dt, b_bc)
        rho_history.append(rho.copy())
        t_history.append(t)

    return np.array(rho_history), np.array(t_history)
