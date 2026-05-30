
import numpy as np
from typing import Tuple, List, Optional, Callable
from utils import solve_quadratic, clamp_value






def build_rigid_body_mass_matrix(
    mass: float,
    cog: np.ndarray,
    I: np.ndarray,
) -> np.ndarray:
    M = np.zeros((6, 6), dtype=float)
    M[0, 0] = mass
    M[1, 1] = mass
    M[2, 2] = mass
    M[0, 4] = mass * cog[2]
    M[0, 5] = -mass * cog[1]
    M[1, 3] = -mass * cog[2]
    M[1, 5] = mass * cog[0]
    M[2, 3] = mass * cog[1]
    M[2, 4] = -mass * cog[0]
    M[3, 1] = -mass * cog[2]
    M[3, 2] = mass * cog[1]
    M[4, 0] = mass * cog[2]
    M[4, 2] = -mass * cog[0]
    M[5, 0] = -mass * cog[1]
    M[5, 1] = mass * cog[0]
    M[3, 3] = I[0]
    M[4, 4] = I[1]
    M[5, 5] = I[2]

    M = 0.5 * (M + M.T)
    return M


def build_hydrostatic_restoring_matrix(
    rho: float,
    g: float,
    area_wp: float,
    I_xx: float,
    I_yy: float,
    I_xy: float,
    z_cob: float,
    z_cog: float,
) -> np.ndarray:
    C = np.zeros((6, 6), dtype=float)
    V_disp = area_wp * abs(z_cob)
    C[2, 2] = rho * g * area_wp
    C[3, 3] = rho * g * I_xx + rho * g * V_disp * (z_cob - z_cog)
    C[4, 4] = rho * g * I_yy + rho * g * V_disp * (z_cob - z_cog)
    C[5, 5] = rho * g * I_xx + rho * g * I_yy
    C[3, 4] = -rho * g * I_xy
    C[4, 3] = C[3, 4]
    C[2, 3] = -rho * g * area_wp * 0.0
    C[3, 2] = C[2, 3]
    C[2, 4] = rho * g * area_wp * 0.0
    C[4, 2] = C[2, 4]
    return C






def bdf2_solve(
    f_ode: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.asarray(y0, dtype=float)
    n_dof = len(y0)
    t0, tf = tspan
    if tf <= t0:
        raise ValueError("tf 必须大于 t0")
    if n_steps < 2:
        raise ValueError("n_steps 至少为 2")
    dt = (tf - t0) / n_steps

    t_arr = np.linspace(t0, tf, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, n_dof), dtype=float)
    y_arr[0, :] = y0


    t1 = t_arr[1]

    y_h = _backward_euler_step(f_ode, t0, y0, dt * 0.5, newton_tol, newton_max_iter)

    y1_be = _backward_euler_step(f_ode, t0, y0, dt, newton_tol, newton_max_iter)

    y1 = 2.0 * y_h - y0

    y1 = _clamp_dofs(y1)
    y_arr[1, :] = y1


    y_prev = y0
    y_curr = y1
    for n in range(1, n_steps):
        t_next = t_arr[n + 1]

        y_guess = 2.0 * y_curr - y_prev
        y_next = _bdf2_newton_solve(
            f_ode, t_next, dt, y_prev, y_curr, y_guess, newton_tol, newton_max_iter
        )
        y_next = _clamp_dofs(y_next)
        y_arr[n + 1, :] = y_next
        y_prev = y_curr
        y_curr = y_next

    return t_arr, y_arr


def _backward_euler_step(
    f: Callable,
    t0: float,
    y0: np.ndarray,
    dt: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    y = y0.copy()
    t = t0 + dt
    for _ in range(max_iter):
        f_val = f(t, y)
        if np.any(np.isnan(f_val)) or np.any(np.isinf(f_val)):

            return y0 + dt * f(t0, y0)
        y_new = y0 + dt * f_val
        if np.linalg.norm(y_new - y) < tol:
            return y_new

        y = 0.5 * y + 0.5 * y_new
    return y


def _bdf2_newton_solve(
    f: Callable,
    t: float,
    dt: float,
    y1: np.ndarray,
    y2: np.ndarray,
    y_guess: np.ndarray,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    y = y_guess.copy()
    for it in range(max_iter):
        f_val = f(t, y)
        if np.any(np.isnan(f_val)) or np.any(np.isinf(f_val)):
            break
        r = 3.0 * y - 4.0 * y2 + y1 - 2.0 * dt * f_val
        if np.linalg.norm(r) < tol:
            break

        y_new = (4.0 * y2 - y1 + 2.0 * dt * f_val) / 3.0
        y_new = _clamp_dofs(y_new)
        diff = np.linalg.norm(y_new - y)

        alpha = 0.7 if diff > 1.0 else 1.0
        y = alpha * y_new + (1.0 - alpha) * y
        if diff < tol:
            break
    return _clamp_dofs(y)


def _clamp_dofs(y: np.ndarray) -> np.ndarray:
    y = y.copy()
    for i in range(3):
        y[i] = clamp_value(y[i], -50.0, 50.0)
    for i in range(3, 6):
        y[i] = clamp_value(y[i], -0.5, 0.5)
    return y






def catenary_mooring_force(
    x_platform: float,
    y_platform: float,
    anchor_pos: np.ndarray,
    unstretched_length: float,
    line_weight: float,
    EA: float,
    horizontal_pretension: float,
) -> np.ndarray:
    dx = x_platform - anchor_pos[0]
    dy = y_platform - anchor_pos[1]

    dx = max(-200.0, min(200.0, dx))
    dy = max(-200.0, min(200.0, dy))
    horiz_dist = np.sqrt(dx ** 2 + dy ** 2)
    if horiz_dist < 1e-6:
        return np.zeros(6)
    w = line_weight
    L0 = unstretched_length
    H = horizontal_pretension

    for _ in range(50):
        if H < 1e-3:
            H = 1e-3
        arg = w * horiz_dist / H
        if arg > 50.0:

            H = max(1e-3, w * horiz_dist / 50.0)
            break
        s = np.sinh(arg)
        c = np.cosh(arg)
        f = (H / w) * s + (EA / w) * (np.arcsinh(arg) - arg) - L0

        ds_dH = c * (-w * horiz_dist / (H * H))
        df = (1.0 / w) * s + (H / w) * ds_dH
        df += (EA / w) * ((1.0 / np.sqrt(1.0 + arg * arg)) - 1.0) * (-w * horiz_dist / (H * H))
        if abs(df) < 1e-15:
            break
        delta = f / df
        H_new = H - delta
        if H_new < 1e-3:
            H_new = 1e-3
        if abs(H_new - H) < 1e-8:
            H = H_new
            break
        H = H_new


    angle = np.arctan2(dy, dx)
    Fx = -H * np.cos(angle)
    Fy = -H * np.sin(angle)
    force = np.zeros(6)
    force[0] = Fx
    force[1] = Fy
    return force






def partition_dofs_brute(
    coupling_weights: np.ndarray,
) -> Tuple[np.ndarray, float]:
    n = coupling_weights.shape[0]
    if n > 10:

        return _partition_greedy(coupling_weights)
    best_disc = float('inf')
    best_mask = np.zeros(n, dtype=int)
    total_subsets = 1 << n
    for mask in range(total_subsets):

        if mask == 0 or mask == (total_subsets - 1):
            continue
        sum0 = 0.0
        sum1 = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                w = coupling_weights[i, j]
                in0_i = (mask >> i) & 1
                in0_j = (mask >> j) & 1
                if in0_i == in0_j:
                    if in0_i:
                        sum0 += w
                    else:
                        sum1 += w
        disc = abs(sum0 - sum1)
        if disc < best_disc:
            best_disc = disc
            best_mask = np.array([(mask >> i) & 1 for i in range(n)])
    return best_mask, best_disc


def _partition_greedy(coupling_weights: np.ndarray) -> Tuple[np.ndarray, float]:
    n = coupling_weights.shape[0]
    mask = np.zeros(n, dtype=int)

    total_w = np.sum(coupling_weights, axis=1)
    order = np.argsort(-total_w)
    sum0 = 0.0
    sum1 = 0.0
    for idx in order:
        if sum0 <= sum1:
            mask[idx] = 0
            sum0 += total_w[idx]
        else:
            mask[idx] = 1
            sum1 += total_w[idx]
    disc = abs(sum0 - sum1)
    return mask, disc


def build_coupling_matrix_from_stiffness(
    K: np.ndarray, threshold: float = 1e-3
) -> np.ndarray:
    n = K.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            kii = abs(K[i, i])
            kjj = abs(K[j, j])
            if kii > 1e-12 and kjj > 1e-12:
                W[i, j] = abs(K[i, j]) / np.sqrt(kii * kjj)
            if W[i, j] < threshold:
                W[i, j] = 0.0
    return W






def simulate_platform_response(
    mass: float = 3.5e7,
    cog: Optional[np.ndarray] = None,
    inertia: Optional[np.ndarray] = None,
    A_add: Optional[np.ndarray] = None,
    B_rad: Optional[np.ndarray] = None,
    C_rest: Optional[np.ndarray] = None,
    wave_force_func: Optional[Callable] = None,
    mooring_config: Optional[List[dict]] = None,
    tspan: Tuple[float, float] = (0.0, 300.0),
    n_steps: int = 600,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    if cog is None:
        cog = np.array([0.0, 0.0, -10.0])
    if inertia is None:
        inertia = np.array([2.5e10, 2.5e10, 3.0e10])
    if A_add is None:

        A_add = np.diag([5.0e6, 5.0e6, 8.0e6, 1.0e9, 1.0e9, 5.0e8])
    if B_rad is None:

        B_rad = np.diag([2.0e5, 2.0e5, 3.0e5, 5.0e7, 5.0e7, 2.0e7])
    if C_rest is None:
        C_rest = np.diag([0.0, 0.0, 2.5e8, 1.5e10, 1.5e10, 5.0e9])

    M = build_rigid_body_mass_matrix(mass, cog, inertia)
    M_total = M + A_add


    K_coupling = M_total + B_rad + C_rest
    W = build_coupling_matrix_from_stiffness(K_coupling)
    partition, disc = partition_dofs_brute(W)


    n_dof = 6
    n_state = 2 * n_dof

    def ode_func(t: float, y: np.ndarray) -> np.ndarray:
        xi = y[:n_dof].copy()
        xi_dot = y[n_dof:].copy()

        for i in range(3):
            xi[i] = max(-50.0, min(50.0, xi[i]))
            xi_dot[i] = max(-10.0, min(10.0, xi_dot[i]))
        for i in range(3, 6):
            xi[i] = max(-0.5, min(0.5, xi[i]))
            xi_dot[i] = max(-0.3, min(0.3, xi_dot[i]))

        F_ext = np.zeros(n_dof)
        if wave_force_func is not None:
            F_ext += wave_force_func(t, xi)
        if mooring_config is not None:
            for mc in mooring_config:
                F_moor = catenary_mooring_force(
                    xi[0], xi[1], mc["anchor"], mc["length"],
                    mc["weight"], mc["EA"], mc["pretension"],
                )
                F_ext += F_moor[:n_dof]






        raise NotImplementedError("ode_func 中的运动方程组装需要实现")

    y0 = np.zeros(n_state)
    t_arr, y_arr = bdf2_solve(ode_func, tspan, y0, n_steps)

    info = {
        "partition": partition,
        "partition_discrepancy": disc,
        "M_total": M_total,
        "B_rad": B_rad,
        "C_rest": C_rest,
    }
    return t_arr, y_arr, info
