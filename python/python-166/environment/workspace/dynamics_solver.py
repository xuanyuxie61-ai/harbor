
import numpy as np
from typing import Callable, Tuple, Optional


def cauchy_theta_method(f: Callable[[float, np.ndarray], np.ndarray],
                       tspan: Tuple[float, float],
                       y0: np.ndarray,
                       n: int,
                       theta: float = 0.5,
                       it_max: int = 20,
                       tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    if theta <= 0 or theta > 1:
        raise ValueError("theta must be in (0, 1]")
    if n < 1:
        raise ValueError("n must be >= 1")

    t0, tf = tspan
    dt = (tf - t0) / n
    m = len(y0)

    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, m))
    y[0] = y0

    for i in range(n):
        ti = t[i]
        yi = y[i]
        tm = ti + theta * dt


        ym = yi.copy()
        for _ in range(it_max):
            ym_new = yi + theta * dt * f(tm, ym)
            if np.linalg.norm(ym_new - ym) < tol:
                ym = ym_new
                break
            ym = ym_new


        y[i + 1] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * yi

    return t, y


def low_storage_rk4(f: Callable[[float, np.ndarray], np.ndarray],
                   tspan: Tuple[float, float],
                   y0: np.ndarray,
                   n: int) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = tspan
    dt = (tf - t0) / n
    m = len(y0)

    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, m))
    y[0] = y0


    a = np.array([0.0,
                  -567301805773.0 / 1357537059087.0,
                  -2404267990393.0 / 2016746695238.0,
                  -3550918686646.0 / 2091501179385.0,
                  -1275806237668.0 / 842570457699.0])
    b = np.array([1432997174477.0 / 9575080441755.0,
                  5161836677717.0 / 13612068292357.0,
                  1720146321549.0 / 2090206949498.0,
                  3134564353537.0 / 4481467310338.0,
                  2277821191437.0 / 14882151754819.0])
    c = np.array([0.0,
                  1432997174477.0 / 9575080441755.0,
                  2526269341429.0 / 6820363962896.0,
                  2006345519317.0 / 3224310063776.0,
                  2802321613138.0 / 2924317926251.0])

    for i in range(n):
        ti = t[i]
        yi = y[i]
        res = np.zeros(m)
        dy = np.zeros(m)

        for s in range(5):
            ts = ti + c[s] * dt
            ys = yi + a[s] * dy
            res = f(ts, ys)
            dy = b[s] * dt * res + dy

        y[i + 1] = yi + dy

    return t, y


def sawtooth_driver(t: float, omega: float = 1.0) -> float:
    T = 2.0 * omega * np.pi
    phase = t + omega * np.pi
    f = np.mod(phase, T) - omega * np.pi
    return f


def driven_harmonic_oscillator(t: float, y: np.ndarray,
                               omega0: float = 1.0,
                               zeta: float = 0.1,
                               omega_drive: float = 1.0) -> np.ndarray:
    u, v = y
    forcing = sawtooth_driver(t, omega_drive)
    dudt = v
    dvdt = -omega0 ** 2 * u - 2.0 * zeta * omega0 * v + forcing
    return np.array([dudt, dvdt])


def cosserat_dynamics_rhs(t: float, state: np.ndarray,
                          L: float, Ns: int,
                          E: float, G: float, A: float,
                          Ixx: float, Iyy: float, J: float,
                          rho: float, F_ext: Optional[Callable] = None,
                          chemical_state: Optional[np.ndarray] = None,
                          chemo_params: Optional[dict] = None) -> np.ndarray:
    dof_per_node = 3
    n_nodes = Ns + 1
    total_dof = n_nodes * dof_per_node

    if len(state) != 2 * total_dof:
        raise ValueError(f"state length {len(state)} != {2*total_dof}")

    q = state[:total_dof]
    qdot = state[total_dof:]

    ds = L / Ns
    if ds < 1e-14:
        raise ValueError("ds too small")


    M_diag = np.zeros(total_dof)
    for i in range(n_nodes):
        base = i * dof_per_node
        M_diag[base] = rho * A
        M_diag[base + 1] = rho * A
        M_diag[base + 2] = rho * (Ixx + Iyy)


    E_eff = E
    if chemical_state is not None and chemo_params is not None:
        from hyperelastic_law import chemo_mechanical_coupling
        E0 = chemo_params.get('E0', E)
        gamma = chemo_params.get('gamma', 0.0)
        beta_chem = chemo_params.get('beta_chem', 0.0)
        E_eff = chemo_mechanical_coupling(chemical_state, 0.0, E0, gamma, beta_chem)

    EA_eff = E_eff * A
    EI = E_eff * max(Ixx, Iyy)


    Kq = np.zeros(total_dof)


    for i in range(n_nodes):
        base = i * dof_per_node
        if i == 0:

            Kq[base] = q[base] * 1.0e12
        elif i == n_nodes - 1:
            Kq[base] = EA_eff * (q[base] - q[base - dof_per_node]) / ds ** 2
        else:
            Kq[base] = -EA_eff * (q[base + dof_per_node] - 2.0 * q[base] + q[base - dof_per_node]) / ds ** 2



    stab_factor = 0.1
    for i in range(n_nodes):
        base = i * dof_per_node + 1
        if i == 0 or i == 1:

            Kq[base] = q[base] * 1.0e6
        elif i == n_nodes - 1 or i == n_nodes - 2:

            if i == n_nodes - 2:
                Kq[base] = EI * (q[base - 2*dof_per_node] - 2.0*q[base - dof_per_node] + q[base]) / ds ** 4
            else:
                Kq[base] = EI * (q[base] - q[base - dof_per_node]) / ds ** 4
        else:
            Kq[base] = EI * (q[base - 2*dof_per_node] - 4.0*q[base - dof_per_node]
                             + 6.0*q[base] - 4.0*q[base + dof_per_node]
                             + q[base + 2*dof_per_node]) / ds ** 4

            Kq[base] += stab_factor * EI * q[base] / ds ** 4


    GJ = G * J
    for i in range(n_nodes):
        base = i * dof_per_node + 2
        if i == 0:
            Kq[base] = q[base] * 1.0e12
        elif i == n_nodes - 1:
            Kq[base] = GJ * (q[base] - q[base - dof_per_node]) / ds ** 2
        else:
            Kq[base] = -GJ * (q[base + dof_per_node] - 2.0 * q[base] + q[base - dof_per_node]) / ds ** 2


    alpha_ray = 0.5
    beta_ray = 0.001
    Cqdot = alpha_ray * M_diag * qdot + beta_ray * Kq * 0.0


    F = np.zeros(total_dof)
    if F_ext is not None:
        for i in range(n_nodes):
            s = i * ds
            f = F_ext(t, s)
            base = i * dof_per_node
            F[base:base + 3] = f


    rhs = F - Cqdot - Kq


    qddot = rhs / M_diag
    qddot = np.where(np.abs(M_diag) < 1e-14, 0.0, qddot)


    dstate = np.zeros(len(state))
    dstate[:total_dof] = qdot
    dstate[total_dof:] = qddot
    return dstate


def integrate_cosserat_dynamics(tspan: Tuple[float, float],
                                q0: np.ndarray, qdot0: np.ndarray,
                                Ns: int, L: float,
                                material_params: dict,
                                n_steps: int = 500,
                                method: str = 'rk4') -> Tuple[np.ndarray, np.ndarray]:
    state0 = np.concatenate([q0, qdot0])

    def rhs(t, s):
        return cosserat_dynamics_rhs(
            t, s, L, Ns,
            material_params['E'], material_params['G'],
            material_params['A'], material_params['Ixx'],
            material_params['Iyy'], material_params['J'],
            material_params['rho']
        )

    if method == 'rk4':
        return low_storage_rk4(rhs, tspan, state0, n_steps)
    elif method == 'cauchy':
        return cauchy_theta_method(rhs, tspan, state0, n_steps, theta=0.5)
    else:
        raise ValueError(f"Unknown method: {method}")
