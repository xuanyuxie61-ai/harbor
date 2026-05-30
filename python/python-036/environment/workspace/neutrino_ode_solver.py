
import numpy as np
from constants import EARTH_RADIUS_KM, KM_TO_EV_INV
from pmns_matrix import build_pmns_matrix, build_mass_matrix
from neutrino_hamiltonian import build_vacuum_hamiltonian, build_matter_hamiltonian


def euler_step(y, t, dt, dydt):
    y = np.asarray(y, dtype=np.complex128)
    f_val = dydt(t, y)
    return y + dt * f_val


def solve_euler(dydt, tspan, y0, n_steps):
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)

    t = np.zeros(n_steps + 1, dtype=np.float64)
    y = np.zeros((n_steps + 1, m), dtype=np.complex128)

    t[0] = t0
    y[0, :] = np.asarray(y0, dtype=np.complex128)

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        y[i + 1, :] = euler_step(y[i, :], t[i], dt, dydt)

    return t, y


def rk4_step(y, t, dt, dydt):
    y = np.asarray(y, dtype=np.complex128)
    k1 = dydt(t, y)
    k2 = dydt(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = dydt(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = dydt(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_rk4(dydt, tspan, y0, n_steps):
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)

    t = np.zeros(n_steps + 1, dtype=np.float64)
    y = np.zeros((n_steps + 1, m), dtype=np.complex128)

    t[0] = t0
    y[0, :] = np.asarray(y0, dtype=np.complex128)

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        y[i + 1, :] = rk4_step(y[i, :], t[i], dt, dydt)

    return t, y


def solve_neutrino_oscillation_ode(
        energy_gev, baseline_km,
        matter_potential_ev=None,
        n_steps=1000, method='rk4',
        theta12=None, theta23=None, theta13=None,
        delta_cp=None, delta_m2_21=None, delta_m2_31=None,
        hierarchy='normal', initial_flavor='electron'
    ):
    from pmns_matrix import get_initial_flavor_state

    if baseline_km < 0:
        raise ValueError("baseline_km must be non-negative")
    if energy_gev <= 0:
        raise ValueError("energy_gev must be positive")

    psi0 = get_initial_flavor_state(initial_flavor)

    if matter_potential_ev is None or matter_potential_ev == 0.0:

        H = build_vacuum_hamiltonian(
            energy_gev, theta12, theta23, theta13, delta_cp,
            delta_m2_21, delta_m2_31, hierarchy
        )
    else:
        H = build_matter_hamiltonian(
            energy_gev, matter_potential_ev,
            theta12, theta23, theta13, delta_cp,
            delta_m2_21, delta_m2_31, hierarchy
        )

    def dydt(t, y):
        return -1j * (H @ y)

    tspan = (0.0, baseline_km)

    if method == 'euler':
        t, y = solve_euler(dydt, tspan, psi0, n_steps)
    elif method == 'rk4':
        t, y = solve_rk4(dydt, tspan, psi0, n_steps)
    elif method == 'matrix_exp':

        t = np.array([0.0, baseline_km])
        L_ev_inv = baseline_km * KM_TO_EV_INV
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        D = np.diag(np.exp(-1j * eigenvalues * L_ev_inv))
        U_prop = eigenvectors @ D @ eigenvectors.conj().T
        psi_final = U_prop @ psi0
        y = np.array([psi0, psi_final], dtype=np.complex128)
    else:
        raise ValueError("method must be 'euler', 'rk4', or 'matrix_exp'")


    P_ee = np.abs(y[:, 0]) ** 2
    P_em = np.abs(y[:, 1]) ** 2
    P_et = np.abs(y[:, 2]) ** 2

    return {
        't': t,
        'P_ee': P_ee,
        'P_em': P_em,
        'P_et': P_et,
        'prob_final': np.array([P_ee[-1], P_em[-1], P_et[-1]], dtype=np.float64),
        'psi_final': y[-1, :]
    }


def solve_varying_matter_ode(
        energy_gev, baseline_km,
        matter_potential_func,
        n_steps=2000, method='rk4',
        theta12=None, theta23=None, theta13=None,
        delta_cp=None, delta_m2_21=None, delta_m2_31=None,
        hierarchy='normal', initial_flavor='electron'
    ):
    from pmns_matrix import get_initial_flavor_state

    psi0 = get_initial_flavor_state(initial_flavor)
    U = build_pmns_matrix(theta12, theta23, theta13, delta_cp)
    M2 = build_mass_matrix(delta_m2_21, delta_m2_31, hierarchy)

    H_vac = (1.0 / (2.0 * energy_gev * 1e9)) * (U @ M2 @ U.conj().T)










    raise NotImplementedError("HOLE 3: dydt 闭包尚未实现")

    tspan = (0.0, baseline_km)

    if method == 'euler':
        t, y = solve_euler(dydt, tspan, psi0, n_steps)
    elif method == 'rk4':
        t, y = solve_rk4(dydt, tspan, psi0, n_steps)
    else:
        raise ValueError("method must be 'euler' or 'rk4' for varying matter")

    P_ee = np.abs(y[:, 0]) ** 2
    P_em = np.abs(y[:, 1]) ** 2
    P_et = np.abs(y[:, 2]) ** 2

    return {
        't': t,
        'P_ee': P_ee,
        'P_em': P_em,
        'P_et': P_et,
        'prob_final': np.array([P_ee[-1], P_em[-1], P_et[-1]], dtype=np.float64),
        'psi_final': y[-1, :]
    }


def r8but_sl(n, mu, a, b):
    x = np.asarray(b, dtype=np.float64).copy()

    for j in range(n - 1, -1, -1):

        diag_idx = mu
        x[j] = x[j] / a[diag_idx, j]
        jlo = max(0, j - mu)
        for i in range(jlo, j):

            a_idx = mu + i - j
            x[i] = x[i] - a[a_idx, j] * x[j]

    return x


def solve_banded_upper_triangular(A_dense, b):
    n = len(b)

    mu = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A_dense[i, j]) > 1e-14:
                mu = max(mu, j - i)


    a_r8but = np.zeros((mu + 1, n), dtype=np.float64)
    for j in range(n):
        for i in range(max(0, j - mu), j + 1):
            a_r8but[mu + i - j, j] = A_dense[i, j]

    return r8but_sl(n, mu, a_r8but, b)
