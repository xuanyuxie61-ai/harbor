# -*- coding: utf-8 -*-

import numpy as np


def chirikov_map_step(xy, K=0.55):
    x, y = xy[0], xy[1]
    y_new = y + K * np.sin(x)
    x_new = x + y_new
    y_new = np.mod(y_new, 2.0 * np.pi)
    x_new = np.mod(x_new, 2.0 * np.pi)
    return np.array([x_new, y_new])


def chirikov_hamiltonian(xy, K=0.55):
    x, y = xy[0], xy[1]
    return 0.5 * y ** 2 + K * np.cos(x)


def chirikov_orbit(n_steps, xy0=None, K=0.55):
    if xy0 is None:
        xy0 = np.array([0.6, 0.7])
    orbit = np.zeros((n_steps + 1, 2))
    energy = np.zeros(n_steps + 1)
    orbit[0] = xy0
    energy[0] = chirikov_hamiltonian(xy0, K)
    for i in range(n_steps):
        orbit[i + 1] = chirikov_map_step(orbit[i], K)
        energy[i + 1] = chirikov_hamiltonian(orbit[i + 1], K)
    return orbit, energy


def pde_hamiltonian(u, v, D2, nonlinear_potential_func=None, dx=1.0):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    kinetic = 0.5 * dx * np.dot(v, v)

    potential = -0.5 * dx * np.dot(u, D2 @ u)
    nonlinear = 0.0
    if nonlinear_potential_func is not None:
        nonlinear = dx * np.sum(nonlinear_potential_func(u))
    H = kinetic + potential + nonlinear
    return H, {"kinetic": kinetic, "potential": potential, "nonlinear": nonlinear}


def velocity_verlet_step(u, v, force_func, dt, mass=1.0):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = force_func(u) / mass
    u_new = u + v * dt + 0.5 * a * dt ** 2
    a_new = force_func(u_new) / mass
    v_new = v + 0.5 * (a + a_new) * dt
    return u_new, v_new


def ensemble_energy_drift(ensemble_u, ensemble_v, D2, dt_steps, force_func,
                          nonlinear_potential_func=None, dx=1.0):
    n_ens, n_dof = ensemble_u.shape
    energy_drift = np.zeros(n_ens)
    for e in range(n_ens):
        u = ensemble_u[e].copy()
        v = ensemble_v[e].copy()
        H0, _ = pde_hamiltonian(u, v, D2, nonlinear_potential_func, dx)
        for _ in range(dt_steps):
            u, v = velocity_verlet_step(u, v, force_func, 0.01)
        H1, _ = pde_hamiltonian(u, v, D2, nonlinear_potential_func, dx)
        if abs(H0) > 1e-15:
            energy_drift[e] = abs(H1 - H0) / abs(H0)
        else:
            energy_drift[e] = abs(H1 - H0)
    return energy_drift
