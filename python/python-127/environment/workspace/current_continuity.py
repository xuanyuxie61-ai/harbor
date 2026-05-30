
import numpy as np


def lax_wendroff_current_continuity(nx, nt, x_max, t_max, sigma, I_source,
                                     bc_type='neumann'):
    nx = int(nx)
    nt = int(nt)
    x_max = float(x_max)
    t_max = float(t_max)

    dx = x_max / (nx - 1)
    dt = t_max / (nt - 1)


    D_eff = 1.0 / sigma
    cfl = D_eff * dt / (dx**2)
    if cfl > 0.5:

        dt = 0.4 * dx**2 / D_eff
        nt = int(np.ceil(t_max / dt)) + 1
        dt = t_max / (nt - 1)
        cfl = D_eff * dt / (dx**2)

    x = np.linspace(0.0, x_max, nx)
    t = np.linspace(0.0, t_max, nt)

    V_array = np.zeros((nx, nt))
    V = np.zeros(nx)


    Vm = np.zeros(nx - 1)
    Jm = np.zeros(nx - 1)

    for it in range(nt):
        if it == 0:

            V = np.zeros(nx)
        else:


            for i in range(nx - 1):
                Vm[i] = 0.5 * (V[i] + V[i + 1])

                J_left = -sigma * (V[i] - (V[i - 1] if i > 0 else V[i])) / dx
                J_right = -sigma * (V[i + 1] - V[i]) / dx
                Jm[i] = 0.5 * (J_left + J_right)


                S_mid = 0.5 * (I_source(x[i], t[it - 1]) + I_source(x[i + 1], t[it - 1]))
                Vm[i] += 0.5 * dt * (S_mid + (J_left - J_right) / dx)


            for i in range(1, nx - 1):
                source = I_source(x[i], t[it])
                V[i] += dt * source

                V[i] += D_eff * dt * (V[i + 1] - 2 * V[i] + V[i - 1]) / (dx**2)


        if bc_type == 'neumann':
            V[0] = V[1]
            V[-1] = V[-2]
        elif bc_type == 'dirichlet':
            V[0] = 0.0
            V[-1] = 0.0
        elif bc_type == 'periodic':
            V[0] = V[-2]
            V[-1] = V[1]

        V_array[:, it] = V

    return V_array, x, t


def current_density_1d(V, x, sigma):
    V = np.asarray(V, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.mean(np.diff(x))

    if V.ndim == 1:
        J = np.zeros_like(V)
        J[1:-1] = -sigma * (V[2:] - V[:-2]) / (2.0 * dx)
        J[0] = J[1]
        J[-1] = J[-2]
        return J
    elif V.ndim == 2:
        nx, nt = V.shape
        J = np.zeros_like(V)
        for it in range(nt):
            J[1:-1, it] = -sigma * (V[2:, it] - V[:-2, it]) / (2.0 * dx)
            J[0, it] = J[1, it]
            J[-1, it] = J[-2, it]
        return J
    else:
        raise ValueError("V 必须为 1D 或 2D 数组")


def compute_charge_conservation_error(V_array, x, t, sigma, I_source_func):
    nx, nt = V_array.shape
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    D_eff = 1.0 / sigma

    dVdt = np.diff(V_array, axis=1) / dt
    d2Vdx2 = np.zeros((nx - 2, nt - 1))

    for it in range(nt - 1):
        for i in range(1, nx - 1):
            d2Vdx2[i - 1, it] = (V_array[i + 1, it] - 2 * V_array[i, it] + V_array[i - 1, it]) / (dx**2)

    source = np.zeros((nx - 2, nt - 1))
    for it in range(nt - 1):
        for i in range(1, nx - 1):
            source[i - 1, it] = I_source_func(x[i], t[it]) / sigma

    error = dVdt[1:-1, :] - D_eff * d2Vdx2 - source
    return error
