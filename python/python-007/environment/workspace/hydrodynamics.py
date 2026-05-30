import numpy as np






def rk3_step(f_func, t, y, dt):
    y = np.asarray(y, dtype=np.float64)

    k1 = dt * np.asarray(f_func(t, y), dtype=np.float64)
    k2 = dt * np.asarray(f_func(t + dt, y + k1), dtype=np.float64)
    k3 = dt * np.asarray(f_func(t + 0.5 * dt, y + 0.25 * k1 + 0.25 * k2), dtype=np.float64)

    y_new = y + (k1 + k2 + 4.0 * k3) / 6.0
    return y_new


def rk3_integrate(f_func, t_span, y0, n_steps):
    y0 = np.asarray(y0, dtype=np.float64)
    t0, tf = t_span
    dt = (tf - t0) / n_steps

    t_array = np.zeros(n_steps + 1, dtype=np.float64)
    y_array = np.zeros((n_steps + 1, len(y0)), dtype=np.float64)

    t_array[0] = t0
    y_array[0] = y0

    for k in range(n_steps):
        y_array[k + 1] = rk3_step(f_func, t_array[k], y_array[k], dt)
        t_array[k + 1] = t_array[k] + dt

    return t_array, y_array






def gradient_2d(field, dx, dy):
    field = np.asarray(field, dtype=np.float64)
    nx, ny = field.shape

    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)


    if nx > 2:
        grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dx)

    if nx > 1:
        grad_x[0, :] = (field[1, :] - field[0, :]) / dx
        grad_x[-1, :] = (field[-1, :] - field[-2, :]) / dx

    if ny > 2:
        grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)
    if ny > 1:
        grad_y[:, 0] = (field[:, 1] - field[:, 0]) / dy
        grad_y[:, -1] = (field[:, -1] - field[:, -2]) / dy

    return grad_x, grad_y


def gradient_1d(field, dx):
    field = np.asarray(field, dtype=np.float64)
    n = len(field)
    grad = np.zeros_like(field)

    if n > 2:
        grad[1:-1] = (field[2:] - field[:-2]) / (2.0 * dx)
    if n > 1:
        grad[0] = (field[1] - field[0]) / dx
        grad[-1] = (field[-1] - field[-2]) / dx

    return grad


def laplacian_1d(field, dx):
    field = np.asarray(field, dtype=np.float64)
    n = len(field)
    lap = np.zeros_like(field)

    if n > 2:
        lap[1:-1] = (field[2:] - 2.0 * field[1:-1] + field[:-2]) / (dx * dx)
    if n > 1:

        lap[0] = (field[1] - field[0]) / (dx * dx)
        lap[-1] = (field[-1] - field[-2]) / (dx * dx)

    return lap


def divergence_cylindrical(v_r, v_phi, v_z, r_grid, dr, dz):
    v_r = np.asarray(v_r, dtype=np.float64)
    v_z = np.asarray(v_z, dtype=np.float64)
    nr, nz = v_r.shape


    dv_z_dz = np.zeros_like(v_z)
    if nz > 2:
        dv_z_dz[:, 1:-1] = (v_z[:, 2:] - v_z[:, :-2]) / (2.0 * dz)
    if nz > 1:
        dv_z_dz[:, 0] = (v_z[:, 1] - v_z[:, 0]) / dz
        dv_z_dz[:, -1] = (v_z[:, -1] - v_z[:, -2]) / dz


    rvr = r_grid.reshape(-1, 1) * v_r
    d_rvr_dr = np.zeros_like(v_r)
    if nr > 2:
        d_rvr_dr[1:-1, :] = (rvr[2:, :] - rvr[:-2, :]) / (2.0 * dr)
    if nr > 1:
        d_rvr_dr[0, :] = (rvr[1, :] - rvr[0, :]) / dr
        d_rvr_dr[-1, :] = (rvr[-1, :] - rvr[-2, :]) / dr


    div = np.zeros_like(v_r)
    for i in range(nr):
        r = r_grid[i]
        if r > 1e-15:
            div[i, :] = d_rvr_dr[i, :] / r + dv_z_dz[i, :]
        else:
            div[i, :] = dv_z_dz[i, :]

    return div


def compute_cfl_timestep(v_r, v_phi, v_z, cs, dr, dz, r_grid, cfl=0.3):
    v_r = np.asarray(v_r, dtype=np.float64)
    v_z = np.asarray(v_z, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)


    v_mag = np.sqrt(v_r ** 2 + v_phi ** 2 + v_z ** 2)


    v_eff = v_mag + cs
    v_eff = np.where(v_eff < 1e-15, 1e-15, v_eff)


    dt_r = np.zeros_like(v_eff)
    for i in range(len(r_grid)):
        dt_r[i, :] = dr / v_eff[i, :]

    dt_z = dz / v_eff

    dt_min = min(np.min(dt_r), np.min(dt_z))
    return cfl * dt_min
