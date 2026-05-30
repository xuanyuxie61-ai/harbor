
import numpy as np


def spring_double_parameters(m1=3.0, m2=5.0, k1=1.0, k2=10.0,
                              t0=0.0, y0=None, tstop=50.0):
    if y0 is None:
        y0 = np.array([0.0, 1.0, 0.0, 0.0])
    return {
        'm1': m1, 'm2': m2, 'k1': k1, 'k2': k2,
        't0': t0, 'y0': y0, 'tstop': tstop
    }


def spring_double_deriv(t, y, params):
    m1 = params['m1']
    m2 = params['m2']
    k1 = params['k1']
    k2 = params['k2']
    u1, v1, u2, v2 = y
    du1dt = v1
    dv1dt = (-k1 * u1 + k2 * (u2 - u1)) / m1
    du2dt = v2
    dv2dt = (-k2 * (u2 - u1)) / m2
    return np.array([du1dt, dv1dt, du2dt, dv2dt])


def rk4_integrate(dydt, tspan, y0, n, args=()):
    y0 = np.asarray(y0, dtype=float).flatten()
    m = len(y0)
    t0, tstop = tspan
    dt = (tstop - t0) / n
    t = np.zeros(n + 1)
    y = np.zeros((n + 1, m))
    t[0] = t0
    y[0, :] = y0
    for i in range(n):
        k1 = dydt(t[i], y[i, :], *args)
        k2 = dydt(t[i] + dt / 2.0, y[i, :] + dt * k1 / 2.0, *args)
        k3 = dydt(t[i] + dt / 2.0, y[i, :] + dt * k2 / 2.0, *args)
        k4 = dydt(t[i] + dt, y[i, :] + dt * k3, *args)
        t[i + 1] = t[i] + dt
        y[i + 1, :] = y[i, :] + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return t, y


def standing_wave_exact(x, t, c=0.2):
    u = np.sin(x) * np.cos(c * t)
    ut = -c * np.sin(x) * np.sin(c * t)
    utt = -c ** 2 * np.sin(x) * np.cos(c * t)
    ux = np.cos(x) * np.cos(c * t)
    uxx = -np.sin(x) * np.cos(c * t)
    return u, ut, utt, ux, uxx


def standing_wave_residual(x, t, c=0.2):
    _, _, utt, _, uxx = standing_wave_exact(x, t, c)
    return utt - c ** 2 * uxx


def seismic_wave_rk4_1d(nx, dx, nt, dt, c, source_time_fn, source_pos,
                         boundary='absorbing'):
    c = np.asarray(c, dtype=float)

    def deriv(t, y):
        u = y[:nx]
        v = y[nx:]
        dudt = v.copy()
        dvdt = np.zeros(nx)






        pass

        if 0 <= source_pos < nx:
            dvdt[source_pos] += source_time_fn(t)
        return np.concatenate([dudt, dvdt])
    
    y0 = np.zeros(2 * nx)
    t, y = rk4_integrate(deriv, (0.0, nt * dt), y0, nt)
    u_history = y[:, :nx]
    return u_history, t


def test_standing_wave_convergence():
    c = 0.5
    t_final = 2.0 * np.pi / c
    errors = []
    dxs = []
    for nx in [41, 81, 161]:
        dx = 2.0 * np.pi / (nx - 1)
        x = np.linspace(0.0, 2.0 * np.pi, nx)

        dt = 0.5 * dx / c
        nt = int(np.ceil(t_final / dt))
        dt = t_final / nt
        c_arr = np.full(nx, c)

        def source_fn(t):
            return 0.0
        u_hist, t = seismic_wave_rk4_1d(nx, dx, nt, dt, c_arr, source_fn, 0,
                                         boundary='reflecting')
        u_num = u_hist[-1, :]
        u_exact, _, _, _, _ = standing_wave_exact(x, t_final, c)

        u_exact_init = np.sin(x)

        err = np.sqrt(np.mean((u_num - u_exact_init * np.cos(c * t_final)) ** 2))
        errors.append(err)
        dxs.append(dx)
    return errors, dxs
