
import numpy as np










def sensitive_photon_deriv(t, y, growth_rate=1.0):
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError("y must have shape (2,).")
    dydt = np.zeros(2, dtype=float)
    dydt[0] = y[1]
    dydt[1] = growth_rate * growth_rate * y[0]
    return dydt


def sensitive_photon_exact(t, y0, growth_rate=1.0):
    t = np.asarray(t, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    eps = y0[0] - 1.0
    n = t.size
    y = np.zeros((n, 2), dtype=float)
    for i in range(n):
        y[i, 0] = (1.0 - eps / 2.0) * np.exp(-growth_rate * t[i]) + (eps / 2.0) * np.exp(growth_rate * t[i])
        y[i, 1] = -(1.0 - eps / 2.0) * np.exp(-growth_rate * t[i]) + (eps / 2.0) * np.exp(growth_rate * t[i])
        y[i, 1] *= growth_rate
    return y






def ode_midpoint_solve(f, a, b, ya, n_steps, theta=0.5, it_max=10):
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if a >= b:
        raise ValueError("Require a < b.")
    ya = np.atleast_1d(np.asarray(ya, dtype=float))
    dim = ya.size

    t = np.linspace(a, b, n_steps + 1)
    y = np.zeros((n_steps + 1, dim), dtype=float)
    y[0, :] = ya
    h = (b - a) / n_steps

    for i in range(n_steps):
        xm = t[i] + theta * h
        ym = y[i, :].copy()
        for _ in range(it_max):
            ym = y[i, :] + theta * h * np.atleast_1d(f(xm, ym))
        y[i + 1, :] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * y[i, :]
    return t, y






def layered_photon_transport(layer_boundaries, layer_properties, y0, n_steps_per_layer=50):
    layer_boundaries = np.asarray(layer_boundaries, dtype=float)
    if len(layer_boundaries) < 2:
        raise ValueError("Need at least 2 boundaries.")
    n_layers = len(layer_boundaries) - 1
    if len(layer_properties) != n_layers:
        raise ValueError("layer_properties length must match number of layers.")

    z_all = []
    y_all = []
    y_current = np.asarray(y0, dtype=float)

    for i in range(n_layers):
        a = layer_boundaries[i]
        b = layer_boundaries[i + 1]
        props = layer_properties[i]
        mu_a = props['mu_a']
        mu_s = props['mu_s']
        g = props['g']
        mu_s_prime = (1.0 - g) * mu_s

        D = 1.0 / (3.0 * (mu_s_prime + mu_a))
        lambda_eff = np.sqrt(mu_a / D) if D > 0 else 0.0

        def f_layer(z, y):
            return sensitive_photon_deriv(z, y, growth_rate=lambda_eff)

        t, y = ode_midpoint_solve(f_layer, a, b, y_current, n_steps_per_layer)
        if i == 0:
            z_all.extend(t)
            y_all.extend(y)
        else:
            z_all.extend(t[1:])
            y_all.extend(y[1:])
        y_current = y[-1, :].copy()

    return np.array(z_all), np.array(y_all)






def fitzhugh_nagumo_deriv(t, y, a=0.7, b=0.8, c=12.5, d=0.5):
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError("y must have shape (2,).")
    v = y[0]
    w = y[1]
    dvdt = v - (v ** 3) / 3.0 - w + d
    dwdt = (v + a - b * w) / c
    return np.array([dvdt, dwdt])






def glycolysis_deriv(t, y, a=0.08, b=0.6):
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError("y must have shape (2,).")
    u = y[0]
    v = y[1]
    dudt = -u + a * v + u * u * v
    dvdt = b - a * v - u * u * v
    return np.array([dudt, dvdt])


def glycolysis_equilibrium(a=0.08, b=0.6):
    denom = a + b * b
    if abs(denom) < 1e-14:
        raise ValueError("Denominator too small in equilibrium calculation.")
    return np.array([b, b / denom])






def refractive_index_from_bio_state(v_membrane, u_metabolite,
                                     n0=1.33, alpha_eo=1e-4, alpha_thermo=1e-3):
    n = n0 + alpha_eo * v_membrane + alpha_thermo * u_metabolite
    return max(n, 1.0)
