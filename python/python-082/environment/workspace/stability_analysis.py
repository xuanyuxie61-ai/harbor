
import numpy as np


def stability_function(z, method='rk4'):
    z = np.asarray(z, dtype=complex)
    method = method.lower()

    if method == 'fe':
        return 1.0 + z
    elif method == 'be':
        return 1.0 / (1.0 - z)
    elif method == 'trapezoidal':
        return (1.0 + 0.5 * z) / (1.0 - 0.5 * z)
    elif method == 'rk4':
        return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0
    elif method == 'ab2':

        return np.ones_like(z)
    elif method == 'bdf2':


        return 1.0 / (1.0 - z + 0.5 * z ** 2)
    else:
        raise ValueError(f"Unknown method: {method}")


def stability_region_grid(method='rk4', xlim=(-5, 5), ylim=(-5, 5),
                          npts=401):
    x = np.linspace(xlim[0], xlim[1], npts)
    y = np.linspace(ylim[0], ylim[1], npts)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    Rval = stability_function(Z, method)
    Rabs = np.abs(Rval)
    return X, Y, Rabs


def is_stable(method, z):
    Rabs = np.abs(stability_function(z, method))
    return Rabs <= 1.0 + 1e-10


def compute_cfl_damage(damage_model, d_state, dt, method='rk4'):
    d_f = d_state[0]

    lambda_eff = -(1.0 / damage_model.epsilon) * (3.0 * d_f ** 2 - damage_model.a_param)
    z = dt * lambda_eff
    cfl = abs(z)
    stable = is_stable(method, z)
    return cfl, stable


def recommend_timestep(damage_model, d_state, method='rk4', safety=0.8):
    d_f = d_state[0]
    lambda_eff = -(1.0 / damage_model.epsilon) * (3.0 * d_f ** 2 - damage_model.a_param)
    lambda_max = abs(lambda_eff)

    if method == 'fe':
        C = 1.0
    elif method == 'rk4':
        C = 2.78
    elif method == 'be':
        C = 1e12
    elif method == 'trapezoidal':
        C = 1e12
    else:
        C = 2.0

    if lambda_max < 1e-14:
        return 1.0
    dt_max = safety * C / lambda_max
    return dt_max


def a_stability_test(method, n_test=1000):

    re_z = -np.random.rand(n_test) * 10.0
    im_z = (np.random.rand(n_test) - 0.5) * 20.0
    z = re_z + 1j * im_z
    Rabs = np.abs(stability_function(z, method))
    max_amp = np.max(Rabs)
    return max_amp <= 1.0 + 1e-6, max_amp


def stability_diagnostic(damage_model, t_span, n_steps, method='rk4'):
    y0 = np.array([0.01, 0.01])
    t_array, y_array = damage_model.rk4_integrate(y0, t_span[0], t_span[1], n_steps)
    dt = t_array[1] - t_array[0]

    cfl_values = []
    violations = 0
    for y in y_array:
        cfl, stable = compute_cfl_damage(damage_model, y, dt, method)
        cfl_values.append(cfl)
        if not stable:
            violations += 1

    return {
        'dt': dt,
        'cfl_mean': np.mean(cfl_values),
        'cfl_max': np.max(cfl_values),
        'violations': violations,
        'violation_ratio': violations / len(cfl_values)
    }
