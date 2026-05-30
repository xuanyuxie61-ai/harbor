
import numpy as np
from scipy.special import lambertw


def lambert_w_approx(x, branch=0):
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.complex128)


    mask_real = (x >= -1.0 / np.e) & (x < np.inf)
    if np.any(mask_real):
        result[mask_real] = lambertw(x[mask_real], k=branch)


    mask_complex = x < -1.0 / np.e
    if np.any(mask_complex):
        result[mask_complex] = lambertw(x[mask_complex], k=branch)

    return result


def flame_deriv(t, y):
    y = np.asarray(y)
    return y ** 2 * (1.0 - y)


def energy_cascade_exact(t, y0=0.01):
    t = np.asarray(t, dtype=np.float64)
    if y0 <= 0.0 or y0 >= 1.0:
        raise ValueError("y0 必须在 (0, 1) 之间")

    a = (1.0 - y0) / y0
    y = np.zeros_like(t)
    for i in range(t.size):
        arg = a * np.exp(a - t.flat[i])
        if arg < -1.0 / np.e:

            y.flat[i] = 0.0
        else:
            w_val = lambertw(arg, k=0)
            y.flat[i] = 1.0 / (np.real(w_val) + 1.0)
    return y


def solve_energy_cascade_rk4(t_span, y0, n_steps=1000):
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = flame_deriv(t[i], y[i])
        k2 = flame_deriv(t[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = flame_deriv(t[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = flame_deriv(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


        if y[i + 1] > 1.0:
            y[i + 1] = 1.0
        elif y[i + 1] < 0.0:
            y[i + 1] = 0.0

    return t, y


def energy_saturation_time(y0, epsilon=0.99):
    if y0 <= 0.0 or y0 >= 1.0:
        return np.inf
    a = (1.0 - y0) / y0
    target = a * np.exp(a) * (1.0 / epsilon - 1.0)
    if target < -1.0 / np.e:
        return np.inf
    w_val = lambertw(target, k=0)
    tau_sat = a - np.real(w_val)
    return float(tau_sat)


def atmospheric_energy_model(intensity, tau, delta=0.001):
    if intensity <= 0:
        intensity = delta
    t_exact = np.array([tau])
    y_exact = energy_cascade_exact(t_exact, y0=intensity)
    return float(y_exact[0])


def test_energy_cascade():
    t = np.linspace(0, 2, 100)
    y0 = 0.01
    y_exact = energy_cascade_exact(t, y0)
    t_num, y_num = solve_energy_cascade_rk4((0, 2), y0, n_steps=500)

    y_interp = np.interp(t, t_num, y_num)
    max_err = np.max(np.abs(y_exact - y_interp))
    assert max_err < 0.05

    tau_sat = energy_saturation_time(y0)
    assert tau_sat > 0
    print("energy_cascade_ode 自测试通过")


if __name__ == "__main__":
    test_energy_cascade()
