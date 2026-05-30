
import numpy as np




PHYSICAL_CONSTANTS = {
    "mu_0": 4.0 * np.pi * 1e-7,
    "rho_core": 1.05e4,
    "sigma_core": 5.0e5,
    "eta_magnetic": 0.8,
    "nu_kinematic": 1.2e-6,
    "kappa_thermal": 5.0e-6,
    "core_radius": 3.48e6,
    "icb_radius": 1.22e6,
    "angular_velocity": 7.2921159e-5,
    "gravity_surface": 10.0,
}


def safe_div(a, b, eps=1e-30):
    b_safe = np.where(np.abs(b) < eps, np.sign(b + eps) * eps, b)
    return a / b_safe


def clip_bounds(x, x_min, x_max):
    return np.clip(x, x_min, x_max)


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(clip_bounds(safe_div(z, r), -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sphere_uniform_sample(n):
    p = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return p / norms


def lindberg_exact_solution(t):
    t = np.atleast_1d(t)
    n = t.size
    y = np.zeros((n, 4))
    dydt = np.zeros((n, 4))

    g1 = 1e4 * (t + 2.0 * np.exp(-t) - 2.0)
    g2 = 1e4 * (1.0 - np.exp(-t) - t * np.exp(-t))

    dg1dt = 1e4 * (1.0 - 2.0 * np.exp(-t))
    dg2dt = 1e4 * (t * np.exp(-t))

    y[:, 0] = np.exp(g1) * (np.cos(g2) + np.sin(g2))
    y[:, 1] = np.exp(g1) * (np.cos(g2) - np.sin(g2))
    y[:, 2] = 1.0 - 2.0 * np.exp(-t)
    y[:, 3] = t * np.exp(-t)

    dydt[:, 0] = (np.exp(g1) * dg1dt * (np.cos(g2) + np.sin(g2))
                  + np.exp(g1) * (-np.sin(g2) + np.cos(g2)) * dg2dt)
    dydt[:, 1] = (np.exp(g1) * dg1dt * (np.cos(g2) - np.sin(g2))
                  + np.exp(g1) * (-np.sin(g2) - np.cos(g2)) * dg2dt)
    dydt[:, 2] = 2.0 * np.exp(-t)
    dydt[:, 3] = (1.0 - t) * np.exp(-t)

    return y, dydt


def lindberg_rhs(t, y):
    _, dydt = lindberg_exact_solution(np.array([t]))
    return dydt[0, :]


def validate_lindberg(integrator_func, dt=0.001, tol=1e-6):
    tspan = np.array([0.0, 0.01])
    y0 = np.array([1.0, 1.0, -1.0, 0.0])
    t, y, _ = integrator_func(lindberg_rhs, tspan, y0, dt, tol)
    y_exact, _ = lindberg_exact_solution(np.array([t[-1]]))
    rel_err = np.linalg.norm(y[-1, :] - y_exact[0, :]) / (np.linalg.norm(y_exact[0, :]) + 1e-30)
    return rel_err


def condition_number_estimate(A):
    s = np.linalg.svd(A, compute_uv=False)
    s_max = np.max(s)
    s_min = np.max(s[s > 1e-15])
    return s_max / s_min
