
import numpy as np


def bisection_method(f, a, b, tol=1e-12, max_iter=100):
    fa = f(a)
    fb = f(b)
    if np.sign(fa) == np.sign(fb):
        raise ValueError("f(a) and f(b) must have opposite signs.")

    it = 0
    while abs(b - a) > tol and it < max_iter:
        c = (a + b) / 2.0
        fc = f(c)
        it += 1
        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    root = (a + b) / 2.0
    return root, it, fa, fb


def expand_bracket(f, a0, b0, max_expand=20, factor=2.0):
    a, b = float(a0), float(b0)
    fa, fb = f(a), f(b)
    if np.sign(fa) != np.sign(fb):
        return a, b

    for _ in range(max_expand):
        if abs(fa) < abs(fb):
            a = a - factor * (b - a)
            fa = f(a)
        else:
            b = b + factor * (b - a)
            fb = f(b)
        if np.sign(fa) != np.sign(fb):
            return a, b

    raise RuntimeError("Failed to find a sign-changing bracket.")


def find_single_sphere_resonance(eps_medium, omega_p=9.0e15,
                                  gamma=1.0e14, eps_inf=9.0,
                                  bracket=None):
    def eps_metal(omega):
        return eps_inf - (omega_p ** 2) / (omega ** 2 + 1j * gamma * omega)

    def target(omega):
        if omega <= 0:
            return 1.0
        return np.real(eps_metal(omega) + 2.0 * eps_medium)

    if bracket is None:
        omega_est = omega_p / np.sqrt(eps_inf + 2.0 * eps_medium)
        a = 0.5 * omega_est
        b = 1.5 * omega_est
    else:
        a, b = bracket

    a, b = expand_bracket(target, a, b)
    root, it, _, _ = bisection_method(target, a, b)
    return root


def find_collective_resonance(positions, polarizability_func,
                               omega_min, omega_max,
                               eps_medium=1.0, num_points=200):
    from dipole_coupling import build_coupling_matrix

    omegas = np.linspace(omega_min, omega_max, num_points)
    figure_of_merit = np.zeros(num_points)

    for i, omg in enumerate(omegas):
        alphas = polarizability_func(omg)
        try:
            A = build_coupling_matrix(positions, alphas, omg, eps_medium)



            eigvals = np.linalg.eigvalsh(A.real)
            if np.any(eigvals <= 0):
                figure_of_merit[i] = 0.0
            else:
                figure_of_merit[i] = np.sum(1.0 / eigvals)
        except Exception:
            figure_of_merit[i] = 0.0


    if np.all(figure_of_merit == 0):
        return (omega_min + omega_max) / 2.0

    idx_max = np.argmax(figure_of_merit)
    if idx_max == 0 or idx_max == num_points - 1:
        return omegas[idx_max]


    o1, o2, o3 = omegas[idx_max - 1], omegas[idx_max], omegas[idx_max + 1]
    f1, f2, f3 = figure_of_merit[idx_max - 1], figure_of_merit[idx_max], figure_of_merit[idx_max + 1]



    denom = f3 - 2.0 * f2 + f1
    if abs(denom) < 1e-30:
        return o2
    omega_res = o2 - 0.5 * (o3 - o1) * (f3 - f1) / denom
    omega_res = np.clip(omega_res, omega_min, omega_max)
    return float(omega_res)
