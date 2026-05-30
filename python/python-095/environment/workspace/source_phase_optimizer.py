
import numpy as np
import math


def zero_chandrupatla(f, x1, x2, epsilon=1.0e-10, delta=0.00001, max_iter=200):
    f1 = f(x1)
    f2 = f(x2)
    calls = 2

    if f1 * f2 > 0:
        raise ValueError("zero_chandrupatla: f(x1) and f(x2) must have opposite signs")

    t = 0.5
    iter_count = 0

    while iter_count < max_iter:
        x0 = x1 + t * (x2 - x1)
        f0 = f(x0)
        calls += 1


        if np.sign(f0) == np.sign(f1):
            x3 = x1
            f3 = f1
            x1 = x0
            f1 = f0
        else:
            x3 = x2
            f3 = f2
            x2 = x1
            f2 = f1
            x1 = x0
            f1 = f0


        if abs(f2) < abs(f1):
            xm = x2
            fm = f2
        else:
            xm = x1
            fm = f1

        tol = 2.0 * epsilon * abs(xm) + 0.5 * delta
        tl = tol / abs(x2 - x1)

        if tl >= 0.5 or abs(fm) < epsilon:
            break


        xi = (x1 - x2) / (x3 - x2)
        ph = (f1 - f2) / (f3 - f2)
        fl = 1.0 - math.sqrt(1.0 - xi)
        fh = math.sqrt(xi)

        if fl < ph < fh:
            al = (x3 - x1) / (x2 - x1)
            a = f1 / (f2 - f1)
            b = f3 / (f2 - f3)
            c = f1 / (f3 - f1)
            d = f2 / (f3 - f2)
            t = a * b + c * d * al
        else:
            t = 0.5

        t = max(t, tl)
        t = min(t, 1.0 - tl)
        iter_count += 1

    return xm, fm, calls


def optimize_source_phase(H_col, d, amplitude, phi_bounds=(0.0, 2.0 * math.pi)):
    H_col = np.asarray(H_col, dtype=complex)
    d = np.asarray(d, dtype=complex)

    def energy_gradient(phi):
        s = amplitude * np.exp(1j * phi)
        p = d + H_col * s

        grad = 2.0 * np.real(np.vdot(p, 1j * H_col * s))
        return grad


    n_brackets = 36
    phis = np.linspace(phi_bounds[0], phi_bounds[1], n_brackets + 1)
    vals = np.array([energy_gradient(p) for p in phis])

    best_phi = phis[0]
    best_energy = np.inf

    for i in range(n_brackets):
        if vals[i] == 0.0:
            candidate = phis[i]
        elif vals[i] * vals[i + 1] < 0:
            try:
                candidate, _, _ = zero_chandrupatla(energy_gradient, phis[i], phis[i + 1])
            except ValueError:
                continue
        else:
            continue

        s = amplitude * np.exp(1j * candidate)
        p = d + H_col * s
        energy = np.vdot(p, p).real
        if energy < best_energy:
            best_energy = energy
            best_phi = candidate


    for p in phi_bounds:
        s = amplitude * np.exp(1j * p)
        p_ = d + H_col * s
        energy = np.vdot(p_, p_).real
        if energy < best_energy:
            best_energy = energy
            best_phi = p

    return best_phi, best_energy
