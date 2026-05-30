
import numpy as np


class RootFinderError(Exception):
    pass


def newton_method(f, df, x0, fatol=1e-12, xatol=1e-12, step_max=50, xmin=-1e6, xmax=1e6):
    xa = float(x0)
    fxa = float(f(xa))
    step = 0.0

    for step_num in range(1, step_max + 1):
        if xa < xmin or xa > xmax:
            raise RootFinderError(f"Iterate left region [{xmin}, {xmax}]: x={xa}")

        if abs(fxa) <= fatol:
            return xa

        if step_num > 1 and abs(step) <= xatol:
            return xa

        fp = float(df(xa))
        if abs(fp) < 1e-15:
            raise RootFinderError("Derivative vanishes in Newton method")

        step = fxa / fp
        xa = xa - step
        fxa = float(f(xa))


    return xa


def brent_method(f, xa, xb, fatol=1e-12, xatol=1e-12, xrtol=1e-12, step_max=100):
    fxa = float(f(xa))
    fxb = float(f(xb))


    if fxa * fxb > 0:
        raise RootFinderError("Brent method requires f(xa)*f(xb) <= 0")

    xc = xa
    fxc = fxa
    d = xb - xa
    e = d

    for step_num in range(step_max + 1):
        if abs(fxc) < abs(fxb):
            xa = xb
            xb = xc
            xc = xa
            fxa = fxb
            fxb = fxc
            fxc = fxa

        xtol = 2.0 * xrtol * abs(xb) + 0.5 * xatol
        xm = 0.5 * (xc - xb)

        if abs(xm) <= xtol:
            return xb
        if abs(fxb) <= fatol:
            return xb


        if abs(e) < xtol or abs(fxa) <= abs(fxb):
            d = xm
            e = d
        else:
            s = fxb / fxa
            if xa == xc:

                p = 2.0 * xm * s
                q = 1.0 - s
            else:

                q = fxa / fxc
                r = fxb / fxc
                p = s * (2.0 * xm * q * (q - r) - (xb - xa) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            else:
                p = -p

            s = e
            e = d

            cond1 = (3.0 * xm * q - abs(xtol * q) <= 2.0 * p)
            cond2 = (abs(0.5 * s * q) <= p)
            if cond1 or cond2:
                d = xm
                e = d
            else:
                d = p / q

        xa = xb
        fxa = fxb

        if abs(d) > xtol:
            xb = xb + d
        elif xm > 0:
            xb = xb + xtol
        else:
            xb = xb - xtol

        fxb = float(f(xb))


        if fxb * fxc > 0:
            xc = xa
            fxc = fxa
            d = xb - xa
            e = d

    return xb


def muller_method(f, x0, x1, x2, fatol=1e-12, step_max=50):
    f0 = float(f(x0))
    f1 = float(f(x1))
    f2 = float(f(x2))

    for step_num in range(step_max):
        if abs(f2) <= fatol:
            return x2


        h0 = x0 - x2
        h1 = x1 - x2
        d0 = f0 - f2
        d1 = f1 - f2

        denom = h0 * h1 * (h0 - h1)
        if abs(denom) < 1e-15:
            break

        a = (h1 * d0 - h0 * d1) / denom
        b = (h0 ** 2 * d1 - h1 ** 2 * d0) / denom
        c = f2

        disc = b ** 2 - 4.0 * a * c
        if disc < 0:
            disc = 0.0

        sqrt_disc = np.sqrt(disc)
        if b >= 0:
            den = b + sqrt_disc
        else:
            den = b - sqrt_disc

        if abs(den) < 1e-15:
            break

        dx = -2.0 * c / den
        x_new = x2 + dx
        f_new = float(f(x_new))


        x0, f0 = x1, f1
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new

    return x2


def solve_equivalent_plastic_strain(yield_stress, hardening_modulus, mu, trial_stress_norm,
                                    tol=1e-12, max_iter=50):
    sqrt23 = np.sqrt(2.0 / 3.0)

    def g(dep):
        return trial_stress_norm - sqrt23 * (yield_stress + hardening_modulus * dep) - 2.0 * mu * dep

    def dg(dep):
        return -sqrt23 * hardening_modulus - 2.0 * mu


    f_trial = trial_stress_norm - sqrt23 * yield_stress
    if f_trial <= 0:
        return 0.0


    dep = newton_method(g, dg, 0.0, fatol=tol, step_max=max_iter)
    return max(0.0, dep)


def solve_volume_constraint(K_modulus, pressure_target, J_guess=1.0):
    def h(J):
        if J <= 0:
            J = 1e-12
        return K_modulus * np.log(J) + pressure_target

    def dh(J):
        if J <= 0:
            J = 1e-12
        return K_modulus / J

    try:
        J_sol = newton_method(h, dh, J_guess, fatol=1e-12, step_max=30)
    except RootFinderError:
        try:
            J_sol = brent_method(h, 0.1, 10.0, fatol=1e-12, step_max=50)
        except RootFinderError:
            J_sol = J_guess

    return J_sol
