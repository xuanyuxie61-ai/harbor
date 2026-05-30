
import numpy as np


def calccf(xi: np.ndarray, c: np.ndarray) -> tuple:
    n = len(xi) - 1
    dx = np.diff(xi)
    divdf1 = np.diff(c[0, :]) / dx
    divdf3 = c[1, 0:n] - 2.0 * divdf1 + c[1, 1:n + 1]
    coefs = np.zeros((4, n))
    coefs[0, :] = c[0, 0:n]
    coefs[1, :] = c[1, 0:n]
    coefs[2, :] = (divdf1 - coefs[1, :] - divdf3) / dx
    coefs[3, :] = divdf3 / (dx * dx)
    return xi, coefs


def spline_eval(breaks: np.ndarray, coefs: np.ndarray, x: float) -> float:
    if x <= breaks[0]:
        i = 0
    elif x >= breaks[-1]:
        i = len(breaks) - 2
    else:
        i = np.searchsorted(breaks, x) - 1
        i = max(0, min(i, len(breaks) - 2))
    dx = x - breaks[i]
    return coefs[0, i] + dx * (coefs[1, i] + dx * (coefs[2, i] + dx * coefs[3, i]))


def muller_method(f, z0: complex, z1: complex, z2: complex,
                  eps1: float = 1e-12, eps2: float = 1e-20,
                  maxit: int = 50) -> complex:
    eps1 = max(eps1, 1e-12)
    eps2 = max(eps2, 1e-20)
    z = [z0, z1, z2]
    fz = [f(zz) for zz in z]

    for _ in range(maxit):

        h0 = z[-2] - z[-3]
        h1 = z[-1] - z[-2]
        d0 = (fz[-2] - fz[-3]) / h0
        d1 = (fz[-1] - fz[-2]) / h1
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = fz[-1]

        disc = np.sqrt(b * b - 4.0 * a * c)
        if abs(b + disc) > abs(b - disc):
            denom = b + disc
        else:
            denom = b - disc
        if abs(denom) < 1e-30:
            denom = 1.0

        dz = -2.0 * c / denom
        z_new = z[-1] + dz
        fz_new = f(z_new)

        z.append(z_new)
        fz.append(fz_new)

        if abs(dz) < eps1 * max(abs(z_new), 1.0):
            break
        if max(abs(fz_new), abs(fz[-2])) <= eps2:
            break

    return z[-1]


def gevp_solve(c_t: np.ndarray, c_t0: np.ndarray) -> tuple:

    c_t0 = 0.5 * (c_t0 + c_t0.T.conj())
    c_t = 0.5 * (c_t + c_t.T.conj())


    eps = 1e-10
    c_t0 += eps * np.eye(c_t0.shape[0])


    lambdas, vectors = np.linalg.eig(np.linalg.solve(c_t0, c_t))

    lambdas = lambdas.real
    idx = np.argsort(-lambdas)
    lambdas = lambdas[idx]
    vectors = vectors[:, idx]
    return lambdas, vectors


def hooke_jeeves(f, x0: np.ndarray, rho: float = 0.85,
                 eps: float = 1e-7, itermax: int = 500) -> tuple:
    nvars = len(x0)
    xbefore = x0.copy().astype(float)
    delta = np.where(xbefore == 0.0, rho, rho * np.abs(xbefore))
    steplength = rho
    iters = 0
    fbefore = f(xbefore)

    def best_nearby(delta_loc, xbase, fbase):
        x = xbase.copy()
        fmin = fbase
        for i in range(nvars):
            for sign in [1.0, -1.0]:
                x[i] += sign * delta_loc[i]
                fn = f(x)
                if fn < fmin:
                    fmin = fn
                else:
                    x[i] -= sign * delta_loc[i]
        return fmin, x

    while iters < itermax and steplength > eps:
        iters += 1
        newf, newx = best_nearby(delta, xbefore, fbefore)

        keep = True
        while newf < fbefore and keep:
            for i in range(nvars):
                if newx[i] <= xbefore[i]:
                    delta[i] = -abs(delta[i])
                else:
                    delta[i] = abs(delta[i])
                tmp = xbefore[i]
                xbefore[i] = newx[i]
                newx[i] = newx[i] + newx[i] - tmp
            fbefore = newf
            newf, newx = best_nearby(delta, newx, fbefore)
            if fbefore <= newf:
                break
            keep = False
            for i in range(nvars):
                if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                    keep = True
                    break

        if steplength >= eps and fbefore <= newf:
            steplength *= rho
            delta *= rho

    return xbefore, fbefore


def variational_masses(correlator_matrix: np.ndarray, t0: int = 2) -> dict:
    nt, nop, _ = correlator_matrix.shape
    lambdas = np.zeros((nt, nop))
    vectors = np.zeros((nt, nop, nop))
    masses = np.zeros((nt - t0 - 1, nop))

    c_t0 = correlator_matrix[t0].real
    for t in range(t0, nt):
        lam, vec = gevp_solve(correlator_matrix[t].real, c_t0)
        lambdas[t] = lam
        vectors[t] = vec.real

    for n in range(nop):
        for t in range(t0, nt - 1):
            if abs(lambdas[t, n]) > 1e-15 and lambdas[t, n] / lambdas[t + 1, n] > 0:
                masses[t - t0, n] = np.log(lambdas[t, n] / lambdas[t + 1, n])
            else:
                masses[t - t0, n] = np.nan

    return {
        "eigenvalues": lambdas,
        "eigenvectors": vectors,
        "masses": masses,
        "t0": t0,
    }


def optimize_smearing_parameter(correlator_func, param_bounds: tuple,
                                t0: int = 2) -> tuple:
    def objective(alpha):
        corr = correlator_func(alpha[0])
        nt = len(corr)
        if nt <= t0 + 3:
            return 1e6

        m_eff = np.zeros(nt - 1)
        for t in range(nt - 1):
            if corr[t] > 1e-15 and corr[t + 1] > 1e-15:
                m_eff[t] = np.log(corr[t] / corr[t + 1])

        plateau = m_eff[t0 + 1:nt // 2]
        if len(plateau) < 2:
            return 1e6
        return np.nanvar(plateau)

    x0 = np.array([(param_bounds[0] + param_bounds[1]) / 2.0])
    xbest, fbest = hooke_jeeves(objective, x0, rho=0.75, eps=1e-4, itermax=100)

    xbest[0] = np.clip(xbest[0], param_bounds[0], param_bounds[1])
    return xbest[0], fbest
