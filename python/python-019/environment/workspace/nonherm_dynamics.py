
import numpy as np


def rkf45_step_complex(f, t, y, h, tol=1e-9):

    a2, a3, a4, a5, a6 = 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 0.5
    b21 = 1.0 / 4.0
    b31, b32 = 3.0 / 32.0, 9.0 / 32.0
    b41, b42, b43 = 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0
    b51, b52, b53, b54 = 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0
    b61, b62, b63, b64, b65 = -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0

    c1, c3, c4, c5, c6 = 16.0 / 135.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0
    d1, d3, d4, d5, d6 = 25.0 / 216.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0

    k1 = f(t, y)
    k2 = f(t + a2 * h, y + h * b21 * k1)
    k3 = f(t + a3 * h, y + h * (b31 * k1 + b32 * k2))
    k4 = f(t + a4 * h, y + h * (b41 * k1 + b42 * k2 + b43 * k3))
    k5 = f(t + a5 * h, y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
    k6 = f(t + a6 * h, y + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))

    y4 = y + h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6)
    y5 = y + h * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6)

    err = np.linalg.norm(y5 - y4)
    if err < 1e-30:
        h_new = 2.0 * h
    else:
        h_new = h * min(5.0, 0.9 * (tol / err) ** 0.2)

    if err > tol:

        return rkf45_step_complex(f, t, y, h_new, tol)

    return y5, t + h, h_new


def evolve_nonhermitian_schrodinger(H_eff, psi0, t_span, dt0=1e-4, tol=1e-10):
    if H_eff.shape[0] != H_eff.shape[1]:
        raise ValueError("H_eff must be square.")
    if psi0.shape[0] != H_eff.shape[0]:
        raise ValueError("psi0 dimension must match H_eff.")

    def rhs(t, psi):
        return -1j * H_eff @ psi

    t0, t1 = t_span
    t = t0
    psi = psi0.copy()
    h = dt0

    t_vals = [t]
    psi_vals = [psi.copy()]
    norms = [np.vdot(psi, psi).real]

    while t < t1:
        h = min(h, t1 - t)
        psi, t, h = rkf45_step_complex(rhs, t, psi, h, tol=tol)
        t_vals.append(t)
        psi_vals.append(psi.copy())
        norms.append(np.vdot(psi, psi).real)

    return np.array(t_vals), np.array(psi_vals), np.array(norms)


def lindblad_evolve_2level(H, L_list, rho0, t_span, dt0=1e-4, tol=1e-10):
    def rhs(t, rho_flat):
        rho = rho_flat.reshape((2, 2))
        drho = -1j * (H @ rho - rho @ H)
        for L in L_list:
            drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        return drho.flatten()

    t0, t1 = t_span
    t = t0
    rho = rho0.copy().flatten()
    h = dt0

    t_vals = [t]
    rho_vals = [rho0.copy()]
    purity = [np.trace(rho0 @ rho0).real]

    while t < t1:
        h = min(h, t1 - t)
        rho, t, h = rkf45_step_complex(rhs, t, rho, h, tol=tol)
        rho_mat = rho.reshape((2, 2))
        t_vals.append(t)
        rho_vals.append(rho_mat.copy())
        purity.append(np.trace(rho_mat @ rho_mat).real)

    return np.array(t_vals), np.array(rho_vals), np.array(purity)


def nonhermitian_lindberg_system(y):
    y0, y1, y2, y3 = y
    dydt = np.zeros(4)
    scale = 1e4
    dydt[0] = scale * y0 * y2 + scale * y1 * y3
    dydt[1] = -scale * y0 * y3 + scale * y1 * y2
    dydt[2] = 1.0 - y2
    dydt[3] = -0.5 * y2 - y3 + 0.5
    return dydt
