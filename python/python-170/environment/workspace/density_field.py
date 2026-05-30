
import numpy as np


def etdrk4_coefficients(L: np.ndarray, dt: float, nx: int, n_contour: int = 64):
    r = np.exp(2.0j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)

    LR = dt * L[:, np.newaxis] + r[np.newaxis, :]

    E = np.exp(dt * L)
    E2 = np.exp(dt * L * 0.5)


    LR_safe = LR.copy()
    tiny = 1e-14
    LR_safe[np.abs(LR_safe) < tiny] = tiny

    Q = dt * np.real(np.mean((np.exp(LR_safe * 0.5) - 1.0) / LR_safe, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR_safe + np.exp(LR_safe) * (4.0 - 3.0 * LR_safe + LR_safe ** 2)) / LR_safe ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR_safe + np.exp(LR_safe) * (-2.0 + LR_safe)) / LR_safe ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR_safe - LR_safe ** 2 + np.exp(LR_safe) * (4.0 - LR_safe)) / LR_safe ** 3, axis=1))

    return E, E2, Q, f1, f2, f3


def solve_burgers_etdrk4(nx: int = 256, nt: int = 10, vis: float = 0.03, tmax: float = 1.0):
    if nx % 2 != 0:
        raise ValueError("nx must be even for FFT.")

    x = np.linspace(-np.pi, np.pi, nx + 1)[:-1]
    u = np.exp(-10.0 * np.sin(0.5 * x) ** 2)
    v = np.fft.fft(u)

    dt = 0.4 / nx ** 2
    nmax = max(1, int(np.round(tmax / dt)))
    jstep = max(1, nmax // max(nt - 1, 1))


    k = np.concatenate((np.arange(0, nx // 2), np.array([0]), np.arange(-nx // 2 + 1, 0)))


    L = 1j * vis * k ** 2


    L = 1j * vis * k ** 2








    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(L, dt, nx)


    g = -0.5 * 1j * k

    uu_list = [u.copy()]
    tt_list = [0.0]

    for i in range(1, nmax + 1):
        t = i * dt
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        if i % jstep == 0 or i == nmax:
            u = np.real(np.fft.ifft(v))
            uu_list.append(u.copy())
            tt_list.append(t)

    uu = np.column_stack(uu_list)
    tt = np.array(tt_list)
    return x, tt, uu


def density_continuum_1d(nx: int = 256, tmax: float = 2.0, nu: float = 0.05, D4: float = 1e-4):
    if nx % 2 != 0:
        nx += 1

    x = np.linspace(-np.pi, np.pi, nx + 1)[:-1]
    dx = x[1] - x[0]


    rho = np.exp(-4.0 * x ** 2)
    rho_hat = np.fft.fft(rho)

    k = np.concatenate((np.arange(0, nx // 2), np.array([0]), np.arange(-nx // 2 + 1, 0)))
    k2 = k.astype(float) ** 2
    k4 = k2 ** 2


    L = -nu * k2 - D4 * k4

    dt = 0.2 / nx ** 2
    nmax = max(1, int(np.round(tmax / dt)))
    nt = min(20, nmax)
    jstep = max(1, nmax // nt)

    E = np.exp(dt * L)
    E2 = np.exp(dt * L * 0.5)


    n_contour = 64
    r = np.exp(2.0j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
    LR = dt * L[:, np.newaxis] + r[np.newaxis, :]
    LR_safe = LR.copy()
    tiny = 1e-14
    LR_safe[np.abs(LR_safe) < tiny] = tiny

    Q = dt * np.real(np.mean((np.exp(LR_safe * 0.5) - 1.0) / LR_safe, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR_safe + np.exp(LR_safe) * (4.0 - 3.0 * LR_safe + LR_safe ** 2)) / LR_safe ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR_safe + np.exp(LR_safe) * (-2.0 + LR_safe)) / LR_safe ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR_safe - LR_safe ** 2 + np.exp(LR_safe) * (4.0 - LR_safe)) / LR_safe ** 3, axis=1))


    g = -1.0j * k

    rho_list = [rho.copy()]
    tt_list = [0.0]














    raise NotImplementedError("HOLE 3: ETDRK4 density_continuum_1d stepping loop not implemented")

    tt = np.array(tt_list)
    rho_out = np.column_stack(rho_list)
    return x, tt, rho_out
