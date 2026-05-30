
import numpy as np


def solve_ks_etdrk4(nx=128, tmax=50.0, dt=0.25, n_snapshots=51):
    if not isinstance(nx, int) or nx < 4:
        raise ValueError("nx must be an integer >= 4")
    if nx % 2 != 0:
        raise ValueError("nx must be even for FFT")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if tmax <= 0:
        raise ValueError("tmax must be positive")

    L_domain = 32.0 * np.pi
    x = L_domain * np.arange(nx) / nx


    u = np.cos(x / 16.0) * (1.0 + np.sin(x / 16.0))
    v = np.fft.fft(u)


    k = np.concatenate([
        np.arange(0, nx // 2),
        np.array([0.0]),
        np.arange(-nx // 2 + 1, 0)
    ]) / 16.0


    L_op = k ** 2 - k ** 4


    E = np.exp(dt * L_op)
    E2 = np.exp(dt * L_op / 2.0)


    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)


    LR = dt * L_op[:, np.newaxis] + r[np.newaxis, :]


    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))


    g = -0.5j * k

    n_steps = int(np.round(tmax / dt))
    if n_steps < 1:
        n_steps = 1


    snapshot_intervals = max(1, int(n_steps // (n_snapshots - 1)))
    actual_snapshots = min(n_snapshots, n_steps // snapshot_intervals + 1)

    u_storage = [u.copy()]
    t_storage = [0.0]








    raise NotImplementedError("Hole 1: ETDRK4 time-stepping loop not implemented")


    if len(t_storage) < actual_snapshots:
        u = np.real(np.fft.ifft(v))
        u_storage.append(u.copy())
        t_storage.append(n_steps * dt)

    u_mat = np.column_stack(u_storage)
    t_vec = np.array(t_storage)

    return x, t_vec, u_mat, k, L_op


def ks_reference_residual(u, x, t, k):
    if u.ndim != 2:
        raise ValueError("u must be 2D array (nx, nt)")
    nx, nt = u.shape
    if len(x) != nx or len(k) != nx:
        raise ValueError("Dimension mismatch between u, x, and k")
    if nt < 2:
        raise ValueError("Need at least 2 time points for time derivative")


    dt_vec = np.diff(t)
    if np.any(dt_vec <= 0):
        raise ValueError("t must be strictly increasing")

    u_t = np.zeros_like(u)

    u_t[:, 0] = (u[:, 1] - u[:, 0]) / (t[1] - t[0])

    u_t[:, -1] = (u[:, -1] - u[:, -2]) / (t[-1] - t[-2])

    for j in range(1, nt - 1):
        u_t[:, j] = (u[:, j + 1] - u[:, j - 1]) / (t[j + 1] - t[j - 1])


    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    u_xxxx = np.zeros_like(u)

    for j in range(nt):
        v = np.fft.fft(u[:, j])
        u_x[:, j] = np.real(np.fft.ifft(1j * k * v))
        u_xx[:, j] = np.real(np.fft.ifft((1j * k) ** 2 * v))
        u_xxxx[:, j] = np.real(np.fft.ifft((1j * k) ** 4 * v))


    residual = u_t + u * u_x + u_xx + u_xxxx
    return residual
