
import numpy as np


def raman_response_function(t, f_R=0.18, tau1=12.2e-15, tau2=32.0e-15):





    raise NotImplementedError("Hole 1: raman_response_function 待实现")


def dispersion_operator(omega, alpha, beta2, beta3, beta4=0.0):
    D = -alpha / 2.0 + 1j * beta2 / 2.0 * omega ** 2 - 1j * beta3 / 6.0 * omega ** 3
    if beta4 != 0.0:
        D += beta4 / 24.0 * omega ** 4
    return D


def nonlinear_operator(A, dt, gamma, omega0, f_R, h_R, shock_term=True):
    n = A.size
    I = np.abs(A) ** 2



    h_R_pad = np.zeros(n)
    h_R_pad[:h_R.size] = h_R[:n]
    I_fft = np.fft.fft(I)
    h_fft = np.fft.fft(h_R_pad)
    S = np.fft.ifft(I_fft * h_fft).real * dt


    response = (1.0 - f_R) * I + S

    if shock_term and omega0 > 1e-30:


        dA = np.zeros(n, dtype=complex)
        dA[0] = (A[1] * response[1] - A[0] * response[0]) / dt
        dA[-1] = (A[-1] * response[-1] - A[-2] * response[-2]) / dt
        for i in range(1, n - 1):
            dA[i] = (A[i + 1] * response[i + 1] - A[i - 1] * response[i - 1]) / (2.0 * dt)
        N_op = 1j * gamma * A * response + 1j * gamma / omega0 * dA
    else:
        N_op = 1j * gamma * A * response

    return N_op


def ssfm_solve(A0, t, z_max, n_steps, alpha, beta2, beta3, gamma, lambda0=1550e-9,
               f_R=0.18, tau1=12.2e-15, tau2=32.0e-15, beta4=0.0,
               noise_ase=None, use_implicit=False):
    if t.size < 2 or A0.size != t.size:
        raise ValueError("ssfm_solve: invalid input dimensions")
    if n_steps < 1 or z_max < 0:
        raise ValueError("ssfm_solve: invalid propagation parameters")

    dt = t[1] - t[0]
    dz = z_max / n_steps
    n = t.size


    df = 1.0 / (n * dt)
    f = np.fft.fftfreq(n, dt)
    omega = 2.0 * np.pi * f

    omega0 = 2.0 * np.pi * 2.99792458e8 / lambda0


    D_op = dispersion_operator(omega, alpha, beta2, beta3, beta4)
    half_disp = np.exp(dz / 2.0 * D_op)


    h_R = raman_response_function(t, f_R, tau1, tau2)

    A = A0.copy()
    z_history = [0.0]
    A_history = [A.copy()]


    if use_implicit:
        from sparse_solver import build_dispersion_matrix_crs, mgmres
        a_crs, ia_crs, ja_crs, nz_num = build_dispersion_matrix_crs(n, dt, beta2, beta3, beta4)


        rows_imp = list(range(n))
        cols_imp = list(range(n))
        vals_imp = [1.0 - dz / 2.0 * D_op[i] for i in range(n)]


        use_implicit = False

    for step in range(n_steps):

        A_tilde = np.fft.ifft(half_disp * np.fft.fft(A))


        N_val = nonlinear_operator(A_tilde, dt, gamma, omega0, f_R, h_R, shock_term=True)
        A_nl = A_tilde * np.exp(dz * N_val / (np.abs(A_tilde) + 1e-30))


        A_nl = A_tilde + dz * N_val


        A = np.fft.ifft(half_disp * np.fft.fft(A_nl))


        if noise_ase is not None and noise_ase.size == n:
            A += noise_ase * np.sqrt(dz)

        z = (step + 1) * dz
        z_history.append(z)
        A_history.append(A.copy())


        if not np.all(np.isfinite(A)):
            raise RuntimeError(f"ssfm_solve: numerical instability at z={z:.3f} m")

    return A, np.array(z_history), A_history


def soliton_order(A0, t, gamma, beta2, T0=None):
    P0 = np.max(np.abs(A0) ** 2)
    if T0 is None:

        I = np.abs(A0) ** 2
        threshold = 0.5 * P0
        above = I > threshold
        if np.any(above):
            T0 = np.sum(above) * (t[1] - t[0]) / 2.0
        else:
            T0 = 1e-12

    if beta2 == 0:
        L_D = np.inf
    else:
        L_D = T0 ** 2 / abs(beta2)

    if gamma * P0 < 1e-30:
        L_NL = np.inf
    else:
        L_NL = 1.0 / (gamma * P0)

    N_sol = np.sqrt(L_D / L_NL) if L_NL > 0 and L_D > 0 else 0.0
    return N_sol, L_D, L_NL


def spectral_width(t, A):
    if t.size < 2 or A.size != t.size:
        return 0.0
    dt = t[1] - t[0]
    spectrum = np.fft.fftshift(np.fft.fft(A))
    freq = np.fft.fftshift(np.fft.fftfreq(t.size, dt))
    power_spec = np.abs(spectrum) ** 2
    max_power = np.max(power_spec)
    if max_power < 1e-30:
        return 0.0
    half_power = max_power / 2.0
    above = power_spec > half_power
    if np.any(above):
        width = np.max(freq[above]) - np.min(freq[above])
        return width
    return 0.0


def temporal_width(t, A):
    if t.size < 2 or A.size != t.size:
        return 0.0
    I = np.abs(A) ** 2
    max_I = np.max(I)
    if max_I < 1e-30:
        return 0.0
    half_I = max_I / 2.0
    above = I > half_I
    if np.any(above):
        return np.max(t[above]) - np.min(t[above])
    return 0.0
