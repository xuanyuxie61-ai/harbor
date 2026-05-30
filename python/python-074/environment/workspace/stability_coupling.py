
import numpy as np


def log_norm(A, p):
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("log_norm: A 必须是方阵。")

    if p == 1:

        B = np.abs(A) - np.diag(np.diag(np.abs(A)))
        c = np.real(np.diag(A))
        d = np.sum(B, axis=0)
        mu = np.max(c + d)
    elif p == 2:

        B = 0.5 * (A + A.conj().T)
        eigvals = np.linalg.eigvalsh(B)
        mu = np.max(eigvals)
    elif p == np.inf:

        B = np.abs(A) - np.diag(np.diag(np.abs(A)))
        c = np.real(np.diag(A))
        d = np.sum(B, axis=1)
        mu = np.max(c + d)
    else:
        raise ValueError("log_norm: p 必须是 1, 2 或 np.inf。")

    return float(mu)


def compute_cfl_matrix_1d(v, k, dx):
    nx = int(np.ceil(1.0 / dx)) + 1
    A = np.zeros((nx, nx))
    A[0, 0] = 1.0
    A[-1, -1] = 1.0

    for i in range(1, nx - 1):
        A[i, i - 1] = -v / (2.0 * dx) - k / (dx ** 2)
        A[i, i] = 2.0 * k / (dx ** 2)
        A[i, i + 1] = v / (2.0 * dx) - k / (dx ** 2)

    return A


def build_fsi_jacobian(nx, ny, nu, dx, dy, dt,
                       mass, damping, stiffness, rho_f, D_cyl,
                       coupling_gain=1.0):
    n_fluid = nx
    n_total = n_fluid + 2

    J = np.zeros((n_total, n_total))


    v_conv = 1.0
    for i in range(n_fluid):
        im = (i - 1) % n_fluid
        ip = (i + 1) % n_fluid
        J[i, im] = -v_conv / (2.0 * dx) - nu / (dx ** 2)
        J[i, i] = 2.0 * nu / (dx ** 2)
        J[i, ip] = v_conv / (2.0 * dx) - nu / (dx ** 2)



    center_i = n_fluid // 2
    J[center_i, n_total - 1] = coupling_gain * 0.1


    J[n_total - 2, n_total - 1] = 1.0
    J[n_total - 1, n_total - 2] = -stiffness / mass
    J[n_total - 1, n_total - 1] = -damping / mass


    lift_sensitivity = coupling_gain * (0.5 * rho_f * v_conv ** 2 * D_cyl) / mass
    for i in range(n_fluid):
        J[n_total - 1, i] = lift_sensitivity * np.sin(2.0 * np.pi * i / n_fluid) / n_fluid

    return J


def gershgorin_bounds(A):
    n = A.shape[0]
    centers = np.real(np.diag(A))
    radii = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))

    lambda_min_est = np.min(centers - radii)
    lambda_max_est = np.max(centers + radii)
    return lambda_min_est, lambda_max_est


def pseudospectrum_abscissa(A, epsilon=1e-3, num_points=100):
    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)
    lambda_max_real = np.max(np.real(eigvals))


    center = lambda_max_real + 1j * np.max(np.imag(eigvals))
    best = lambda_max_real

    for _ in range(num_points):
        z = center + epsilon * (np.random.randn() + 1j * np.random.randn())
        M = z * np.eye(n) - A
        sigma_min = np.min(np.linalg.svd(M, compute_uv=False))
        if sigma_min <= epsilon:
            best = max(best, np.real(z))

    return best


def analyze_stability(nx=20, nu=0.01, mass=1.0, damping=0.1,
                      stiffness=10.0, rho_f=1.0, D_cyl=0.1):
    dx = 1.0 / (nx - 1)
    dy = dx
    dt = 0.01

    J = build_fsi_jacobian(nx, 1, nu, dx, dy, dt,
                           mass, damping, stiffness, rho_f, D_cyl)

    mu_1 = log_norm(J, 1)
    mu_2 = log_norm(J, 2)
    mu_inf = log_norm(J, np.inf)

    lam_min, lam_max = gershgorin_bounds(J)
    eigvals = np.linalg.eigvals(J)
    spectral_abscissa = np.max(np.real(eigvals))

    report = {
        'mu_1': mu_1,
        'mu_2': mu_2,
        'mu_inf': mu_inf,
        'gershgorin_min': lam_min,
        'gershgorin_max': lam_max,
        'spectral_abscissa': spectral_abscissa,
        'stable_mu1': mu_1 < 0,
        'stable_spectral': spectral_abscissa < 0,
    }
    return report
