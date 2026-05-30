
import numpy as np


def r83s_matvec(n, a, x):
    b = np.zeros(n, dtype=float)
    for j in range(n):
        i_start = max(0, j - 1)
        i_end = min(n, j + 2)
        for i in range(i_start, i_end):
            b[i] += a[i - j + 1] * x[j]
    return b


def conjugate_gradient_r83s(n, a, b, x0, tol=1e-12, max_iter=None):
    if max_iter is None:
        max_iter = n
    x = x0.astype(float).copy()
    b = b.astype(float).copy()

    ap = r83s_matvec(n, a, x)
    r = b - ap
    p = r.copy()

    for _ in range(max_iter):
        ap = r83s_matvec(n, a, p)
        pap = np.dot(p, ap)
        pr = np.dot(p, r)
        if abs(pap) < 1e-30:
            break
        alpha = pr / pap
        x += alpha * p
        r -= alpha * ap
        rap = np.dot(r, ap)
        beta = -rap / pap
        p = r + beta * p
        if np.linalg.norm(r) < tol:
            break

    return x


def build_1d_waveguide_matrix(epsilon_eff, k0, h, boundary='PEC'):
    N = epsilon_eff.size
    if N < 3:
        raise ValueError("At least 3 grid points required.")
    if h <= 0:
        raise ValueError("Grid spacing h must be positive.")

    inv_h2 = 1.0 / (h ** 2)
    diag = -2.0 * inv_h2 + (k0 ** 2) * np.real(epsilon_eff)

    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = diag[i]
        if i > 0:
            A[i, i - 1] = inv_h2
        if i < N - 1:
            A[i, i + 1] = inv_h2

    if boundary == 'PEC':
        A[0, 0] += inv_h2
        A[N - 1, N - 1] += inv_h2
    elif boundary == 'PML':

        sigma = np.zeros(N)
        pml_width = max(1, N // 10)
        for i in range(pml_width):
            sigma[i] = (i + 1) ** 2 * 0.1
            sigma[N - 1 - i] = (i + 1) ** 2 * 0.1
        A += np.diag(1j * k0 * sigma)
    else:
        raise ValueError("Unknown boundary condition.")

    a_r83s = np.array([inv_h2, np.mean(diag), inv_h2])
    return A, a_r83s


def solve_waveguide_modes(epsilon_eff, k0, h, num_modes=5, boundary='PEC'):
    A, _ = build_1d_waveguide_matrix(epsilon_eff, k0, h, boundary)
    N = A.shape[0]
    if num_modes > N:
        num_modes = N


    if np.allclose(A, A.T.conj()):
        eigvals, eigvecs = np.linalg.eigh(A)
    else:
        eigvals, eigvecs = np.linalg.eig(A)


    idx = np.argsort(-np.real(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    betas = np.sqrt(eigvals[:num_modes])
    modes = eigvecs[:, :num_modes]


    for m in range(num_modes):
        norm = np.sqrt(np.trapezoid(np.abs(modes[:, m]) ** 2, dx=h))
        if norm > 0:
            modes[:, m] /= norm

    return betas, modes


def effective_permittivity_mim_waveguide(epsilon_metal, epsilon_dielectric,
                                         width_metal, width_dielectric, wavelength):
    if width_dielectric <= 0:
        raise ValueError("Dielectric width must be positive.")
    ratio = width_dielectric / wavelength
    correction = 1.0 + 2.0 * (epsilon_dielectric / (abs(epsilon_metal) + 1e-20)) / ratio
    return epsilon_dielectric * correction
