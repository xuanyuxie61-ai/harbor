
import numpy as np
from fem_acoustics import C_AIR


def rectangular_room_modes(Lx, Ly, Lz, max_order=5):
    modes = []
    for l in range(max_order + 1):
        for m in range(max_order + 1):
            for n in range(max_order + 1):
                if l == 0 and m == 0 and n == 0:
                    continue
                f = (C_AIR / 2.0) * np.sqrt(
                    (l / Lx) ** 2 + (m / Ly) ** 2 + (n / Lz) ** 2
                )
                modes.append({
                    'l': l, 'm': m, 'n': n,
                    'frequency': f,
                    'wavelength': C_AIR / f if f > 0 else np.inf
                })
    modes.sort(key=lambda x: x['frequency'])
    return modes


def schroeder_frequency(room_volume, total_surface_area, absorption_coeff_avg):
    A_eq = total_surface_area * absorption_coeff_avg
    if A_eq < 1e-14:
        A_eq = 1e-14

    f_s = 2000.0 * np.sqrt(absorption_coeff_avg * total_surface_area / room_volume)
    return f_s


def zero_rc_brent(func, a, b, tol=1e-10, max_iter=100):
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise ValueError("Root not bracketed: f(a) and f(b) must have opposite signs")

    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol_act or abs(fb) < tol:
            return b
        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:

                p = 2.0 * m * s
                q = 1.0 - s
            else:

                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3.0 * m * q - abs(tol_act * q)
            min2 = abs(e * q)
            if 2.0 * p < min(min1, min2):
                e = d
                d = p / q
            else:
                d = e = m
        a, fa = b, fb
        if abs(d) > tol_act:
            b += d
        else:
            b += np.sign(m) * tol_act
        fb = func(b)
    return b


def inverse_iteration(K_sparse, M_sparse, max_iter=50, tol=1e-10):
    n = K_sparse.n
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)

    from sparse_linalg import conjugate_gradient

    for iteration in range(max_iter):
        b = M_sparse.mv(x)

        x_new = conjugate_gradient(K_sparse, b, x0=x, tol=1e-8, max_iter=min(n, 500))

        Mx = M_sparse.mv(x_new)
        norm_m = np.sqrt(np.dot(x_new, Mx))
        if norm_m < 1e-14:
            break
        x_new = x_new / norm_m

        Kx = K_sparse.mv(x_new)
        Mx = M_sparse.mv(x_new)
        lam = np.dot(x_new, Kx) / np.dot(x_new, Mx)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new



    freq = 0.0
    return freq, x, lam


def rayleigh_quotient_iteration(K_sparse, M_sparse, shift_guess, max_iter=20, tol=1e-12):
    n = K_sparse.n
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    mu = shift_guess

    from sparse_linalg import conjugate_gradient, assemble_sparse_from_triplets

    for iteration in range(max_iter):

        A_rows, A_cols, A_vals = [], [], []
        for i in range(K_sparse.nnz):
            A_rows.append(K_sparse.rows[i])
            A_cols.append(K_sparse.cols[i])
            A_vals.append(K_sparse.vals[i])
        for i in range(M_sparse.nnz):
            A_rows.append(M_sparse.rows[i])
            A_cols.append(M_sparse.cols[i])
            A_vals.append(-mu * M_sparse.vals[i])
        A_shift = assemble_sparse_from_triplets(A_rows, A_cols, A_vals, n)

        for i in range(n):
            A_rows.append(i)
            A_cols.append(i)
            A_vals.append(1e-8)
        A_shift = assemble_sparse_from_triplets(A_rows, A_cols, A_vals, n)

        b = M_sparse.mv(x)
        x_new = conjugate_gradient(A_shift, b, x0=x, tol=1e-7, max_iter=min(n, 500))
        Mx = M_sparse.mv(x_new)
        norm_m = np.sqrt(np.abs(np.dot(x_new, Mx)))
        if norm_m < 1e-14:
            break
        x_new = x_new / norm_m
        Kx = K_sparse.mv(x_new)
        Mx = M_sparse.mv(x_new)
        mu_new = np.dot(x_new, Kx) / np.dot(x_new, Mx)
        if abs(mu_new - mu) < tol:
            mu = mu_new
            x = x_new
            break
        mu = mu_new
        x = x_new

    omega = np.sqrt(abs(mu))
    freq = omega * C_AIR / (2.0 * np.pi)
    return freq, x, mu


def modal_participation_factor(mode_shape, force_dof):
    return mode_shape[force_dof]


def spherical_mode_integral(mode_shape, p, center, radius, n_samples=1000):
    from quadrature_rules import ball01_sample

    samples = ball01_sample(n_samples)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-14)
    samples = samples / norms * radius + center


    vals = []
    for s in samples:
        dists = np.linalg.norm(p - s, axis=1)
        idx = np.argmin(dists)
        vals.append(mode_shape[idx])


    integral = 4.0 * np.pi * radius ** 2 * np.mean(vals)
    return integral


def compute_modal_density(room_volume, freq):
    c = C_AIR

    n_f = 4.0 * np.pi * room_volume * freq ** 2 / (c ** 3)
    return n_f


def modal_overlap_factor(modes, damping_ratio=0.01):
    mof_values = []
    for mode in modes:
        f = mode['frequency']
        n_f = compute_modal_density(400.0, f)
        mof = n_f * damping_ratio * f
        mof_values.append({'frequency': f, 'mof': mof})
    return mof_values
