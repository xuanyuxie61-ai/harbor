
import numpy as np
from spectral_quadrature import legendre_nodes_weights, sparse_grid_gauss_legendre






def poisson_nucleation_events(lambda_rate, t_end, event_num=None, seed=42):
    rng = np.random.default_rng(seed)
    if event_num is None:
        n_total = rng.poisson(lam=lambda_rate * t_end)
    else:
        n_total = event_num

    if n_total <= 0:
        return np.array([0.0]), np.array([0.0]), 0

    w = np.zeros(n_total + 1)
    w[1:] = rng.exponential(scale=1.0 / lambda_rate, size=n_total)
    t = np.cumsum(w)


    mask = t <= t_end
    t = t[mask]
    w = w[mask]
    n_total = t.size - 1 if t.size > 0 else 0
    return t, w, n_total






def breakage_frequency_lehr(V, C_B=0.5, sigma=0.072, rho_l=800.0):
    V = np.asarray(V, dtype=float)
    V = np.clip(V, 1e-15, None)
    d_eq = (6.0 * V / np.pi) ** (1.0 / 3.0)
    return C_B * np.sqrt(sigma / (rho_l * d_eq**3))


def daughter_distribution_uniform(V_parent, V_daughter):
    return 2.0


def coalescence_kernel_prince_blanch(V_i, V_j, epsilon=0.1, sigma=0.072,
                                      rho_l=800.0, h0_hf_ratio=10.0):
    V_i = np.asarray(V_i, dtype=float)
    V_j = np.asarray(V_j, dtype=float)
    V_i = np.clip(V_i, 1e-15, None)
    V_j = np.clip(V_j, 1e-15, None)

    r_i = (3.0 * V_i / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_j = (3.0 * V_j / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_eq = 2.0 * r_i * r_j / (r_i + r_j)


    omega = 1.43 * epsilon ** (1.0 / 3.0) * (r_i + r_j) ** 2.0


    t_contact = (r_i + r_j) ** (2.0 / 3.0) / (epsilon ** (1.0 / 3.0) + 1e-12)


    t_drainage = np.sqrt(rho_l * r_eq ** 3.0 / (16.0 * sigma + 1e-12)) * np.log(h0_hf_ratio)


    h = np.exp(-t_contact / (t_drainage + 1e-12))
    h = np.clip(h, 0.0, 1.0)

    return omega * h






def moment_source_qmom(moments, xi, wi, rho_l=800.0, sigma=0.072, epsilon=0.1):





    raise NotImplementedError("Hole 3: 请实现 moment_source_qmom 的矩源项计算")


def wheeler_algorithm(moments, n_nodes=2):
    moments = np.asarray(moments, dtype=float)
    n = n_nodes
    if len(moments) < 2 * n:
        raise ValueError("Need at least 2*n moments")

    m0 = moments[0]
    if m0 <= 0:
        m0 = 1e-12
    mu = moments / m0

    if n == 2:





        det = -mu[0] * mu[2] + mu[1] ** 2
        if abs(det) < 1e-30:

            x0 = max(mu[1], 1e-15)
            xi = np.array([x0 * 0.9, x0 * 1.1])
            wi = np.array([m0 * 0.5, m0 * 0.5])
            return xi, wi

        b = (mu[2] ** 2 - mu[1] * mu[3]) / det
        a = (mu[0] * mu[3] - mu[1] * mu[2]) / det

        disc = a ** 2 - 4.0 * b
        if disc < 0.0:
            disc = 0.0
        sqrt_disc = np.sqrt(disc)
        x1 = (a - sqrt_disc) / 2.0
        x2 = (a + sqrt_disc) / 2.0


        x1 = max(x1, 1e-15)
        x2 = max(x2, 1e-15)

        if abs(x2 - x1) < 1e-15:
            w1 = w2 = m0 * 0.5
        else:
            w1 = (m0 * x2 - moments[1]) / (x2 - x1)
            w2 = (moments[1] - m0 * x1) / (x2 - x1)

        w1 = max(w1, 0.0)
        w2 = max(w2, 0.0)


        if w1 + w2 < 1e-15 or x1 > 1e3 * x2 or x2 > 1e3 * x1:
            x_avg = max(mu[1], 1e-15)
            xi = np.array([x_avg * 0.8, x_avg * 1.2])
            wi = np.array([m0 * 0.5, m0 * 0.5])
            return xi, wi

        return np.array([x1, x2]), np.array([w1, w2])

    else:

        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                idx = i + j
                if idx < len(mu):
                    H[i, j] = mu[idx]
        H = H + 1e-14 * np.eye(n)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            xi = np.clip(eigenvalues, 1e-15, None)
            wi = m0 * (eigenvectors[0, :] ** 2)
            wi = np.clip(wi, 0.0, None)
        except np.linalg.LinAlgError:
            xi = np.linspace(1e-6, 1.0, n)
            wi = np.ones(n) * m0 / n
        return xi, wi


def qmom_integrate_pbe(m0_init, t_span, dt, n_nodes=2, **kwargs):
    t0, tf = t_span
    t_array = np.arange(t0, tf + dt, dt)
    nt = len(t_array)
    moments_hist = np.zeros((nt, 4))
    moments = np.asarray(m0_init, dtype=float).copy()
    moments_hist[0] = moments

    for it in range(1, nt):
        try:
            xi, wi = wheeler_algorithm(moments, n_nodes=n_nodes)
            S = moment_source_qmom(moments, xi, wi, **kwargs)
        except Exception:

            S = np.zeros(4)


        moments = moments + dt * S
        moments = np.clip(moments, 0.0, None)
        moments_hist[it] = moments

    return t_array, moments_hist
