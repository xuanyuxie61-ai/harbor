
import numpy as np






def chirikov_map_step(x, y, k):
    y_next = y + k * np.sin(x)
    x_next = (x + y_next) % (2.0 * np.pi)
    return x_next, y_next


def chirikov_jacobian(x, k):
    c = k * np.cos(x)
    return np.array([[1.0, c], [1.0, 1.0 + c]])


def lyapunov_exponent_standard_map(k, n_iter=5000, n_burn=100):
    x = np.random.rand() * 2.0 * np.pi
    y = np.random.rand() * 2.0 * np.pi
    v = np.random.randn(2)
    v = v / np.linalg.norm(v)
    lam = 0.0
    for t in range(n_burn + n_iter):
        J = chirikov_jacobian(x, k)
        v = J @ v
        r = np.linalg.norm(v)
        v = v / r
        x, y = chirikov_map_step(x, y, k)
        if t >= n_burn:
            lam += np.log(max(r, 1e-30))
    return lam / n_iter






def frobenius_norm(A):
    A = np.asarray(A, dtype=complex)
    return np.sqrt(np.sum(np.abs(A) ** 2))


def spectral_condition_number(A):
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    s_max = np.max(s)
    s_min = np.max(s[s > 1e-15])
    return s_max / s_min


def gershgorin_discs(A):
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    centers = np.diag(A)
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    return centers, radii


def power_iteration_eigenvalue(A, max_iter=1000, tol=1e-12):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    lam_old = 0.0
    for _ in range(max_iter):
        Av = A @ v
        lam = np.dot(v, Av)
        v = Av / np.linalg.norm(Av)
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return lam, v






def empirical_cdf(data, x_grid):
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)
    cdf = np.searchsorted(data, x_grid, side='right') / n
    return cdf


def kolmogorov_smirnov_stat(data1, data2):
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    all_data = np.concatenate([data1, data2])
    grid = np.linspace(np.min(all_data), np.max(all_data), 1000)
    cdf1 = empirical_cdf(data1, grid)
    cdf2 = empirical_cdf(data2, grid)
    return np.max(np.abs(cdf1 - cdf2))


def wasserstein2_distance(data1, data2):
    data1 = np.sort(np.asarray(data1, dtype=float))
    data2 = np.sort(np.asarray(data2, dtype=float))

    n = max(len(data1), len(data2))
    q = np.linspace(0, 1, n)
    d1_q = np.quantile(data1, q)
    d2_q = np.quantile(data2, q)
    return np.sqrt(np.mean((d1_q - d2_q) ** 2))


def moment_statistics(samples):
    s = np.asarray(samples, dtype=float)
    n = len(s)
    mu = np.mean(s)
    var = np.var(s, ddof=1)
    std = np.sqrt(var)
    if std < 1e-15:
        return mu, var, 0.0, -3.0
    skew = np.mean((s - mu) ** 3) / std ** 3
    kurt = np.mean((s - mu) ** 4) / std ** 4 - 3.0
    return mu, var, skew, kurt






def sample_unit_sphere_uniform(d, n_samples):
    X = np.random.randn(n_samples, d)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def spherical_triangle_area(v1, v2, v3):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    s = 0.5 * (a + b + c)

    tan_E_4 = np.sqrt(np.tan(0.5 * s) * np.tan(0.5 * (s - a)) *
                      np.tan(0.5 * (s - b)) * np.tan(0.5 * (s - c)))
    E = 4.0 * np.arctan(tan_E_4)
    return E


def test_uq_analysis():

    lam_small = lyapunov_exponent_standard_map(0.5, n_iter=2000, n_burn=200)
    assert lam_small < 0.1, f"Small k should be near-integrable, got lambda={lam_small}"
    lam_large = lyapunov_exponent_standard_map(2.5, n_iter=2000, n_burn=200)
    assert lam_large > 0.05, f"Large k should be chaotic, got lambda={lam_large}"

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(frobenius_norm(A), np.sqrt(30.0), atol=1e-12)

    centers, radii = gershgorin_discs(A)
    eigs = np.linalg.eigvals(A)
    for ev in eigs:
        in_disc = any(np.abs(ev - c) <= r + 1e-10 for c, r in zip(centers, radii))
        assert in_disc, "Gershgorin theorem violated"

    d1 = np.random.randn(1000)
    d2 = np.random.randn(1000)
    ks = kolmogorov_smirnov_stat(d1, d2)
    assert ks < 0.2

    pts = sample_unit_sphere_uniform(3, 100)
    norms = np.linalg.norm(pts, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)
    print("uq_analysis: all self-tests passed")


if __name__ == "__main__":
    test_uq_analysis()
