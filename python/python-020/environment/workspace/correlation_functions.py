# -*- coding: utf-8 -*-
import numpy as np
from utils import magnetic_length





def find_nearest_neighbors(m, nr, R, ns, S):
    R = np.asarray(R, dtype=float)
    S = np.asarray(S, dtype=float)

    if R.shape != (m, nr):
        raise ValueError(f"R 形状应为 ({m}, {nr})，实际为 {R.shape}")
    if S.shape != (m, ns):
        raise ValueError(f"S 形状应为 ({m}, {ns})，实际为 {S.shape}")

    nearest_idx = np.full(ns, -1, dtype=int)
    min_dists = np.full(ns, np.inf, dtype=float)

    for js in range(ns):
        dist_min = np.inf
        idx_min = -1
        s_vec = S[:, js]
        for jr in range(nr):
            diff = R[:, jr] - s_vec
            dist = np.sqrt(np.sum(diff ** 2))
            if dist < dist_min:
                dist_min = dist
                idx_min = jr
        nearest_idx[js] = idx_min
        min_dists[js] = dist_min

    return nearest_idx, min_dists






def hyperball_distance_stats(m_dim, n_samples, seed=42):
    np.random.seed(seed)


    p = np.random.randn(m_dim, n_samples)
    norms = np.linalg.norm(p, axis=0)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    p = p / norms
    u = np.random.uniform(0.0, 1.0, n_samples)
    radii = u ** (1.0 / m_dim)
    points = p * radii


    distances = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(points[:, i] - points[:, j])
            distances.append(d)
    distances = np.array(distances)

    if len(distances) == 0:
        return 0.0, 0.0, distances

    mu = np.mean(distances)
    if len(distances) > 1:
        var = np.var(distances, ddof=1)
    else:
        var = 0.0
    return mu, var, distances


def hypercube_surface_distance_stats(m_dim, n_samples, seed=42):
    np.random.seed(seed + 1)
    points = np.random.uniform(0.0, 1.0, (m_dim, n_samples))

    for i in range(n_samples):
        dim = np.random.randint(0, m_dim)
        face = np.random.choice([0.0, 1.0])
        points[dim, i] = face

    distances = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(points[:, i] - points[:, j])
            distances.append(d)
    distances = np.array(distances)

    if len(distances) == 0:
        return 0.0, 0.0, distances

    mu = np.mean(distances)
    if len(distances) > 1:
        var = np.var(distances, ddof=1)
    else:
        var = 0.0
    return mu, var, distances






def two_point_correlation(z, lB, r_bins=60, r_max=None):
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 2:
        raise ValueError("至少需要2个电子")

    distances = []
    for i in range(N):
        for j in range(i + 1, N):
            distances.append(abs(z[i] - z[j]))
    distances = np.array(distances)

    if r_max is None:
        r_max = np.max(distances) * 1.2 if len(distances) > 0 else 5.0 * lB
    if r_max <= 0:
        r_max = 1.0

    g2, r_edges = np.histogram(distances, bins=r_bins, range=(0.0, r_max))
    bin_widths = np.diff(r_edges)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])


    R_sys = np.max(np.abs(z)) * 1.1 if N > 0 else 5.0 * lB
    area = np.pi * R_sys ** 2
    n_density = N / area

    for i in range(len(g2)):
        rc = r_centers[i]
        dr = bin_widths[i]
        shell = 2.0 * np.pi * rc * dr
        if shell < 1e-15:
            g2[i] = 0.0
            continue
        norm = 0.5 * N * (N - 1) * shell / area
        if norm < 1e-15:
            g2[i] = 0.0
        else:
            g2[i] = g2[i] / norm

    return r_edges, g2, r_centers


def density_correlation_function(n_grid, dx, dy, q_max=None, n_q=40):
    n_grid = np.asarray(n_grid, dtype=float)
    Nx, Ny = n_grid.shape


    n_fft = np.fft.fft2(n_grid)
    n_fft_shift = np.fft.fftshift(n_fft)


    qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx)) * 2.0 * np.pi
    qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy)) * 2.0 * np.pi
    QX, QY = np.meshgrid(qx, qy, indexing='ij')
    q_mag = np.sqrt(QX ** 2 + QY ** 2)


    N_total = np.sum(n_grid)
    if N_total < 1e-14:
        N_total = 1.0
    S_grid = np.abs(n_fft_shift) ** 2 / N_total


    if q_max is None:
        q_max = np.max(q_mag)
    q_bins = np.linspace(0.0, q_max, n_q + 1)
    q_vals = 0.5 * (q_bins[:-1] + q_bins[1:])
    S_q = np.zeros(n_q)
    counts = np.zeros(n_q)

    for i in range(Nx):
        for j in range(Ny):
            q = q_mag[i, j]
            bin_idx = np.searchsorted(q_bins, q) - 1
            if 0 <= bin_idx < n_q:
                S_q[bin_idx] += S_grid[i, j]
                counts[bin_idx] += 1

    for i in range(n_q):
        if counts[i] > 0:
            S_q[i] /= counts[i]

    return q_vals, S_q





def test_correlation_functions():
    print("=" * 60)
    print("[correlation_functions.py] 关联函数测试")
    print("=" * 60)


    print("\n1. 最近邻搜索测试:")
    m, nr, ns = 2, 5, 3
    R = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    S = np.array([[0.3, 2.5, 4.2],
                  [0.0, 0.0, 0.0]], dtype=float)
    idx, dists = find_nearest_neighbors(m, nr, R, ns, S)
    print(f"   查询点 S 的最近邻索引: {idx}")
    print(f"   最小距离: {dists}")


    print("\n2. 超球距离统计测试:")
    for dim in [2, 3, 5]:
        mu, var, dists = hyperball_distance_stats(dim, 200)
        print(f"   dim={dim}: 平均距离={mu:.4f}, 方差={var:.6f}")


    print("\n3. 超立方体表面距离统计测试:")
    for dim in [2, 3]:
        mu, var, dists = hypercube_surface_distance_stats(dim, 200)
        print(f"   dim={dim}: 平均距离={mu:.4f}, 方差={var:.6f}")


    print("\n4. 量子霍尔两点关联函数测试:")
    B = 10.0
    lB = magnetic_length(B, 1.0)
    N = 12
    np.random.seed(42)
    theta = np.random.uniform(0.0, 2.0 * np.pi, N)
    r = np.sqrt(np.random.uniform(0.0, 1.0, N)) * np.sqrt(6.0 * N) * lB * 0.4
    z = r * np.exp(1j * theta)
    r_edges, g2, r_centers = two_point_correlation(z, lB, r_bins=30)
    print(f"   电子数 N={N}")
    print(f"   g2 前3个值: {g2[:3]}")


    print("\n5. 密度关联函数测试:")
    n_grid = np.random.rand(32, 32)
    q_vals, S_q = density_correlation_function(n_grid, dx=0.1, dy=0.1)
    print(f"   q 范围: [{q_vals[0]:.4f}, {q_vals[-1]:.4f}]")
    print(f"   S(q) 范围: [{np.min(S_q):.4f}, {np.max(S_q):.4f}]")

    print("\n[correlation_functions.py] 测试完成。\n")


if __name__ == "__main__":
    test_correlation_functions()
