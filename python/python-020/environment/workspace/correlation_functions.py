# -*- coding: utf-8 -*-
"""
correlation_functions.py
关联函数与最近邻分析

核心物理：
  在分数量子霍尔效应中，电子-电子关联函数 g^(2)(r) 是表征
  强关联态的核心量。对于各向同性均匀系统：

      g^(2)(r) = V² / N(N-1) · ⟨Σ_{i≠j} δ(r - r_i + r_j)⟩

  其中 V 为系统面积。在小距离极限下，Laughlin态的行为为：
      g^(2)(r) ~ (r/l_B)^{2m}    (r → 0)

  这表明电子之间存在 m 阶零点（m-th order zero），
  即任意两个电子不能靠近到距离 ~ l_B 以内。

  静态结构因子 S(q) 与 g^(2)(r) 通过Fourier变换联系：
      S(q) = 1 + n ∫ d²r [g^(2)(r) - 1] e^{-iq·r}

  在分数量子霍尔态中，S(q) 在 q → 0 时表现出特征性的
  能隙行为（ROTON极小值）：
      S(q) ~ q^{2m} / (2m)!    (q → 0)

  模块还包含对电子构型的最近邻分析，用于识别Wigner晶体
  序和准粒子激发的空间分布特征。

本模块融合原项目：
  - 1219_test_nearest（最近邻搜索）
  - 552_hyperball_distance（超球距离统计）
  - 561_hypercube_surface_distance（超立方体表面距离统计）
"""
import numpy as np
from utils import magnetic_length

# ============================================================================
# 1. 最近邻搜索（融合原项目 1219_test_nearest）
# ============================================================================

def find_nearest_neighbors(m, nr, R, ns, S):
    """
    对于每个S中的点，找到R中最近的点。

    算法：
        对每个 s ∈ S：
            min_dist = ∞
            对每个 r ∈ R：
                d = ||r - s||
                若 d < min_dist：
                    min_dist = d, nearest = r

    参数:
        m  : int, 空间维数
        nr : int, R中点数
        R  : ndarray, shape (m, nr)，数据点
        ns : int, S中点数
        S  : ndarray, shape (m, ns)，查询点

    返回:
        nearest_idx : ndarray, shape (ns,), 最近邻在R中的索引
        min_dists   : ndarray, shape (ns,), 最小距离
    """
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


# ============================================================================
# 2. 高维距离统计（融合原项目 552_hyperball_distance, 561_hypercube_surface_distance）
# ============================================================================

def hyperball_distance_stats(m_dim, n_samples, seed=42):
    """
    在m维单位超球内随机取点，计算配对距离的统计量。

    分布：超球内均匀分布的点满足：
        p(r) = m · r^{m-1}    (0 ≤ r ≤ 1)

    参数:
        m_dim     : int, 空间维数
        n_samples : int, 采样数
        seed      : int, 随机种子

    返回:
        mu        : float, 平均距离
        var       : float, 距离方差
        distances : ndarray, 所有配对距离
    """
    np.random.seed(seed)
    # 在单位超球内均匀采样：
    # 先生成m维高斯随机向量，再归一化，最后按 r = u^{1/m} 缩放
    p = np.random.randn(m_dim, n_samples)
    norms = np.linalg.norm(p, axis=0)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    p = p / norms
    u = np.random.uniform(0.0, 1.0, n_samples)
    radii = u ** (1.0 / m_dim)
    points = p * radii

    # 计算配对距离
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
    """
    在m维单位超立方体表面上随机取点，计算配对距离的统计量。

    参数:
        m_dim     : int, 空间维数
        n_samples : int, 采样数
        seed      : int, 随机种子

    返回:
        mu        : float, 平均距离
        var       : float, 距离方差
        distances : ndarray, 所有配对距离
    """
    np.random.seed(seed + 1)
    points = np.random.uniform(0.0, 1.0, (m_dim, n_samples))
    # 将点投影到超立方体表面：随机选择一个维度设为0或1
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


# ============================================================================
# 3. 量子霍尔效应关联函数
# ============================================================================

def two_point_correlation(z, lB, r_bins=60, r_max=None):
    """
    计算量子霍尔系统的两点关联函数 g^(2)(r)。

    参数:
        z      : ndarray, 电子复坐标
        lB     : float, 磁长度
        r_bins : int, 径向分箱数
        r_max  : float or None

    返回:
        r_edges : ndarray, 距离边界
        g2      : ndarray, 关联函数
        r_centers : ndarray, 距离中心
    """
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

    # 二维环形归一化
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
    """
    计算密度-密度关联函数（静态结构因子的Fourier变换）。

    S(q) = (1/N) ⟨ρ(q) ρ(-q)⟩
         = (1/N) |Σ_j e^{-iq·r_j}|²

    参数:
        n_grid : ndarray, 二维密度格点
        dx, dy : float, 格点间距
        q_max  : float or None
        n_q    : int, 波矢分箱数

    返回:
        q_vals : ndarray
        S_q    : ndarray
    """
    n_grid = np.asarray(n_grid, dtype=float)
    Nx, Ny = n_grid.shape

    # Fourier变换
    n_fft = np.fft.fft2(n_grid)
    n_fft_shift = np.fft.fftshift(n_fft)

    # 波矢
    qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx)) * 2.0 * np.pi
    qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy)) * 2.0 * np.pi
    QX, QY = np.meshgrid(qx, qy, indexing='ij')
    q_mag = np.sqrt(QX ** 2 + QY ** 2)

    # S(q) = |n(q)|² / N
    N_total = np.sum(n_grid)
    if N_total < 1e-14:
        N_total = 1.0
    S_grid = np.abs(n_fft_shift) ** 2 / N_total

    # 径向平均
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


# ============================================================================
# 4. 测试接口
# ============================================================================
def test_correlation_functions():
    """测试关联函数模块。"""
    print("=" * 60)
    print("[correlation_functions.py] 关联函数测试")
    print("=" * 60)

    # 测试最近邻
    print("\n1. 最近邻搜索测试:")
    m, nr, ns = 2, 5, 3
    R = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    S = np.array([[0.3, 2.5, 4.2],
                  [0.0, 0.0, 0.0]], dtype=float)
    idx, dists = find_nearest_neighbors(m, nr, R, ns, S)
    print(f"   查询点 S 的最近邻索引: {idx}")
    print(f"   最小距离: {dists}")

    # 测试超球距离统计
    print("\n2. 超球距离统计测试:")
    for dim in [2, 3, 5]:
        mu, var, dists = hyperball_distance_stats(dim, 200)
        print(f"   dim={dim}: 平均距离={mu:.4f}, 方差={var:.6f}")

    # 测试超立方体表面距离统计
    print("\n3. 超立方体表面距离统计测试:")
    for dim in [2, 3]:
        mu, var, dists = hypercube_surface_distance_stats(dim, 200)
        print(f"   dim={dim}: 平均距离={mu:.4f}, 方差={var:.6f}")

    # 测试量子霍尔关联函数
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

    # 测试密度关联（傅里叶）
    print("\n5. 密度关联函数测试:")
    n_grid = np.random.rand(32, 32)
    q_vals, S_q = density_correlation_function(n_grid, dx=0.1, dy=0.1)
    print(f"   q 范围: [{q_vals[0]:.4f}, {q_vals[-1]:.4f}]")
    print(f"   S(q) 范围: [{np.min(S_q):.4f}, {np.max(S_q):.4f}]")

    print("\n[correlation_functions.py] 测试完成。\n")


if __name__ == "__main__":
    test_correlation_functions()
