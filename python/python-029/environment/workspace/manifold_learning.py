"""
manifold_learning.py
=====================
核数据流形学习与维度约减模块

基于种子项目 1052_sammon_data 的 Sammon 非线性映射思想，
本模块为核素质量表、能级密度等核数据提供流形学习分析：
1. Sammon 映射：保持数据点间距离的低维嵌入
2. 核素在 (N, Z) 平面上的质量曲面流形分析
3. 结合能曲面的局部线性嵌入 (LLE)
4. 核数据聚类与分类

核心公式
--------
Sammon 映射目标函数:
    E = (1 / Σ_{i<j} d_{ij}) Σ_{i<j} (d_{ij} - d*_{ij})² / d_{ij}

其中 d_{ij} 为原始空间距离，d*_{ij} 为低维空间距离。

核素质量曲面:
    M(N, Z) = Z m_p + N m_n - B(N, Z)/c²

在 (N, Z, M) 空间中，稳定核素形成一条三维流形
(滴线附近的谷)。

结合能每核子的梯度:
    ∇(B/A) = (∂(B/A)/∂N, ∂(B/A)/∂Z)

稳定谷的条件: ∇(B/A) ≈ 0
"""

import numpy as np


def euclidean_distance_matrix(X):
    """计算点集 X 的成对欧氏距离矩阵。"""
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i, :] - X[j, :])
            D[i, j] = d
            D[j, i] = d
    return D


def sammon_mapping(X, n_components=2, max_iter=300, alpha=0.3, tol=1e-9):
    """
    Sammon 非线性映射。

    将高维数据 X 映射到低维空间 Y，尽量保持局部距离结构。

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        高维数据。
    n_components : int
        目标维度。
    max_iter : int
        最大迭代次数。
    alpha : float
        学习率。
    tol : float
        收敛容差。

    Returns
    -------
    Y : ndarray
        低维嵌入。
    stress_history : list
        应力函数历史。
    """
    n = X.shape[0]
    D = euclidean_distance_matrix(X)
    # 避免零距离
    D = np.where(D < 1e-12, 1e-12, D)

    # 初始化低维坐标 (PCA 或随机)
    Y = np.random.randn(n, n_components) * 0.01
    # 使用前两维主成分初始化
    if n_features := X.shape[1] >= n_components:
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        for d in range(n_components):
            if d < len(idx):
                Y[:, d] = (X - X.mean(axis=0)) @ eigvecs[:, idx[d]]

    # 归一化
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-12)

    stress_history = []
    denom = np.sum(D[D > 0])

    for iteration in range(max_iter):
        D_star = euclidean_distance_matrix(Y)
        D_star = np.where(D_star < 1e-12, 1e-12, D_star)

        # 计算应力
        mask = D > 0
        stress = np.sum(((D[mask] - D_star[mask]) ** 2) / D[mask]) / denom
        stress_history.append(stress)

        if iteration > 10 and abs(stress_history[-1] - stress_history[-2]) < tol:
            break

        # 梯度下降
        for i in range(n):
            grad = np.zeros(n_components)
            for j in range(n):
                if i == j:
                    continue
                diff = Y[i, :] - Y[j, :]
                factor = -2.0 * (D[i, j] - D_star[i, j]) / (D[i, j] * D_star[i, j])
                grad += factor * diff
            Y[i, :] -= alpha * grad

    return Y, stress_history


def nuclear_mass_manifold(Z_range, N_range):
    """
    生成核素质量曲面数据并分析其流形结构。

    对 (N, Z) 平面上的每个核素计算结合能，
    形成三维点集 (N, Z, B/A)。

    Parameters
    ----------
    Z_range, N_range : range
        原子序数和中子数范围。

    Returns
    -------
    X : ndarray
        高维特征 (N, Z, BE, S_n, S_p)。
    labels : list
        核素标签。
    """
    from nuclear_data_io import Nuclide

    data = []
    labels = []
    for Z in Z_range:
        for N in N_range:
            A = Z + N
            if A < Z or A < 1:
                continue
            nuc = Nuclide(Z, A)
            BE_per_A = nuc.binding_energy() / A
            Sn = nuc.neutron_separation_energy()
            Sp = nuc.proton_separation_energy()
            # 特征向量
            features = [float(N), float(Z), BE_per_A, Sn, Sp]
            data.append(features)
            labels.append(f"{A}{nuc._element_symbol() if hasattr(nuc, '_element_symbol') else ''}")

    return np.array(data), labels


def local_linear_embedding(X, n_neighbors=5, n_components=2):
    """
    局部线性嵌入 (LLE) 的简化实现。

    对核数据流形进行局部线性嵌入。

    算法步骤:
    1. 对每个点找 k 近邻
    2. 用邻居线性重构该点 (最小二乘)
    3. 在低维空间保持重构权重
    """
    n = X.shape[0]
    D = euclidean_distance_matrix(X)

    # 找 k 近邻
    neighbors = np.argsort(D, axis=1)[:, 1:n_neighbors + 1]

    # 计算重构权重
    W = np.zeros((n, n))
    for i in range(n):
        Xi = X[neighbors[i], :] - X[i, :]
        G = Xi @ Xi.T
        # 正则化
        G += 1e-3 * np.eye(n_neighbors)
        w = np.linalg.solve(G, np.ones(n_neighbors))
        w = w / np.sum(w)
        W[i, neighbors[i]] = w

    # 构造 M = (I - W)^T (I - W)
    I = np.eye(n)
    M = (I - W).T @ (I - W)

    # 特征分解，取最小的 n_components+1 个特征值对应的特征向量
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)
    # 跳过第一个 (常数向量，特征值≈0)
    Y = eigvecs[:, idx[1:n_components + 1]]
    return Y


def cluster_nuclides_by_stability(X, n_clusters=3):
    """
    使用 k-means 对核素按稳定性聚类。

    聚类特征: 结合能每核子、分离能。
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_


def magic_number_detection(Z_list, N_list, BE_list):
    """
    通过结合能二阶差分检测幻数。

    Δ₂B(N) = B(N+1) - 2B(N) + B(N-1)

    在幻数处 Δ₂B 出现峰值。
    """
    Z_list = np.array(Z_list)
    N_list = np.array(N_list)
    BE_list = np.array(BE_list)

    magic_numbers = []
    for Z in np.unique(Z_list):
        mask = Z_list == Z
        Ns = N_list[mask]
        BEs = BE_list[mask]
        idx = np.argsort(Ns)
        Ns = Ns[idx]
        BEs = BEs[idx]
        if len(Ns) < 3:
            continue
        delta2 = BEs[2:] - 2 * BEs[1:-1] + BEs[:-2]
        # 找峰值
        for i, d2 in enumerate(delta2):
            if d2 > 2.0:  # 阈值
                magic_numbers.append((Z, int(Ns[i + 1])))

    return magic_numbers


def binding_energy_gradient_flow(Z_range, N_range):
    """
    计算结合能梯度流场，用于分析稳定谷的方向。

    Returns
    -------
    Z_grid, N_grid : ndarray
        网格坐标。
    dBE_dZ, dBE_dN : ndarray
        梯度分量。
    """
    from nuclear_data_io import Nuclide

    Z_vals = list(Z_range)
    N_vals = list(N_range)
    Z_grid, N_grid = np.meshgrid(Z_vals, N_vals)
    BE_grid = np.zeros_like(Z_grid, dtype=float)

    for i, N in enumerate(N_vals):
        for j, Z in enumerate(Z_vals):
            A = N + Z
            if A >= Z and A > 0:
                nuc = Nuclide(Z, A)
                BE_grid[i, j] = nuc.binding_energy() / A
            else:
                BE_grid[i, j] = np.nan

    # 计算梯度
    dBE_dZ, dBE_dN = np.gradient(BE_grid)
    return Z_grid, N_grid, dBE_dZ, dBE_dN


if __name__ == "__main__":
    # 自检
    # 生成测试数据 ( helix )
    t = np.linspace(0, 4 * np.pi, 100)
    X_test = np.column_stack([np.cos(t), np.sin(t), t])

    Y_sammon, stress = sammon_mapping(X_test, n_components=2, max_iter=200)
    print(f"Sammon 最终应力: {stress[-1]:.6f}")

    Y_lle = local_linear_embedding(X_test, n_neighbors=5, n_components=2)
    print(f"LLE 嵌入形状: {Y_lle.shape}")

    # 核素流形
    X_nuc, labels = nuclear_mass_manifold(range(20, 30), range(20, 40))
    print(f"核素数据形状: {X_nuc.shape}")

    # 幻数检测
    magic = magic_number_detection(X_nuc[:, 1].astype(int), X_nuc[:, 0].astype(int), X_nuc[:, 2] * (X_nuc[:, 0] + X_nuc[:, 1]))
    print(f"检测到的幻数 (Z, N): {magic[:10]}")
