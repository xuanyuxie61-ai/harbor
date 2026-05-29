"""
voronoi_partition.py
基于种子项目 1398_voronoi_plot 的 Voronoi 划分思想

在核素图 (N,Z) 平面上，Voronoi 划分可用于：
1. 将核素空间分割为不同的反应路径区域
2. 在核数据插值中确定最近邻核素
3. 划分 r 过程冻结线（freeze-out line）附近的核素域

Voronoi 单元定义：
    V_i = { x in R^2 : ||x - g_i|| <= ||x - g_j|| for all j != i }

在核数据插值中，对于任意查询点 x，找到其最近生成点 g_i，
则 x 处的核数据可用 g_i 对应核素的数据近似。
"""

import numpy as np


def voronoi_nearest_generator(query_points, generators, metric='euclidean', p=2):
    """
    对每个查询点找到最近的 Voronoi 生成点。

    参数:
        query_points : ndarray, shape (nq, dim)
        generators : ndarray, shape (ng, dim)
        metric : str, 'euclidean', 'manhattan', 'chebyshev'
        p : float, L_p 范数的 p 值

    返回:
        indices : ndarray, shape (nq,), 最近生成点索引
        distances : ndarray, shape (nq,), 距离
    """
    query_points = np.asarray(query_points, dtype=float)
    generators = np.asarray(generators, dtype=float)

    if metric == 'manhattan':
        dists = np.sum(np.abs(query_points[:, None, :] - generators[None, :, :]), axis=2)
    elif metric == 'chebyshev':
        dists = np.max(np.abs(query_points[:, None, :] - generators[None, :, :]), axis=2)
    elif metric == 'euclidean':
        dists = np.sqrt(np.sum((query_points[:, None, :] - generators[None, :, :]) ** 2, axis=2))
    else:
        # 通用 L_p
        dists = np.sum(np.abs(query_points[:, None, :] - generators[None, :, :]) ** p, axis=2) ** (1.0 / p)

    indices = np.argmin(dists, axis=1)
    distances = dists[np.arange(len(query_points)), indices]
    return indices, distances


def voronoi_cell_centroid(generators, density_samples, indices):
    """
    计算每个 Voronoi 单元的质心（用于 CVT 迭代）。

    参数:
        generators : ndarray, shape (ng, dim)
        density_samples : ndarray, shape (ns, dim)
        indices : ndarray, shape (ns,), 每个样本的最近生成点索引

    返回:
        centroids : ndarray, shape (ng, dim)
    """
    ng = generators.shape[0]
    dim = generators.shape[1]
    centroids = np.zeros((ng, dim))
    counts = np.zeros(ng)
    for i in range(len(density_samples)):
        idx = indices[i]
        centroids[idx] += density_samples[i]
        counts[idx] += 1
    # 避免除零
    counts = np.where(counts < 1, 1, counts)
    centroids = centroids / counts[:, None]
    return centroids


def interpolate_nuclear_data(query_nz, known_nz, known_data, metric='euclidean'):
    """
    使用 Voronoi 最近邻插值核数据。

    参数:
        query_nz : ndarray, shape (nq, 2), 查询点 (N,Z)
        known_nz : ndarray, shape (nk, 2), 已知核素坐标
        known_data : ndarray, shape (nk,), 已知数据
        metric : str, 距离度量

    返回:
        interpolated : ndarray, shape (nq,), 插值结果
    """
    indices, _ = voronoi_nearest_generator(query_nz, known_nz, metric=metric)
    interpolated = known_data[indices]
    return interpolated


def partition_nuclear_chart(nuclides, n_partitions, max_iter=50):
    """
    对核素图进行 Voronoi 划分，用于区分不同的核合成区域。

    参数:
        nuclides : list of tuple, [(Z,N,A), ...]
        n_partitions : int, 分区数
        max_iter : int, Lloyd 迭代次数

    返回:
        generators : ndarray, shape (n_partitions, 2), 分区中心 (Z,N)
        labels : ndarray, shape (len(nuclides),), 每个核素所属分区
    """
    coords = np.array([(z, n) for z, n, a in nuclides], dtype=float)
    if len(coords) == 0:
        return np.array([]), np.array([])

    # 初始化生成点
    z_min, z_max = coords[:, 0].min(), coords[:, 0].max()
    n_min, n_max = coords[:, 1].min(), coords[:, 1].max()
    generators = np.zeros((n_partitions, 2))
    generators[:, 0] = np.linspace(z_min, z_max, n_partitions)
    generators[:, 1] = np.linspace(n_min, n_max, n_partitions)

    for _ in range(max_iter):
        indices, _ = voronoi_nearest_generator(coords, generators)
        new_gens = voronoi_cell_centroid(generators, coords, indices)
        # 保持未分配单元不变
        for i in range(n_partitions):
            if np.all(new_gens[i] == 0):
                new_gens[i] = generators[i]
        generators = new_gens

    labels, _ = voronoi_nearest_generator(coords, generators)
    return generators, labels


def test_voronoi_partition():
    """自包含测试"""
    np.random.seed(42)
    generators = np.random.rand(10, 2)
    queries = np.random.rand(100, 2)
    indices, dists = voronoi_nearest_generator(queries, generators)
    print(f"[voronoi_partition] Nearest generator indices range: [{indices.min()}, {indices.max()}]")

    # 测试核数据插值
    known_nz = np.array([[50, 80], [82, 126], [92, 146]], dtype=float)
    known_data = np.array([1.0, 2.0, 3.0])
    query_nz = np.array([[60, 90], [85, 130]], dtype=float)
    interp = interpolate_nuclear_data(query_nz, known_nz, known_data)
    print(f"[voronoi_partition] Interpolated values: {interp}")


if __name__ == "__main__":
    test_voronoi_partition()
