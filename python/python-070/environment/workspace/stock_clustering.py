"""
stock_clustering.py
渔业种群栖息地快速聚类模块

整合算法：Charles Elkan 的快速 K-Means 算法（基于 kmeans_fast）

核心科学应用：
利用海洋环境数据（温度、盐度、深度、叶绿素浓度等）
对渔业栖息地进行聚类，识别不同的鱼类种群区域（stock units）。

数学基础：
1. K-Means 目标函数：
   J = \sum_{i=1}^n \sum_{k=1}^K r_{ik} ||x_i - \mu_k||^2
   其中 r_{ik} ∈ {0,1} 为隶属指示变量

2. Elkan 加速：利用三角不等式避免不必要的距离计算
   ||x_i - \mu_j|| >= max(0, ||x_i - \mu_{c(i)}|| - ||\mu_{c(i)} - \mu_j||)
   若下界已大于到最近中心的距离，则无需计算 ||x_i - \mu_j||

3. 渔业应用中的特征工程：
   特征向量 x = [SST, SSH, Chl-a, Depth, Bottom_temp]
   其中 SST=海面温度, SSH=海面高度, Chl-a=叶绿素浓度
"""

import numpy as np
from utils import NumericalConfig


def euclidean_distance(a, b):
    """
    计算欧几里得距离
    ||a - b||_2 = sqrt( \sum (a_i - b_i)^2 )
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.sqrt(np.sum(diff ** 2))


def calcdist(data, center):
    """
    计算数据点集到单个中心的距离

    Parameters
    ----------
    data : ndarray, shape (n, dim)
    center : ndarray, shape (dim,)

    Returns
    -------
    dist : ndarray, shape (n,)
    """
    data = np.asarray(data, dtype=float)
    center = np.asarray(center, dtype=float)
    diff = data - center
    return np.sqrt(np.sum(diff ** 2, axis=1))


def alldist(centers):
    """
    计算所有中心对之间的距离矩阵

    Parameters
    ----------
    centers : ndarray, shape (k, dim)

    Returns
    -------
    distmat : ndarray, shape (k, k)
    """
    k = centers.shape[0]
    distmat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            d = euclidean_distance(centers[i], centers[j])
            distmat[i, j] = d
            distmat[j, i] = d
    return distmat


def furthest_first_init(data, k):
    """
    最远优先初始化：选择相互距离最大的 k 个中心

    算法：
    1. 随机选择第一个中心
    2. 每次选择距离已有中心最远的点作为新中心

    Parameters
    ----------
    data : ndarray, shape (n, dim)
    k : int
        聚类数

    Returns
    -------
    centers : ndarray, shape (k, dim)
    mincenter : ndarray, shape (n,)
    mindist : ndarray, shape (n,)
    lower : ndarray, shape (n, k)
    computed : int
    """
    n, dim = data.shape
    centers = np.zeros((k, dim), dtype=float)

    # 第一个中心：数据均值附近随机扰动
    centers[0, :] = np.mean(data, axis=0) + 0.01 * np.std(data, axis=0) * np.random.randn(dim)

    mincenter = np.zeros(n, dtype=int)
    mindist = np.full(n, np.inf, dtype=float)
    lower = np.zeros((n, k), dtype=float)
    computed = 0

    for j in range(1, k):
        # 计算每个点到最近已有中心的距离
        dists = calcdist(data, centers[j - 1, :])
        update = dists < mindist
        mindist[update] = dists[update]
        mincenter[update] = j - 1
        lower[:, j - 1] = dists
        computed += n

        # 选择距离已有中心最远的点
        farthest_idx = np.argmax(mindist)
        centers[j, :] = data[farthest_idx, :]

    return centers, mincenter, mindist, lower, computed


def kmeans_fast(data, k, init_centers=None, max_iter=100, tol=1e-6):
    """
    Charles Elkan 快速 K-Means 算法（简化但完整的实现）

    Parameters
    ----------
    data : ndarray, shape (n, dim)
        数据点
    k : int
        聚类数
    init_centers : ndarray, optional
        初始中心，shape (k, dim)
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值

    Returns
    -------
    centers : ndarray, shape (k, dim)
        最终聚类中心
    labels : ndarray, shape (n,)
        每个点的聚类标签
    inertia : float
        聚类内平方和
    """
    n, dim = data.shape

    if init_centers is None:
        centers, labels, mindist, lower, _ = furthest_first_init(data, k)
    else:
        centers = np.asarray(init_centers, dtype=float).copy()
        labels = np.zeros(n, dtype=int)
        mindist = np.full(n, np.inf, dtype=float)
        lower = np.zeros((n, k), dtype=float)
        for j in range(k):
            dists = calcdist(data, centers[j, :])
            lower[:, j] = dists
            update = dists < mindist
            mindist[update] = dists[update]
            labels[update] = j

    # 预计算中心间距离
    centdist = 0.5 * alldist(centers) + np.diag(np.full(k, np.inf))
    pop = np.zeros(k, dtype=int)
    for j in range(k):
        pop[j] = np.sum(labels == j)

    old_labels = labels.copy()

    for iteration in range(max_iter):
        # E-step：分配点到最近中心（使用三角不等式剪枝）
        nndist = np.min(centdist + np.diag(np.full(k, np.inf)), axis=1)
        mobile = np.where(mindist > nndist[labels])[0]

        recalculated = np.zeros(n, dtype=bool)

        for j in range(k):
            # 剪枝：若下界足够大则跳过
            mdm = mindist[mobile]
            mcm = labels[mobile]
            track = np.where(mdm > centdist[mcm, j])[0]
            if len(track) == 0:
                continue

            alt = np.where(mdm[track] > lower[mobile[track], j])[0]
            if len(alt) == 0:
                continue

            track1 = mobile[track[alt]]

            # 重新计算到当前最近中心的距离
            redo = track1[~recalculated[track1]]
            if len(redo) > 0:
                c_redo = labels[redo]
                for jj in np.unique(c_redo):
                    rp = redo[c_redo == jj]
                    udist = calcdist(data[rp, :], centers[jj, :])
                    lower[rp, jj] = udist
                    mindist[rp] = udist
                recalculated[redo] = True

            # 检查是否需要计算到中心 j 的距离
            track2 = np.where(mindist[track1] > centdist[labels[track1], j])[0]
            track1 = track1[track2]
            if len(track1) == 0:
                continue

            track4 = np.where(lower[track1, j] < mindist[track1])[0]
            if len(track4) == 0:
                continue

            track5 = track1[track4]
            jdist = calcdist(data[track5, :], centers[j, :])
            lower[track5, j] = jdist

            # 更新分配
            update = jdist < mindist[track5]
            track3 = track5[update]
            mindist[track3] = jdist[update]
            labels[track3] = j

        # M-step：重新计算中心
        diff = np.where(labels != old_labels)[0]
        if len(diff) == 0:
            break

        diffj = np.unique(np.concatenate([labels[diff], old_labels[diff]]))
        diffj = diffj[diffj >= 0]

        for j in diffj:
            track = np.where(labels == j)[0]
            pop[j] = len(track)
            if pop[j] == 0:
                continue
            centers[j, :] = np.mean(data[track, :], axis=0)

        # 更新距离下界
        for j in diffj:
            offset = euclidean_distance(centers[j, :], centers[j, :])
            if offset == 0:
                continue
            track = np.where(labels == j)[0]
            mindist[track] += offset
            lower[:, j] = np.maximum(lower[:, j] - offset, 0.0)

        # 重新计算中心间距离
        centdist = 0.5 * alldist(centers) + np.diag(np.full(k, np.inf))
        recalculated = np.zeros(n, dtype=bool)

        old_labels = labels.copy()

        # 检查收敛
        if len(diff) < tol * n and iteration > 0:
            break

    # 计算最终惯性
    inertia = 0.0
    for j in range(k):
        track = np.where(labels == j)[0]
        if len(track) > 0:
            inertia += np.sum(calcdist(data[track, :], centers[j, :]) ** 2)

    return centers, labels, inertia


def cluster_habitat_zones(n_stations, env_features, n_zones=4):
    """
    对渔业调查站点进行栖息地分区聚类

    Parameters
    ----------
    n_stations : int
        站点数
    env_features : ndarray, shape (n_stations, n_features)
        环境特征矩阵（已标准化）
    n_zones : int
        分区数

    Returns
    -------
    zones : ndarray
        每个站点的分区标签
    zone_centers : ndarray
        分区中心
    zone_stats : dict
        分区统计信息
    """
    if env_features.ndim == 1:
        env_features = env_features.reshape(-1, 1)

    # 标准化特征
    mean_feat = np.mean(env_features, axis=0)
    std_feat = np.std(env_features, axis=0)
    std_feat = np.where(std_feat < NumericalConfig.EPS, NumericalConfig.EPS, std_feat)
    normalized = (env_features - mean_feat) / std_feat

    centers, zones, inertia = kmeans_fast(normalized, n_zones)

    zone_stats = {
        'inertia': inertia,
        'n_points_per_zone': [int(np.sum(zones == j)) for j in range(n_zones)],
        'zone_centers_original_scale': centers * std_feat + mean_feat,
    }

    return zones, centers, zone_stats
