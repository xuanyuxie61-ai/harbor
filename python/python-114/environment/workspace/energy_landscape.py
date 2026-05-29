"""
energy_landscape.py

DNA 损伤修复分子动力学 —— 修复蛋白构象能量景观的非线性降维与插值

基于种子项目:
  - 1052_sammon_data: Sammon 映射数据生成（高维到低维的非线性投影）
  - 927_pwl_interp_2d: 二维网格上的分段线性插值

科学背景:
  DNA 损伤修复蛋白（如 RAD51、BRCA2、DNA-PKcs）在执行功能时经历
  大幅度的构象变化。这些高维构象空间（数千个自由度）的自由能景观
  (free energy landscape, FEL) 决定了蛋白的动力学路径和修复效率。

  本模块实现：
  1. Sammon-like 非线性降维，将高维构象坐标投影到 2D 反应坐标平面；
  2. 在该平面上进行分段线性插值，构建光滑的自由能表面；
  3. 计算能垒高度、过渡态位置与最小能量路径 (MEP)。
"""

import numpy as np
from typing import Tuple, Optional


def sammon_mapping(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 300,
    alpha: float = 0.3,
    tol: float = 1e-5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Sammon 非线性映射 (Sammon Mapping)。

    目标函数（应力函数）:
        E = (1 / Σ_{i<j} d*_{ij}) * Σ_{i<j} (d*_{ij} - d_{ij})^2 / d*_{ij}

    其中:
        d*_{ij} = ||x_i - x_j|| 为高维空间中的欧氏距离
        d_{ij}  = ||y_i - y_j|| 为低维空间中的欧氏距离

    优化采用最速下降法，迭代更新:
        y_k^{(t+1)} = y_k^{(t)} - α * (∂E / ∂y_k) / |∂²E / ∂y_k²|

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        高维数据，例如 MD 轨迹中的蛋白骨架二面角或 Cα 坐标。
    n_components : int
        目标维度（通常为 2）。
    max_iter : int
        最大迭代次数。
    alpha : float
        学习率。
    tol : float
        收敛容差。
    random_state : int
        随机种子。

    Returns
    -------
    Y : ndarray, shape (n_samples, n_components)
        低维嵌入坐标。
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    if n_samples < 2:
        raise ValueError("need at least 2 samples")

    rng = np.random.RandomState(random_state)

    # 计算高维距离矩阵
    dist_high = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(X[i, :] - X[j, :])
            dist_high[i, j] = d
            dist_high[j, i] = d

    # 避免除零
    dist_high = np.where(dist_high < 1e-12, 1e-12, dist_high)

    sum_dist = np.sum(dist_high)
    if sum_dist < 1e-12:
        raise ValueError("all pairwise distances are zero")

    # 初始低维坐标：PCA 前两个主成分
    X_centered = X - np.mean(X, axis=0)
    cov = X_centered.T @ X_centered / n_samples
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    Y = X_centered @ eigvecs[:, idx[:n_components]]

    # 缩放使得距离尺度匹配
    scale = np.mean(dist_high) / (np.mean(np.linalg.norm(Y[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)) + 1e-12)
    Y *= scale

    # Sammon 迭代优化
    for iteration in range(max_iter):
        dist_low = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.linalg.norm(Y[i, :] - Y[j, :])
                dist_low[i, j] = max(d, 1e-12)
                dist_low[j, i] = dist_low[i, j]

        # 计算梯度
        delta = np.zeros_like(Y)
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    continue
                diff = Y[i, :] - Y[j, :]
                d_low = dist_low[i, j]
                d_high = dist_high[i, j]
                factor = -2.0 / sum_dist * (d_high - d_low) / (d_high * d_low)
                delta[i, :] += factor * diff

        # 近似二阶导数做归一化
        second_order = np.zeros(n_samples)
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    continue
                d_low = dist_low[i, j]
                d_high = dist_high[i, j]
                second_order[i] += -2.0 / sum_dist / (d_high * d_low)

        second_order = np.abs(second_order) + 1e-12

        # 更新
        Y_new = Y.copy()
        for i in range(n_samples):
            Y_new[i, :] -= alpha * delta[i, :] / second_order[i]

        # 收敛判断
        stress_change = np.linalg.norm(Y_new - Y) / (np.linalg.norm(Y) + 1e-12)
        Y = Y_new

        if stress_change < tol:
            break

    return Y


def pwl_interp_2d(
    xd: np.ndarray,
    yd: np.ndarray,
    zd: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
) -> np.ndarray:
    """
    二维网格上的分段线性 (PWL) 插值。

    对每个查询点 (xi(k), yi(k)):
      1. 确定其所在矩形单元 [xd(i), xd(i+1)] × [yd(j), yd(j+1)]
      2. 将矩形沿对角线分为两个三角形
      3. 在对应三角形上做重心坐标线性插值:
             z = α z_a + β z_b + γ z_c
         其中 α+β+γ=1，由点在三角形内的重心坐标决定。

    Parameters
    ----------
    xd : ndarray, shape (nxd,)
        单调递增的 x 网格坐标。
    yd : ndarray, shape (nyd,)
        单调递增的 y 网格坐标。
    zd : ndarray, shape (nxd, nyd)
        网格节点上的函数值。
    xi, yi : ndarray, shape (ni,)
        查询点坐标。

    Returns
    -------
    zi : ndarray, shape (ni,)
        插值结果。对于超出网格范围的点返回 np.inf。
    """
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    zd = np.asarray(zd, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    yi = np.asarray(yi, dtype=np.float64)

    nxd = len(xd)
    nyd = len(yd)
    ni = len(xi)

    if zd.shape != (nxd, nyd):
        raise ValueError(f"zd shape {zd.shape} does not match ({nxd}, {nyd})")

    zi = np.full(ni, np.inf, dtype=np.float64)

    for k in range(ni):
        # 二分查找确定 i
        i = np.searchsorted(xd, xi[k], side='right') - 1
        if i < 0 or i >= nxd - 1:
            continue

        j = np.searchsorted(yd, yi[k], side='right') - 1
        if j < 0 or j >= nyd - 1:
            continue

        # 判断点位于哪半个三角形
        # 对角线: y - yd[j] = (yd[j+1]-yd[j])/(xd[i+1]-xd[i]) * (x - xd[i])
        slope = (yd[j + 1] - yd[j]) / (xd[i + 1] - xd[i] + 1e-18)
        diag_y = yd[j] + slope * (xi[k] - xd[i])

        if yi[k] < diag_y:
            # 下三角形: (xd[i], yd[j]), (xd[i+1], yd[j]), (xd[i], yd[j+1])
            dxa = xd[i + 1] - xd[i]
            dya = 0.0
            dxb = 0.0
            dyb = yd[j + 1] - yd[j]
            dxi = xi[k] - xd[i]
            dyi = yi[k] - yd[j]
            det = dxa * dyb - dya * dxb
            if abs(det) < 1e-18:
                continue
            alpha = (dxi * dyb - dyi * dxb) / det
            beta = (dxa * dyi - dya * dxi) / det
            gamma = 1.0 - alpha - beta
            if alpha < -1e-9 or beta < -1e-9 or gamma < -1e-9:
                continue
            zi[k] = alpha * zd[i + 1, j] + beta * zd[i, j + 1] + gamma * zd[i, j]
        else:
            # 上三角形: (xd[i+1], yd[j+1]), (xd[i], yd[j+1]), (xd[i+1], yd[j])
            dxa = xd[i] - xd[i + 1]
            dya = yd[j + 1] - yd[j + 1]
            dxb = xd[i + 1] - xd[i + 1]
            dyb = yd[j] - yd[j + 1]
            dxi = xi[k] - xd[i + 1]
            dyi = yi[k] - yd[j + 1]
            det = dxa * dyb - dya * dxb
            if abs(det) < 1e-18:
                continue
            alpha = (dxi * dyb - dyi * dxb) / det
            beta = (dxa * dyi - dya * dxi) / det
            gamma = 1.0 - alpha - beta
            if alpha < -1e-9 or beta < -1e-9 or gamma < -1e-9:
                continue
            zi[k] = alpha * zd[i, j + 1] + beta * zd[i + 1, j] + gamma * zd[i + 1, j + 1]

    return zi


def build_free_energy_surface(
    dihedral_angles: np.ndarray,
    temperature: float = 310.0,
    grid_n: int = 40,
) -> dict:
    """
    基于 MD 轨迹的二面角数据构建二维自由能表面。

    自由能公式（玻尔兹曼反转）:
        F(q1, q2) = -k_B T ln P(q1, q2)

    其中 P(q1, q2) 为反应坐标 (q1, q2) 上的联合概率密度，通过
    Sammon 映射得到 q1, q2，再进行直方图统计和 PWL 插值得到光滑表面。

    Parameters
    ----------
    dihedral_angles : ndarray, shape (n_frames, n_dihedrals)
        MD 轨迹中的蛋白二面角（弧度）。
    temperature : float
        模拟温度 (K)。
    grid_n : int
        插值网格分辨率。

    Returns
    -------
    result : dict
        包含网格坐标、自由能矩阵、极小/极大值位置。
    """
    # 1. Sammon 降维到 2D
    Y = sammon_mapping(dihedral_angles, n_components=2, max_iter=200, alpha=0.2)

    # 2. 二维直方图统计概率密度
    q1, q2 = Y[:, 0], Y[:, 1]

    # 边界
    q1_min, q1_max = np.min(q1), np.max(q1)
    q2_min, q2_max = np.min(q2), np.max(q2)

    # 扩展边界 10%
    margin1 = 0.1 * (q1_max - q1_min)
    margin2 = 0.1 * (q2_max - q2_min)
    q1_min -= margin1
    q1_max += margin1
    q2_min -= margin2
    q2_max += margin2

    H, edges1, edges2 = np.histogram2d(
        q1, q2, bins=grid_n,
        range=[[q1_min, q1_max], [q2_min, q2_max]]
    )

    # 概率密度
    dx1 = edges1[1] - edges1[0]
    dx2 = edges2[1] - edges2[0]
    P = H / (np.sum(H) * dx1 * dx2 + 1e-18)

    # TODO (Hole 2): 根据玻尔兹曼反转原理计算自由能表面:
    #   F(q1, q2) = -k_B T * ln(P(q1, q2) + eps)
    # 其中 P 为二维直方图得到的概率密度，temperature 为模拟温度 (K)。
    # 计算结果最初单位为焦耳 (J)，需要除以元电荷 e = 1.602176634e-19 C
    # 转换为电子伏特 (eV)。
    # 关键科学知识点:
    #   - 玻尔兹曼分布与自由能的关系: F = -k_B T ln P
    #   - 单位换算: 1 eV = 1.602176634e-19 J
    # 注意: 需处理 P=0 时的数值鲁棒性（加 eps 避免 log(0)）。
    raise NotImplementedError("Hole 2: 待实现自由能玻尔兹曼反转计算")

    # 3. 在网格节点上构建 PWL 插值所需的节点值
    # 使用直方图网格中心
    xc = 0.5 * (edges1[:-1] + edges1[1:])
    yc = 0.5 * (edges2[:-1] + edges2[1:])

    # 4. 寻找极小值（稳定构象）和极大值（过渡态）
    # 简单局部极值搜索
    local_min = []
    local_max = []
    for i in range(1, grid_n - 1):
        for j in range(1, grid_n - 1):
            neighborhood = F[i - 1:i + 2, j - 1:j + 2]
            if F[i, j] == np.min(neighborhood):
                local_min.append((xc[i], yc[j], F[i, j]))
            if F[i, j] == np.max(neighborhood):
                local_max.append((xc[i], yc[j], F[i, j]))

    # 5. 使用 PWL 插值在细网格上重采样
    fine_n = 80
    xi_fine = np.linspace(q1_min, q1_max, fine_n)
    yi_fine = np.linspace(q2_min, q2_max, fine_n)
    XI, YI = np.meshgrid(xi_fine, yi_fine)
    ZI = pwl_interp_2d(xc, yc, F.T, XI.ravel(), YI.ravel())
    ZI = ZI.reshape(fine_n, fine_n)

    # 处理外插产生的 inf
    ZI = np.where(np.isfinite(ZI), ZI, np.max(F))

    return {
        "sammon_coords": Y,
        "free_energy_ev": F,
        "grid_x": xc,
        "grid_y": yc,
        "fine_x": xi_fine,
        "fine_y": yi_fine,
        "fine_z": ZI,
        "local_minima": local_min,
        "local_maxima": local_max,
        "barrier_height_ev": float(np.max(F) - np.min(F)) if len(local_max) > 0 else 0.0,
    }


def simplex_vertex_coordinates(n_dim: int) -> np.ndarray:
    """
    计算 n 维空间中正则单形的顶点坐标。

    几何约束:
      1) 质心在原点: (1/(n+1)) Σ v_k = 0
      2) 每个顶点到质心距离为 1: ||v_k|| = 1
      3) 任意两个顶点间距离相等: ||v_i - v_j|| = const
      4) 任意两个顶点夹角: arccos(-1/n)

    构造方法:
        前 n 个顶点取 n×n 单位矩阵的列，第 n+1 个顶点取:
            a = (1 - sqrt(1+n)) / n
        然后平移使质心为零，再缩放使范数为 1。
    """
    if n_dim < 1:
        raise ValueError("n_dim must be >= 1")

    x = np.zeros((n_dim, n_dim + 1), dtype=np.float64)
    for j in range(n_dim):
        x[j, j] = 1.0

    a = (1.0 - np.sqrt(1.0 + n_dim)) / n_dim
    x[:, n_dim] = a

    # 平移至质心为零
    centroid = np.mean(x, axis=1, keepdims=True)
    x -= centroid

    # 缩放
    s = np.linalg.norm(x[:, 0])
    if s < 1e-15:
        raise ValueError("degenerate simplex")
    x /= s

    return x
