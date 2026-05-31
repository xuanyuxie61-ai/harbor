"""
rbf_pressure_field.py
径向基函数插值重建气泡周围三维压力场

核心物理模型:
1. 径向基函数（RBF）插值:
   给定 N 个散点数据 (x_i, p_i)，构造插值函数:
   p(x) = Σ_{j=1}^{N} w_j * φ(||x - x_j||)
   其中 φ(r) 为径向基函数。

2. 常用 RBF 核函数:
   - 多元二次 (MQ): φ(r) = sqrt(r² + r0²)
   - 薄板样条 (TPS): φ(r) = r² log(r)
   - 高斯: φ(r) = exp(-r²/r0²)
   - 逆多元二次 (IMQ): φ(r) = 1/sqrt(r² + r0²)

3. 权重求解:
   由插值条件 p(x_i) = p_i 得到线性系统:
   A w = p,  A_{ij} = φ(||x_i - x_j||)

4. 压力梯度重构:
   ∇p(x) = Σ_j w_j * φ'(||x - x_j||) * (x - x_j) / ||x - x_j||

映射来源:
- 1014_rbf_interp_2d: 2D RBF 插值框架 → 3D 压力场重建
"""

import numpy as np
from numpy.linalg import solve, lstsq


def phi_mq(r, r0):
    """多元二次径向基函数: φ(r) = sqrt(r² + r0²)"""
    return np.sqrt(r**2 + r0**2)


def phi_tps(r, r0):
    """薄板样条径向基函数: φ(r) = r² log(r)"""
    r = np.where(r < 1e-15, 1e-15, r)
    return r**2 * np.log(r / (r0 + 1e-15))


def phi_gaussian(r, r0):
    """高斯径向基函数: φ(r) = exp(-r²/r0²)"""
    return np.exp(-r**2 / (r0**2 + 1e-30))


def phi_imq(r, r0):
    """逆多元二次径向基函数: φ(r) = 1/sqrt(r² + r0²)"""
    return 1.0 / np.sqrt(r**2 + r0**2 + 1e-30)


def rbf_weights(m, nd, xd, r0, phi_func, pd):
    """
    计算 RBF 插值权重。
    对应 1014_rbf_interp_2d 中的 rbf_weight。

    参数:
        m: 空间维数
        nd: 数据点数
        xd: m x nd 数据点坐标
        r0: 尺度参数
        phi_func: 径向基函数
        pd: nd 个已知压力值
    返回:
        w: nd 个权重
    """
    A = np.zeros((nd, nd), dtype=float)
    for i in range(nd):
        for j in range(nd):
            r = np.linalg.norm(xd[:, i] - xd[:, j])
            A[i, j] = phi_func(r, r0)

    # 正则化处理以提高稳定性
    A += np.eye(nd) * 1e-10 * np.trace(A) / nd

    try:
        w = solve(A, pd)
    except np.linalg.LinAlgError:
        w = lstsq(A, pd, rcond=None)[0]
    return w


def rbf_interpolate(m, nd, xd, r0, phi_func, w, ni, xi):
    """
    RBF 插值计算。
    对应 1014_rbf_interp_2d 中的 rbf_interp。

    参数:
        m: 空间维数
        nd: 数据点数
        xd: m x nd 数据点
        r0: 尺度参数
        phi_func: 径向基函数
        w: nd 个权重
        ni: 插值点数
        xi: m x ni 插值点坐标
    返回:
        pi: ni 个插值结果
    """
    pi = np.zeros(ni, dtype=float)
    for i in range(ni):
        d = xd - xi[:, i][:, None]
        r = np.sqrt(np.sum(d**2, axis=0))
        v = phi_func(r, r0)
        pi[i] = np.dot(v, w)
    return pi


def rbf_gradient(m, nd, xd, r0, phi_func, w, xi):
    """
    使用 RBF 重构压力梯度 ∇p。

    参数:
        xi: m x 1 单个查询点
    返回:
        grad: m 维梯度向量
    """
    grad = np.zeros(m, dtype=float)
    for j in range(nd):
        diff = xi[:, 0] - xd[:, j]
        r = np.linalg.norm(diff)
        r = max(r, 1e-15)

        if phi_func == phi_mq:
            dphi = r / np.sqrt(r**2 + r0**2)
        elif phi_func == phi_tps:
            dphi = 2.0 * r * np.log(r / (r0 + 1e-15)) + r
        elif phi_func == phi_gaussian:
            dphi = -2.0 * r / (r0**2 + 1e-30) * np.exp(-r**2 / (r0**2 + 1e-30))
        elif phi_func == phi_imq:
            dphi = -r / ((r**2 + r0**2) ** 1.5 + 1e-30)
        else:
            dphi = 0.0

        grad += w[j] * dphi * diff / r
    return grad


def pressure_laplacian_rbf(m, nd, xd, r0, phi_func, w, xi):
    """
    使用 RBF 重构压力 Laplacian ∇²p。
    在不可压缩无粘流动中，∇²p = -ρ ∇·(u·∇u)。
    """
    laplacian = 0.0
    for j in range(nd):
        diff = xi[:, 0] - xd[:, j]
        r = np.linalg.norm(diff)
        r = max(r, 1e-15)

        if phi_func == phi_mq:
            # d²φ/dr² = r0² / (r² + r0²)^(3/2)
            d2phi = r0**2 / ((r**2 + r0**2) ** 1.5 + 1e-30)
        elif phi_func == phi_gaussian:
            d2phi = (4.0 * r**2 / (r0**4 + 1e-30) - 2.0 / (r0**2 + 1e-30)) * np.exp(-r**2 / (r0**2 + 1e-30))
        else:
            d2phi = 0.0

        # 一维径向 Laplacian 推广到 m 维: ∇²φ = d²φ/dr² + (m-1)/r * dφ/dr
        if phi_func == phi_mq:
            dphi = r / np.sqrt(r**2 + r0**2)
            lap_phi = d2phi + (m - 1.0) / r * dphi
        elif phi_func == phi_gaussian:
            dphi = -2.0 * r / (r0**2 + 1e-30) * np.exp(-r**2 / (r0**2 + 1e-30))
            lap_phi = d2phi + (m - 1.0) / r * dphi
        else:
            lap_phi = 0.0

        laplacian += w[j] * lap_phi
    return laplacian


def adaptive_rbf_scale(xd):
    """
    自适应选择 RBF 尺度参数 r0。
    经验公式: r0 = 0.5 * (平均最近邻距离)
    """
    nd = xd.shape[1]
    min_dists = []
    for i in range(nd):
        dists = np.sqrt(np.sum((xd - xd[:, i][:, None])**2, axis=0))
        dists[i] = np.inf
        min_dists.append(np.min(dists))
    return 0.5 * np.mean(min_dists)


def reconstruct_3d_pressure_field(bubble_center, bubble_radius, p_wall, p_far,
                                  n_data=100, n_eval=50, rbf_type='mq'):
    """
    重建气泡周围三维压力场。

    参数:
        bubble_center: [3] 气泡中心
        bubble_radius: 当前半径
        p_wall: 气泡壁压力
        p_far: 远场压力
        n_data: 数据点数量
        n_eval: 评估网格分辨率
        rbf_type: 'mq', 'tps', 'gaussian', 'imq'
    返回:
        xi_grid: 评估点坐标
        p_eval: 评估点压力
    """
    phi_map = {
        'mq': phi_mq,
        'tps': phi_tps,
        'gaussian': phi_gaussian,
        'imq': phi_imq,
    }
    phi_func = phi_map.get(rbf_type, phi_mq)

    # 在气泡周围生成数据点
    m = 3
    xd = np.random.randn(m, n_data)
    norms = np.sqrt(np.sum(xd**2, axis=0))
    norms = np.maximum(norms, 1e-15)
    xd = xd / norms

    # 在 [R, 5R] 范围内分布
    radii = bubble_radius * (1.0 + 4.0 * np.random.uniform(0.0, 1.0, size=n_data))
    xd = xd * radii + bubble_center[:, None]

    # 数据点压力：近似解析解（衰减球面波）
    pd = p_far + (p_wall - p_far) * (bubble_radius / radii)

    r0 = adaptive_rbf_scale(xd)
    w = rbf_weights(m, n_data, xd, r0, phi_func, pd)

    # 在规则网格上评估
    x_eval = np.linspace(bubble_center[0] - 3*bubble_radius,
                         bubble_center[0] + 3*bubble_radius, n_eval)
    y_eval = np.linspace(bubble_center[1] - 3*bubble_radius,
                         bubble_center[1] + 3*bubble_radius, n_eval)
    z_eval = np.linspace(bubble_center[2] - 3*bubble_radius,
                         bubble_center[2] + 3*bubble_radius, n_eval)

    X, Y, Z = np.meshgrid(x_eval, y_eval, z_eval, indexing='ij')
    xi_grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    ni = xi_grid.shape[1]
    p_eval = rbf_interpolate(m, n_data, xd, r0, phi_func, w, ni, xi_grid)

    return xi_grid, p_eval
