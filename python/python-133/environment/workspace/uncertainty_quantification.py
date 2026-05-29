"""
uncertainty_quantification.py
==============================
基于稀疏网格随机配点法的不确定性量化 (UQ)

基于种子项目 1105_sparse_grid_hermite 融合重构。

科学背景：
---------
聚合反应动力学参数（如 kp, kt, kd）在实验测定中存在显著不确定性，
通常可用高斯随机变量描述：

    log(k_i) ~ N(μ_i, σ_i²)

这些参数的不确定性通过动力学方程传播，导致分子量分布预测
存在置信区间。本模块采用稀疏网格 Gauss-Hermite 随机配点法
(Smolyak construction) 进行高效不确定性传播。

Smolyak 稀疏网格公式：
----------------------
对于 d 维积分，Smolyak 构造为：

    A(q,d) = Σ_{q-d+1 ≤ |i| ≤ q} (-1)^{q-|i|} C(d-1, q-|i|) (Q^{i_1} ⊗ ... ⊗ Q^{i_d})

其中 Q^{i} 为一维 Gauss-Hermite 求积规则，阶数为 2^{i}-1（开型）。

一维 Gauss-Hermite 求积：

    ∫_{-∞}^{+∞} f(x) exp(-x²) dx ≈ Σ_{j=1}^{n} w_j f(x_j)

节点 x_j 为 n 阶 Hermite 多项式 H_n(x) 的根，权重：

    w_j = 2^{n-1} n! sqrt(π) / [n² H_{n-1}(x_j)²]

统计量估计：
    E[f]   ≈ Σ_k w_k f(x_k)
    Var[f] ≈ Σ_k w_k f(x_k)² - (E[f])²
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from math import comb as nchoosek


def hermite_abscissa(order: int) -> np.ndarray:
    """
    计算 Gauss-Hermite 求积节点。
    节点为 physicist's Hermite 多项式 H_n(x) 的根。

    利用 numpy.polynomial.hermite.hermgauss 直接计算。
    """
    from numpy.polynomial.hermite import hermgauss
    if order <= 0:
        return np.array([0.0])
    x, _ = hermgauss(order)
    return x


def hermite_weights(order: int) -> np.ndarray:
    """
    计算 Gauss-Hermite 求积权重（ physicist's 形式，含 exp(-x²) 权重）。
    """
    from numpy.polynomial.hermite import hermgauss
    if order <= 0:
        return np.array([np.sqrt(np.pi)])
    _, w = hermgauss(order)
    return w


def comp_next(n: int, k: int, a: Optional[np.ndarray] = None,
              more: bool = False, h: int = 0, t: int = 0) -> Tuple[np.ndarray, bool, int, int]:
    """
    计算整数 n 的 k-部分组合 (compositions)。
    基于 comp_next.m 的算法。

    组合序列例如 n=6, k=3：
      (6,0,0), (5,1,0), (4,2,0), ..., (0,0,6)
    """
    if a is None or not more:
        a = np.zeros(k, dtype=int)
        a[0] = n
        h = 0
        t = n
        more = (a[-1] != n)
        return a, more, h, t

    if 1 < t:
        h = 0
    h += 1
    t = a[h - 1]
    a[h - 1] = 0
    a[0] = t - 1
    a[h] += 1
    more = (a[-1] != n)
    return a, more, h, t


def level_to_order_open(dim_num: int, level_1d: np.ndarray) -> np.ndarray:
    """
    将 1D 水平映射为求积阶数（开型规则）。
    对于 Gauss-Hermite 规则：order = 2^{level+1} - 1
    但此处采用更简洁的映射：order = 2*level + 1
    """
    level_1d = np.asarray(level_1d, dtype=int)
    order = 2 * level_1d + 1
    order = np.maximum(order, 1)
    return order


def product_weight_herm(dim_num: int, order_1d: np.ndarray, order_nd: int) -> np.ndarray:
    """
    计算 d 维乘积网格上各点的 Gauss-Hermite 权重乘积。
    基于 product_weight_herm.m 的思想。
    """
    weights = np.ones(order_nd)
    # 利用广播构造乘积权重
    # 为简化，假设 dim_num 较小，采用循环
    # 实际上 order_nd = prod(order_1d)
    # 这里直接构造全组合
    grids = [hermite_weights(int(o)) for o in order_1d]
    # 使用 meshgrid 的展平方式
    if dim_num == 1:
        return grids[0]

    # 递归构造
    w_curr = grids[0]
    for d in range(1, dim_num):
        w_new = []
        for wi in w_curr:
            for wj in grids[d]:
                w_new.append(wi * wj)
        w_curr = np.array(w_new)

    return w_curr


def multigrid_index_z(dim_num: int, order_1d: np.ndarray, order_nd: int) -> np.ndarray:
    """
    生成乘积网格的指标。
    基于 multigrid_index_z.m：指标从 -(order-1)/2 到 +(order-1)/2
    """
    grids = []
    for d in range(dim_num):
        od = int(order_1d[d])
        base = (od - 1) // 2
        grids.append(np.arange(-base, base + 1))

    if dim_num == 1:
        return grids[0].reshape(1, -1)

    # 构造笛卡尔积
    mesh = np.array(np.meshgrid(*grids, indexing='ij'))
    indices = mesh.reshape(dim_num, -1)
    return indices


def sparse_grid_herm_size(dim_num: int, level_max: int) -> int:
    """
    计算 Smolyak 稀疏网格的节点数（近似上界）。
    """
    point_num = 0
    level_min = max(0, level_max + 1 - dim_num)
    for level in range(level_min, level_max + 1):
        a = None
        more = False
        h = 0
        t = 0
        while True:
            a, more, h, t = comp_next(level, dim_num, a, more, h, t)
            order_1d = level_to_order_open(dim_num, a)
            point_num += int(np.prod(order_1d))
            if not more:
                break
    return point_num


def sparse_grid_hermite(dim_num: int, level_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 d 维 Smolyak 稀疏网格 Gauss-Hermite 节点与权重。
    基于 sparse_grid_herm.m 的算法。

    返回：
        grid_point : (dim_num, n_points) 节点坐标
        grid_weight: (n_points,) 权重
    """
    point_num = sparse_grid_herm_size(dim_num, level_max)
    grid_point = np.zeros((dim_num, point_num))
    grid_weight = np.zeros(point_num)
    point_num2 = 0

    level_min = max(0, level_max + 1 - dim_num)

    for level in range(level_min, level_max + 1):
        level_1d = None
        more = False
        h = 0
        t = 0

        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            order_1d = level_to_order_open(dim_num, level_1d)
            order_nd = int(np.prod(order_1d))

            # 计算该子网格权重
            w2 = product_weight_herm(dim_num, order_1d, order_nd)

            # Smolyak 系数
            coeff = ((-1) ** (level_max - level)
                     * nchoosek(dim_num - 1, level_max - level))

            # 生成指标
            idx = multigrid_index_z(dim_num, order_1d, order_nd)
            base2 = np.round((order_1d - 1) / 2).astype(int)

            for pt in range(order_nd):
                # 计算实际节点坐标
                pt_coord = np.zeros(dim_num)
                for d in range(dim_num):
                    abs_idx = int(idx[d, pt])
                    # 映射到 Hermite 节点
                    herm_x = hermite_abscissa(int(order_1d[d]))
                    # idx 从 -base 到 +base
                    map_idx = abs_idx + base2[d]
                    map_idx = max(0, min(map_idx, int(order_1d[d]) - 1))
                    pt_coord[d] = herm_x[map_idx]

                # 检查是否已存在
                found = False
                for pt2 in range(point_num2):
                    if np.allclose(grid_point[:, pt2], pt_coord, atol=1.0e-10):
                        grid_weight[pt2] += coeff * w2[pt]
                        found = True
                        break

                if not found:
                    grid_point[:, point_num2] = pt_coord
                    grid_weight[point_num2] = coeff * w2[pt]
                    point_num2 += 1

            if not more:
                break

    # 截断到实际大小
    grid_point = grid_point[:, :point_num2]
    grid_weight = grid_weight[:point_num2]
    return grid_point, grid_weight


def propagate_uncertainty(model_func: Callable[[np.ndarray], float],
                          dim_num: int,
                          level_max: int = 3,
                          param_means: Optional[np.ndarray] = None,
                          param_stds: Optional[np.ndarray] = None) -> dict:
    """
    使用稀疏网格 Gauss-Hermite 配点法传播参数不确定性。

    模型：Y = model_func(ξ)，其中 ξ_i ~ N(μ_i, σ_i²)
    对数正态参数的处理：ξ 为标准正态变量，实际参数为
        k_i = exp(μ_i + σ_i ξ_i)

    返回统计量：
        mean, variance, std, skewness, kurtosis
    """
    if param_means is None:
        param_means = np.zeros(dim_num)
    if param_stds is None:
        param_stds = np.ones(dim_num)

    points, weights = sparse_grid_hermite(dim_num, level_max)
    n_points = points.shape[1]

    values = np.zeros(n_points)
    for i in range(n_points):
        xi = points[:, i]
        # 变换到实际参数空间
        params = param_means + param_stds * xi
        try:
            values[i] = model_func(params)
        except Exception:
            values[i] = 0.0

    # 归一化权重（数值鲁棒性）
    total_weight = np.sum(weights)
    if abs(total_weight) < 1.0e-15:
        total_weight = 1.0
    weights_norm = weights / total_weight

    mean_val = np.dot(weights_norm, values)
    var_val = np.dot(weights_norm, values ** 2) - mean_val ** 2
    var_val = max(var_val, 0.0)
    std_val = np.sqrt(var_val)

    # 高阶矩
    if std_val > 1.0e-12:
        skew = np.dot(weights_norm, (values - mean_val) ** 3) / (std_val ** 3)
        kurt = np.dot(weights_norm, (values - mean_val) ** 4) / (std_val ** 4)
    else:
        skew = 0.0
        kurt = 3.0

    return {
        'mean': mean_val,
        'variance': var_val,
        'std': std_val,
        'skewness': skew,
        'kurtosis': kurt,
        'points': points,
        'weights': weights_norm,
        'values': values,
    }


def sensitivity_index_sobol(values: np.ndarray,
                            weights: np.ndarray,
                            points: np.ndarray,
                            dim_num: int) -> np.ndarray:
    """
    基于稀疏网格配点的一阶 Sobol 敏感度指数近似估计。

    S_i = Var(E[Y | X_i]) / Var(Y)

    采用条件期望的网格近似：
        E[Y | X_i = x] ≈ Σ_{j: X_i^{(j)} ≈ x} w_j Y_j / Σ w_j
    """
    total_var = np.dot(weights, (values - np.dot(weights, values)) ** 2)
    total_var = max(total_var, 1.0e-15)

    S1 = np.zeros(dim_num)
    for d in range(dim_num):
        # 对第 d 维进行条件分组
        unique_vals = np.unique(np.round(points[d, :], 6))
        cond_var = 0.0
        for uv in unique_vals:
            mask = np.isclose(points[d, :], uv, atol=1.0e-5)
            if np.sum(mask) == 0:
                continue
            w_sub = weights[mask]
            y_sub = values[mask]
            w_sum = np.sum(w_sub)
            if w_sum < 1.0e-15:
                continue
            cond_mean = np.dot(w_sub, y_sub) / w_sum
            cond_var += w_sum * cond_mean ** 2

        S1[d] = (cond_var - np.dot(weights, values) ** 2) / total_var
        S1[d] = max(0.0, min(1.0, S1[d]))

    return S1
