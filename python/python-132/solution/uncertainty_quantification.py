"""
uncertainty_quantification.py
=============================
不确定性量化与随机采样模块。

本模块融合以下种子项目：
- 六边形区域蒙特卡洛积分（源自项目 529_hexagon_monte_carlo）
- 随机列联表生成（源自项目 042_asa144/rcont）

科学背景
--------
精馏塔设计中的参数不确定性（如传质系数、物性数据、操作条件）会显著影响
能效与分离效果。本模块通过蒙特卡洛方法量化这些不确定性。

六边形区域蒙特卡洛：
    将操作参数空间 (T, P, R) 映射到六边形区域进行随机采样，
    计算积分：
        I = (A/n) Σ f(x_i, y_i)

随机列联表（rcont）：
    生成具有给定行和与列和的随机矩阵，用于模拟组分流量分布的随机性。
    给定总进料量，随机分配各组分在各塔板上的流量。

敏感性指标：
    Sobol 一阶指标：
        S_i = V_{X_i}(E_{X_{~i}}(Y|X_i)) / V(Y)
    本模块采用朴素蒙特卡洛估计。
"""

import numpy as np
from utils import ensure_positive


# ---------------------------------------------------------------------------
# 六边形区域采样与积分（源自项目 529_hexagon_monte_carlo）
# ---------------------------------------------------------------------------

def hexagon01_sample(n):
    """
    在单位正六边形内均匀采样。
    六边形顶点：(1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2),
               (-1,0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)。

    Parameters
    ----------
    n : int
        样本数。

    Returns
    -------
    x, y : ndarray, shape (n,)
        采样坐标。
    """
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    count = 0
    while count < n:
        # 在包围矩形 [-1,1]×[-sqrt(3)/2, sqrt(3)/2] 内均匀采样
        xr = np.random.uniform(-1.0, 1.0, size=n)
        yr = np.random.uniform(-np.sqrt(3.0)/2.0, np.sqrt(3.0)/2.0, size=n)
        # 判断是否在六边形内
        inside = (np.abs(xr) <= 1.0) & (np.abs(yr) <= np.sqrt(3.0)/2.0) & \
                 (np.abs(xr) + np.abs(yr) / np.sqrt(3.0) <= 1.0)
        valid = np.where(inside)[0]
        take = min(len(valid), n - count)
        x[count:count+take] = xr[valid[:take]]
        y[count:count+take] = yr[valid[:take]]
        count += take
    return x, y


def hexagon01_area():
    """单位正六边形面积：3*sqrt(3)/2"""
    return 3.0 * np.sqrt(3.0) / 2.0


def hexagon_monte_carlo_integrate(f_func, n_samples=10000):
    """
    在六边形区域内使用蒙特卡洛积分计算函数平均值。

    Parameters
    ----------
    f_func : callable
        函数 f(x, y)。
    n_samples : int
        样本数。

    Returns
    -------
    result : float
        积分估计值。
    """
    area = hexagon01_area()
    x, y = hexagon01_sample(n_samples)
    vals = f_func(x, y)
    vals = np.asarray(vals, dtype=float)
    return float(area * np.mean(vals))


# ---------------------------------------------------------------------------
# 随机列联表生成（源自项目 042_asa144/rcont）
# ---------------------------------------------------------------------------

def rcont_random_table(nrow, ncol, nrowt, ncolt, seed=None):
    """
    生成具有给定行和与列和的随机二维表。

    用于模拟精馏塔各板各组分流量的随机分配：
        行和 = 各板总摩尔流量
        列和 = 各组分总摩尔流量

    Parameters
    ----------
    nrow, ncol : int
        行数与列数。
    nrowt : ndarray, shape (nrow,)
        行和（各塔板总流量）。
    ncolt : ndarray, shape (ncol,)
        列和（各组分总流量）。
    seed : int, optional
        随机种子。

    Returns
    -------
    matrix : ndarray, shape (nrow, ncol)
        随机表。
    """
    if seed is not None:
        np.random.seed(seed)

    nrowt = np.asarray(nrowt, dtype=int)
    ncolt = np.asarray(ncolt, dtype=int)

    if np.sum(nrowt) != np.sum(ncolt):
        # 调整使总和相等
        diff = np.sum(nrowt) - np.sum(ncolt)
        if diff > 0:
            ncolt[-1] += diff
        else:
            nrowt[-1] -= diff

    ntotal = np.sum(nrowt)
    nvect = np.arange(1, ntotal + 1)

    # 随机置换
    perm = np.random.permutation(ntotal)
    nvect_perm = nvect[perm]

    # 构造列前缀和
    nsubt = np.cumsum(ncolt)

    matrix = np.zeros((nrow, ncol), dtype=int)
    ii = 0
    for i in range(nrow):
        limit = nrowt[i]
        for k in range(limit):
            for j in range(ncol):
                if nvect_perm[ii] <= nsubt[j]:
                    ii += 1
                    matrix[i, j] += 1
                    break

    return matrix


def random_flow_distribution(n_trays, nc, total_flows, component_totals, n_samples=5, seed=42):
    """
    生成多组随机流量分布。

    Parameters
    ----------
    n_trays : int
        塔板数。
    nc : int
        组分数。
    total_flows : ndarray, shape (n_trays,)
        各板总流量 [mol/s]。
    component_totals : ndarray, shape (nc,)
        各组分总流量 [mol/s]。
    n_samples : int
        样本数。
    seed : int
        随机种子。

    Returns
    -------
    samples : list of ndarray
        随机流量分布列表，每个元素 shape (n_trays, nc)。
    """
    total_flows = np.asarray(total_flows, dtype=int)
    component_totals = np.asarray(component_totals, dtype=int)

    # 缩放到整数
    scale = 1000
    total_flows_i = (total_flows * scale).astype(int)
    component_totals_i = (component_totals * scale).astype(int)

    samples = []
    for s in range(n_samples):
        mat = rcont_random_table(n_trays, nc, total_flows_i, component_totals_i, seed=seed + s)
        samples.append(mat.astype(float) / scale)

    return samples


# ---------------------------------------------------------------------------
# 敏感性分析
# ---------------------------------------------------------------------------

def sobol_first_order_index_mc(model_func, param_names, param_ranges,
                                n_samples=2048):
    """
    使用蒙特卡洛估计 Sobol 一阶敏感性指标。

    S_i ≈ V_{X_i}(E_{X_{~i}}(Y|X_i)) / V(Y)

    Parameters
    ----------
    model_func : callable
        模型函数 f(params_dict) -> float。
    param_names : list of str
        参数名。
    param_ranges : list of tuple
        各参数范围 (low, high)。
    n_samples : int
        样本数。

    Returns
    -------
    S1 : dict
        各参数的一阶敏感性指标。
    VY : float
        总方差。
    """
    n_params = len(param_names)

    # 生成两个独立样本矩阵
    A = np.random.rand(n_samples, n_params)
    B = np.random.rand(n_samples, n_params)

    # 映射到实际范围
    def map_params(X):
        params = {}
        for i, name in enumerate(param_names):
            low, high = param_ranges[i]
            params[name] = low + X[:, i] * (high - low)
        return params

    params_A = map_params(A)
    params_B = map_params(B)

    Y_A = np.array([model_func({name: params_A[name][j] for name in param_names})
                    for j in range(n_samples)], dtype=float)
    Y_B = np.array([model_func({name: params_B[name][j] for name in param_names})
                    for j in range(n_samples)], dtype=float)

    VY = np.var(np.concatenate([Y_A, Y_B]))
    if VY < 1e-15:
        VY = 1e-15

    S1 = {}
    for i, name in enumerate(param_names):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        params_ABi = map_params(A_Bi)
        Y_ABi = np.array([model_func({n: params_ABi[n][j] for n in param_names})
                          for j in range(n_samples)], dtype=float)

        # 估计条件方差
        mean_A = np.mean(Y_A)
        mean_B = np.mean(Y_B)
        V_i = np.mean(Y_B * (Y_ABi - Y_A))
        S1[name] = float(V_i / VY)
        S1[name] = float(np.clip(S1[name], -1.0, 1.0))

    return S1, VY


def uncertainty_propagation_mc(model_func, param_distributions, n_samples=5000):
    """
    通过蒙特卡洛传播参数不确定性。

    Parameters
    ----------
    model_func : callable
        模型函数。
    param_distributions : dict
        {param_name: (dist_type, params)}，
        dist_type 为 'uniform' 或 'normal'。
    n_samples : int
        样本数。

    Returns
    -------
    mean : float
        输出均值。
    std : float
        输出标准差。
    ci_95 : tuple
        95% 置信区间。
    """
    outputs = []
    for _ in range(n_samples):
        sample_params = {}
        for name, (dist_type, params) in param_distributions.items():
            if dist_type == 'uniform':
                sample_params[name] = np.random.uniform(params[0], params[1])
            elif dist_type == 'normal':
                sample_params[name] = np.random.normal(params[0], params[1])
            else:
                sample_params[name] = params[0]
        try:
            out = model_func(sample_params)
            outputs.append(float(out))
        except Exception:
            outputs.append(np.nan)

    outputs = np.array(outputs, dtype=float)
    outputs = outputs[~np.isnan(outputs)]

    if len(outputs) == 0:
        return 0.0, 0.0, (0.0, 0.0)

    mean = float(np.mean(outputs))
    std = float(np.std(outputs))
    ci_95 = (float(np.percentile(outputs, 2.5)), float(np.percentile(outputs, 97.5)))
    return mean, std, ci_95
