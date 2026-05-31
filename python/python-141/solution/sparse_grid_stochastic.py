"""
稀疏网格随机配置模块
======================
基于种子项目 1055_sandia_sgmgg 的核心算法改造。

在高维金融衍生品定价问题中，传统张量积Quadrature遭遇"维度灾难"。
稀疏网格(Sparse Grid)通过Smolyak构造，在保持精度的同时大幅降低节点数。

数学背景:
---------
Smolyak稀疏网格公式（基于Clenshaw-Curtis/高斯节点）:

    A(q,d) = Σ_{|i|≤q} (Δ_{i_1} ⊗ ... ⊗ Δ_{i_d})

其中 Δ_i = Q_i - Q_{i-1} 为差分算子，Q_i 为一维i级Quadrature。

组合系数:
    对于每个多指数 i = (i_1, ..., i_d)，其组合系数为:
        c_i = (-1)^{q-|i|} · C(d-1, q-|i|)
    其中 |i| = i_1 + ... + i_d，q = d + ℓ 为稀疏网格层级。

在金融随机波动率模型中，稀疏网格用于：
1. 高维参数空间中的模型校准目标函数积分
2. 多因子波动率模型的期望计算（替代全张量积Monte Carlo）
3. 敏感性分析（Greeks）的数值微分

参考:
    Nobile, Tempone, Webster, "A Sparse Grid Stochastic Collocation Method
    for Partial Differential Equations with Random Input Data",
    SIAM J. Numer. Anal., 46(5), 2309-2345, 2008.
"""

import numpy as np
from math import comb


def sandia_sgmgg_coef_naive(dim_num, point_num, sparse_index):
    """
    计算稀疏网格组合系数（朴素实现）。

    定义:
    ------
    点J是点I的"邻居"，当且仅当 sparse_index[:,J] 的每个分量
    等于或比 sparse_index[:,I] 大1。

    系数公式:
        coef(I) = Σ_{J∈Neighbor(I)} (-1)^{Σ_k (sparse_index[k,J] - sparse_index[k,I])}

    参数:
    ------
    dim_num     : int, 维度
    point_num   : int, 点数
    sparse_index: ndarray, 形状 (dim_num, point_num)

    返回:
    ------
    ndarray, 形状 (point_num,), 组合系数
    """
    sparse_index = np.asarray(sparse_index, dtype=np.int64)
    coef = np.zeros(point_num, dtype=np.int64)
    for j1 in range(point_num):
        for j2 in range(point_num):
            neighbor = True
            term = 1
            for i in range(dim_num):
                dif = sparse_index[i, j2] - sparse_index[i, j1]
                if dif == 0:
                    pass
                elif dif == 1:
                    term = -term
                else:
                    neighbor = False
                    break
            if neighbor:
                coef[j1] += term
    return coef


def sandia_sgmgg_coef_inc2(m, n1, s1, c1, s2):
    """
    计算将候选点S2加入活跃集后的增量组合系数。

    算法思想:
    ---------
    设活跃集 S1 有 N1 个点，系数为 C1。
    候选点 S2 加入后，新的系数 C3 满足:
        C3(1:N1)  = C1 - Σ_{j: min(S1(:,j), S2) = S1(:,j)} C1(j)
        C3(N1+1)  = 1

    参数:
    ------
    m  : int, 维度
    n1 : int, 活跃集点数
    s1 : ndarray, 形状 (m, n1), 活跃集多指数
    c1 : ndarray, 形状 (n1,), 活跃集系数
    s2 : ndarray, 形状 (m,), 候选点多指数

    返回:
    ------
    ndarray, 形状 (n1+1,), 新系数
    """
    s1 = np.asarray(s1, dtype=np.int64)
    c1 = np.asarray(c1, dtype=np.int64)
    s2 = np.asarray(s2, dtype=np.int64)
    c3 = np.zeros(n1 + 1, dtype=np.int64)
    c3[:n1] = c1.copy()
    c3[n1] = 1

    # 临时存储非活跃索引的系数（应最终归零，用于数值校验）
    n4 = 0
    c4 = np.zeros(n1, dtype=np.int64)
    s4 = np.zeros((m, n1), dtype=np.int64)

    for j in range(n1):
        s_min = np.minimum(s1[:, j], s2)
        k = -1
        # 检查 S_min 是否等于 S1 中某点
        for j2 in range(n1):
            if np.array_equal(s1[:, j2], s_min):
                k = j2
                break
        if k >= 0:
            c3[k] -= c1[j]
        else:
            # 检查是否等于非活跃集中的某点
            found = False
            for j2 in range(n4):
                if np.array_equal(s4[:, j2], s_min):
                    c4[j2] -= c1[j]
                    found = True
                    break
            if not found:
                s4[:, n4] = s_min
                c4[n4] = -c1[j]
                n4 += 1

    # 校验：非活跃集系数应为零
    if np.any(c4[:n4] != 0):
        raise RuntimeError("增量系数计算出错：非活跃索引残留非零系数")

    return c3


def generate_sparse_grid_indices(dim_num, level):
    """
    生成Smolyak稀疏网格的多指数集合。

    索引约束（基于Clenshaw-Curtis规则，节点数 n_i = 2^{i-1}+1, i≥2; n_1=1）:
        |i| = i_1 + ... + i_d ≤ d + level

    参数:
    ------
    dim_num : int, 维度
    level   : int, 稀疏网格层级 ℓ

    返回:
    ------
    ndarray, 形状 (dim_num, n_points)
    """
    if dim_num <= 0 or level < 0:
        raise ValueError("dim_num>0, level>=0")

    max_sum = dim_num + level
    indices = []

    def recurse(dim, current_sum, current_idx):
        if dim == dim_num:
            if current_sum <= max_sum:
                indices.append(current_idx.copy())
            return
        remaining_dims = dim_num - dim - 1
        min_val = 1
        # 剪枝：剩余维度至少取1，因此当前维度最大值受限
        max_val = max_sum - current_sum - remaining_dims
        max_val = max(max_val, min_val)
        for v in range(min_val, max_val + 1):
            current_idx[dim] = v
            recurse(dim + 1, current_sum + v, current_idx)

    recurse(0, 0, np.zeros(dim_num, dtype=np.int64))
    if not indices:
        return np.zeros((dim_num, 0), dtype=np.int64)
    return np.array(indices, dtype=np.int64).T


def clenshaw_curtis_nodes_weights(level):
    """
    生成Clenshaw-Curtis一维节点与权重。

    规则:
        n_1 = 1,   x_1 = 0,   w_1 = 2
        n_i = 2^{i-1} + 1  (i ≥ 2)
        x_j = cos(π (j-1) / (n_i - 1)),  j = 1, ..., n_i
        w_j 通过FFT计算（基于代数精度的闭式公式）

    本实现使用简化但数值稳定的权重公式。
    """
    if level == 0:
        return np.array([0.0]), np.array([2.0])
    n = 2 ** level + 1
    # 节点: x_j = -cos(π j / (n-1)) 映射到 [-1,1]
    j = np.arange(n)
    x = -np.cos(np.pi * j / (n - 1))
    # 权重: 基于Chebyshev多项式积分的闭式解
    w = np.zeros(n)
    if n == 1:
        w[0] = 2.0
        return x, w
    # 内部权重
    theta = np.pi * j / (n - 1)
    for i in range(n):
        wi = 1.0
        for k in range(1, (n - 1) // 2 + 1):
            b = 2.0 if (2 * k == n - 1) else 1.0
            wi -= b * np.cos(2.0 * k * theta[i]) / (4.0 * k * k - 1.0)
        w[i] = wi * 2.0 / (n - 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return x, w


class SparseGridIntegrator:
    """
    Smolyak稀疏网格积分器，用于高维期望计算。
    """

    def __init__(self, dim_num, level):
        self.dim_num = dim_num
        self.level = level
        self.indices = generate_sparse_grid_indices(dim_num, level)
        self.coef = sandia_sgmgg_coef_naive(dim_num, self.indices.shape[1], self.indices)
        # 预计算一维节点和权重
        max_level_per_dim = np.max(self.indices) if self.indices.size > 0 else 1
        self._1d_nodes = {}
        self._1d_weights = {}
        for lvl in range(max_level_per_dim + 1):
            x, w = clenshaw_curtis_nodes_weights(lvl)
            self._1d_nodes[lvl] = x
            self._1d_weights[lvl] = w

    def integrate(self, func, domain=None):
        """
        在指定域上积分 func(x)。

        参数:
        ------
        func   : callable, func(x) 其中 x 形状为 (dim_num,)
        domain : list of (a,b) 元组, 默认 [-1,1]^d

        返回:
        ------
        float, 积分值
        """
        if domain is None:
            domain = [(-1.0, 1.0)] * self.dim_num
        if len(domain) != self.dim_num:
            raise ValueError("domain维度与dim_num不匹配")

        total = 0.0
        n_points = self.indices.shape[1]
        for p in range(n_points):
            idx = self.indices[:, p]
            c = self.coef[p]
            if c == 0:
                continue
            # 构造张量积节点
            nodes_list = [self._1d_nodes[int(idx[d])] for d in range(self.dim_num)]
            weights_list = [self._1d_weights[int(idx[d])] for d in range(self.dim_num)]
            # 递归遍历张量积
            def tensor_product_recurse(dim, current_x, current_w):
                nonlocal total
                if dim == self.dim_num:
                    # 线性变换到实际域
                    x_transformed = np.zeros(self.dim_num, dtype=np.float64)
                    jac = 1.0
                    for d in range(self.dim_num):
                        a, b = domain[d]
                        x_transformed[d] = (current_x[d] + 1.0) * 0.5 * (b - a) + a
                        jac *= 0.5 * (b - a)
                    total += c * current_w * func(x_transformed) * jac
                    return
                for xi, wi in zip(nodes_list[dim], weights_list[dim]):
                    current_x[dim] = xi
                    tensor_product_recurse(dim + 1, current_x, current_w * wi)

            tensor_product_recurse(0, np.zeros(self.dim_num), 1.0)

        return total

    def get_total_points(self):
        """统计实际参与计算的非零系数节点总数。"""
        n_points = self.indices.shape[1]
        count = 0
        for p in range(n_points):
            if self.coef[p] == 0:
                continue
            idx = self.indices[:, p]
            prod = 1
            for d in range(self.dim_num):
                prod *= len(self._1d_nodes[int(idx[d])])
            count += prod
        return count


def sparse_grid_expectation_heston(dim_num, level, payoff_func, params):
    """
    使用稀疏网格计算Heston模型衍生品的期望价格。

    参数:
    ------
    dim_num    : int, 维度（资产价格 + 波动率 + 时间）
    level      : int, 稀疏网格层级
    payoff_func: callable, 收益函数
    params     : dict, 模型参数 {S0, v0, r, kappa, theta, sigma, rho, T}

    返回:
    ------
    float, 折现后期望收益
    """
    S0 = params['S0']
    v0 = params['v0']
    r = params['r']
    T = params['T']

    # 定义积分域（对数价格空间与波动率空间）
    # 使用 3σ 法则确定边界
    logS_std = np.sqrt(v0 * T)
    domain = [
        (np.log(S0) - 3*logS_std, np.log(S0) + 3*logS_std),  # log S
        (max(v0 * 0.1, 1e-4), v0 * 3.0)                       # v
    ]
    if dim_num > 2:
        for _ in range(dim_num - 2):
            domain.append((-3.0, 3.0))

    sg = SparseGridIntegrator(dim_num, level)

    def integrand(x):
        S = np.exp(x[0])
        v = max(x[1], 1e-8)
        return payoff_func(S, v)

    expectation = sg.integrate(integrand, domain)
    return np.exp(-r * T) * expectation
