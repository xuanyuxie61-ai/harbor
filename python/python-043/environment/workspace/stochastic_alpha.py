"""
stochastic_alpha.py — 随机 α-效应的稀疏网格随机配置法

融合以下种子项目：
- 1056_sandia_sparse : Smolyak 稀疏网格多维积分

功能：
1. 将 α-效应中的湍流强度建模为随机参数
2. 使用稀疏网格随机配置法（Sparse Grid Stochastic Collocation）
   计算磁场均值、方差及反转概率
3. 提供 Gauss-Legendre、Clenshaw-Curtis 等一维求积规则

核心数学模型：
-------------
α-效应的随机参数化：
  α(r,θ; ξ) = α₀(r,θ) · (1 + σ_α · Σ_{i=1}^D ξ_i · φ_i(r,θ))

其中 ξ = (ξ₁, ..., ξ_D) ∈ Γ = [-1,1]^D 为随机变量，
φ_i(r,θ) 为空间模态（取为球谐函数 Y_{l_i}^{m_i}）。

稀疏网格积分（Smolyak 构造）：
  Q_L^D[f] = Σ_{|i| ≤ L+D-1} (-1)^{L+D-|i|} · C(L+D-|i|-1, D-1)
             · (Q_{i₁}¹ ⊗ ... ⊗ Q_{i_D}¹)[f]

其中 Q_{i_k}¹ 为一维 L-level 求积规则。

目标泛函：
  均值:   E[m_z(t)] = ∫ m_z(t; ξ) ρ(ξ) dξ
  方差:   Var[m_z(t)] = E[m_z²] - E[m_z]²
  反转概率: P_rev(t) = P(m_z(t) · m_z(0) < 0)
"""

import numpy as np
from itertools import product


class SparseGridCollocation:
    """
    Smolyak 稀疏网格随机配置法。
    源自 1056_sandia_sparse 的 sparse_grid 思想。
    """

    def __init__(self, dim, level, rule='cc'):
        """
        参数：
          dim   : 随机维度 D
          level : 稀疏网格层数 L
          rule  : 'cc' (Clenshaw-Curtis), 'gl' (Gauss-Legendre)
        """
        self.dim = dim
        self.level = level
        self.rule = rule
        self.points, self.weights = self._build_sparse_grid()
        self.n_points = len(self.weights)

    def _one_d_rule(self, level_k):
        """
        一维求积规则。
        Clenshaw-Curtis: 节点 x_j = cos(jπ/n), j=0,...,n
        Gauss-Legendre:  节点与权重由 Legendre 多项式零点确定
        """
        if level_k == 0:
            return np.array([0.0]), np.array([2.0])

        if self.rule == 'cc':
            n = 2 ** level_k
            if n == 1:
                return np.array([0.0]), np.array([2.0])
            j = np.arange(n + 1)
            x = np.cos(j * np.pi / n)
            # Clenshaw-Curtis 权重（Barycentric 公式）
            w = np.ones(n + 1)
            w[0] = 0.5
            w[-1] = 0.5
            # 归一化到 [-1,1] 上权重和为 2
            w = w * 2.0 / n
            return x, w
        elif self.rule == 'gl':
            # Gauss-Legendre: 使用 numpy 的 Legendre 多项式零点近似
            n = level_k + 1
            # 近似 GL 节点（Chebyshev 节点作为近似）
            j = np.arange(1, n + 1)
            x = np.cos((2 * j - 1) * np.pi / (2 * n))
            w = np.ones(n) * 2.0 / n
            return x, w
        else:
            raise ValueError(f"未知求积规则: {self.rule}")

    def _build_sparse_grid(self):
        """
        构建 Smolyak 稀疏网格。
        使用层级索引集：
          H(D,L) = {i ∈ ℕ^D : |i| = i₁+...+i_D ≤ L+D-1}
        """
        points_list = []
        weights_list = []

        # 预计算所有层级的一维规则
        rules_1d = {}
        for l in range(self.level + 1):
            rules_1d[l] = self._one_d_rule(l)

        # 层级差分规则
        def diff_rule(l):
            """差分规则 ΔQ_l = Q_l - Q_{l-1}。"""
            if l == 0:
                return rules_1d[0]
            x_h, w_h = rules_1d[l]
            x_l, w_l = rules_1d[l - 1]
            # 简化为仅取新节点
            return x_h, w_h

        # Smolyak 构造
        max_sum = self.level + self.dim - 1
        for multi_index in product(range(self.level + 1), repeat=self.dim):
            if sum(multi_index) > max_sum:
                continue

            # 计算组合系数
            coeff = self._smolyak_coefficient(multi_index, self.level, self.dim)
            if abs(coeff) < 1e-15:
                continue

            # 张量积节点与权重
            x_coords = [rules_1d[m][0] for m in multi_index]
            w_coords = [rules_1d[m][1] for m in multi_index]

            for pt_tuple in product(*x_coords):
                w = coeff
                for wi, idx in zip(w_coords, pt_tuple):
                    # 找到对应权重
                    pos = np.argmin(np.abs(wi[0] - idx))
                    w *= wi[1][pos]
                points_list.append(np.array(pt_tuple))
                weights_list.append(w)

        if len(points_list) == 0:
            return np.zeros((1, self.dim)), np.array([1.0])

        points = np.array(points_list)
        weights = np.array(weights_list)

        # 合并重复节点
        unique_pts = []
        unique_w = []
        tol = 1e-10
        for i, pt in enumerate(points):
            found = False
            for j, upt in enumerate(unique_pts):
                if np.linalg.norm(pt - upt) < tol:
                    unique_w[j] += weights[i]
                    found = True
                    break
            if not found:
                unique_pts.append(pt)
                unique_w.append(weights[i])

        return np.array(unique_pts), np.array(unique_w)

    def _smolyak_coefficient(self, multi_index, L, D):
        """
        Smolyak 组合系数：
          c(i) = (-1)^{L+D-|i|} · C(D-1, L+D-|i|-1)
        其中 |i| = i₁+...+i_D。
        """
        s = sum(multi_index)
        if s > L + D - 1:
            return 0.0
        k = L + D - s - 1
        if k < 0 or k > D - 1:
            return 0.0
        # 组合数 C(D-1, k)
        comb = np.math.comb(D - 1, k)
        return (-1) ** k * comb

    def integrate(self, func):
        """
        对函数 func(ξ) 进行稀疏网格积分。
        func: 输入 (n_points, dim) 数组，返回 (n_points, ...) 值数组。
        """
        values = func(self.points)
        # 加权求和
        result = np.zeros_like(values[0])
        for i in range(self.n_points):
            result += self.weights[i] * values[i]
        return result

    def compute_statistics(self, func):
        """
        计算随机函数的均值与方差。
        """
        values = func(self.points)
        mean = np.zeros_like(values[0])
        mean_sq = np.zeros_like(values[0])
        for i in range(self.n_points):
            mean += self.weights[i] * values[i]
            mean_sq += self.weights[i] * values[i] ** 2
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0.0)
        return mean, variance


class StochasticAlphaDynamo:
    """
    随机 α-效应发电机模型。
    将 α 的参数不确定性通过稀疏网格传播到磁场均值与反转概率。
    """

    def __init__(self, base_alpha, sigma_alpha, spatial_modes,
                 dim_random=3, sg_level=3):
        """
        参数：
          base_alpha    : 基础 α-效应场 (N,)
          sigma_alpha   : 随机扰动标准差
          spatial_modes : 空间模态列表 [(l1,m1), (l2,m2), ...]
          dim_random    : 随机维度
          sg_level      : 稀疏网格层数
        """
        self.base_alpha = base_alpha
        self.sigma_alpha = sigma_alpha
        self.spatial_modes = spatial_modes[:dim_random]
        self.dim_random = dim_random
        self.sg = SparseGridCollocation(dim_random, sg_level, rule='cc')

    def alpha_realization(self, xi, nodes, theta, phi):
        """
        给定随机样本 ξ，生成 α-效应实现。
        """
        alpha = self.base_alpha.copy()
        for i, (l, m) in enumerate(self.spatial_modes):
            # 空间模态：P_l^m(cosθ) · cos(mφ)
            from special_functions import associated_legendre
            P_lm = associated_legendre(l, abs(m), np.cos(theta))
            mode = P_lm * np.cos(m * phi)
            alpha += self.sigma_alpha * xi[i] * mode
        return alpha

    def estimate_reversal_probability(self, dipole_trajectories):
        """
        基于稀疏网格样本的偶极矩轨迹，估计反转概率。

        参数：
          dipole_trajectories: (n_samples, n_times) 偶极矩时间序列
        """
        n_samples, n_times = dipole_trajectories.shape
        sign0 = np.sign(dipole_trajectories[:, 0])
        reversals = np.zeros(n_times)
        for t in range(1, n_times):
            sign_t = np.sign(dipole_trajectories[:, t])
            reversals[t] = np.mean(sign_t * sign0 < 0)
        return reversals
