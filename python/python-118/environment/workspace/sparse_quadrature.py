"""
sparse_quadrature.py
多维稀疏网格高斯求积模块

融合种子项目：
- 1104_sparse_grid_gl: Smolyak稀疏网格构造 + 多维高斯-勒让德求积

Smolyak构造:
    A(q,d) = \sum_{q-d+1 \leq |\mathbf{l}|_1 \leq q} (-1)^{q - |\mathbf{l}|_1}
             \binom{d-1}{q - |\mathbf{l}|_1} (\bigotimes_{i=1}^d U^{l_i})

其中 U^{l} 为一维Gauss-Legendre规则，阶数 m_l = 2^{l} - 1 (l>=1)。

用于合金参数空间的高维积分与不确定性量化:
    \int_{\Omega} f(T, C_B, \gamma, D) d\Omega
其中 \Omega 为相图参数空间 [T_min, T_max] x [C_min, C_max] x ...
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from itertools import combinations


class SparseGridGL:
    """Smolyak稀疏网格高斯-勒让德求积。"""

    def __init__(self, dim, level_max):
        """
        参数:
            dim: 空间维数
            level_max: 最大层级 (q = level_max)
        """
        self.dim = int(dim)
        self.level_max = int(level_max)
        self.level_min = max(0, level_max + 1 - dim)

    def _level_to_order(self, level):
        """层级到一维节点数的映射: m(l) = 2^l - 1 (l>=1), m(0)=1。"""
        if level == 0:
            return 1
        return 2 ** level - 1

    def _generate_compositions(self, n, k):
        """生成n的k-部分组合 (compositions)。"""
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for rest in self._generate_compositions(n - i, k - 1):
                yield (i,) + rest

    def build_grid(self):
        """
        构造稀疏网格点与权重。
        返回:
            points: (n_points, dim)
            weights: (n_points,)
        """
        point_dict = {}  # 用于去重

        for level in range(self.level_min, self.level_max + 1):
            # 生成所有满足 |l|_1 = level 的层级向量
            for level_1d in self._generate_compositions(level, self.dim):
                # 计算组合系数
                coeff = (-1) ** (self.level_max - level)
                from math import comb
                coeff *= comb(self.dim - 1, self.level_max - level)

                # 一维节点与权重
                orders = [self._level_to_order(l) for l in level_1d]
                # 对每个维度生成GL点
                grids_1d = []
                weights_1d = []
                for o in orders:
                    if o == 1:
                        x = np.array([0.0])
                        w = np.array([2.0])
                    else:
                        x, w = leggauss(o)
                    grids_1d.append(x)
                    weights_1d.append(w)

                # 张量积
                # 使用迭代方式构造多维点
                indices = [np.arange(len(g)) for g in grids_1d]
                import itertools
                for idx in itertools.product(*indices):
                    point = np.array([grids_1d[d][idx[d]] for d in range(self.dim)])
                    weight = np.prod([weights_1d[d][idx[d]] for d in range(self.dim)])
                    total_weight = coeff * weight

                    key = tuple(np.round(point, decimals=12))
                    if key in point_dict:
                        point_dict[key] += total_weight
                    else:
                        point_dict[key] = total_weight

        points = np.array([np.array(k) for k in point_dict.keys()])
        weights = np.array([point_dict[k] for k in point_dict.keys()])
        return points, weights

    def integrate(self, func, domain_bounds):
        """
        在指定域上积分 func(x)，domain_bounds: [(a1,b1), (a2,b2), ...]。
        func: callable, 输入 (n_points, dim) 输出 (n_points,)。
        """
        points, weights = self.build_grid()
        # 将[-1,1]^d映射到实际域
        scaled_points = np.zeros_like(points)
        scale_factors = []
        for d in range(self.dim):
            a, b = domain_bounds[d]
            scaled_points[:, d] = 0.5 * (b - a) * points[:, d] + 0.5 * (b + a)
            scale_factors.append(0.5 * (b - a))
        jacobian = np.prod(scale_factors)

        f_vals = func(scaled_points)
        return jacobian * np.sum(weights * f_vals)


class AlloyPhaseSpaceSampler:
    """使用稀疏网格对合金参数空间进行采样。"""

    def __init__(self, param_names, param_bounds, level_max=3):
        """
        param_names: list of str
        param_bounds: list of (min, max)
        """
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.dim = len(param_names)
        self.grid = SparseGridGL(self.dim, level_max)

    def sample(self, callback=None):
        """
        生成参数空间采样点。
        callback: 对每个采样点执行的回调函数 func(params_dict)。
        返回:
            points: (n_points, dim)
            weights: (n_points,)
        """
        points, weights = self.grid.build_grid()
        scaled = np.zeros_like(points)
        for d in range(self.dim):
            a, b = self.param_bounds[d]
            scaled[:, d] = 0.5 * (b - a) * points[:, d] + 0.5 * (b + a)

        if callback is not None:
            for i in range(scaled.shape[0]):
                params = {self.param_names[d]: scaled[i, d] for d in range(self.dim)}
                callback(params)
        return scaled, weights

    def compute_expectation(self, func_values, domain_bounds):
        """计算期望值 \int f(x) w(x) dx / \int w(x) dx。"""
        _, weights = self.grid.build_grid()
        jacobian = 1.0
        for d in range(self.dim):
            a, b = domain_bounds[d]
            jacobian *= 0.5 * (b - a)
        total_weight = jacobian * np.sum(weights)
        return jacobian * np.sum(weights * func_values) / (total_weight + 1e-15)
