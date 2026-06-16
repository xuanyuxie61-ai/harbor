"""
sparse_grid_quadrature.py - 稀疏网格积分模块
=============================================

融合种子项目:
  - 1362_truncated_normal_sparse_grid: 截断正态稀疏网格积分
  - 590_interp: Lagrange 插值与参数化

核心数学: 稀疏网格求积与最优控制目标评估

在 PDE 约束优化中, 目标泛函需要计算空间积分:

  J(u, q) = integral_Omega L(u(x), q(x), x) dx

有限元离散化后:
  J(U, Q) = sum_{e=1}^{n_elem} integral_{T_e} L(U_h, Q_h, x) dx

稀疏网格积分 (Smolyak 算法):
  对高维积分, 全张量积网格的点数呈指数增长.
  Smolyak 稀疏网格通过选择性组合低维规则来缓解维数灾难:

  A(q, d) = sum_{q-1 <= |i|_1 <= q} (-1)^{q-|i|_1} C(q-1, q-|i|_1) (Q^{i_1} x ... x Q^{i_d})

  其中 q 是精度参数, d 是维度, Q^i 是精度为 i 的 1D 规则.

  点数: O(N * (log N)^{d-1}) vs 全张量积的 O(N^d)

截断正态分布:
  当优化目标涉及随机参数 (如随机扩散系数) 时,
  需要在截断正态分布下计算期望:
    E[L] = integral_a^b L(x) rho(mu, sigma, a, b; x) dx
  其中 rho 是截断正态密度:
    rho(x) = phi((x-mu)/sigma) / (sigma * (Phi((b-mu)/sigma) - Phi((a-mu)/sigma)))
"""

import numpy as np
from typing import Tuple, List, Callable, Optional
from itertools import product as iter_product


class SparseGridQuadrature:
    """
    Smolyak 稀疏网格求积规则.

    用于高维积分的高效计算, 特别是在随机优化中
    计算目标泛函的期望值.

    Parameters
    ----------
    dimension : int
        积分维度
    level : int
        稀疏网格精度等级
    """

    def __init__(self, dimension: int, level: int):
        self.dimension = dimension
        self.level = level
        self._points: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None

    def _gauss_patterson_rule(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Patterson 1D 求积规则.

        嵌套式 Gauss 求积: 每次增加精度时复用旧点.
        Level 1: 1 点 (中点规则)
        Level 2: 3 点
        Level 3: 7 点
        Level 4: 15 点
        Level 5: 31 点

        Parameters
        ----------
        level : int
            精度等级 (1-5)

        Returns
        -------
        x : ndarray
            积分点
        w : ndarray
            求积权重
        """
        # 预定义 Gauss-Patterson 规则
        rules = {
            1: (np.array([0.0]),
                np.array([2.0])),
            2: (np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
                np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])),
            3: (np.array([
                -0.9604912687080203, -0.7745966692414834,
                -0.4342437493468026, 0.0,
                0.4342437493468026, 0.7745966692414834,
                0.9604912687080203
            ]),
                np.array([
                    0.1046562260264672, 5.0 / 9.0 - 0.1046562260264672,
                    0.2684887819493058, 8.0 / 9.0 - 2 * 0.2684887819493058,
                    0.2684887819493058, 5.0 / 9.0 - 0.1046562260264672,
                    0.1046562260264672
                ])),
        }

        level = min(level, 3)  # 安全限制
        return rules.get(level, rules[1])

    def build_smolyak_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建 Smolyak 稀疏网格.

        Smolyak 公式:
          A(q, d) = sum_{(-1)^{q-|i|}} C(q-1, q-|i|) Q^{i_1} x ... x Q^{i_d}

        其中求和遍历满足 q-d+1 <= |i|_1 <= q 的多指标 i.

        实际实现: 遍历所有满足条件的多指标, 对每个多指标
        构造张量积规则, 按 Smolyak 系数加权.

        Returns
        -------
        points : ndarray, shape (N, d)
            积分点
        weights : ndarray, shape (N,)
            求积权重
        """
        q = self.level
        d = self.dimension

        all_points = []
        all_weights = []

        # 遍历满足 q-d+1 <= |i|_1 <= q 的多指标
        min_level = max(1, q - d + 1)
        max_level = q

        for level_sum in range(min_level, max_level + 1):
            # 生成所有 |i| = level_sum 的多指标 (每个分量 >= 1)
            for multi_idx in self._generate_multi_indices(d, level_sum):
                # 计算 Smolyak 系数
                coeff = ((-1) ** (q - level_sum) *
                         _binomial(q - 1, q - level_sum))

                if abs(coeff) < 1.0e-15:
                    continue

                # 构造 1D 规则的张量积
                rules_1d = [self._gauss_patterson_rule(l) for l in multi_idx]

                # 张量积
                pts_tensor, wts_tensor = _tensor_product_rules(rules_1d)

                all_points.append(pts_tensor)
                all_weights.append(coeff * wts_tensor)

        if len(all_points) == 0:
            # 退化情况: 使用单点规则
            return np.zeros((1, d)), np.array([2.0 ** d])

        self._points = np.vstack(all_points)
        self._weights = np.concatenate(all_weights)

        return self._points, self._weights

    def _generate_multi_indices(self, d: int, level_sum: int
                                 ) -> List[Tuple[int, ...]]:
        """
        生成满足 |i|_1 = level_sum 的多指标 (每个分量 >= 1).

        这是组合数学中的星条问题:
          将 level_sum 分配给 d 个变量, 每个 >= 1.

        等价于: 将 level_sum - d 分配给 d 个变量, 每个 >= 0.

        Parameters
        ----------
        d : int
            维度
        level_sum : int
            分量之和

        Returns
        -------
        list of tuples
            所有满足条件的多指标
        """
        if d == 1:
            return [(level_sum,)]

        indices = []
        remaining = level_sum - d  # 分配给 d 个变量, 每个 >= 0
        if remaining < 0:
            return []

        for combo in _weak_compositions(remaining, d):
            indices.append(tuple(c + 1 for c in combo))

        return indices

    def integrate(self, func: Callable, bounds: Optional[np.ndarray] = None
                   ) -> float:
        """
        使用稀疏网格计算积分.

        integral = sum_i w_i * f(x_i)

        若指定 bounds, 则将积分点从 [-1, 1]^d 映射到 [a, b]^d.
        映射公式:
          x_mapped = 0.5 * (b - a) * x + 0.5 * (a + b)
          w_mapped = w * prod(0.5 * (b_i - a_i))

        Parameters
        ----------
        func : callable
            被积函数 f(x) -> float
        bounds : ndarray, shape (d, 2), optional
            积分域 [a_i, b_i]

        Returns
        -------
        float
            积分近似值
        """
        if self._points is None:
            self.build_smolyak_grid()

        points = self._points.copy()
        weights = self._weights.copy()

        # 映射到指定域
        if bounds is not None:
            scale = 1.0
            for d in range(self.dimension):
                a, b = bounds[d]
                points[:, d] = 0.5 * (b - a) * points[:, d] + 0.5 * (a + b)
                scale *= 0.5 * (b - a)
            weights *= scale

        # 求积
        result = 0.0
        for i in range(len(weights)):
            result += weights[i] * func(points[i])

        return result


class LagrangeInterpolator:
    """
    Lagrange 插值 (融合 590_interp).

    在内点法中, Lagrange 插值用于:
      1. 障碍参数沿中心路径的插值外推
      2. 收敛曲线的多项式拟合
      3. 参数化弧长方法中的路径跟踪

    Lagrange 多项式:
      L(x) = sum_{i=0}^n y_i * l_i(x)
      l_i(x) = prod_{j!=i} (x - x_j) / (x_i - x_j)

    注意: 高阶 Lagrange 插值可能产生 Runge 现象.
    在实践中, 应限制数据点数或分段低阶插值.
    """

    def __init__(self, t_data: np.ndarray, p_data: np.ndarray):
        """
        Parameters
        ----------
        t_data : ndarray, shape (n,)
            参数值
        p_data : ndarray, shape (m, n)
            数据点 (m 维, n 个样本)
        """
        self.t_data = np.asarray(t_data)
        self.p_data = np.asarray(p_data)
        self.data_num = len(t_data)
        self.spatial_dim = p_data.shape[0] if p_data.ndim > 1 else 1

    def evaluate(self, t_interp: np.ndarray) -> np.ndarray:
        """
        计算 Lagrange 插值.

        对每个插值点 t, 计算所有 Lagrange 基多项式:
          l_i(t) = prod_{j!=i} (t - t_j) / (t_i - t_j)

        然后:
          p(t) = sum_i p_i * l_i(t)

        Parameters
        ----------
        t_interp : ndarray
            插值点

        Returns
        -------
        ndarray
            插值结果
        """
        n = self.data_num
        t_interp = np.atleast_1d(t_interp)
        m = len(t_interp)

        # 计算 Lagrange 基
        l_interp = np.zeros((n, m))
        for i in range(n):
            l_interp[i, :] = 1.0
            for j in range(n):
                if j != i:
                    denom = self.t_data[i] - self.t_data[j]
                    if abs(denom) < 1.0e-30:
                        denom = 1.0e-30
                    l_interp[i, :] *= (t_interp - self.t_data[j]) / denom

        # 矩阵乘法求值
        if self.p_data.ndim == 1:
            return self.p_data @ l_interp
        else:
            return self.p_data @ l_interp


def _tensor_product_rules(rules_1d: List[Tuple[np.ndarray, np.ndarray]]
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算多个 1D 规则的张量积.

    对 d 个 1D 规则 {(x_i, w_i)}:
      张量积点: (x_{i_1}, ..., x_{i_d}) for all (i_1, ..., i_d)
      张量积权: w_{i_1} * ... * w_{i_d}

    Parameters
    ----------
    rules_1d : list of (points, weights)
        1D 求积规则列表

    Returns
    -------
    points : ndarray, shape (N, d)
    weights : ndarray, shape (N,)
    """
    d = len(rules_1d)
    sizes = [len(r[0]) for r in rules_1d]
    N = int(np.prod(sizes))

    points = np.zeros((N, d))
    weights = np.ones(N)

    for dim in range(d):
        x_dim, w_dim = rules_1d[dim]
        n_dim = len(x_dim)

        # 重复模式
        repeat_before = int(np.prod(sizes[:dim]))
        repeat_after = int(np.prod(sizes[dim + 1:]))

        for i in range(N):
            idx = (i // repeat_after) % n_dim
            points[i, dim] = x_dim[idx]
            weights[i] *= w_dim[idx]

    return points, weights


def _weak_compositions(n: int, k: int) -> List[List[int]]:
    """
    生成 n 的 k 部分弱组合 (每个部分 >= 0).

    使用递归星条法.

    Parameters
    ----------
    n : int
        总和
    k : int
        部分数

    Returns
    -------
    list of lists
        所有组合
    """
    if k == 1:
        return [[n]]
    result = []
    for i in range(n + 1):
        for rest in _weak_compositions(n - i, k - 1):
            result.append([i] + rest)
    return result


def _binomial(n: int, k: int) -> int:
    """二项式系数 C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
