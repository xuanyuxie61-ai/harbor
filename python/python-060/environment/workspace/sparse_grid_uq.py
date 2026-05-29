"""
稀疏网格不确定性量化模块

本模块基于 Smolyak 稀疏网格构造，对平流层臭氧模型中的
化学参数不确定性进行高维数值积分与敏感性分析。

科学背景:
平流层化学模型包含大量不确定参数 (反应速率常数、光解截面、
传输系数等)。传统蒙特卡洛方法在高维空间收敛缓慢。
稀疏网格通过张量积的稀疏组合，以较少的样本点达到
高维积分的多项式精度。

科学公式:
1. 高维积分 (期望值):
   E[f] = ∫_{[0,1]^d} f(x) dx ≈ Σ_{i=1}^N w_i f(x_i)

2. Smolyak 构造:
   A(q,d) = Σ_{q-d+1 ≤ |l| ≤ q} (-1)^{q-|l|} × C(d-1, q-|l|) × (U^{l1} ⊗ ... ⊗ U^{ld})
   其中 U^l 为一维求积规则，|l| = l1 + ... + ld

3. Clenshaw-Curtis 一维规则:
   节点: x_j = cos(π j / m), j = 0, ..., m
   权重: 基于 Chebyshev 多项式插值
   特点: 嵌套性 (细网格包含粗网格节点)

4. 敏感性指标 (Sobol 一阶):
   S_i = Var_{x_i}(E_{x_~i}[f|x_i]) / Var(f)

融入原项目: 1103_sparse_grid_cc (Clenshaw-Curtis 稀疏网格)
"""

import numpy as np
from typing import Tuple, List, Callable, Optional
from itertools import product


class ClenshawCurtisRule:
    """
    Clenshaw-Curtis 一维求积规则
    """

    def __init__(self):
        self.rules_cache = {}

    def get_rule(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定层级的 CC 规则

        Parameters
        ----------
        level : int
            层级 (0, 1, 2, ...)
            level 0: 1个点
            level l: 2^l + 1 个点

        Returns
        -------
        x, w : ndarray
            节点和权重
        """
        if level in self.rules_cache:
            return self.rules_cache[level]

        if level == 0:
            x = np.array([0.5])
            w = np.array([1.0])
        else:
            n = 2 ** level + 1
            # 节点: [-1, 1] -> 映射到 [0, 1]
            x_cheb = np.cos(np.pi * np.arange(n) / (n - 1))
            x = 0.5 * (x_cheb + 1.0)  # 映射到 [0,1]

            # 权重 (基于 FFT 的算法)
            w = self._cc_weights(n)
            w = 0.5 * w  # Jacobian of transformation

        self.rules_cache[level] = (x, w)
        return x, w

    def _cc_weights(self, n: int) -> np.ndarray:
        """
        计算 Clenshaw-Curtis 权重
        """
        if n == 1:
            return np.array([2.0])

        w = np.zeros(n)
        theta = np.pi * np.arange(n) / (n - 1)

        # 使用简化权重公式
        for j in range(n):
            if j == 0 or j == n - 1:
                c = 1.0
            else:
                c = 2.0

            val = 0.0
            for k in range(0, n // 2):
                if 2 * k == n - 1:
                    continue
                b = 1.0
                if k == 0:
                    b = 1.0
                else:
                    b = 2.0
                val += b / (4.0 * k * k - 1.0) * np.cos(2.0 * k * theta[j])
            w[j] = c * val / (n - 1)

        # 归一化
        w = w / np.sum(w) * 2.0
        return w

    def get_nested_points(self, level: int) -> np.ndarray:
        """
        获取层级 l 的嵌套新增节点
        """
        if level == 0:
            return np.array([0.5])

        x_all, _ = self.get_rule(level)
        x_prev, _ = self.get_rule(level - 1)

        # 找出新增节点
        new_points = []
        for x in x_all:
            if not any(abs(x - xp) < 1e-14 for xp in x_prev):
                new_points.append(x)
        return np.array(new_points)


class SparseGridGenerator:
    """
    Smolyak 稀疏网格生成器
    """

    def __init__(self, dim: int, level_max: int):
        """
        Parameters
        ----------
        dim : int
            维度
        level_max : int
            最大层级
        """
        if dim < 1 or level_max < 0:
            raise ValueError("dim >= 1 且 level_max >= 0")

        self.dim = dim
        self.level_max = level_max
        self.cc = ClenshawCurtisRule()

    def compute_size(self) -> int:
        """
        计算稀疏网格点数
        """
        q = self.level_max + self.dim
        count = 0

        for l_vec in product(range(self.level_max + 1), repeat=self.dim):
            l_sum = sum(l_vec)
            if q - self.dim <= l_sum <= q - self.dim:
                count += 1

        # 更精确的计算
        # 使用闭式估计
        if self.dim == 1:
            return 2 ** self.level_max + 1
        else:
            # 粗略估计
            return int((self.level_max + 1) ** self.dim * 0.5)

    def generate_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成稀疏网格点和权重

        Returns
        -------
        points : ndarray
            网格点 (n_points, dim)
        weights : ndarray
            权重 (n_points,)
        """
        q = self.level_max + self.dim
        points_dict = {}  # 去重

        # 遍历所有满足 |l| <= q 的多重指标
        for l_vec in self._multi_indices(q):
            l_sum = sum(l_vec)
            coeff = (-1) ** (q - l_sum) * self._n_choose_k(self.dim - 1, q - l_sum)

            if coeff == 0:
                continue

            # 获取各维度规则
            rules_1d = []
            for d in range(self.dim):
                x_d, w_d = self.cc.get_rule(l_vec[d])
                rules_1d.append((x_d, w_d))

            # 张量积
            indices = [range(len(r[0])) for r in rules_1d]
            for idx in product(*indices):
                point = np.array([rules_1d[d][0][idx[d]] for d in range(self.dim)])
                weight = coeff * np.prod([rules_1d[d][1][idx[d]] for d in range(self.dim)])

                key = tuple(np.round(point, 14))
                if key in points_dict:
                    points_dict[key] += weight
                else:
                    points_dict[key] = weight

        points = np.array([np.array(k) for k in points_dict.keys()])
        weights = np.array(list(points_dict.values()))

        # 归一化权重
        if len(weights) > 0:
            weights = weights / np.sum(weights)

        return points, weights

    def _multi_indices(self, q: int):
        """
        生成满足 dim <= |l| <= q 的多重指标
        """
        from itertools import combinations_with_replacement

        # 简化: 生成所有满足 sum(l) <= q 的 l
        def generate_indices(dim, max_sum, current=[]):
            if dim == 0:
                if sum(current) <= max_sum:
                    yield tuple(current)
                return
            for i in range(max_sum + 1):
                if sum(current) + i <= max_sum:
                    yield from generate_indices(dim - 1, max_sum, current + [i])

        yield from generate_indices(self.dim, q)

    def _n_choose_k(self, n: int, k: int) -> int:
        """
        组合数 C(n, k)
        """
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result


class OzoneModelUQ:
    """
    臭氧模型的不确定性量化分析器
    """

    def __init__(self, dim: int = 5, level_max: int = 3):
        """
        Parameters
        ----------
        dim : int
            不确定参数维度
        level_max : int
            稀疏网格最大层级
        """
        self.dim = dim
        self.level_max = level_max
        self.grid_gen = SparseGridGenerator(dim, level_max)

        # 参数范围 (归一化到 [0,1]):
        # 0: 总臭氧柱敏感度
        # 1: 反应速率 A 修正因子
        # 2: 活化能 Ea 修正因子
        # 3: 涡旋扩散系数 Kzz 修正因子
        # 4: 光解速率 J 修正因子
        self.param_names = ['O3_column', 'A_factor', 'Ea_factor',
                            'Kzz_factor', 'J_factor']

    def map_parameters(self, xi: np.ndarray) -> dict:
        """
        将 [0,1]^d 的参数映射到物理参数
        """
        params = {}
        # 使用 Beta 分布转换: 参数集中在 0.5 附近
        params['A_factor'] = 0.5 + xi[0]  # [0.5, 1.5]
        params['Ea_factor'] = 0.8 + 0.4 * xi[1]  # [0.8, 1.2]
        params['Kzz_factor'] = 0.2 + 2.0 * xi[2]  # [0.2, 2.2]
        params['J_factor'] = 0.7 + 0.6 * xi[3]  # [0.7, 1.3]
        params['temp_offset'] = -5.0 + 10.0 * xi[4]  # [-5, 5] K
        return params

    def model_response(self, xi: np.ndarray,
                       base_o3_column: float = 300.0) -> float:
        """
        简化的模型响应函数: 臭氧柱对参数的响应
        使用参数化近似模型
        """
        params = self.map_parameters(xi)

        # 简化响应面模型
        # O3_column ≈ base * (A_factor^α1) * (Ea_factor^α2) * ...
        o3 = base_o3_column
        o3 *= params['A_factor'] ** 0.3
        o3 *= np.exp(-0.5 * (params['Ea_factor'] - 1.0) ** 2 / 0.04)
        o3 *= params['Kzz_factor'] ** (-0.2)
        o3 *= params['J_factor'] ** 0.4
        o3 += 2.0 * params['temp_offset']

        # 添加参数交互项
        interaction = 10.0 * (xi[0] - 0.5) * (xi[2] - 0.5)
        o3 += interaction

        return np.clip(o3, 100.0, 600.0)

    def compute_statistics(self) -> dict:
        """
        使用稀疏网格计算统计量
        """
        points, weights = self.grid_gen.generate_grid()

        if len(points) == 0:
            return {}

        values = np.array([self.model_response(p) for p in points])

        # 期望值
        mean = np.sum(weights * values)

        # 方差
        variance = np.sum(weights * (values - mean) ** 2)
        std = np.sqrt(max(variance, 0.0))

        # 分位数 (通过排序近似)
        sorted_indices = np.argsort(values)
        cum_weights = np.cumsum(weights[sorted_indices])

        def quantile(q: float) -> float:
            idx = np.searchsorted(cum_weights, q)
            idx = min(idx, len(values) - 1)
            return values[sorted_indices[idx]]

        q05 = quantile(0.05)
        q25 = quantile(0.25)
        q50 = quantile(0.50)
        q75 = quantile(0.75)
        q95 = quantile(0.95)

        return {
            'n_points': len(points),
            'mean': mean,
            'variance': variance,
            'std': std,
            'q05': q05,
            'q25': q25,
            'q50': q50,
            'q75': q75,
            'q95': q95,
            'points': points,
            'weights': weights,
            'values': values
        }

    def sobol_first_order(self, n_monte_carlo: int = 5000) -> dict:
        """
        使用蒙特卡洛估计 Sobol 一阶敏感性指标
        """
        np.random.seed(42)

        # 生成两组样本
        A = np.random.rand(n_monte_carlo, self.dim)
        B = np.random.rand(n_monte_carlo, self.dim)

        f_A = np.array([self.model_response(a) for a in A])
        f_B = np.array([self.model_response(b) for b in B])

        total_var = np.var(np.concatenate([f_A, f_B]))
        if total_var < 1e-20:
            return {f'X{i}': 0.0 for i in range(self.dim)}

        sobol = {}
        for i in range(self.dim):
            A_B = A.copy()
            A_B[:, i] = B[:, i]
            f_AB = np.array([self.model_response(ab) for ab in A_B])

            # 一阶 Sobol 指标
            S_i = np.mean(f_B * (f_AB - f_A)) / (total_var + 1e-30)
            sobol[f'X{i}_{self.param_names[i]}'] = np.clip(S_i, 0.0, 1.0)

        return sobol
