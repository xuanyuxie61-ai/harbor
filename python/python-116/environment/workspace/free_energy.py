"""
free_energy.py
自由能计算的稀疏网格积分模块

本模块利用稀疏网格（Sparse Grid）高斯求积对脂质双分子层的
多维自由能面进行数值积分，计算配分函数与各热力学量。

参考种子项目: 1055_sandia_sgmgg (稀疏网格组合系数)

物理背景:
    双层膜的集体变量（Collective Variables, CV）空间维度通常较高：
    膜厚度 d、面积 A、取向序 S_2、曲率 H、等等。
    全张量积网格的节点数随维度指数增长（"维度灾难"）。
    稀疏网格通过 Smolyak 构造，在保持多项式精确度的同时，
    将节点数从 O(N^D) 降低到 O(N (log N)^{D-1})。

    配分函数:
        Z = ∫ exp(-βF(x)) dx
    其中 x ∈ ℝ^D 为集体变量，β = 1/(k_B T)。

    稀疏网格近似:
        Z ≈ Σ_{i} w_i exp(-βF(x_i))
    其中 {x_i, w_i} 为稀疏网格节点与权重。
"""

import numpy as np


class SparseGridIntegration:
    """
    稀疏网格积分器（基于种子项目 1055_sandia_sgmgg 的组合系数算法）。
    """

    def __init__(self, dim_num, level_max):
        """
        Parameters
        ----------
        dim_num : int
            维度数 D。
        level_max : int
            最大稀疏网格层数 L。
        """
        if dim_num < 1:
            raise ValueError("dim_num 必须至少为 1。")
        if level_max < 0:
            raise ValueError("level_max 必须非负。")
        self.dim_num = dim_num
        self.level_max = level_max
        self.points = []
        self.weights = []
        self._build_sparse_grid()

    def _index_to_level(self, index):
        """
        将稀疏索引映射到一维规则层数。
        规则: level = index（Clenshaw-Curtis 类型）。
        """
        return index

    def _univariate_nodes_weights(self, level):
        """
        一维 Clenshaw-Curtis 型节点与权重（简化版）。

        节点数: n = 2^level + 1  (level ≥ 1), n=1 (level=0)
        节点: x_j = cos(jπ/n), j=0..n
        权重: 通过离散余弦变换计算（此处用梯形法则近似）。
        """
        if level == 0:
            return np.array([0.0]), np.array([2.0])
        n = (1 << level) + 1
        j = np.arange(n)
        x = np.cos(j * np.pi / (n - 1))
        # 梯形权重（端点 0.5，内部 1.0）
        w = np.ones(n)
        w[0] = 0.5
        w[-1] = 0.5
        w = w * (2.0 / (n - 1))
        return x, w

    def _build_sparse_grid(self):
        """
        通过 Smolyak 构造建立 D 维稀疏网格。

        稀疏索引集:
            I(L,D) = { i ∈ ℕ^D : |i|_1 ≤ L + D - 1 }
        每个索引 i 对应一个全张量积规则 TP(i)。
        组合系数（种子项目 1055_sandia_sgmgg）:
            c(i) = Σ_{j∈N(i)} (-1)^{|j-i|_1}
            其中 N(i) = {j : j_k ∈ {i_k, i_k+1} 对所有 k}
        """
        dim = self.dim_num
        L = self.level_max

        # 生成满足 |i|_1 ≤ L + dim - 1 的所有索引
        indices = []

        def gen_indices(pos, current, current_sum):
            if pos == dim:
                if current_sum <= L + dim - 1:
                    indices.append(current.copy())
                return
            for idx in range(L + 1):
                if current_sum + idx > L + dim - 1:
                    break
                current.append(idx)
                gen_indices(pos + 1, current, current_sum + idx)
                current.pop()

        gen_indices(0, [], 0)

        # 对每个索引计算张量积节点和组合系数
        point_dict = {}
        for idx in indices:
            # 获取各维节点和权重
            nodes_list = []
            weights_list = []
            for d in range(dim):
                x_d, w_d = self._univariate_nodes_weights(idx[d])
                nodes_list.append(x_d)
                weights_list.append(w_d)

            # 组合系数（简化实现：只考虑邻居差为 0 或 1 的情况）
            coef = self._compute_coefficient_naive(idx, indices)
            if coef == 0:
                continue

            # 张量积节点
            grids = np.meshgrid(*nodes_list, indexing='ij')
            w_grids = np.meshgrid(*weights_list, indexing='ij')

            flat_pts = np.stack([g.ravel() for g in grids], axis=1)
            flat_w = np.prod(np.stack([g.ravel() for g in w_grids], axis=0), axis=0)

            for pt, w in zip(flat_pts, flat_w):
                key = tuple(np.round(pt, 12))
                if key not in point_dict:
                    point_dict[key] = 0.0
                point_dict[key] += coef * w

        self.points = np.array([np.array(k) for k in point_dict.keys()])
        self.weights = np.array(list(point_dict.values()))

    def _compute_coefficient_naive(self, idx, all_indices):
        """
        计算稀疏网格组合系数（受种子项目 1055_sandia_sgmgg_coef_naive 启发）。

        c(i) = Σ_{j∈N(i)} (-1)^{|j-i|_1}
        其中 j 是 i 的邻居，即对每个分量 j_k ∈ {i_k, i_k+1}。
        """
        idx = np.asarray(idx)
        dim = len(idx)
        coef = 0
        # 枚举所有 2^dim 个邻居
        for mask in range(1 << dim):
            j = idx.copy()
            diff_sum = 0
            valid = True
            for d in range(dim):
                if mask & (1 << d):
                    j[d] += 1
                    diff_sum += 1
                # 检查层数和约束
                if j[d] > self.level_max + 1:
                    valid = False
                    break
            if not valid:
                continue
            # 检查 j 是否在有效索引集中（|j|_1 ≤ L+dim-1）
            if np.sum(j) <= self.level_max + dim - 1:
                coef += ((-1) ** diff_sum)
        return coef

    def integrate(self, func):
        """
        对函数 func(x) 进行稀疏网格积分。

        ∫_{[-1,1]^D} f(x) dx ≈ Σ_i w_i f(x_i)
        """
        total = 0.0
        for pt, w in zip(self.points, self.weights):
            total += w * func(pt)
        return total

    def partition_function(self, energy_func, beta):
        """
        计算配分函数 Z = ∫ exp(-β E(x)) dx。
        """
        def integrand(x):
            return np.exp(-beta * energy_func(x))
        return self.integrate(integrand)

    def free_energy(self, energy_func, beta):
        """
        计算 Helmholtz 自由能:
            F = - (1/β) ln Z
        """
        Z = self.partition_function(energy_func, beta)
        if Z <= 0 or not np.isfinite(Z):
            return np.inf
        return -np.log(Z) / beta

    def expectation(self, observable_func, energy_func, beta):
        """
        计算热力学期望值:
            <O> = (1/Z) ∫ O(x) exp(-β E(x)) dx
        """
        def num(x):
            return observable_func(x) * np.exp(-beta * energy_func(x))

        Z = self.partition_function(energy_func, beta)
        if Z <= 0:
            return 0.0
        return self.integrate(num) / Z


class FreeEnergyCalculator:
    """
    脂质双分子层自由能计算工具。
    """

    @staticmethod
    def maier_saupe_free_energy(S, T, Tc, J_coupling=2.5):
        """
        Maier-Saupe 平均场自由能（单位: kJ/mol）。

        F(S) = (1/2) J S² - k_B T ln[∫_{-1}^{1} exp((3/2)J S x² / (k_B T)) dx]

        在平均场近似下:
            F(S)/N = (J/2) S² - k_B T ln I(S)
        其中 I(S) = ∫_0^1 exp[(3J S / (2k_B T)) (x² - 1/3)] dx

        相变: 当 T < T_c 时，F(S) 在 S>0 处有极小值（有序相）；
              当 T > T_c 时，唯一极小值在 S=0（无序相）。
        """
        if T <= 0:
            raise ValueError("温度必须为正。")
        kb = 0.008314  # kJ/(mol·K)
        if abs(T - Tc) < 1e-6:
            T = Tc + 1e-6
        # 简化计算：使用 Landau 展开近似
        # F ≈ a (T-Tc)/Tc * S² + b S⁴ + c S⁶
        a = J_coupling
        b = J_coupling * 0.5
        c = J_coupling * 0.1
        tau = (T - Tc) / Tc
        f = a * tau * S ** 2 + b * S ** 4 + c * S ** 6
        return f

    @staticmethod
    def landau_expansion_coefficients(T, Tc, J=2.5):
        """
        返回 Landau 展开系数 (a, b, c)。

        Landau 自由能展开:
            F(S) = a τ S² + b S⁴ + c S⁶,  τ = (T - T_c) / T_c
        """
        # TODO: 请补全 Landau 展开系数 (a, b, c) 的计算
        # 注意: 这里的返回值将直接用于构造 dF/dS = 0 的多项式系数
        raise NotImplementedError("landau_expansion_coefficients 方法需要补全")

    @staticmethod
    def transition_temperature_estimate(J=2.5, S_init=0.8):
        """
        用自洽方程估计相变温度:
            S = <P_2(cosθ)> = ∫ P_2(x) exp[β J S P_2(x)] dx / Z
        在 S→0 处线性化得到 T_c ≈ 0.22 J / k_B。
        """
        kb = 0.008314
        Tc = 0.220 * J / kb
        return Tc
