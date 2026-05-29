"""
uncertainty_quantify.py
=======================
基于稀疏网格（Sparse Grid）Smolyak 构造与高斯-埃尔米特（Gauss-Hermite）
积分规则，对超构表面制造误差进行不确定性量化（UQ）。

本模块源自项目 1105_sparse_grid_hermite 的核心算法，将多维稀疏网格
积分应用于工艺参数涨落的统计传播分析。

科学背景：
在纳米加工中，纳米柱的高度、宽度和位置存在随机涨落：
    h_i = h₀ + δh_i,   w_i = w₀ + δw_i,   (x_i, y_i) = (x₀, y₀) + (δx_i, δy_i)
其中 δh, δw, δx, δy 通常服从高斯分布 N(0, σ²)。

制造误差导致相位响应的随机变化：
    ΔΦ = Φ(h₀+δh, w₀+δw, x₀+δx, y₀+δy) - Φ(h₀, w₀, x₀, y₀)

不确定性量化的目标是计算相位响应的统计矩：
    μ_Φ   = E[ΔΦ]   ≈ ∫ ΔΦ(ξ) w(ξ) dξ
    σ_Φ²  = E[(ΔΦ - μ)²]
    S_Φ   = E[(ΔΦ - μ)³] / σ³   (偏度)
    K_Φ   = E[(ΔΦ - μ)⁴] / σ⁴   (峰度)

其中 ξ 为归一化随机变量，w(ξ) = (2π)^{-d/2} exp(-|ξ|²/2) 为高斯权重。

采用 Smolyak 稀疏网格在多维奇异积分中避免维数灾难：
    A(q,d) = Σ_{q-d+1 ≤ |l| ≤ q} (-1)^{q-|l|} C(d-1, q-|l|) ⊗_{n=1}^d U_n^{l_n}
"""

import numpy as np
from itertools import combinations_with_replacement


class UncertaintyQuantify:
    """
    使用稀疏网格高斯-埃尔米特积分进行制造误差不确定性量化。
    """

    def __init__(self, dim_num=4, level_max=3):
        """
        Parameters
        ----------
        dim_num : int
            随机维度数（如：h, w, x, y → dim=4）
        level_max : int
            Smolyak 稀疏网格层数
        """
        self.dim_num = dim_num
        self.level_max = level_max

    # ------------------------------------------------------------------
    # 一维 Gauss-Hermite 积分点与权重
    # ------------------------------------------------------------------
    @staticmethod
    def hermite_gauss_rule(order):
        """
        计算 Gauss-Hermite 积分规则：
            ∫_{-∞}^{∞} f(x) exp(-x²) dx ≈ Σ_i w_i f(x_i)

        使用 numpy 的 hermite 多项式根。

        Parameters
        ----------
        order : int
            积分点数量

        Returns
        -------
        x : ndarray, shape (order,)
        w : ndarray, shape (order,)
            注意：权重对应 exp(-x²) 权重，非标准正态
        """
        from numpy.polynomial.hermite import hermgauss
        x, w = hermgauss(order)
        return x, w

    def level_to_order_open(self, level):
        """
        将稀疏网格层数映射到一维积分阶数。
        对于 Gauss-Hermite：order = 2*level + 1。
        """
        return 2 * level + 1

    # ------------------------------------------------------------------
    # Smolyak 稀疏网格构造
    # ------------------------------------------------------------------
    def sparse_grid_hermite(self):
        """
        构造稀疏网格点与权重。

        Returns
        -------
        points : ndarray, shape (n_points, dim_num)
        weights : ndarray, shape (n_points,)
        """
        dim = self.dim_num
        level_max = self.level_max
        level_min = max(0, level_max + 1 - dim)

        grid_points = []
        grid_weights = []

        for level in range(level_min, level_max + 1):
            # 生成所有满足 |l| = level 的 dim 维分解
            # 使用递归/组合生成
            for comp in self._comp_next(level, dim):
                level_1d = np.array(comp, dtype=np.int32)
                order_1d = np.array([self.level_to_order_open(l) for l in level_1d])

                # 一维规则的乘积网格
                x_1d_list = []
                w_1d_list = []
                for d_idx in range(dim):
                    x_d, w_d = self.hermite_gauss_rule(order_1d[d_idx])
                    x_1d_list.append(x_d)
                    w_1d_list.append(w_d)

                # 笛卡尔积
                import itertools
                for indices in itertools.product(*[range(len(xd)) for xd in x_1d_list]):
                    point = np.array([x_1d_list[d][indices[d]] for d in range(dim)])
                    weight = 1.0
                    for d in range(dim):
                        weight *= w_1d_list[d][indices[d]]

                    # Smolyak 组合系数
                    coeff = (-1) ** (level_max - level)
                    # 组合数 C(dim-1, level_max - level)
                    from math import comb
                    coeff *= comb(dim - 1, level_max - level)
                    weight *= coeff

                    grid_points.append(point)
                    grid_weights.append(weight)

        if len(grid_points) == 0:
            # 退化情况
            return np.zeros((1, dim)), np.ones(1)

        points = np.array(grid_points)
        weights = np.array(grid_weights)

        # 合并重复点（简化：使用容差判断）
        points_unique = []
        weights_unique = []
        tol = 1e-10
        for i in range(len(points)):
            found = False
            for j in range(len(points_unique)):
                if np.linalg.norm(points[i] - points_unique[j]) < tol:
                    weights_unique[j] += weights[i]
                    found = True
                    break
            if not found:
                points_unique.append(points[i])
                weights_unique.append(weights[i])

        return np.array(points_unique), np.array(weights_unique)

    def _comp_next(self, n, k):
        """
        生成所有 k 元非负整数组合，使其和为 n。
        """
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for tail in self._comp_next(n - i, k - 1):
                yield (i,) + tail

    # ------------------------------------------------------------------
    # 统计矩计算
    # ------------------------------------------------------------------
    def propagate_moments(self, model_func, sigma_params):
        """
        传播制造误差，计算模型输出的统计矩。

        Parameters
        ----------
        model_func : callable
            model_func(ξ) → scalar or ndarray，ξ 为标准正态随机变量
            （注意：实际物理参数 = μ + σ * ξ）
        sigma_params : ndarray, shape (dim_num,)
            各维度的标准差 σ

        Returns
        -------
        stats : dict
            包含 mean, variance, std, skewness, kurtosis
        """
        points, weights = self.sparse_grid_hermite()
        n_points = len(points)

        # 物理参数转换：ξ → μ + σ ξ
        # 这里 model_func 已经内部处理，我们传入 ξ
        values = []
        for i in range(n_points):
            val = model_func(points[i])
            values.append(val)
        values = np.array(values)

        # 权重归一化
        w_norm = weights / np.sum(weights)

        mean = np.sum(w_norm * values)
        variance = np.sum(w_norm * (values - mean) ** 2)
        std = np.sqrt(variance)

        if std > 1e-15:
            skewness = np.sum(w_norm * (values - mean) ** 3) / std ** 3
            kurtosis = np.sum(w_norm * (values - mean) ** 4) / std ** 4
        else:
            skewness = 0.0
            kurtosis = 3.0

        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'points': points,
            'weights': w_norm,
            'values': values,
        }

    def phase_sensitivity_analysis(self, base_params, sigma_params,
                                    phase_model):
        """
        对超构表面相位响应进行敏感性分析和 Sobol 型一阶指数估计。

        base_params : ndarray (h0, w0, x0, y0, ...)
        sigma_params : ndarray
        phase_model : callable(params) → phase [rad]
        """
        dim = self.dim_num

        def wrapped_model(xi):
            params = base_params + sigma_params * xi
            return phase_model(params)

        stats = self.propagate_moments(wrapped_model, sigma_params)

        # 一阶 Sobol 指数（主效应）的简化估计
        # S_i ≈ Var(E[Y|X_i]) / Var(Y)
        sobol_first = np.zeros(dim)
        total_var = stats['variance']

        if total_var > 1e-15:
            for i in range(dim):
                # 条件期望的方差：固定 x_i，对其他维度积分
                points, weights = self.sparse_grid_hermite()
                # 简化：使用已有的网格，按 x_i 分组平均
                unique_xi = np.unique(np.round(points[:, i], decimals=8))
                conditional_means = []
                conditional_weights = []
                for ux in unique_xi:
                    mask = np.abs(points[:, i] - ux) < 1e-7
                    w_sum = np.sum(weights[mask])
                    if np.sum(mask) > 0 and w_sum > 1e-15:
                        wm = weights[mask] / w_sum
                        cm = np.sum(wm * stats['values'][mask])
                        conditional_means.append(cm)
                        conditional_weights.append(w_sum)
                if len(conditional_means) > 1:
                    cw = np.array(conditional_weights)
                    cw = cw / np.sum(cw)
                    cm_arr = np.array(conditional_means)
                    var_cond = np.sum(cw * (cm_arr - np.mean(cm_arr)) ** 2)
                    sobol_first[i] = var_cond / total_var

        stats['sobol_first'] = sobol_first
        return stats


def demo():
    """演示：对简化的相位模型进行不确定性量化。"""
    uq = UncertaintyQuantify(dim_num=3, level_max=4)

    # 简化模型：相位 ≈ k0 * (n_eff - 1) * h
    k0 = 2.0 * np.pi / 1.55e-6
    n_si = 3.48

    def phase_model(params):
        """params = [δh, δw, δx]（已归一化）"""
        h0 = 0.6e-6
        w0 = 0.3e-6
        sigma_h = 0.02e-6
        sigma_w = 0.01e-6
        sigma_x = 0.005e-6
        h = h0 + sigma_h * params[0]
        w = w0 + sigma_w * params[1]
        # 有效折射率随宽度变化
        n_eff = 1.0 + (n_si - 1.0) * (w / 0.5e-6) ** 0.7
        phi = k0 * (n_eff - 1.0) * h
        # 位置误差导致相位倾斜（一阶近似）
        phi += k0 * sigma_x * params[2] * 0.1
        return phi

    base = np.array([0.6e-6, 0.3e-6, 0.0])
    sigma = np.array([0.02e-6, 0.01e-6, 0.005e-6])

    stats = uq.phase_sensitivity_analysis(base, sigma, phase_model)
    print("[uncertainty_quantify] 相位响应统计矩:")
    print(f"  均值 μ = {stats['mean']:.4f} rad = {np.degrees(stats['mean']):.2f}°")
    print(f"  方差 σ² = {stats['variance']:.4e}")
    print(f"  标准差 σ = {stats['std']:.4f} rad = {np.degrees(stats['std']):.2f}°")
    print(f"  偏度 S = {stats['skewness']:.4f}")
    print(f"  峰度 K = {stats['kurtosis']:.4f}")
    print(f"  Sobol 一阶指数: h={stats['sobol_first'][0]:.3f}, "
          f"w={stats['sobol_first'][1]:.3f}, x={stats['sobol_first'][2]:.3f}")
    return stats


if __name__ == "__main__":
    demo()
