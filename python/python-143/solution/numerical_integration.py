"""
numerical_integration.py
========================
Fekete 点高斯积分与金融期望计算

本模块基于以下种子项目融合:
- 678_line_fekete_rule: Fekete 点计算 → 高维金融积分的节点选取与权重计算

核心数学模型:
--------------
1.  Fekete 点定义:
    对紧集 K ⊂ ℂ 和多项式空间 P_m, Fekete 点是使 Vandermonde 行列式
    最大化的点集 {x_1, ..., x_m}:
        V(x_1,...,x_m) = det[ φ_j(x_k) ]_{j,k=1}^m
    其中 {φ_j} 为 P_m 的一组基 (如 Chebyshev 多项式).

    Fekete 点具有良好的插值和积分性质, Lebesgue 常数增长缓慢.

2.  Chebyshev 多项式基:
    在区间 [a,b] 上, 经仿射变换到 [-1,1]:
        T_n(x) = cos(n arccos(x))
    前三项:
        T_0(x) = 1
        T_1(x) = x
        T_2(x) = 2x² - 1

    Vandermonde 矩阵 (Chebyshev 基):
        V_{kj} = T_{j-1}(x_k),   j=1,...,m; k=1,...,n

3.  积分权重计算:
    对基函数 φ_j, 矩:
        μ_j = ∫_a^b φ_j(x) dx
    权重 w 满足:
        V^T w = μ
    其中 V 为 Vandermonde 矩阵.
    Fekete 子集由 w 的非零分量索引确定.

4.  金融应用: 期望收益积分
    对策略收益函数 g(S_T), 在风险中性测度下:
        E[g(S_T)] = ∫ g(S_T(x)) φ(x) dx
    其中 φ(x) 为标准正态密度.
    采用 Fekete 点进行数值积分:
        E[g] ≈ Σ_{j=1}^m w_j g(S_T(x_j))
    相比等距节点, Fekete 点可有效抑制 Runge 现象.

5.  多维积分 (张量积推广):
    对 d 维问题, 采用稀疏网格或全张量积:
        ∫_{[a,b]^d} f(x) dx ≈ Σ_{j_1} ... Σ_{j_d} w_{j_1}...w_{j_d} f(x_{j_1},...,x_{j_d})
    对 d=2 (如价格-波动率平面), 全张量积节点数为 m².
"""

import numpy as np
from typing import Tuple, Optional


class ChebyshevBasis:
    """
    Chebyshev 多项式基函数.
    """

    @staticmethod
    def evaluate(m: int, x: np.ndarray) -> np.ndarray:
        """
        计算前 m 个 Chebyshev 多项式在 x 处的值.
        T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)

        Parameters
        ----------
        m : int
            多项式个数.
        x : np.ndarray
            计算点, 应在 [-1,1] 内.

        Returns
        -------
        V : np.ndarray, shape (len(x), m)
            Vandermonde 矩阵.
        """
        n = len(x)
        V = np.zeros((n, m))
        V[:, 0] = 1.0
        if m > 1:
            V[:, 1] = x
        for j in range(2, m):
            V[:, j] = 2.0 * x * V[:, j - 1] - V[:, j - 2]
        return V

    @staticmethod
    def moments(m: int, a: float, b: float) -> np.ndarray:
        """
        计算 Chebyshev 基在 [a,b] 上的积分矩.
            μ_j = ∫_a^b T_j( (2x-a-b)/(b-a) ) dx
        经变量替换 t = (2x-a-b)/(b-a), x = (b-a)/2 * t + (a+b)/2, dx = (b-a)/2 dt:
            μ_j = (b-a)/2 ∫_{-1}^1 T_j(t) dt
        """
        mu = np.zeros(m)
        scale = (b - a) / 2.0
        # ∫_{-1}^1 T_0(t) dt = 2
        mu[0] = scale * 2.0
        # ∫_{-1}^1 T_1(t) dt = 0
        if m > 1:
            mu[1] = 0.0
        # 对 j ≥ 2: ∫ T_j(t) dt = [T_{j+1}/(j+1) - T_{j-1}/(j-1)] / 2 |_{-1}^1
        # 在 t=1: T_j(1)=1; 在 t=-1: T_j(-1)=(-1)^j
        for j in range(2, m):
            val_at_1 = 1.0 / (j + 1) - 1.0 / (j - 1)
            val_at_m1 = ((-1.0) ** (j + 1)) / (j + 1) - ((-1.0) ** (j - 1)) / (j - 1)
            mu[j] = scale * 0.5 * (val_at_1 - val_at_m1)
        return mu


class FeketeQuadrature:
    """
    Fekete 点数值积分, 基于 678_line_fekete_rule 的思想.
    """

    def __init__(self, a: float = -1.0, b: float = 1.0):
        self.a = a
        self.b = b

    def compute_fekete_points(self, m: int, n_samples: int = 200) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算近似 Fekete 点.

        Parameters
        ----------
        m : int
            基函数个数 (期望的 Fekete 点数).
        n_samples : int
            候选样本点数, 必须 ≥ m.

        Returns
        -------
        nf : int
            实际选中的 Fekete 点数.
        xf : np.ndarray
            Fekete 点坐标.
        wf : np.ndarray
            积分权重.
        vf : np.ndarray
            Vandermonde 子矩阵.
        """
        if n_samples < m:
            n_samples = m

        # 候选点: Chebyshev 节点 (第一类)
        # x_k = cos( (2k-1)π / (2n) ), k=1,...,n
        k = np.arange(1, n_samples + 1)
        x_candidates = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n_samples))
        # 映射到 [a,b]
        x_candidates = 0.5 * (self.b - self.a) * x_candidates + 0.5 * (self.a + self.b)

        # 计算 Vandermonde 矩阵
        # 先将候选点映射回 [-1,1]
        t_candidates = (2.0 * x_candidates - self.a - self.b) / (self.b - self.a)
        V_tall = ChebyshevBasis.evaluate(m, t_candidates)   # n_samples × m
        V_wide = V_tall.T                                    # m × n_samples

        # 矩向量 (m,)
        mom = ChebyshevBasis.moments(m, self.a, self.b)

        # 求解 V_wide w = mom 的最小二乘解
        # MATLAB: w = V \ mom  ->  min ||V w - mom||
        w, _, _, _ = np.linalg.lstsq(V_wide, mom, rcond=None)

        # 选取非零权重对应的点
        tol = 1e-10 * np.max(np.abs(w)) if np.max(np.abs(w)) > 0 else 1e-10
        ind = np.where(np.abs(w) > tol)[0]
        nf = len(ind)

        if nf == 0:
            # 退化: 取权重绝对值最大的 m 个
            ind = np.argsort(np.abs(w))[-m:]
            nf = len(ind)

        xf = x_candidates[ind]
        wf = w[ind]
        vf = V_tall[ind, :]   # nf × m, 与原始 MATLAB 的 vf=(v(:,ind))' 一致

        # 归一化权重使其和为区间长度
        sum_w = np.sum(wf)
        if abs(sum_w) > 1e-12:
            wf = wf * (self.b - self.a) / sum_w

        return nf, xf, wf, vf

    def integrate(self, f: callable, m: int = 10, n_samples: int = 200) -> float:
        """
        使用 Fekete 点数值积分计算 ∫_a^b f(x) dx.
        """
        nf, xf, wf, _ = self.compute_fekete_points(m, n_samples)
        return float(np.sum(wf * f(xf)))


class FinancialExpectation:
    """
    金融期望计算工具.
    """

    @staticmethod
    def expected_payoff_fekete(payoff_func: callable,
                                m: int = 15,
                                a: float = -5.0,
                                b: float = 5.0) -> float:
        """
        使用 Fekete 点计算风险中性期望.
        将积分 ∫_{-∞}^∞ payoff(x) φ(x) dx 截断到 [a,b],
        并用 Fekete 点近似.

        权重修正: 包含标准正态密度 φ(x_j).
        """
        fq = FeketeQuadrature(a, b)
        nf, xf, wf, _ = fq.compute_fekete_points(m)

        # 标准正态密度
        phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xf ** 2)

        # 复合权重
        composite_weights = wf * phi
        return float(np.sum(composite_weights * payoff_func(xf)))

    @staticmethod
    def expected_shortfall_integral(returns: np.ndarray,
                                     confidence: float = 0.95,
                                     m: int = 20) -> float:
        """
        利用数值积分计算期望损失:
            ES_α = (1/(1-α)) ∫_{-∞}^{-VaR_α} (-x) f(x) dx
        采用核密度估计 + Fekete 积分.
        """
        if len(returns) < 10:
            return 0.0

        # 核密度估计
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(returns)
        var = -np.percentile(returns, (1.0 - confidence) * 100.0)

        # 积分区间
        a = np.min(returns) - 3.0 * np.std(returns)
        b = -var

        if b <= a:
            return 0.0

        fq = FeketeQuadrature(a, b)
        nf, xf, wf, _ = fq.compute_fekete_points(m)

        integrand = (-xf) * kde(xf)
        integral = np.sum(wf * integrand)
        return float(integral / (1.0 - confidence))


class MultidimensionalQuadrature:
    """
    多维数值积分 (全张量积).
    """

    @staticmethod
    def tensor_product_2d(f: callable,
                          m1: int = 8, m2: int = 8,
                          a1: float = -1.0, b1: float = 1.0,
                          a2: float = -1.0, b2: float = 1.0) -> float:
        """
        二维张量积积分:
            ∫_{a1}^{b1} ∫_{a2}^{b2} f(x,y) dy dx
            ≈ Σ_i Σ_j w_i^{(1)} w_j^{(2)} f(x_i, y_j)
        """
        fq1 = FeketeQuadrature(a1, b1)
        nf1, x1, w1, _ = fq1.compute_fekete_points(m1)

        fq2 = FeketeQuadrature(a2, b2)
        nf2, x2, w2, _ = fq2.compute_fekete_points(m2)

        total = 0.0
        for i in range(nf1):
            for j in range(nf2):
                total += w1[i] * w2[j] * f(x1[i], x2[j])

        return float(total)
