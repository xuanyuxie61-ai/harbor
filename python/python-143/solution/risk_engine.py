"""
risk_engine.py
==============
高频交易风险矩阵计算与凸几何分析

本模块基于以下种子项目融合:
- 981_r8ge: 通用矩阵运算, 共轭梯度法, 高斯消元 → 大规模协方差矩阵的线性系统求解
- 952_quadrilateral: 四边形几何计算 → 风险空间的可行域凸包分析

核心数学模型:
--------------
1.  组合收益与风险:
    设策略组合有 n 个资产/因子, 权重向量 w ∈ ℝⁿ.
    组合收益:  R_p = w^T r
    组合方差:  σ_p² = w^T Σ w
    其中 Σ 为收益协方差矩阵, 半正定对称.

2.  协方差矩阵估计:
    采用指数加权移动平均 (EWMA):
        Σ_t = λ Σ_{t-1} + (1-λ) r_t r_t^T
    衰减因子 λ ∈ (0,1), 高频场景典型值 λ ≈ 0.94 (RiskMetrics).
    等价的连续形式为微分方程:
        dΣ/dt = -(1-λ) Σ + (1-λ) r r^T

3.  风险价值 (VaR) 与高阶矩:
    在正态假设下:
        VaR_α = μ_p + z_α σ_p
    其中 z_α 为标准正态 α 分位数.
    考虑峰度修正的 Cornish-Fisher 展开:
        z_α^{CF} = z_α + (z_α² - 1) S / 6 + (z_α³ - 3z_α) K / 24
                   - (2z_α³ - 5z_α) S² / 36
    其中 S 为偏度, K 为超额峰度.

4.  共轭梯度法求解风险约束优化 (基于 981_r8ge/r8ge_cg):
    最小方差组合问题:
        min_w   0.5 w^T Σ w
        s.t.    1^T w = 1
    KKT 条件给出线性系统:
        [ Σ   1 ] [ w ]   [ 0 ]
        [ 1^T 0 ] [ ν ] = [ 1 ]
    对大规模 n, 采用共轭梯度法 (CG) 求解, 避免显式求逆.
    CG 迭代:
        r_0 = b - A w_0,   p_0 = r_0
        α_k = (r_k^T r_k) / (p_k^T A p_k)
        w_{k+1} = w_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
        β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + β_k p_k
    对正定矩阵 A, CG 理论上 n 步内收敛.

5.  风险空间凸几何 (基于 quadrilateral 思想):
    在 (μ, σ, S, K) 四维风险空间中, 可行投资组合构成凸集.
    对二维投影 (μ, σ), 有效前沿为凸包的上边界.
    四边形面积度量:
        A = 0.5 |Σ (x_i y_{i+1} - x_{i+1} y_i)|
    面积越大, 可行集越广, 分散化机会越多.
"""

import numpy as np
from typing import Tuple, Optional, List


class CovarianceEstimator:
    """
    协方差矩阵估计器.
    """

    def __init__(self, n_assets: int, decay: float = 0.94):
        if not (0.0 < decay < 1.0):
            raise ValueError("衰减因子 λ 必须在 (0,1) 内.")
        if n_assets <= 0:
            raise ValueError("资产数必须为正.")

        self.n_assets = n_assets
        self.decay = decay
        self.cov = np.eye(n_assets) * 1e-4
        self.mean = np.zeros(n_assets)
        self.t = 0

    def update(self, returns: np.ndarray):
        """
        指数加权更新.
            Σ_t = λ Σ_{t-1} + (1-λ) (r - μ)(r - μ)^T
            μ_t = λ μ_{t-1} + (1-λ) r
        """
        if len(returns) != self.n_assets:
            raise ValueError("收益向量维度不匹配.")

        lam = self.decay
        self.mean = lam * self.mean + (1.0 - lam) * returns
        dev = returns - self.mean
        self.cov = lam * self.cov + (1.0 - lam) * np.outer(dev, dev)
        self.t += 1

    def get_covariance(self) -> np.ndarray:
        """返回当前协方差矩阵估计."""
        return self.cov.copy()

    def get_correlation(self) -> np.ndarray:
        """返回相关系数矩阵."""
        diag = np.sqrt(np.diag(self.cov))
        if np.any(diag < 1e-12):
            return np.eye(self.n_assets)
        corr = self.cov / np.outer(diag, diag)
        # 数值稳定性修正
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)


class ConjugateGradientSolver:
    """
    共轭梯度法求解线性系统 A x = b.
    基于 981_r8ge 中 r8ge_cg 的思想.
    """

    def __init__(self, max_iter: Optional[int] = None, tol: float = 1e-10):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, A: np.ndarray, b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        求解 A x = b.

        Parameters
        ----------
        A : np.ndarray, shape (n,n)
            对称正定矩阵.
        b : np.ndarray, shape (n,)
            右端项.
        x0 : np.ndarray, optional
            初始猜测.

        Returns
        -------
        x : np.ndarray
            近似解.
        """
        n = len(b)
        max_iter = self.max_iter if self.max_iter is not None else n

        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()

        r = b - A.dot(x)
        p = r.copy()
        rs_old = np.dot(r, r)

        for _ in range(max_iter):
            Ap = A.dot(p)
            pAp = np.dot(p, Ap)

            if abs(pAp) < 1e-18:
                break

            alpha = rs_old / pAp
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.dot(r, r)

            if np.sqrt(rs_new) < self.tol:
                break

            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        return x


class MinimumVariancePortfolio:
    """
    最小方差组合优化.
    """

    def __init__(self, cov_estimator: CovarianceEstimator):
        self.cov_est = cov_estimator

    def solve(self) -> np.ndarray:
        """
        求解 min 0.5 w^T Σ w  s.t. sum(w) = 1.
        通过增广矩阵使用 CG 法.
        """
        n = self.cov_est.n_assets
        Sigma = self.cov_est.get_covariance()

        # 正则化确保正定性
        Sigma += np.eye(n) * 1e-8

        # 增广系统 (n+1) x (n+1)
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = Sigma
        A_aug[:n, n] = 1.0
        A_aug[n, :n] = 1.0

        b_aug = np.zeros(n + 1)
        b_aug[n] = 1.0

        cg = ConjugateGradientSolver()
        sol = cg.solve(A_aug, b_aug)
        w = sol[:n]

        # 投影到单纯形
        w = np.maximum(w, 0.0)
        sum_w = np.sum(w)
        if sum_w > 0:
            w /= sum_w
        else:
            w = np.ones(n) / n

        return w

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """计算组合方差."""
        Sigma = self.cov_est.get_covariance()
        return float(weights.T @ Sigma @ weights)


class RiskMetrics:
    """
    风险指标计算器.
    """

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        历史模拟法 VaR.
            VaR_α = -quantile(returns, 1-α)
        """
        if len(returns) == 0:
            return 0.0
        return -np.percentile(returns, (1.0 - confidence) * 100.0)

    @staticmethod
    def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        期望损失 (CVaR/ES):
            ES_α = -E[ R | R ≤ -VaR_α ]
        """
        if len(returns) == 0:
            return 0.0
        var = RiskMetrics.value_at_risk(returns, confidence)
        tail = returns[returns <= -var]
        if len(tail) == 0:
            return var
        return -np.mean(tail)

    @staticmethod
    def cornish_fisher_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Cornish-Fisher 展开修正 VaR, 考虑偏度和峰度.
        """
        if len(returns) < 4:
            return RiskMetrics.value_at_risk(returns, confidence)

        mu = np.mean(returns)
        sigma = np.std(returns)
        if sigma < 1e-12:
            return 0.0

        S = np.mean(((returns - mu) / sigma) ** 3)
        K = np.mean(((returns - mu) / sigma) ** 4) - 3.0

        from scipy.stats import norm
        z_alpha = norm.ppf(1.0 - confidence)

        z_cf = (z_alpha
                + (z_alpha ** 2 - 1.0) * S / 6.0
                + (z_alpha ** 3 - 3.0 * z_alpha) * K / 24.0
                - (2.0 * z_alpha ** 3 - 5.0 * z_alpha) * (S ** 2) / 36.0)

        return -(mu + z_cf * sigma)

    @staticmethod
    def max_drawdown(cumulative: np.ndarray) -> float:
        """最大回撤."""
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    @staticmethod
    def calmar_ratio(returns: np.ndarray, cumulative: np.ndarray) -> float:
        """
        Calmar 比率:
            C = μ_R / MDD
        """
        mdd = RiskMetrics.max_drawdown(cumulative)
        mean_ret = np.mean(returns)
        if mdd < 1e-12:
            return 0.0
        return mean_ret / mdd


class RiskGeometry:
    """
    风险空间几何分析, 基于 952_quadrilateral 思想.
    """

    @staticmethod
    def convex_hull_area_2d(points: np.ndarray) -> float:
        """
        二维点集的凸包面积 (Graham scan 简化版).
        对风险空间投影 (如收益-标准差平面) 计算可行域面积.
        """
        if len(points) < 3:
            return 0.0

        # 去重
        pts = np.unique(points, axis=0)
        if len(pts) < 3:
            return 0.0

        # 按极角排序
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        sorted_pts = pts[sorted_idx]

        # 鞋带公式计算多边形面积
        n = len(sorted_pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += sorted_pts[i, 0] * sorted_pts[j, 1]
            area -= sorted_pts[j, 0] * sorted_pts[i, 1]
        return abs(area) * 0.5

    @staticmethod
    def is_convex_quadrilateral(quad: np.ndarray) -> bool:
        """
        判断四边形是否为凸.
        对顶点按顺序计算叉积符号.
        """
        if quad.shape != (2, 4):
            raise ValueError("quad 必须是 2x4 数组.")

        signs = []
        for i in range(4):
            p0 = quad[:, i]
            p1 = quad[:, (i + 1) % 4]
            p2 = quad[:, (i + 2) % 4]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            signs.append(np.sign(cross))

        signs = [s for s in signs if s != 0]
        if len(signs) == 0:
            return True
        return all(s == signs[0] for s in signs)
