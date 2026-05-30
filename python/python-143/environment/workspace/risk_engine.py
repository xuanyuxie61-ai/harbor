
import numpy as np
from typing import Tuple, Optional, List


class CovarianceEstimator:

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
        if len(returns) != self.n_assets:
            raise ValueError("收益向量维度不匹配.")

        lam = self.decay
        self.mean = lam * self.mean + (1.0 - lam) * returns
        dev = returns - self.mean
        self.cov = lam * self.cov + (1.0 - lam) * np.outer(dev, dev)
        self.t += 1

    def get_covariance(self) -> np.ndarray:
        return self.cov.copy()

    def get_correlation(self) -> np.ndarray:
        diag = np.sqrt(np.diag(self.cov))
        if np.any(diag < 1e-12):
            return np.eye(self.n_assets)
        corr = self.cov / np.outer(diag, diag)

        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)


class ConjugateGradientSolver:

    def __init__(self, max_iter: Optional[int] = None, tol: float = 1e-10):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, A: np.ndarray, b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> np.ndarray:
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

    def __init__(self, cov_estimator: CovarianceEstimator):
        self.cov_est = cov_estimator

    def solve(self) -> np.ndarray:
        n = self.cov_est.n_assets
        Sigma = self.cov_est.get_covariance()


        Sigma += np.eye(n) * 1e-8


        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = Sigma
        A_aug[:n, n] = 1.0
        A_aug[n, :n] = 1.0

        b_aug = np.zeros(n + 1)
        b_aug[n] = 1.0

        cg = ConjugateGradientSolver()
        sol = cg.solve(A_aug, b_aug)
        w = sol[:n]


        w = np.maximum(w, 0.0)
        sum_w = np.sum(w)
        if sum_w > 0:
            w /= sum_w
        else:
            w = np.ones(n) / n

        return w

    def portfolio_variance(self, weights: np.ndarray) -> float:
        Sigma = self.cov_est.get_covariance()
        return float(weights.T @ Sigma @ weights)


class RiskMetrics:

    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        if len(returns) == 0:
            return 0.0
        return -np.percentile(returns, (1.0 - confidence) * 100.0)

    @staticmethod
    def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
        if len(returns) == 0:
            return 0.0
        var = RiskMetrics.value_at_risk(returns, confidence)
        tail = returns[returns <= -var]
        if len(tail) == 0:
            return var
        return -np.mean(tail)

    @staticmethod
    def cornish_fisher_var(returns: np.ndarray, confidence: float = 0.95) -> float:
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
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    @staticmethod
    def calmar_ratio(returns: np.ndarray, cumulative: np.ndarray) -> float:
        mdd = RiskMetrics.max_drawdown(cumulative)
        mean_ret = np.mean(returns)
        if mdd < 1e-12:
            return 0.0
        return mean_ret / mdd


class RiskGeometry:

    @staticmethod
    def convex_hull_area_2d(points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0


        pts = np.unique(points, axis=0)
        if len(pts) < 3:
            return 0.0


        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_idx = np.argsort(angles)
        sorted_pts = pts[sorted_idx]


        n = len(sorted_pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += sorted_pts[i, 0] * sorted_pts[j, 1]
            area -= sorted_pts[j, 0] * sorted_pts[i, 1]
        return abs(area) * 0.5

    @staticmethod
    def is_convex_quadrilateral(quad: np.ndarray) -> bool:
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
