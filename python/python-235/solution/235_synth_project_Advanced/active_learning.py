"""
active_learning.py — 主动学习参数空间探索
==========================================
种子项目映射:
  1143_ALRLDA (RLDA/主动学习) → Fisher信息驱动的主动学习

物理: 使用主动学习高效搜索BSM参数空间, 最大化新物理发现潜力.
"""
import numpy as np
import math


def fisher_information(significance, params, epsilon=1e-3):
    """
    计算Fisher信息矩阵 (数值差分):
      F_ij = -∂²ln L / ∂θ_i ∂θ_j
    """
    n_params = len(params)
    F = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(i, n_params):
            # 二阶中心差分
            params_pp = params.copy()
            params_pm = params.copy()
            params_mp = params.copy()
            params_mm = params.copy()

            params_pp[i] += epsilon
            params_pp[j] += epsilon
            params_pm[i] += epsilon
            params_pm[j] -= epsilon
            params_mp[i] -= epsilon
            params_mp[j] += epsilon
            params_mm[i] -= epsilon
            params_mm[j] -= epsilon

            F_pp = significance(params_pp)
            F_pm = significance(params_pm)
            F_mp = significance(params_mp)
            F_mm = significance(params_mm)

            F[i, j] = -(F_pp - F_pm - F_mp + F_mm) / (4 * epsilon**2)
            F[j, i] = F[i, j]

    return F


def sherman_morrison_update(A_inv, u, v):
    """
    Sherman-Morrison秩1更新:
      (A + uv^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
    """
    A_inv_u = A_inv @ u
    v_A_inv = v @ A_inv
    denom = 1.0 + v @ A_inv_u
    if abs(denom) < 1e-30:
        return A_inv
    return A_inv - np.outer(A_inv_u, v_A_inv) / denom


class ActiveLearningExplorer:
    """主动学习参数空间探索器."""

    def __init__(self, n_params, param_bounds, objective_func, seed=42):
        self.n_params = n_params
        self.param_bounds = param_bounds
        self.objective_func = objective_func
        self.rng = np.random.RandomState(seed)

        self.points = []
        self.scores = []
        self.fisher_matrix = None

    def initialize(self, n_initial=8):
        """拉丁超立方初始采样."""
        for dim in range(self.n_params):
            lo, hi = self.param_bounds[dim]
            pts = np.linspace(lo, hi, n_initial + 1)[:-1]
            self.rng.shuffle(pts)
            if dim == 0:
                grid = pts[:, None]
            else:
                grid = np.hstack([grid, pts[:, None]])

        for pt in grid:
            score = self.objective_func(pt)
            self.points.append(pt)
            self.scores.append(score)

        self._update_fisher()

    def _update_fisher(self):
        """更新Fisher信息矩阵."""
        if len(self.points) < 3:
            return

        pts = np.array(self.points)
        scores = np.array(self.scores)

        # 简化: 使用经验协方差逆作为Fisher近似
        if self.n_params == 1:
            var = np.var(scores) + 1e-6
            self.fisher_matrix = np.array([[1.0 / var]])
        else:
            # 加权样本协方差
            weights = np.maximum(scores, 0)
            if np.sum(weights) < 1e-10:
                weights = np.ones(len(weights))
            weights /= np.sum(weights)

            mean = np.average(pts, axis=0, weights=weights)
            cov = np.zeros((self.n_params, self.n_params))
            for i, pt in enumerate(pts):
                d = pt - mean
                cov += weights[i] * np.outer(d, d)
            cov += np.eye(self.n_params) * 1e-6  # 正则化

            self.fisher_matrix = np.linalg.inv(cov)

    def acquire_next(self, n_candidates=50):
        """采集下一个点: 最大化预期改进 (EI)."""
        candidates = []
        for _ in range(n_candidates):
            pt = np.array([
                self.rng.uniform(lo, hi) for lo, hi in self.param_bounds
            ])
            candidates.append(pt)

        # 简化EI: 使用Fisher信息加权
        if self.fisher_matrix is None:
            return candidates[0]

        best_score = max(self.scores)
        best_pt = None
        best_ei = -np.inf

        for pt in candidates:
            # 预期改进近似
            pts_arr = np.array(self.points)
            dists = [np.sqrt((pt - p) @ self.fisher_matrix @ (pt - p))
                     for p in pts_arr]
            min_dist = min(dists) if dists else 1.0

            # 探索-利用平衡
            score_est = self.objective_func(pt)
            ei = score_est + 0.1 * min_dist

            if ei > best_ei:
                best_ei = ei
                best_pt = pt

        return best_pt

    def run_exploration(self, n_initial=8, n_iterations=10, n_candidates=30):
        """运行完整的主动学习循环."""
        self.initialize(n_initial)

        uncertainties = []
        volumes = []

        for it in range(n_iterations):
            next_pt = self.acquire_next(n_candidates)
            score = self.objective_func(next_pt)
            self.points.append(next_pt)
            self.scores.append(score)
            self._update_fisher()

            # 记录不确定性
            if self.fisher_matrix is not None:
                try:
                    cov = np.linalg.inv(self.fisher_matrix)
                    unc = np.sqrt(np.diag(cov))
                    uncertainties.append(unc.tolist())
                    vol = np.sqrt(np.linalg.det(cov))
                    volumes.append(vol)
                except np.linalg.LinAlgError:
                    pass

        return {
            'points': self.points,
            'scores': self.scores,
            'n_initial': n_initial,
            'n_iterations': n_iterations,
            'uncertainties': uncertainties,
            'volumes': volumes,
        }
