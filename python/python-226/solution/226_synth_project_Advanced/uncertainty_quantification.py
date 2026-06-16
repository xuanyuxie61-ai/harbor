"""
不确定度量化与条件线性规划界模块
==============================
实现中微子振荡参数反演的置信区间估计，
使用条件线性规划（CLP）方法和Multiplier Bootstrap。

核心方法：
1. 条件线性规划界（CLP Bounds）：
   在约束条件 Aᵀν ≥ q 下求解：
   min/max νᵀ b̂₀(X)

   用于反演参数的界估计。

2. Multiplier Bootstrap置信区间：
   使用指数分布权重进行重采样
   W_i ~ Exp(1), E[W] = 1
   θ̃* = (1/n) Σ W_i × c_i
   重复B次获得Bootstrap分布

3. 交叉拟合（Cross-fitting）：
   K折交叉验证避免过拟合偏差
   将数据分为K份，轮流用作训练和测试

数据来源：
- 1223_gev26_clpbounds: CLP界估计、交叉拟合、Bootstrap
- 1275_cognizant-ai-labs_red-paper: GP不确定度量化
"""

import numpy as np
from typing import Tuple, List, Optional, Callable


class ConditionalLinearProgram:
    """
    条件线性规划求解器。

    用于计算反演参数的界：
    min/max νᵀ b̂₀(X)
    subject to: Aᵀν ≥ q

    其中：
    - ν: 对偶变量
    - b̂₀(X): 第一阶段估计
    - A: 约束矩阵
    - q: 约束右端
    """

    def __init__(self, A: np.ndarray, q: np.ndarray):
        """
        初始化CLP。

        参数：
            A: 约束矩阵 (r × k)
            q: 约束右端 (k,)
        """
        self.A = A
        self.q = q
        self.vertices = None

    def enumerate_vertices(self) -> np.ndarray:
        """
        枚举多面体 {ν: Aᵀν ≥ q} 的所有顶点。

        方法：对于小规模问题，枚举所有基可行解

        返回：
            vertices: 顶点列表
        """
        r, k = self.A.shape

        # 对于3维问题，直接枚举
        vertices = []

        # 选择k个约束作为等式
        from itertools import combinations

        for indices in combinations(range(k), r):
            A_sub = self.A[:, indices].T
            q_sub = self.q[list(indices)]

            try:
                nu = np.linalg.solve(A_sub, q_sub)
                # 检查是否满足所有约束
                if np.all(self.A.T @ nu >= self.q - 1e-10):
                    vertices.append(nu)
            except np.linalg.LinAlgError:
                continue

        self.vertices = np.array(vertices) if vertices else np.empty((0, r))
        return self.vertices

    def solve_bound(self, b_hat: np.ndarray, direction: str = 'min') -> float:
        """
        求解线性规划界。

        min/max νᵀ b̂
        subject to: ν ∈ vertices

        参数：
            b_hat: 第一阶段估计 (r,)
            direction: 'min' 或 'max'

        返回：
            bound: 界值
        """
        if self.vertices is None or len(self.vertices) == 0:
            self.enumerate_vertices()

        if len(self.vertices) == 0:
            return 0.0

        # 在所有顶点上计算目标值
        objective_values = self.vertices @ b_hat

        if direction == 'min':
            return np.min(objective_values)
        else:
            return np.max(objective_values)


class CrossFittingEstimator:
    """
    交叉拟合估计器。

    用于避免过拟合偏差的第一阶段估计。

    K折交叉拟合：
    1. 将数据分为K份
    2. 对每折k：
       - 使用其他K-1折训练模型
       - 在第k折上进行预测
    3. 合并所有折的预测

    估计器类型：
    - OLS: 普通最小二乘
    - Ridge: 岭回归
    """

    def __init__(self, n_folds: int = 5, estimator_type: str = 'ols'):
        """
        初始化交叉拟合。

        参数：
            n_folds: 折数
            estimator_type: 估计器类型
        """
        self.n_folds = n_folds
        self.estimator_type = estimator_type

    def fit_predict(self, X: np.ndarray, y: np.ndarray,
                    sample_weight: np.ndarray = None) -> np.ndarray:
        """
        交叉拟合并预测。

        参数：
            X: 特征矩阵 (N, D)
            y: 目标变量 (N,)
            sample_weight: 样本权重

        返回：
            y_hat: 交叉拟合预测 (N,)
        """
        N = len(X)
        y_hat = np.zeros(N)

        # 随机打乱
        indices = np.random.permutation(N)
        fold_size = N // self.n_folds

        for fold in range(self.n_folds):
            # 测试集索引
            start = fold * fold_size
            end = start + fold_size if fold < self.n_folds - 1 else N
            test_idx = indices[start:end]
            train_idx = np.setdiff1d(indices, test_idx)

            # 训练
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]

            if self.estimator_type == 'ols':
                # OLS: β = (XᵀX)⁻¹Xᵀy
                try:
                    beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                    y_hat[test_idx] = X_test @ beta
                except np.linalg.LinAlgError:
                    y_hat[test_idx] = np.mean(y_train)

            elif self.estimator_type == 'ridge':
                # Ridge: β = (XᵀX + λI)⁻¹Xᵀy
                lam = 1.0
                XtX = X_train.T @ X_train
                Xty = X_train.T @ y_train
                beta = np.linalg.solve(XtX + lam * np.eye(X_train.shape[1]), Xty)
                y_hat[test_idx] = X_test @ beta

        return y_hat


class MultiplierBootstrap:
    """
    Multiplier Bootstrap置信区间。

    使用指数分布权重进行重采样：
    W_i ~ Exp(1) - 1 (中心化)

    Bootstrap估计：
    θ̃* = (1/n) Σ (W_i + 1) × c_i

    重复B次获得Bootstrap分布，计算分位数作为置信区间。
    """

    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        """
        初始化Bootstrap。

        参数：
            n_bootstrap: Bootstrap次数
            confidence_level: 置信水平
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

    def compute_ci(self, contributions: np.ndarray, seed: int = None) -> dict:
        """
        计算置信区间。

        参数：
            contributions: 个体贡献 c_i (N,)
            seed: 随机种子

        返回：
            ci: 置信区间结果
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        N = len(contributions)
        point_estimate = np.mean(contributions)

        # Bootstrap
        bootstrap_estimates = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # 指数分布权重 (中心化)
            W = rng.exponential(1.0, N) - 1.0

            # Bootstrap估计
            theta_boot = np.mean((W + 1) * contributions)
            bootstrap_estimates[b] = theta_boot

        # 置信区间
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

        # 标准误
        se = np.std(bootstrap_estimates)

        return {
            'point_estimate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': se,
            'confidence_level': self.confidence_level,
            'bootstrap_mean': np.mean(bootstrap_estimates),
            'bootstrap_std': np.std(bootstrap_estimates),
        }


class ParameterUncertainty:
    """
    中微子振荡参数的不确定度量化。

    结合CLP界和Bootstrap方法计算参数的置信区间。
    """

    def __init__(self):
        """初始化"""
        self.clp = None
        self.bootstrap = MultiplierBootstrap()

    def compute_parameter_bounds(self, b_hat_samples: np.ndarray,
                                  A: np.ndarray, q: np.ndarray,
                                  seed: int = None) -> dict:
        """
        计算参数界。

        参数：
            b_hat_samples: 第一阶段估计样本 (N, r)
            A: 约束矩阵
            q: 约束右端
            seed: 随机种子

        返回：
            bounds: 参数界结果
        """
        self.clp = ConditionalLinearProgram(A, q)
        self.clp.enumerate_vertices()

        N = len(b_hat_samples)

        # 对每个样本计算界
        min_bounds = np.zeros(N)
        max_bounds = np.zeros(N)

        for i in range(N):
            min_bounds[i] = self.clp.solve_bound(b_hat_samples[i], 'min')
            max_bounds[i] = self.clp.solve_bound(b_hat_samples[i], 'max')

        # Bootstrap置信区间
        min_ci = self.bootstrap.compute_ci(min_bounds, seed)
        max_ci = self.bootstrap.compute_ci(max_bounds, seed)

        return {
            'min_bound': min_ci,
            'max_bound': max_ci,
            'mean_min_bound': np.mean(min_bounds),
            'mean_max_bound': np.mean(max_bounds),
        }

    def profile_likelihood_ci(self, chi2_func: Callable, param_name: str,
                               best_fit: dict, all_params: list,
                               delta_chi2: float = 1.0) -> dict:
        """
        基于轮廓似然的置信区间。

        对于单个参数θ_i，置信区间由下式给出：
        χ²(θ_i) ≤ χ²_min + Δχ²

        其中Δχ² = 1.0对应68% CL（1个感兴趣参数）

        参数：
            chi2_func: χ²函数
            param_name: 参数名
            best_fit: 最佳拟合参数
            all_params: 参数名列表
            delta_chi2: Δχ²阈值

        返回：
            ci: 置信区间
        """
        chi2_min = chi2_func(best_fit)
        threshold = chi2_min + delta_chi2

        # 在参数范围内搜索
        from parameter_inversion import golden_section_search

        x0 = best_fit[param_name]

        # 向左搜索
        def f_left(dx):
            params = best_fit.copy()
            params[param_name] = x0 - dx
            return abs(chi2_func(params) - threshold)

        result_left = golden_section_search(f_left, 0, abs(x0) * 0.5, tol=1e-6)

        # 向右搜索
        def f_right(dx):
            params = best_fit.copy()
            params[param_name] = x0 + dx
            return abs(chi2_func(params) - threshold)

        result_right = golden_section_search(f_right, 0, abs(x0) * 0.5 + 0.1, tol=1e-6)

        return {
            'ci_lower': x0 - result_left['x_min'],
            'ci_upper': x0 + result_right['x_min'],
            'best_fit': x0,
            'chi2_min': chi2_min,
        }


def create_constraint_matrix(n_params: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建CLP约束矩阵。

    对于中微子混合参数，约束包括：
    - 混合角范围: 0 ≤ θ ≤ π/2
    - 质量平方差范围
    - 幺正性约束

    参数：
        n_params: 参数数

    返回：
        A: 约束矩阵
        q: 约束右端
    """
    # 简化的约束：参数范围
    r = n_params
    k = 2 * n_params  # 上下界

    A = np.zeros((r, k))
    q = np.zeros(k)

    for i in range(n_params):
        A[i, 2*i] = 1.0    # 上界约束
        A[i, 2*i+1] = -1.0  # 下界约束
        q[2*i] = 0.0       # θ ≤ π/2
        q[2*i+1] = -np.pi/2  # θ ≥ 0

    return A, q
