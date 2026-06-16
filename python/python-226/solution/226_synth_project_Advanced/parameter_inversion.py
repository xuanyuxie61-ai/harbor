"""
中微子振荡参数反演模块
=====================
使用黄金分割搜索和高斯过程代理模型反演中微子混合参数。

核心反演问题：
给定观测到的振荡概率 P_obs(E_i, L_j)，反演参数向量 θ = (θ₁₂, θ₁₃, θ₂₃, δ, Δm²₂₁, Δm²₃₁)

数学表达：
min_θ χ²(θ) = Σ_{i,j} [P_obs(E_i, L_j) - P_theory(E_i, L_j; θ)]² / σ²_{ij}

优化方法：
1. 黄金分割搜索（1D线搜索）
2. 梯度下降 + 线搜索
3. 高斯过程代理加速

数据来源：
- 476_golden_section: 黄金分割搜索
- 1275_cognizant-ai-labs_red-paper: 高斯过程代理模型
"""

import numpy as np
from typing import Tuple, Callable, Optional, List


def golden_section_search(f: Callable, a: float, b: float,
                          tol: float = 1e-8, max_iter: int = 1000) -> dict:
    """
    黄金分割搜索求单变量函数最小值。

    算法：
    在区间 [a, b] 内放置两个试验点：
    x₁ = a + (1-g)(b-a),  x₂ = a + g(b-a)
    其中 g = (√5-1)/2 ≈ 0.618

    若 f(x₁) < f(x₂): 新区间 [a, x₂]
    若 f(x₁) > f(x₂): 新区间 [x₁, b]

    每次迭代只需一次新函数计算。
    收敛速度: 线性，比例因子 g ≈ 0.618

    参数：
        f: 目标函数 f(x)
        a, b: 搜索区间
        tol: 区间宽度容差
        max_iter: 最大迭代次数

    返回：
        result: {'x_min': 最优点, 'f_min': 最小值, 'iterations': 迭代次数, 'interval': 最终区间}
    """
    g = (np.sqrt(5) - 1) / 2  # 黄金比例 ≈ 0.618

    # 初始试验点
    x1 = a + (1 - g) * (b - a)
    x2 = a + g * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    iterations = 0

    for _ in range(max_iter):
        iterations += 1

        # 检查收敛
        if abs(b - a) < tol:
            break

        if f1 < f2:
            # 最优点在 [a, x2] 内
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - g) * (b - a)
            f1 = f(x1)
        else:
            # 最优点在 [x1, b] 内
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + g * (b - a)
            f2 = f(x2)

    x_min = (a + b) / 2
    f_min = f(x_min)

    return {
        'x_min': x_min,
        'f_min': f_min,
        'iterations': iterations,
        'interval': (a, b),
        'converged': abs(b - a) < tol,
    }


class GaussianProcessSurrogate:
    """
    高斯过程代理模型。

    用于加速中微子振荡概率的正向计算。

    数学模型：
    f(x) ~ GP(m(x), k(x, x'))
    m(x) = 0 (零均值)
    k(x, x') = σ² exp(-||x-x'||²/(2l²)) (RBF核)

    训练：
    最大化对数边际似然：
    log p(y|X) = -½ y^T K⁻¹ y - ½ log|K| - N/2 log(2π)

    预测：
    μ* = k(X*, X) [K + σ_n²I]⁻¹ y
    σ²* = k(X*, X*) - k(X*, X) [K + σ_n²I]⁻¹ k(X, X*)
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        """
        初始化GP模型。

        参数：
            length_scale: RBF核长度尺度
            noise: 观测噪声方差
        """
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None

    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF（平方指数）核函数。

        k(x, x') = σ² exp(-||x-x'||²/(2l²))

        参数：
            X1: (N1, D)
            X2: (N2, D)

        返回：
            K: (N1, N2) 核矩阵
        """
        # 计算距离矩阵
        sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
                   np.sum(X2**2, axis=1) - \
                   2 * X1 @ X2.T

        return np.exp(-0.5 * sq_dists / self.length_scale**2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练GP模型。

        参数：
            X: 训练输入 (N, D)
            y: 训练输出 (N,)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        N = len(X)

        # 计算核矩阵
        K = self.rbf_kernel(X, X) + self.noise * np.eye(N)

        # Cholesky分解
        try:
            L = np.linalg.cholesky(K)
            # 求解 K α = y
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.L = L
        except np.linalg.LinAlgError:
            # 使用正则化
            K_reg = K + 1e-4 * np.eye(N)
            self.alpha = np.linalg.solve(K_reg, y)
            self.L = None

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GP预测。

        参数：
            X_star: 预测点 (N*, D)

        返回：
            mu: 预测均值 (N*,)
            sigma: 预测标准差 (N*,)
        """
        if self.X_train is None:
            raise RuntimeError("模型尚未训练")

        # k(X*, X)
        K_star = self.rbf_kernel(X_star, self.X_train)

        # 均值: μ* = k* α
        mu = K_star @ self.alpha

        # 方差
        K_star_star = self.rbf_kernel(X_star, X_star)

        if self.L is not None:
            v = np.linalg.solve(self.L, K_star.T)
            sigma_sq = np.diag(K_star_star) - np.sum(v**2, axis=0)
        else:
            K_inv = np.linalg.inv(self.rbf_kernel(self.X_train, self.X_train) +
                                   self.noise * np.eye(len(self.X_train)))
            sigma_sq = np.diag(K_star_star) - np.sum(K_star @ K_inv * K_star, axis=1)

        sigma_sq = np.maximum(sigma_sq, 1e-10)
        sigma = np.sqrt(sigma_sq)

        return mu, sigma


class NeutrinoParameterInversion:
    """
    中微子振荡参数反演器。

    反演参数：
    θ = (θ₁₂, θ₁₃, θ₂₃, δ_CP, Δm²₂₁, Δm²₃₁)

    目标函数：
    χ²(θ) = Σ_i (P_obs_i - P_theory_i(θ))² / σ_i²

    优化策略：
    1. 使用GP代理模型加速正向计算
    2. 黄金分割搜索进行线搜索
    3. 坐标下降法优化多维参数
    """

    def __init__(self, forward_model: Callable, observation_data: dict):
        """
        初始化反演器。

        参数：
            forward_model: 正向计算函数 P_theory(params) → probabilities
            observation_data: 观测数据字典 {'energies': ..., 'baselines': ..., 'probs': ..., 'errors': ...}
        """
        self.forward_model = forward_model
        self.obs = observation_data

        # 参数范围
        self.param_bounds = {
            'theta12': (0, np.pi/2),
            'theta13': (0, np.pi/4),
            'theta23': (0, np.pi/2),
            'delta_cp': (0, 2*np.pi),
            'delta_m2_21': (1e-6, 1e-3),  # eV²
            'delta_m2_31': (2e-3, 3e-3),  # eV²
        }

        # GP代理
        self.gp = GaussianProcessSurrogate(length_scale=0.1)
        self.gp_trained = False

    def chi_squared(self, params: dict) -> float:
        """
        计算χ²目标函数。

        χ² = Σ_i (P_obs_i - P_theory_i)² / σ_i²

        参数：
            params: 参数字典

        返回：
            chi2: χ²值
        """
        try:
            P_theory = self.forward_model(params)
        except Exception:
            return 1e10  # 无效参数返回大值

        P_obs = self.obs['probs']
        sigma = self.obs.get('errors', np.ones_like(P_obs) * 0.01)

        # 防止除零
        sigma = np.maximum(sigma, 1e-10)

        chi2 = np.sum(((P_obs - P_theory) / sigma) ** 2)

        return chi2

    def line_search_1d(self, param_name: str, current_params: dict,
                       step_size: float = 0.01) -> dict:
        """
        对单个参数进行黄金分割线搜索。

        参数：
            param_name: 参数名
            current_params: 当前参数值
            step_size: 搜索步长

        返回：
            result: 线搜索结果
        """
        x0 = current_params[param_name]
        bounds = self.param_bounds[param_name]

        # 定义1D目标函数
        def f_1d(x):
            params = current_params.copy()
            params[param_name] = x
            return self.chi_squared(params)

        # 搜索区间
        a = max(bounds[0], x0 - step_size)
        b = min(bounds[1], x0 + step_size)

        if a >= b:
            return {'x_min': x0, 'f_min': f_1d(x0)}

        return golden_section_search(f_1d, a, b)

    def coordinate_descent(self, initial_params: dict, max_cycles: int = 20,
                           tol: float = 1e-6) -> dict:
        """
        坐标下降法优化。

        算法：
        对每个参数依次进行1D线搜索，循环直到收敛。

        参数：
            initial_params: 初始参数值
            max_cycles: 最大循环次数
            tol: 收敛容差

        返回：
            result: 优化结果
        """
        params = initial_params.copy()
        param_names = list(self.param_bounds.keys())

        chi2_history = []

        for cycle in range(max_cycles):
            chi2_old = self.chi_squared(params)
            chi2_history.append(chi2_old)

            # 对每个参数进行线搜索
            for name in param_names:
                result = self.line_search_1d(name, params)
                params[name] = result['x_min']

            chi2_new = self.chi_squared(params)

            # 检查收敛
            if abs(chi2_old - chi2_new) < tol:
                break

        return {
            'optimal_params': params,
            'chi2_min': self.chi_squared(params),
            'cycles': cycle + 1,
            'chi2_history': chi2_history,
            'converged': abs(chi2_old - chi2_new) < tol if chi2_history else False,
        }

    def train_gp_surrogate(self, n_training: int = 100):
        """
        训练GP代理模型。

        在参数空间中采样训练点，计算正向模型，训练GP。

        参数：
            n_training: 训练样本数
        """
        param_names = list(self.param_bounds.keys())
        D = len(param_names)

        # Latin Hypercube采样
        X_train = np.zeros((n_training, D))
        for d in range(D):
            bounds = self.param_bounds[param_names[d]]
            X_train[:, d] = np.random.uniform(bounds[0], bounds[1], n_training)

        # 计算训练目标
        y_train = np.zeros(n_training)
        for i in range(n_training):
            params = {name: X_train[i, d] for d, name in enumerate(param_names)}
            y_train[i] = self.chi_squared(params)

        # 标准化
        self.y_mean = np.mean(y_train)
        self.y_std = np.std(y_train) + 1e-10
        y_normalized = (y_train - self.y_mean) / self.y_std

        # 训练GP
        self.gp.fit(X_train, y_normalized)
        self.gp_trained = True
        self.gp_param_names = param_names

    def predict_chi2_gp(self, params: dict) -> float:
        """
        使用GP代理预测χ²。

        参数：
            params: 参数字典

        返回：
            chi2_pred: 预测的χ²值
        """
        if not self.gp_trained:
            raise RuntimeError("GP代理未训练")

        x = np.array([[params[name] for name in self.gp_param_names]])
        mu, _ = self.gp.predict(x)

        # 反标准化
        chi2_pred = mu[0] * self.y_std + self.y_mean

        return max(0, chi2_pred)  # χ²非负


def create_synthetic_observation(true_params: dict, energies: np.ndarray,
                                  baselines: np.ndarray, noise_level: float = 0.01,
                                  seed: int = 42) -> dict:
    """
    创建合成观测数据。

    参数：
        true_params: 真实参数值
        energies: 能量数组
        baselines: 基线长度数组
        noise_level: 噪声水平
        seed: 随机种子

    返回：
        observation: 观测数据字典
    """
    rng = np.random.RandomState(seed)

    from neutrino_physics import pmns_matrix, hamiltonian_vacuum
    from matrix_evolution import NeutrinoEvolution

    n_obs = len(energies) * len(baselines)
    probs_true = np.zeros(n_obs)

    evolution = NeutrinoEvolution('matrix_exp')

    idx = 0
    for E in energies:
        for L in baselines:
            U = pmns_matrix(true_params['theta12'], true_params['theta13'],
                           true_params['theta23'], true_params.get('delta_cp', 0))

            H = hamiltonian_vacuum(U, true_params['delta_m2_21'],
                                  true_params['delta_m2_31'], E)

            U_evolution = evolution.evolve_constant_H(H, L)
            probs_true[idx] = np.abs(U_evolution[1, 0]) ** 2  # P(νe→νμ)
            idx += 1

    # 添加噪声
    probs_obs = probs_true + rng.normal(0, noise_level, n_obs)
    probs_obs = np.clip(probs_obs, 0, 1)

    return {
        'energies': energies,
        'baselines': baselines,
        'probs': probs_obs,
        'errors': np.ones(n_obs) * noise_level,
        'probs_true': probs_true,
    }
