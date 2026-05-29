"""
compass_search_calibration.py
冰盖流变参数无导数优化标定 — 罗盘搜索 (Compass Search)

基于种子项目 204_compass_search 与 1224_test_optimization 的直接搜索思想，
实现无需梯度信息的坐标搜索算法，用于标定 Glen 流动律参数 (A_0, Q, n) 及
各向异性增强因子，使模型预测的表面流速与观测数据之间的残差最小。

优化目标函数:
    J(\theta) = \frac{1}{2} \sum_{i=1}^{N_{obs}} w_i \left( u_i^{obs} - u_i^{model}(\theta) \right)^2
              + \frac{\lambda}{2} \| \theta - \theta_{prior} \|_{\Sigma^{-1}}^2

其中 \theta = [\log A_0, Q, n, E_{max}]^T 为待标定参数，
w_i 为观测权重，第二项为先验正则化。

算法: Compass Search (Coordinate Search)
  1. 初始化参数 \theta_0，步长 \Delta_0 > 0
  2. 对每个坐标方向 e_j 与 ± 方向进行探测:
       若 J(\theta_k \pm \Delta_k e_j) < J(\theta_k)，则接受该步
  3. 若所有方向均失败: \Delta_{k+1} = \Delta_k / 2 (收缩)
  4. 若某方向成功: \theta_{k+1} = \theta_{success}, \Delta 不变 (扩张可选)
  5. 终止: \Delta < \Delta_{tol} 或 k > k_{max}

数值特性:
  - 对非光滑、带噪声的目标函数具有鲁棒性
  - 保证收敛到一阶稳定点 (针对光滑函数)
  - 参数边界保护: 搜索中自动投影到可行域
"""

import numpy as np
from typing import Callable, Tuple, Optional


class CompassSearchOptimizer:
    """
    罗盘搜索优化器，用于冰盖流变参数标定。
    """

    def __init__(self,
                 theta_init: np.ndarray,
                 theta_lower: np.ndarray,
                 theta_upper: np.ndarray,
                 delta_init: float = 1.0,
                 delta_tol: float = 1e-8,
                 k_max: int = 10000,
                 contraction_factor: float = 0.5):
        """
        初始化优化器。

        参数:
            theta_init: 初始参数向量
            theta_lower: 参数下界
            theta_upper: 参数上界
            delta_init: 初始步长
            delta_tol: 步长容差 (收敛判据)
            k_max: 最大迭代次数
            contraction_factor: 收缩因子 (默认 0.5)
        """
        self.theta = np.asarray(theta_init, dtype=np.float64).copy()
        self.theta_lower = np.asarray(theta_lower, dtype=np.float64)
        self.theta_upper = np.asarray(theta_upper, dtype=np.float64)
        self.delta = float(delta_init)
        self.delta_tol = float(delta_tol)
        self.k_max = int(k_max)
        self.contraction_factor = float(contraction_factor)
        self.history = []

        if len(self.theta) != len(self.theta_lower) or len(self.theta) != len(self.theta_upper):
            raise ValueError("theta_init, theta_lower, theta_upper must have the same length.")

    def _project_to_bounds(self, theta: np.ndarray) -> np.ndarray:
        """将参数投影到可行域 [lower, upper]。"""
        return np.clip(theta, self.theta_lower, self.theta_upper)

    def _evaluate_objective(self,
                            obj_func: Callable[[np.ndarray], float],
                            theta: np.ndarray) -> float:
        """安全地评估目标函数，捕获异常并返回大值。"""
        try:
            val = float(obj_func(theta))
            if not np.isfinite(val):
                return 1e20
            return val
        except Exception:
            return 1e20

    def optimize(self, obj_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """
        执行罗盘搜索优化。

        参数:
            obj_func: 目标函数 J(\theta)

        返回:
            theta_opt: 最优参数
            J_opt: 最优目标值
        """
        m = len(self.theta)
        theta_k = self._project_to_bounds(self.theta.copy())
        J_k = self._evaluate_objective(obj_func, theta_k)

        self.history = [(theta_k.copy(), J_k, self.delta)]

        for k in range(self.k_max):
            if self.delta < self.delta_tol:
                break

            improved = False
            theta_candidate = theta_k.copy()
            J_candidate = J_k

            # 探测 2m 个坐标方向
            for j in range(m):
                for sign in [-1.0, 1.0]:
                    theta_trial = theta_k.copy()
                    theta_trial[j] += sign * self.delta
                    theta_trial = self._project_to_bounds(theta_trial)

                    # 避免重复评估同一点
                    if np.allclose(theta_trial, theta_k, atol=1e-14):
                        continue

                    J_trial = self._evaluate_objective(obj_func, theta_trial)

                    if J_trial < J_candidate:
                        J_candidate = J_trial
                        theta_candidate = theta_trial
                        improved = True

            if improved:
                theta_k = theta_candidate
                J_k = J_candidate
            else:
                self.delta *= self.contraction_factor

            self.history.append((theta_k.copy(), J_k, self.delta))

        self.theta = theta_k
        return theta_k, J_k


def build_calibration_objective(
    observed_velocities: np.ndarray,
    model_velocity_func: Callable[[np.ndarray], np.ndarray],
    weights: Optional[np.ndarray] = None,
    prior_theta: Optional[np.ndarray] = None,
    prior_sigma: Optional[np.ndarray] = None,
    regularization_lambda: float = 0.01
) -> Callable[[np.ndarray], float]:
    """
    构建标定用的目标函数 (加权最小二乘 + 先验正则化)。

    参数:
        observed_velocities: 观测流速 (N_obs,)
        model_velocity_func: 模型流速预测函数 model_vel = f(theta)
        weights: 观测权重 (N_obs,)
        prior_theta: 先验参数
        prior_sigma: 先验标准差
        regularization_lambda: 正则化强度

    返回:
        objective: 目标函数 callable(theta) -> float
    """
    obs = np.asarray(observed_velocities, dtype=np.float64)
    w = np.ones_like(obs) if weights is None else np.asarray(weights, dtype=np.float64)

    def objective(theta: np.ndarray) -> float:
        try:
            model_vel = np.asarray(model_velocity_func(theta), dtype=np.float64)
            if model_vel.shape != obs.shape:
                return 1e20
        except Exception:
            return 1e20

        residual = obs - model_vel
        misfit = 0.5 * np.sum(w * (residual ** 2))

        # 先验正则化
        if prior_theta is not None and prior_sigma is not None:
            prior_term = 0.5 * regularization_lambda * np.sum(
                ((theta - prior_theta) / prior_sigma) ** 2
            )
            misfit += prior_term

        return float(misfit)

    return objective


def demo_calibration_problem(theta_true: np.ndarray,
                              noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """
    生成一个演示标定问题: 已知真值参数，加噪声观测。

    模型:
        u_i^{model} = a \cdot \theta_0^{\theta_2} \cdot \exp(-\theta_1 / T_i)

    返回:
        theta_true, observed, model_func
    """
    theta_true = np.asarray(theta_true, dtype=np.float64)
    a0, q, n = theta_true[:3]

    # 合成温度序列
    T_data = np.linspace(240.0, 270.0, 50)

    def model_func(theta: np.ndarray) -> np.ndarray:
        a, qv, nv = theta[:3]
        u = a * (T_data ** nv) * np.exp(-qv / T_data)
        return u

    u_true = model_func(theta_true)
    noise = noise_level * np.mean(np.abs(u_true)) * np.random.randn(len(T_data))
    observed = u_true + noise

    return theta_true, observed, model_func
