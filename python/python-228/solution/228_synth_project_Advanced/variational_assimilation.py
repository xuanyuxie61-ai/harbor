"""
variational_assimilation.py — 变分数据同化 (4D-Var)
=====================================================

融合种子项目:
  - 1124_rspence821505_Variational-Data-Consistent-Assimilation

本模块实现 4D-Var 方法用于将蒙特卡罗模拟结果与解析剖面进行融合:

1. 代价函数:
   J(x0) = 0.5 * (x0 - xb)^T B^{-1} (x0 - xb)
           + 0.5 * sum_k (H_k * M_{0->k}(x0) - y_k)^T R_k^{-1} (...)

   其中:
     x0     : 初始状态 (shower 参数)
     xb     : 背景场 (先验估计)
     B      : 背景误差协方差
     M      : 非线性模型 (级联方程求解器)
     H_k    : 观测算子 (在观测点采样)
     y_k    : 观测值 (MC 模拟结果)
     R_k    : 观测误差协方差

2. 伴随模型 (用于梯度计算):
   dJ/dx0 = B^{-1} * (x0 - xb) + sum_k M_{0->k}^T * H_k^T * R_k^{-1} * (...)

3. 增量 4D-Var:
   x0^{n+1} = x0^n + alpha_n * delta_x
   delta_x = -B * grad_J  (预条件梯度下降)

4. 物理约束:
   总能量守恒: sum_k dE_k = E0
   非负性:     dE_k >= 0  (所有深度)
   光滑性:     ||d^2 E / dt^2|| <= C
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from material_properties import MaterialSpec


@dataclass
class AssimilationConfig:
    """数据同化配置"""
    n_iterations: int = 50        # 最大迭代次数
    convergence_tol: float = 1e-6 # 收敛阈值
    bg_error_std: float = 0.1     # 背景误差标准差 (相对)
    obs_error_std: float = 0.05   # 观测误差标准差 (相对)
    regularization: float = 0.01  # Tikhonov 正则化参数
    learning_rate: float = 0.5    # 梯度下降步长
    lbfgs_memory: int = 10        # L-BFGS 记忆长度


@dataclass
class AssimilationResult:
    """数据同化结果"""
    optimal_parameters: List[float]    # 最优参数
    cost_function_history: List[float] # J 的迭代历史
    gradient_norm_history: List[float] # ||grad J|| 历史
    converged: bool
    n_iterations_used: int
    final_cost: float
    analysis_profile: List[float]      # 分析后的剖面
    background_profile: List[float]    # 背景剖面
    observation_profile: List[float]   # 观测剖面 (MC)


class VariationalAssimilator:
    """
    4D-Var 变分数据同化器

    将解析纵向剖面 (背景) 与蒙特卡罗模拟结果 (观测) 融合,
    得到最优的 shower 参数估计。
    """

    def __init__(
        self,
        material: MaterialSpec,
        config: Optional[AssimilationConfig] = None,
    ):
        self.mat = material
        self.config = config or AssimilationConfig()

    def run_assimilation(
        self,
        background_params: List[float],
        observations: List[float],
        obs_depths: List[float],
        forward_model: Callable[[List[float], List[float]], List[float]],
    ) -> AssimilationResult:
        """
        运行 4D-Var 数据同化

        参数:
          background_params: 初始参数 [t_max, sigma, amplitude]
          observations:      观测值 (MC 模拟的 dE/dt)
          obs_depths:        观测深度 [X0]
          forward_model:     正向模型 (参数 -> 剖面)

        返回:
          AssimilationResult
        """
        cfg = self.config
        n_params = len(background_params)
        n_obs = len(observations)

        # 初始化
        x = list(background_params)
        xb = list(background_params)

        # 背景协方差 (对角)
        B_diag = [(cfg.bg_error_std * abs(xb[i]) + 1e-6) ** 2 for i in range(n_params)]

        # 观测协方差 (对角)
        R_diag = [(cfg.obs_error_std * max(abs(observations[k]), 1e-6)) ** 2 for k in range(n_obs)]

        # 背景剖面
        bg_profile = forward_model(xb, obs_depths)

        # L-BFGS 存储
        s_history = []  # delta_x
        y_history = []  # delta_grad
        grad_prev = None

        cost_history = []
        grad_norm_history = []

        for iteration in range(cfg.n_iterations):
            # 正向模拟
            model_output = forward_model(x, obs_depths)

            # 计算代价函数
            # J_b = 0.5 * sum_i (x_i - xb_i)^2 / B_i
            J_b = 0.0
            for i in range(n_params):
                J_b += 0.5 * (x[i] - xb[i]) ** 2 / B_diag[i]

            # J_o = 0.5 * sum_k (H_k(x) - y_k)^2 / R_k
            J_o = 0.0
            innovation = [0.0] * n_obs
            for k in range(n_obs):
                innovation[k] = model_output[k] - observations[k]
                J_o += 0.5 * innovation[k] ** 2 / R_diag[k]

            # 正则化
            J_reg = cfg.regularization * sum(xi ** 2 for xi in x)

            J_total = J_b + J_o + J_reg
            cost_history.append(J_total)

            # 计算梯度 (伴随模型)
            grad = self._compute_gradient(
                x, xb, B_diag, model_output, observations, R_diag,
                obs_depths, forward_model, n_params, n_obs,
            )

            grad_norm = math.sqrt(sum(g * g for g in grad))
            grad_norm_history.append(grad_norm)

            # 收敛检查
            if grad_norm < cfg.convergence_tol:
                return AssimilationResult(
                    optimal_parameters=x,
                    cost_function_history=cost_history,
                    gradient_norm_history=grad_norm_history,
                    converged=True,
                    n_iterations_used=iteration + 1,
                    final_cost=J_total,
                    analysis_profile=model_output,
                    background_profile=bg_profile,
                    observation_profile=list(observations),
                )

            # L-BFGS 方向
            if iteration > 0 and grad_prev is not None:
                delta_x = [x[j] - x_prev[j] for j in range(n_params)]
                delta_g = [grad[j] - grad_prev[j] for j in range(n_params)]
                sy = sum(s * y for s, y in zip(delta_x, delta_g))
                if sy > 1e-15:
                    s_history.append(delta_x)
                    y_history.append(delta_g)
                    if len(s_history) > cfg.lbfgs_memory:
                        s_history.pop(0)
                        y_history.pop(0)

                direction = self._lbfgs_direction(grad, s_history, y_history)
            else:
                # 预条件梯度 (对角 B)
                direction = [-B_diag[i] * grad[i] for i in range(n_params)]

            # 线搜索 (Armijo 回溯)
            alpha = cfg.learning_rate
            x_prev = list(x)
            grad_prev = list(grad)

            for _ in range(20):
                x_trial = [x[i] + alpha * direction[i] for i in range(n_params)]
                trial_output = forward_model(x_trial, obs_depths)

                J_trial_b = sum(0.5 * (x_trial[i] - xb[i]) ** 2 / B_diag[i]
                               for i in range(n_params))
                J_trial_o = sum(0.5 * (trial_output[k] - observations[k]) ** 2 / R_diag[k]
                               for k in range(n_obs))
                J_trial = J_trial_b + J_trial_o

                if J_trial < J_total - 1e-4 * alpha * sum(g * d for g, d in zip(grad, direction)):
                    break
                alpha *= 0.5

            x = [x[i] + alpha * direction[i] for i in range(n_params)]

            # 物理约束: 参数非负
            x = [max(xi, 1e-6) for xi in x]

        # 未收敛
        final_output = forward_model(x, obs_depths)
        return AssimilationResult(
            optimal_parameters=x,
            cost_function_history=cost_history,
            gradient_norm_history=grad_norm_history,
            converged=False,
            n_iterations_used=cfg.n_iterations,
            final_cost=cost_history[-1] if cost_history else float('inf'),
            analysis_profile=final_output,
            background_profile=bg_profile,
            observation_profile=list(observations),
        )

    def _compute_gradient(
        self, x, xb, B_diag, model_output, observations, R_diag,
        obs_depths, forward_model, n_params, n_obs,
    ) -> List[float]:
        """
        计算代价函数梯度 (有限差分近似伴随)

        dJ/dx_i = (x_i - xb_i)/B_i
                  + sum_k (H_k(x) - y_k)/R_k * dH_k/dx_i

        其中 dH_k/dx_i 用有限差分近似:
          dH_k/dx_i ≈ (H_k(x + e_i*h) - H_k(x - e_i*h)) / (2*h)
        """
        grad = [0.0] * n_params
        eps = 1e-5

        for i in range(n_params):
            # 背景项
            grad[i] += (x[i] - xb[i]) / B_diag[i]

            # 观测项 (有限差分)
            x_plus = list(x)
            x_minus = list(x)
            h = max(abs(x[i]) * eps, eps)
            x_plus[i] += h
            x_minus[i] -= h

            out_plus = forward_model(x_plus, obs_depths)
            out_minus = forward_model(x_minus, obs_depths)

            for k in range(n_obs):
                innovation = model_output[k] - observations[k]
                dH_dxi = (out_plus[k] - out_minus[k]) / (2.0 * h)
                grad[i] += innovation / R_diag[k] * dH_dxi

        return grad

    def _lbfgs_direction(
        self, grad: List[float],
        s_history: List[List[float]],
        y_history: List[List[float]],
    ) -> List[float]:
        """
        L-BFGS 两步递推计算搜索方向

        算法 (Nocedal & Wright):
          q = grad
          for i = k-1, ..., k-m:
            alpha_i = rho_i * s_i^T * q
            q = q - alpha_i * y_i
          r = H0 * q  (H0 = gamma * I)
          for i = k-m, ..., k-1:
            beta = rho_i * y_i^T * r
            r = r + s_i * (alpha_i - beta)
          direction = -r
        """
        n = len(grad)
        m = len(s_history)

        if m == 0:
            return [-g for g in grad]

        # rho_i = 1 / (y_i^T * s_i)
        rhos = []
        for s, y in zip(s_history, y_history):
            sy = sum(si * yi for si, yi in zip(s, y))
            rhos.append(1.0 / max(sy, 1e-15))

        # 前向递推
        q = list(grad)
        alphas = [0.0] * m
        for i in range(m - 1, -1, -1):
            alphas[i] = rhos[i] * sum(s_history[i][j] * q[j] for j in range(n))
            for j in range(n):
                q[j] -= alphas[i] * y_history[i][j]

        # 初始 Hessian 近似
        yy = sum(y_history[-1][j] ** 2 for j in range(n))
        sy = sum(s_history[-1][j] * y_history[-1][j] for j in range(n))
        gamma = sy / max(yy, 1e-15)
        r = [gamma * q[j] for j in range(n)]

        # 后向递推
        for i in range(m):
            beta = rhos[i] * sum(y_history[i][j] * r[j] for j in range(n))
            for j in range(n):
                r[j] += s_history[i][j] * (alphas[i] - beta)

        return [-ri for ri in r]


def shower_forward_model(
    material: MaterialSpec,
    E0_MeV: float,
) -> Callable[[List[float], List[float]], List[float]]:
    """
    构建 shower 正向模型 (参数化剖面)

    参数: [t_max_shift, sigma_scale, amplitude]
    返回: 在给定深度上的能量沉积剖面

    剖面模型:
      Gamma(t) = A * (t/t_max)^a * exp(b*(t_max - t))
    """
    t_max_base = material.critical_depth_X0(E0_MeV)

    def model(params: List[float], depths: List[float]) -> List[float]:
        t_max_shift = params[0]
        sigma_scale = max(params[1], 0.1)
        amplitude = max(params[2], 0.01)

        t_max = t_max_base + t_max_shift
        a = max(math.log(E0_MeV / material.Ec_MeV) / math.log(2.0), 0.5)
        b = 0.5 * sigma_scale

        profile = []
        for t in depths:
            if t <= 0 or t_max <= 0:
                profile.append(0.0)
                continue
            ratio = t / t_max
            log_val = (
                math.log(amplitude)
                + a * math.log(max(ratio, 1e-30))
                + b * (t_max - t)
            )
            if log_val > 500:
                profile.append(float('inf'))
            elif log_val < -700:
                profile.append(0.0)
            else:
                profile.append(math.exp(log_val))
        return profile

    return model
