"""
inverse_problem.py — PDE 约束优化反问题
============================================
来源项目: 1255_hyper-mpc_hypermpc_code (MPC 预测/优化框架),
          1259_EmperorAiphaton_MetricDistortion (LP 失真度量)

本模块实现 PDE 约束反问题的数学框架:
  min_{u,θ}  J(u, θ) = (1/2)||u - u_obs||²_Q + (α/2)||θ||²
  s.t.       F(u, θ) = 0   (PDE 约束)
             u|_Γ = g       (边界条件)

其中:
  u: 状态变量 (PDE 解)
  θ: 待估参数 (扩散系数、反应速率等)
  u_obs: 观测数据
  Q: 观测权重矩阵
  α: Tikhonov 正则化参数
  F: PDE 残差算子

伴随方程 (用于梯度计算):
  F_u^T λ = Q(u - u_obs)
  ∇_θ J = F_θ^T λ + αθ

其中 λ 为伴随变量 (Lagrange 乘子), F_u 和 F_θ 分别为
PDE 关于状态和参数的 Fréchet 导数.

L-BFGS 优化 (拟 Newton):
  θ_{k+1} = θ_k - α_k H_k ∇J(θ_k)
  其中 H_k 为 Hessian 逆的低秩近似:
  H_k = (I - ρ_k s_k y_k^T) H_{k-1} (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
  s_k = θ_{k+1} - θ_k,  y_k = ∇J_{k+1} - ∇J_k
  ρ_k = 1 / (y_k^T s_k)

在 ADMM 分解中:
  每个子域的局部反问题独立求解,
  接口处通过共识约束 θ_i = z 协调.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List, Dict
from config import EPS_NUM


# ============================================================
#  反问题配置
# ============================================================
class InverseProblemConfig:
    """PDE 约束反问题配置"""

    def __init__(self, n_params: int, n_observations: int,
                 alpha_reg: float = 1e-3,
                 max_outer_iter: int = 50,
                 max_inner_iter: int = 100,
                 grad_tol: float = 1e-6):
        self.n_params = n_params
        self.n_observations = n_observations
        self.alpha_reg = alpha_reg
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.grad_tol = grad_tol


# ============================================================
#  Tikhonov 正则化泛函
# ============================================================
def tikhonov_objective(params: np.ndarray, misfit: float,
                       alpha: float, prior: Optional[np.ndarray] = None) -> float:
    """Tikhonov 正则化目标泛函

    J(θ) = D(u(θ), u_obs) + (α/2) ||θ - θ_prior||²

    其中:
      D: 数据失配泛函 (如 L2 范数的平方)
      α: 正则化参数 (控制光滑度 vs 数据拟合的权衡)
      θ_prior: 先验参数估计

    最优 α 选择 (L-curve 方法):
      log D vs log R 曲线的拐点, 其中 R = ||θ - θ_prior||²

    L-curve 曲率:
      κ(α) = (D'R'' - D''R') / (D'² + R'²)^{3/2}
    """
    reg = 0.0
    if prior is not None:
        diff = params - prior
        reg = 0.5 * alpha * np.sum(diff ** 2)
    else:
        reg = 0.5 * alpha * np.sum(params ** 2)
    return misfit + reg


def tikhonov_gradient(params: np.ndarray, grad_misfit: np.ndarray,
                       alpha: float, prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Tikhonov 正则化梯度

    ∇J = ∇D + α(θ - θ_prior)
    """
    grad_reg = alpha * params
    if prior is not None:
        grad_reg = alpha * (params - prior)
    return grad_misfit + grad_reg


# ============================================================
#  L-BFGS 优化器
# ============================================================
class LBFGSOptimizer:
    """L-BFGS 拟 Newton 优化器

    L-BFGS (Limited-memory BFGS) 维护最近 m 步的 (s, y) 对:
      s_k = θ_{k+1} - θ_k
      y_k = g_{k+1} - g_k

    通过两段递推 (two-loop recursion) 计算 H_k g_k:
      q = g_k
      for i = k-1, k-2, ..., k-m:
        α_i = ρ_i s_i^T q
        q = q - α_i y_i
      r = H_0 q  (H_0 = γ_k I, γ_k = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1})
      for i = k-m, k-m+1, ..., k-1:
        β = ρ_i y_i^T r
        r = r + s_i (α_i - β)
      H_k g_k = r
    """

    def __init__(self, n_params: int, memory: int = 10):
        self.n_params = n_params
        self.memory = memory
        self.s_history: List[np.ndarray] = []
        self.y_history: List[np.ndarray] = []
        self.rho_history: List[float] = []
        self.prev_params: Optional[np.ndarray] = None
        self.prev_grad: Optional[np.ndarray] = None

    def compute_direction(self, grad: np.ndarray) -> np.ndarray:
        """计算搜索方向 d = -H_k g_k (L-BFGS 两段递推)"""
        m = len(self.s_history)

        if m == 0:
            # 最陡下降
            return -grad

        q = grad.copy()
        alphas = np.zeros(m)

        # 第一段递推 (从最新到最旧)
        for i in range(m - 1, -1, -1):
            alphas[i] = self.rho_history[i] * np.dot(self.s_history[i], q)
            q -= alphas[i] * self.y_history[i]

        # 初始 Hessian 近似: γ_k = s^T y / y^T y
        s_last = self.s_history[-1]
        y_last = self.y_history[-1]
        yy = np.dot(y_last, y_last)
        gamma = np.dot(s_last, y_last) / (yy + EPS_NUM) if yy > EPS_NUM else 1.0
        r = gamma * q

        # 第二段递推 (从最旧到最新)
        for i in range(m):
            beta = self.rho_history[i] * np.dot(self.y_history[i], r)
            r += self.s_history[i] * (alphas[i] - beta)

        return -r

    def update(self, params: np.ndarray, grad: np.ndarray):
        """更新 L-BFGS 记忆"""
        if self.prev_params is not None:
            s = params - self.prev_params
            y = grad - self.prev_grad
            sy = np.dot(s, y)

            if sy > EPS_NUM:  # 曲率条件
                if len(self.s_history) >= self.memory:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
                    self.rho_history.pop(0)
                self.s_history.append(s)
                self.y_history.append(y)
                self.rho_history.append(1.0 / sy)

        self.prev_params = params.copy()
        self.prev_grad = grad.copy()


# ============================================================
#  PDE 约束反问题求解器
# ============================================================
def solve_inverse_problem(forward_solver: Callable,
                           observation_operator: Callable,
                           observations: np.ndarray,
                           initial_params: np.ndarray,
                           config: InverseProblemConfig) -> dict:
    """求解 PDE 约束反问题 (L-BFGS)

    算法:
      1. 给定初始参数 θ_0
      2. 正演: 求解 F(u, θ_k) = 0 → u_k
      3. 计算数据失配: D = (1/2)||G(u_k) - u_obs||²
      4. 伴随: 求解 F_u^T λ = G^T(G(u_k) - u_obs) → λ_k
      5. 梯度: g_k = -F_θ^T λ_k + αθ_k
      6. L-BFGS 更新: θ_{k+1} = θ_k - η_k H_k g_k

    Args:
        forward_solver: 正演求解器 θ → u
        observation_operator: 观测算子 u → Gu
        observations: 观测数据
        initial_params: 初始参数
        config: 反问题配置

    Returns:
        dict: optimal_params, objective_history, gradient_history
    """
    params = initial_params.copy()
    alpha = config.alpha_reg
    optimizer = LBFGSOptimizer(config.n_params)

    obj_history = []
    grad_norm_history = []
    misfit_history = []

    for k in range(config.max_outer_iter):
        # 1. 正演求解
        u = forward_solver(params)

        # 2. 观测与失配
        Gu = observation_operator(u)
        residual = Gu - observations
        misfit = 0.5 * np.sum(residual ** 2)

        # 3. Tikhonov 目标
        obj = tikhonov_objective(params, misfit, alpha)
        obj_history.append(obj)
        misfit_history.append(misfit)

        # 4. 梯度 (简化: 使用有限差分近似)
        grad_misfit = _finite_difference_gradient(
            forward_solver, observation_operator, observations, params
        )
        grad = tikhonov_gradient(params, grad_misfit, alpha)
        grad_norm = np.linalg.norm(grad)
        grad_norm_history.append(grad_norm)

        # 5. 收敛判据
        if grad_norm < config.grad_tol:
            break

        # 6. L-BFGS 方向
        direction = optimizer.compute_direction(grad)

        # 7. 线搜索 (回溯 Armijo)
        step_size = _armijo_line_search(
            lambda p: tikhonov_objective(
                p,
                _compute_misfit(forward_solver, observation_operator, observations, p),
                alpha
            ),
            params, direction, obj, grad
        )

        # 8. 更新参数
        params = params + step_size * direction
        optimizer.update(params, grad)

    return {
        'optimal_params': params,
        'objective_history': obj_history,
        'gradient_norm_history': grad_norm_history,
        'misfit_history': misfit_history,
        'n_iterations': k + 1,
        'final_gradient_norm': grad_norm_history[-1] if grad_norm_history else float('inf'),
    }


# ============================================================
#  辅助函数
# ============================================================
def _finite_difference_gradient(forward_solver: Callable,
                                 observation_operator: Callable,
                                 observations: np.ndarray,
                                 params: np.ndarray,
                                 eps: float = 1e-6) -> np.ndarray:
    """有限差分梯度近似 (中心差分)

    ∂J/∂θ_i ≈ (J(θ + εe_i) - J(θ - εe_i)) / (2ε)
    """
    n = len(params)
    grad = np.zeros(n)

    for i in range(n):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps

        u_plus = forward_solver(params_plus)
        u_minus = forward_solver(params_minus)

        Gu_plus = observation_operator(u_plus)
        Gu_minus = observation_operator(u_minus)

        misfit_plus = 0.5 * np.sum((Gu_plus - observations)**2)
        misfit_minus = 0.5 * np.sum((Gu_minus - observations)**2)

        grad[i] = (misfit_plus - misfit_minus) / (2.0 * eps)

    return grad


def _compute_misfit(forward_solver: Callable,
                     observation_operator: Callable,
                     observations: np.ndarray,
                     params: np.ndarray) -> float:
    """计算数据失配"""
    u = forward_solver(params)
    Gu = observation_operator(u)
    return 0.5 * np.sum((Gu - observations)**2)


def _armijo_line_search(f: Callable, x: np.ndarray, d: np.ndarray,
                         f0: float, grad: np.ndarray,
                         c1: float = 1e-4, rho: float = 0.5,
                         max_ls: int = 20) -> float:
    """Armijo 回溯线搜索

    找到满足 Armijo 条件的步长 α:
      f(x + αd) ≤ f(x) + c₁ α ∇f^T d

    从 α=1 开始, 每次乘以 ρ.
    """
    alpha = 1.0
    slope = np.dot(grad, d)

    for _ in range(max_ls):
        x_new = x + alpha * d
        f_new = f(x_new)
        if f_new <= f0 + c1 * alpha * slope:
            return alpha
        alpha *= rho

    return alpha


# ============================================================
#  LP 失真度量 (来源: 1259_EmperorAiphaton_MetricDistortion)
# ============================================================
def compute_metric_distortion(cost_matrix: np.ndarray,
                               weights: np.ndarray) -> float:
    """计算加权度量失真 (来源: 1259)

    度量失真 (metric distortion):
      D = max_{i,j} (c(i,j) * w_i) / min_k c(k,j)

    其中 c 为成本矩阵, w 为权重向量.

    在 ADMM 中用于衡量子域间解的不一致性:
      distortion = max_interface ||x_i - x_j|| / min_interface ||x_i - x_j||

    Returns:
        失真值 (≥ 1)
    """
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return 1.0

    max_cost = 0.0
    min_cost = float('inf')

    for j in range(m):
        col = cost_matrix[:, j]
        weighted = col * weights[:n] if len(weights) >= n else col
        col_max = np.max(weighted)
        col_min = np.min(col[col > EPS_NUM]) if np.any(col > EPS_NUM) else EPS_NUM
        max_cost = max(max_cost, col_max)
        min_cost = min(min_cost, col_min)

    if min_cost < EPS_NUM:
        return float('inf')
    return max_cost / min_cost
