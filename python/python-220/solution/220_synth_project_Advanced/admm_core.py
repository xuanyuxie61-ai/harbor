"""
admm_core.py — 分布式 ADMM 核心算法
============================================
来源项目: 1255_hyper-mpc_hypermpc_code (MPC 框架/参数预测)

本模块实现分布式 ADMM (Alternating Direction Method of Multipliers) 的核心算法.

标准 ADMM 形式:
  min  f(x) + g(z)
  s.t. Ax + Bz = c

增广 Lagrangian:
  L_ρ(x, z, λ) = f(x) + g(z) + (ρ/2)||Ax + Bz - c + λ||²

ADMM 迭代:
  x^{k+1} = argmin_x L_ρ(x, z^k, λ^k)
  z^{k+1} = argmin_z L_ρ(x^{k+1}, z, λ^k)
  λ^{k+1} = λ^k + (Ax^{k+1} + Bz^{k+1} - c)

Consensus ADMM (分布式形式):
  min  Σ_i f_i(x_i)
  s.t. x_i = z  (对所有子域 i)

  x-更新: x_i^{k+1} = argmin_{x_i} [f_i(x_i) + (ρ/2)||x_i - z^k + u_i^k||²]
  z-更新: z^{k+1} = (1/N) Σ_i x_i^{k+1}  (均值)
  u-更新: u_i^{k+1} = u_i^k + x_i^{k+1} - z^{k+1}

收敛条件 (Boyd et al. 2011):
  原始残差: r^k = Ax^k + Bz^k - c → 0
  对偶残差: s^k = ρA^T B(z^k - z^{k-1}) → 0
  停止条件: ||r^k|| ≤ ε_pri  and  ||s^k|| ≤ ε_dual

其中:
  ε_pri = √n ε_abs + ε_rel max(||Ax^k||, ||Bz^k||, ||c||)
  ε_dual = √p ε_abs + ε_rel ||ρ A^T λ^k||

超松弛 (over-relaxation):
  x̂^{k+1} = α x^{k+1} + (1-α)(c - Bz^k)  (α ∈ (1, 2))
  用 x̂ 替代 Ax^{k+1} 更新对偶变量.
"""

import numpy as np
from typing import Callable, Optional, List, Dict, Tuple
from config import ADMMConfig, EPS_NUM


# ============================================================
#  ADMM 数据结构
# ============================================================
class ADMMState:
    """ADMM 迭代状态

    Attributes:
        x_locals: 各子域局部变量列表
        z_consensus: 全局共识变量
        u_duals: 各子域对偶变量列表 (scaled form: u = λ/ρ)
        rho: 当前罚参数
        iteration: 当前迭代次数
        primal_residuals: 原始残差历史
        dual_residuals: 对偶残差历史
        objective_values: 目标函数历史
    """

    def __init__(self, n_subdomains: int, n_vars: int):
        self.n_subdomains = n_subdomains
        self.n_vars = n_vars

        # 初始化变量
        self.x_locals = [np.zeros(n_vars) for _ in range(n_subdomains)]
        self.z_consensus = np.zeros(n_vars)
        self.u_duals = [np.zeros(n_vars) for _ in range(n_subdomains)]

        # 历史记录
        self.primal_residuals: List[float] = []
        self.dual_residuals: List[float] = []
        self.objective_values: List[float] = []
        self.rho_history: List[float] = []
        self.iteration = 0


# ============================================================
#  共识 ADMM 求解器
# ============================================================
def consensus_admm(local_objectives: List[Callable],
                    n_vars: int,
                    config: ADMMConfig,
                    x_init: Optional[List[np.ndarray]] = None,
                    callback: Optional[Callable] = None) -> dict:
    """分布式 Consensus ADMM 求解器

    问题形式:
      min  Σ_{i=1}^{N} f_i(x_i)
      s.t. x_i = z  for all i

    算法 (Boyd et al. 2011):
      for k = 0, 1, 2, ...
        // x-更新 (各子域独立求解)
        for i = 1, ..., N:
          x_i^{k+1} = argmin_{x_i} [f_i(x_i) + (ρ/2)||x_i - z^k + u_i^k||²]

        // z-更新 (共识)
        x_bar^{k+1} = (1/N) Σ_i x_i^{k+1}
        z^{k+1} = x_bar^{k+1}

        // 对偶更新 (scaled form)
        u_i^{k+1} = u_i^k + x_i^{k+1} - z^{k+1}

    Args:
        local_objectives: 各子域局部目标函数列表
        n_vars: 变量维度
        config: ADMM 参数配置
        x_init: 初始值 (可选)
        callback: 每步回调函数

    Returns:
        dict: x_opt, z_opt, state (含完整历史)
    """
    N = len(local_objectives)
    state = ADMMState(N, n_vars)

    # 初始化
    if x_init is not None:
        for i in range(N):
            state.x_locals[i] = x_init[i].copy()
    state.z_consensus = np.mean(state.x_locals, axis=0)

    rho = config.rho
    z_prev = state.z_consensus.copy()

    for k in range(config.max_iter):
        state.iteration = k

        # ---- x-更新 (各子域独立) ----
        for i in range(N):
            x_i = _solve_local_subproblem(
                local_objectives[i], state.z_consensus, state.u_duals[i], rho
            )
            state.x_locals[i] = x_i

        # ---- z-更新 (共识/均值) ----
        z_prev = state.z_consensus.copy()
        state.z_consensus = np.mean(state.x_locals, axis=0)

        # ---- 超松弛 (over-relaxation) ----
        # x_hat = α x + (1-α)(z - u)
        alpha = config.alpha_overrelax
        if alpha != 1.0:
            x_hat = [alpha * x_i + (1.0 - alpha) * state.z_consensus
                     for x_i in state.x_locals]
        else:
            x_hat = state.x_locals

        # ---- 对偶更新 ----
        for i in range(N):
            state.u_duals[i] = state.u_duals[i] + x_hat[i] - state.z_consensus

        # ---- 残差计算 ----
        # 原始残差: r_i = x_i - z
        primal_res = np.sqrt(sum(np.sum((x_i - state.z_consensus)**2) for x_i in state.x_locals))
        # 对偶残差: s = ρ(z^k - z^{k-1})
        dual_res = rho * np.sqrt(n_vars) * np.linalg.norm(state.z_consensus - z_prev)

        state.primal_residuals.append(primal_res)
        state.dual_residuals.append(dual_res)
        state.rho_history.append(rho)

        # 目标值
        total_obj = sum(local_objectives[i](state.x_locals[i]) for i in range(N))
        state.objective_values.append(total_obj)

        # 回调
        if callback is not None:
            callback(k, state)

        # ---- 收敛判据 ----
        eps_pri = np.sqrt(n_vars * N) * config.abs_tol + config.rel_tol * max(
            np.sqrt(sum(np.sum(x_i**2) for x_i in state.x_locals)),
            np.sqrt(N * n_vars) * np.linalg.norm(state.z_consensus)
        )
        eps_dual = np.sqrt(n_vars) * config.abs_tol + config.rel_tol * rho * np.linalg.norm(
            np.concatenate(state.u_duals)
        )

        if primal_res < eps_pri and dual_res < eps_dual:
            break

        # ---- 自适应罚参数 ----
        if config.use_adaptive_penalty:
            rho = _adaptive_penalty_update(rho, primal_res, dual_res, config)

    return {
        'z_opt': state.z_consensus,
        'x_opt': state.x_locals,
        'u_opt': state.u_duals,
        'n_iterations': state.iteration + 1,
        'state': state,
        'converged': (state.primal_residuals[-1] < eps_pri if state.primal_residuals else False),
    }


# ============================================================
#  局部子问题求解
# ============================================================
def _solve_local_subproblem(objective: Callable, z: np.ndarray,
                             u: np.ndarray, rho: float) -> np.ndarray:
    """求解 ADMM 局部子问题

    min_x  f(x) + (ρ/2)||x - z + u||²

    展开:
      = f(x) + (ρ/2)(x^T x - 2x^T(z-u) + ||z-u||²)
      = f(x) + (ρ/2)||x - (z-u)||² + const

    对于二次目标 f(x) = (1/2)x^T Q x + c^T x:
      最优解: x* = (Q + ρI)^{-1} (ρ(z-u) - c)

    对于一般目标: 使用梯度下降或 NM.

    此处实现: 近端算子 (proximal operator) 的梯度近似.
    """
    n = len(z)
    x = z - u  # 增广项的中心

    # 如果目标函数可微, 用梯度法; 否则用 NM
    # 尝试少量梯度步
    lr = 0.1 / (rho + 1.0)
    for _ in range(5):
        grad = _numerical_gradient(objective, x)
        residual = rho * (x - z + u)
        total_grad = grad + residual
        x = x - lr * total_grad

    return x


def _numerical_gradient(f: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """数值梯度 (中心差分)"""
    n = len(x)
    grad = np.zeros(n)
    f0 = f(x)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * eps)
    return grad


# ============================================================
#  自适应罚参数更新
# ============================================================
def _adaptive_penalty_update(rho: float, primal_res: float, dual_res: float,
                              config: ADMMConfig) -> float:
    """自适应罚参数更新策略 (Boyd et al. 2011, Section 3.4.1)

    残差比率: μ = ||r|| / ||s||

    若 μ > μ_threshold: (原始残差过大 → 增大 ρ)
      ρ ← τ_incr * ρ
      λ ← λ / τ_incr  (维持 λ/ρ 的一致性)

    若 μ < 1/μ_threshold: (对偶残差过大 → 减小 ρ)
      ρ ← ρ / τ_decr

    理论保证: 自适应策略不影响 ADMM 的收敛性, 但可显著加速.
    """
    mu = primal_res / (dual_res + EPS_NUM)

    if mu > config.mu:
        rho = min(rho * config.tau_incr, config.rho_max)
    elif mu < 1.0 / config.mu:
        rho = max(rho / config.tau_decr, config.rho_min)

    return rho


# ============================================================
#  全局 ADMM (非 consensus, 标准分裂)
# ============================================================
def standard_admm(f_obj: Callable, g_obj: Callable,
                   A: np.ndarray, B: np.ndarray, c_vec: np.ndarray,
                   config: ADMMConfig,
                   x0: Optional[np.ndarray] = None,
                   z0: Optional[np.ndarray] = None) -> dict:
    """标准 ADMM (分裂形式)

    min  f(x) + g(z)
    s.t. Ax + Bz = c

    迭代:
      x^{k+1} = argmin f(x) + (ρ/2)||Ax + Bz^k - c + u^k||²
      z^{k+1} = argmin g(z) + (ρ/2)||Ax^{k+1} + Bz - c + u^k||²
      u^{k+1} = u^k + Ax^{k+1} + Bz^{k+1} - c
    """
    n = A.shape[1] if A.ndim > 1 else len(A)
    p = B.shape[1] if B.ndim > 1 else len(B)

    x = x0.copy() if x0 is not None else np.zeros(n)
    z = z0.copy() if z0 is not None else np.zeros(p)
    u = np.zeros(p)  # scaled dual variable

    rho = config.rho
    history = {'primal_res': [], 'dual_res': [], 'objective': []}

    for k in range(config.max_iter):
        z_old = z.copy()

        # x-update (简化: 梯度法)
        x = _solve_standard_subproblem_x(f_obj, A, B, z, c_vec, u, rho, n)

        # z-update
        z = _solve_standard_subproblem_z(g_obj, A, x, B, c_vec, u, rho, p)

        # dual update
        u = u + A @ x + B @ z - c_vec

        # residuals
        primal_res = np.linalg.norm(A @ x + B @ z - c_vec)
        dual_res = rho * np.linalg.norm(A.T @ B @ (z - z_old))

        history['primal_res'].append(primal_res)
        history['dual_res'].append(dual_res)
        history['objective'].append(f_obj(x) + g_obj(z))

        # convergence
        if primal_res < config.abs_tol and dual_res < config.abs_tol:
            break

    return {
        'x_opt': x,
        'z_opt': z,
        'u_opt': u,
        'n_iterations': k + 1,
        'history': history,
    }


def _solve_standard_subproblem_x(f_obj: Callable, A: np.ndarray,
                                   B: np.ndarray, z: np.ndarray,
                                   c_vec: np.ndarray, u: np.ndarray,
                                   rho: float, n: int) -> np.ndarray:
    """x-子问题: min f(x) + (ρ/2)||Ax + Bz - c + u||²"""
    x = np.zeros(n)
    lr = 0.01 / (rho + 1.0)
    AtA = A.T @ A if A.ndim > 1 else np.outer(A, A)

    for _ in range(10):
        grad_f = _numerical_gradient(f_obj, x)
        Ax = A @ x if A.ndim > 1 else A * x
        residual = A.T @ (Ax + B @ z - c_vec + u) if A.ndim > 1 else A * (Ax + B @ z - c_vec + u)
        grad = grad_f + rho * residual
        x = x - lr * grad
    return x


def _solve_standard_subproblem_z(g_obj: Callable, A: np.ndarray,
                                   x: np.ndarray, B: np.ndarray,
                                   c_vec: np.ndarray, u: np.ndarray,
                                   rho: float, p: int) -> np.ndarray:
    """z-子问题: min g(z) + (ρ/2)||Ax + Bz - c + u||²"""
    z = np.zeros(p)
    lr = 0.01 / (rho + 1.0)

    for _ in range(10):
        grad_g = _numerical_gradient(g_obj, z)
        Bz = B @ z if B.ndim > 1 else B * z
        Ax = A @ x if A.ndim > 1 else A * x
        residual = B.T @ (Ax + Bz - c_vec + u) if B.ndim > 1 else B * (Ax + Bz - c_vec + u)
        grad = grad_g + rho * residual
        z = z - lr * grad
    return z
